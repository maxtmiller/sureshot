"""Many requests at once: nothing is lost, nothing errors, and identical requests ask the model once."""
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import httpx
import pytest
import uvicorn
from sureshot import Cache, Router, Store, review_queue, stats

Q = {"q": {"type": "noul", "instructions": "?"}}


class Model:
    """Tier and System 2 in one: counts the questions it is asked, takes a moment, and is confident about even states."""

    def __init__(self, identity, delay=0.002):
        self.identity, self.delay, self.asked, self.lock = identity, delay, [], threading.Lock()

    def _note(self, questions, state):
        with self.lock:
            self.asked.extend((state, key) for key in questions)
        time.sleep(self.delay)

    def decide(self, state, questions, images=()):
        self._note(questions, state)
        confidence = 0.95 if int(state.split()[-1]) % 2 == 0 else 0.5
        return {k: {"answer": True, "confidence": confidence, "probabilities": {}} for k in questions}

    def answer(self, state, questions, images=()):
        self._note(questions, state)
        return {k: {"answer": True, "reason": "r"} for k in questions}


def test_threads_sharing_a_router_lose_nothing(tmp_path):
    store, cache = Store(tmp_path / "s.db"), Cache(tmp_path / "c.db")
    tier, claude = Model("tier"), Model("claude")
    router = Router(tier, claude, threshold=0.8, log=store, audit=0.3, cache=cache, backoff=0)
    states = [f"state {i % 100}" for i in range(600)]  # 100 distinct states, each asked 6 times
    with ThreadPoolExecutor(24) as pool:
        results = list(pool.map(lambda s: router.decide(s, Q), states))
    ids = [r["id"] for r in results]
    assert len(set(ids)) == 600 and len(store.results()) == 600
    # Every state's tier question was asked once and cached for the repeats, and System 2 at most once per state.
    assert sorted(tier.asked) == sorted((f"state {i}", "q") for i in range(100))
    assert len({s for s, _ in claude.asked}) == len(claude.asked) <= 100
    assert all(r["answers"]["q"]["answer"] is True for r in results)
    report = stats(store)
    assert report["decisions"] == 600 and report["cache"]["hits"] + report["cache"]["misses"] >= 600


def test_identical_requests_at_once_ask_the_model_once(tmp_path):
    cache = Cache(tmp_path / "c.db")
    tier, claude = Model("tier", delay=0.1), Model("claude", delay=0.1)
    router = Router(tier, claude, threshold=0.8, cache=cache, backoff=0)
    start = threading.Barrier(16)

    def go(_):
        start.wait()
        return router.decide("state 1", Q)  # an odd state: unsure, so System 2 is asked too

    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(go, range(16)))
    assert len(tier.asked) == 1 and len(claude.asked) == 1
    assert all(r["escalated"] == ["q"] and r["answers"]["q"]["source"] == "system2" for r in results)
    assert router._flights == {}  # no lock is left behind


def test_overlapping_question_sets_neither_deadlock_nor_repeat_work(tmp_path):
    cache = Cache(tmp_path / "c.db")
    tier = Model("tier", delay=0.05)
    router = Router(tier, None, threshold=0.0, cache=cache)
    many = {name: {"type": "noul", "instructions": name} for name in "abcd"}
    asks = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "b", "c", "d"), ("d", "c", "b", "a")]
    with ThreadPoolExecutor(len(asks)) as pool:
        futures = [pool.submit(router.decide, "state 2", {k: many[k] for k in names}) for names in asks]
        for f in futures:
            f.result(timeout=10)  # a deadlock would time out here
    assert sorted(key for _, key in tier.asked) == ["a", "b", "c", "d"]


def test_a_failing_call_releases_the_waiting_requests(tmp_path):
    class Breaks(Model):
        def decide(self, state, questions, images=()):
            self._note(questions, state)
            if len(self.asked) == 1:
                raise ValueError("bad input")  # not an outage, so it is raised, not retried
            return super().decide(state, questions, images)

    router = Router(Breaks("tier", delay=0.05), None, threshold=0.0, cache=Cache(tmp_path / "c.db"))
    with ThreadPoolExecutor(4) as pool:
        futures = [pool.submit(router.decide, "state 2", Q) for _ in range(4)]
        outcomes = [f.exception(timeout=10) for f in futures]
    assert sum(isinstance(o, ValueError) for o in outcomes) == 1 and sum(o is None for o in outcomes) == 3
    assert router._flights == {}


def test_reads_do_not_block_writes(tmp_path):
    store, cache = Store(tmp_path / "s.db"), Cache(tmp_path / "c.db")
    for db in (store, cache):
        with db._db() as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    router = Router(Model("tier", delay=0), Model("claude", delay=0), threshold=0.8, log=store, audit=0.5)
    stop, errors = threading.Event(), []

    def read():
        while not stop.is_set():
            try:
                stats(store)
                review_queue(store)
            except Exception as e:  # noqa: BLE001 (any failure is the point of this test)
                errors.append(repr(e))

    readers = [threading.Thread(target=read, daemon=True) for _ in range(4)]
    [t.start() for t in readers]
    try:
        with ThreadPoolExecutor(12) as pool:
            ids = list(pool.map(lambda i: router.decide("x" * 5000 + f" state {i}", Q)["id"], range(300)))
    finally:
        stop.set()
        [t.join(timeout=10) for t in readers]
    assert errors == [] and len(set(ids)) == 300 and len(store.results()) == 300


def test_http_server_handles_concurrent_clients(tmp_path, monkeypatch):
    import sureshot.api as api

    monkeypatch.setenv("SURESHOT_API_KEY", "secret")
    store = Store(tmp_path / "s.db")
    monkeypatch.setattr(api, "_router", Router(Model("tier"), Model("claude"), threshold=0.8, log=store, audit=0.3, cache=Cache(tmp_path / "c.db")))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(api.app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.02)
    try:
        auth = {"Authorization": "Bearer secret"}

        def client(n):
            codes = []
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as c:
                for j in range(10):
                    r = c.post("/decide", json={"state": f"state {n * 10 + j}", "questions": Q}, headers=auth)
                    codes.append(r.status_code)
                    if j == 5:
                        codes.append(c.post("/feedback", json={"id": r.json()["id"], "labels": {"q": True}}, headers=auth).status_code)
                        codes.append(c.get("/stats", headers=auth).status_code)
            return codes

        with ThreadPoolExecutor(24) as pool:
            codes = [code for batch in pool.map(client, range(24)) for code in batch]
        assert set(codes) == {200} and len(codes) == 24 * 12
        assert len(store.results()) == 240 and len(store.labels()) == 24
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_model_calls_share_one_connection_pool(monkeypatch):
    # httpx.post builds a client per call, which holds the GIL for ~10 ms and capped a process near 30 model calls a
    # second. Every backend must go through the shared client instead.
    import sureshot.adapters.base as backends
    from sureshot import ChatDecider

    seen = []

    def handler(request):
        seen.append(request.url.path)
        top = [{"token": "yes", "logprob": -0.1}, {"token": "no", "logprob": -2.4}]
        return httpx.Response(200, json={"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(backends, "_client", client)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("a model call built its own client"))
    decider = ChatDecider("m", "http://local/v1")
    many = {f"q{i}": {"type": "noul", "instructions": "?"} for i in range(40)}
    with ThreadPoolExecutor(16) as pool:
        answers = list(pool.map(lambda _: decider.decide("state", many), range(4)))
    assert len(seen) == 160 and all(a["q7"]["answer"] is True for a in answers)
    assert backends._client is client and not client.is_closed
