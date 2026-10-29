import pytest
import hunch.adapters.base as backends
from hunch.adapters.base import normalize
from hunch.questions import answer_schema, validate
from hunch.router import Router

QUESTIONS = {
    "needs_prep": {"type": "noul", "instructions": "Does this event need preparation?"},
    "importance": {"type": "score", "instructions": "How important?", "criteria": ["Optional", "Low", "Normal", "High", "Critical"]},
    "event_type": {"type": "choice", "instructions": "What kind of event?", "criteria": {"meeting": "A meeting", "interview": "An interview"}},
}

# Raw System 1 output for each question type, as an adapter's parse function returns it.
RAW = {
    "needs_prep": {"type": "noul", "noul": 0.966},
    "importance": {"type": "score", "probabilities": {"0": 0.02, "1": 0.02, "2": 0.21, "3": 0.41, "4": 0.34}, "score": 3.03, "confidence": 0.456},
    "event_type": {"type": "choice", "probabilities": {"meeting": 0.02, "interview": 0.98}, "choice": "interview", "confidence": 0.976},
}


class FakeDecider:
    def decide(self, state, questions, images=()):
        return {k: normalize(questions[k], RAW[k]) for k in questions}


class FakeClaude:
    def __init__(self):
        self.asked = None

    def answer(self, state, questions, images=()):
        self.asked = list(questions)
        return {k: {"answer": 4 if questions[k]["type"] == "score" else True, "reason": "because"} for k in questions}


def test_normalize_each_type():
    a = {k: normalize(QUESTIONS[k], RAW[k]) for k in QUESTIONS}
    assert a["needs_prep"]["answer"] is True and a["needs_prep"]["confidence"] == pytest.approx(0.966)
    assert a["importance"]["answer"] == 3 and a["importance"]["confidence"] == 0.456 and a["importance"]["score"] == 3.03
    assert a["event_type"]["answer"] == "interview" and a["event_type"]["confidence"] == 0.976
    low = normalize(QUESTIONS["needs_prep"], {"noul": 0.1})
    assert low["answer"] is False and low["confidence"] == pytest.approx(0.9)


def test_http_decider_adapts_any_api(monkeypatch):
    import httpx
    from hunch import HTTPDecider

    seen = {}

    def post(url, json, headers, timeout):
        seen.update(url=url, json=json, headers=headers)
        return httpx.Response(200, json={"out": RAW}, request=httpx.Request("POST", url))

    monkeypatch.setattr(backends, "post", post)
    decider = HTTPDecider("https://example.test/classify", lambda s, q, i: {"text": s, "q": q},
                          lambda data, questions: data["out"], api_key="k")
    answers = decider.decide("event", QUESTIONS)
    assert seen["url"] == "https://example.test/classify" and seen["json"]["text"] == "event"
    assert seen["headers"] == {"Authorization": "Bearer k"}
    assert answers["event_type"]["answer"] == "interview" and answers["needs_prep"]["answer"] is True


def test_http_reasoner_adapts_any_api(monkeypatch):
    import httpx
    from hunch import HTTPReasoner, prompt

    seen = {}

    def post(url, json, headers, timeout):
        seen.update(url=url, json=json, headers=headers)
        return httpx.Response(200, json={"text": {"importance": {"answer": 4, "reason": "ok"}}}, request=httpx.Request("POST", url))

    monkeypatch.setattr(backends, "post", post)
    reasoner = HTTPReasoner("https://llm.test/answer", lambda s, q, i: {"input": prompt(s, q)}, lambda data, q: data["text"], api_key="k")
    result = Router(FakeDecider(), reasoner).decide("event", QUESTIONS)
    assert seen["url"] == "https://llm.test/answer" and "importance" in seen["json"]["input"] and "needs_prep" not in seen["json"]["input"]
    assert result["answers"]["importance"]["answer"] == 4 and result["answers"]["importance"]["source"] == "system2"


def test_only_unsure_questions_escalate():
    claude = FakeClaude()
    result = Router(FakeDecider(), claude, threshold=0.8).decide("event", QUESTIONS)
    assert result["unsure"] == ["importance"] and result["escalated"] == ["importance"] and claude.asked == ["importance"]
    assert result["answers"]["importance"]["source"] == "system2" and result["answers"]["importance"]["answer"] == 4
    assert result["answers"]["importance"]["hunch"]["answer"] == 3
    assert result["answers"]["event_type"]["source"] == "hunch"


def test_per_question_threshold_and_no_escalation():
    claude = FakeClaude()
    result = Router(FakeDecider(), claude, threshold=0.8).decide("event", QUESTIONS, thresholds={"importance": 0.4, "event_type": 0.99})
    assert result["unsure"] == ["event_type"] and claude.asked == ["event_type"]
    kept = Router(FakeDecider(), claude, threshold=0.8).decide("event", QUESTIONS, escalate=False)
    assert kept["unsure"] == ["importance"] and kept["escalated"] == [] and kept["answers"]["importance"]["source"] == "hunch"
    assert Router(FakeDecider(), None).decide("event", QUESTIONS)["escalated"] == []


def test_schema_and_validation():
    schema = answer_schema(QUESTIONS)
    assert schema["properties"]["event_type"]["properties"]["answer"]["enum"] == ["meeting", "interview"]
    assert schema["properties"]["importance"]["properties"]["answer"]["enum"] == [0, 1, 2, 3, 4]
    assert schema["properties"]["needs_prep"]["properties"]["answer"] == {"type": "boolean"}
    with pytest.raises(ValueError):
        validate({"x": {"type": "choice", "criteria": {}}})


def test_api_requires_token(monkeypatch):
    from fastapi.testclient import TestClient
    import hunch.api as api

    monkeypatch.setenv("HUNCH_API_KEY", "secret")
    monkeypatch.setattr(api, "_router", Router(FakeDecider(), FakeClaude()))
    client = TestClient(api.app)
    body = {"state": "event", "questions": QUESTIONS}
    assert client.post("/decide", json=body).status_code == 401
    assert client.post("/decide", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = client.post("/decide", json=body, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200 and ok.json()["escalated"] == ["importance"]
    bad = client.post("/decide", json={"state": "x", "questions": {}}, headers={"Authorization": "Bearer secret"})
    assert bad.status_code == 422


def _logprob_response(url, tops):
    import httpx

    content = [{"token": t, "logprob": lp} for t, lp in tops]
    return httpx.Response(200, json={"choices": [{"logprobs": {"content": [{"top_logprobs": content}]}}]}, request=httpx.Request("POST", url))


def test_chat_decider_turns_logprobs_into_confidence(monkeypatch):
    import math
    import httpx
    from hunch import ChatDecider

    sent = []

    def post(url, json, headers, timeout):
        sent.append(json)
        text = json["messages"][1]["content"][-1]["text"]
        if "yes or no" in text:
            return _logprob_response(url, [("Yes", math.log(0.9)), ("no", math.log(0.1)), ("The", math.log(0.01))])
        if "How important" in text:
            return _logprob_response(url, [("3", math.log(0.5)), ("4", math.log(0.5))])
        return _logprob_response(url, [(" B", math.log(0.96)), ("A", math.log(0.04))])

    monkeypatch.setattr(backends, "post", post)
    answers = ChatDecider("m", "http://local/v1", api_key="k").decide("event", QUESTIONS)
    assert answers["needs_prep"]["answer"] is True and answers["needs_prep"]["confidence"] == pytest.approx(0.9)
    assert answers["event_type"]["answer"] == "interview" and answers["event_type"]["confidence"] == pytest.approx(0.96)
    assert answers["importance"]["confidence"] == pytest.approx(0.5)
    assert all(s["logprobs"] and s["max_tokens"] == 1 for s in sent) and len(sent) == 3


def test_chat_reasoner_uses_structured_outputs(monkeypatch):
    import httpx
    from hunch import ChatReasoner

    seen = {}

    def post(url, json, headers, timeout):
        seen.update(url=url, json=json)
        body = {"choices": [{"message": {"content": '{"importance": {"answer": 4, "reason": "ok"}}'}}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(backends, "post", post)
    result = Router(FakeDecider(), ChatReasoner("m", "http://local/v1")).decide("event", QUESTIONS)
    assert seen["url"] == "http://local/v1/chat/completions" and seen["json"]["response_format"]["json_schema"]["strict"]
    assert result["answers"]["importance"]["answer"] == 4


def test_router_from_env(monkeypatch):
    from hunch import ChatDecider, ChatReasoner, router_from_env

    for k, v in {"HUNCH_SYSTEM1_MODEL": "a", "HUNCH_SYSTEM1_URL": "http://one/v1",
                 "HUNCH_SYSTEM2": "chat", "HUNCH_SYSTEM2_MODEL": "b", "HUNCH_SYSTEM2_URL": "http://two/v1", "HUNCH_THRESHOLD": "0.9"}.items():
        monkeypatch.setenv(k, v)
    router = router_from_env()
    assert isinstance(router.system1, ChatDecider) and router.system1.url == "http://one/v1/chat/completions"
    assert isinstance(router.system2, ChatReasoner) and router.threshold == 0.9
    monkeypatch.setenv("HUNCH_SYSTEM2", "none")
    assert router_from_env().system2 is None
    monkeypatch.setenv("HUNCH_SYSTEM2", "bogus")
    with pytest.raises(ValueError):
        router_from_env()


def _calibration_store(tmp_path):
    from hunch import Store

    return Store(tmp_path / "hunch.db")


class Scripted:
    """System 1 that answers the first question with a given confidence, and is right only when told so."""

    def __init__(self, confidence):
        self.confidence = confidence

    def decide(self, state, questions, images=()):
        c = self.confidence
        return {"needs_prep": {"answer": True, "confidence": c, "probabilities": {"true": c, "false": 1 - c}}}


def test_log_label_and_audit(tmp_path):
    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    # System 2 disagrees (False); audit=1 means every confident answer is also checked and weighted 1.
    class Slow:
        def answer(self, state, questions, images=()):
            return {k: {"answer": False, "reason": "no"} for k in questions}

    result = Router(Scripted(0.9), Slow(), log=store, audit=1.0).decide("event", q)
    assert result["escalated"] == [] and result["answers"]["needs_prep"]["source"] == "hunch"
    assert result["answers"]["needs_prep"]["audit"]["answer"] is False
    row = store.labeled()[0]
    assert row["kind"] == "proxy" and row["weight"] == 1.0 and row["truth"] is False and row["answer"] is True
    store.label(result["id"], "needs_prep", True)
    assert store.labeled()[0]["kind"] == "human" and store.labeled()[0]["truth"] is True
    with pytest.raises(ValueError):
        store.label(result["id"], "needs_prep", "yes")
    with pytest.raises(KeyError):
        store.label("nope", "needs_prep", True)


def test_calibrate_finds_threshold_and_router_uses_it(tmp_path):
    from hunch import calibrate

    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    # 0.95-confidence answers are always right; 0.6-confidence answers are right half the time.
    for i in range(120):
        confidence = 0.95 if i % 2 else 0.6
        result = Router(Scripted(confidence), None, log=store).decide("event", q)
        store.label(result["id"], "needs_prep", i % 4 != 0 if confidence == 0.6 else True)
    report = calibrate(store, target_accuracy=0.95, min_samples=30)["needs_prep"]
    assert report["threshold"] == 0.95 and report["trusted_accuracy"] == 1.0 and report["escalation_rate"] == pytest.approx(0.5)
    assert store.thresholds() == {"needs_prep": 0.95}
    claude = FakeClaude()
    # 0.9 was fine under the default 0.8 but is now below the learned 0.95 threshold.
    assert Router(Scripted(0.9), claude, log=store).decide("event", q)["escalated"] == ["needs_prep"]
    assert Router(Scripted(0.9), claude, log=store).decide("event", q, thresholds={"needs_prep": 0.5})["escalated"] == []
    assert calibrate(store, min_samples=1000)["needs_prep"]["threshold"] is None


def test_evaluate_compares_candidates(tmp_path):
    from hunch import evaluate

    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    for _ in range(40):
        store.label(Router(Scripted(0.9), None, log=store).decide("event", q)["id"], "needs_prep", True)
    report = evaluate(store, {"good": Scripted(0.99), "bad": FlippedScripted()}, min_samples=30)
    assert report["good"]["needs_prep"]["accuracy"] == 1.0 and report["good"]["needs_prep"]["threshold"] == 0.99
    assert report["bad"]["needs_prep"]["accuracy"] == 0.0 and report["bad"]["needs_prep"]["threshold"] is None


class FlippedScripted:
    def decide(self, state, questions, images=()):
        return {"needs_prep": {"answer": False, "confidence": 0.9, "probabilities": {}}}


def test_api_feedback_and_calibrate(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import hunch.api as api

    monkeypatch.setenv("HUNCH_API_KEY", "secret")
    store = _calibration_store(tmp_path)
    monkeypatch.setattr(api, "_router", Router(FakeDecider(), FakeClaude(), log=store))
    client, auth = TestClient(api.app), {"Authorization": "Bearer secret"}
    decision = client.post("/decide", json={"state": "event", "questions": QUESTIONS}, headers=auth).json()
    assert client.post("/feedback", json={"id": decision["id"], "labels": {"importance": 4}}).status_code == 401
    assert client.post("/feedback", json={"id": decision["id"], "labels": {"importance": 4}}, headers=auth).status_code == 200
    assert client.post("/feedback", json={"id": decision["id"], "labels": {"importance": 9}}, headers=auth).status_code == 422
    assert client.post("/feedback", json={"id": "nope", "labels": {"importance": 4}}, headers=auth).status_code == 404
    assert "importance" in client.post("/calibrate", json={}, headers=auth).json()
    monkeypatch.setattr(api, "_router", Router(FakeDecider()))
    assert client.post("/calibrate", json={}, headers=auth).status_code == 400


def test_latency_is_reported_and_logged(tmp_path):
    import time

    class Slow(FakeClaude):
        def answer(self, state, questions, images=()):
            time.sleep(0.02)
            return super().answer(state, questions, images)

    store = _calibration_store(tmp_path)
    result = Router(FakeDecider(), Slow(), log=store).decide("event", QUESTIONS)
    latency = result["latency_ms"]
    assert latency["system2"] >= 20 and latency["total"] >= latency["system2"] + latency["system1"] - 0.2
    assert Router(FakeDecider(), Slow()).decide("event", {"event_type": QUESTIONS["event_type"]})["latency_ms"]["system2"] is None


def test_golden_run(tmp_path):
    import json

    store = _calibration_store(tmp_path)
    router = Router(FakeDecider(), FakeClaude(), log=store)
    # System 1 says importance 3 (unsure), System 2 says 4: the second case expects 3, so System 2 gets it wrong.
    cases = [{"state": "a", "expected": {"importance": 4, "event_type": "interview", "needs_prep": True}},
             {"state": "b", "expected": {"importance": 3, "event_type": "meeting"}}]
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in cases))
    report = router.test(path, questions=QUESTIONS)
    assert report["cases"] == 2 and store.labeled() == []
    assert report["questions"]["importance"] == {"n": 2, "accuracy": 0.5, "system1_accuracy": 0.5, "escalation_rate": 1.0}
    assert report["questions"]["event_type"]["accuracy"] == 0.5 and report["questions"]["event_type"]["escalation_rate"] == 0.0
    assert report["overall"]["n"] == 5 and report["overall"]["accuracy"] == pytest.approx(3 / 5)
    assert {f["key"] for f in report["failures"]} == {"importance", "event_type"}
    assert report["latency_ms"]["system1"][50] >= 0 and report["latency_ms"]["system2"] is not None
    with pytest.raises(ValueError, match="case 0"):
        router.test([{"state": "a", "expected": {"importance": 9}}], questions=QUESTIONS)


def test_extra_fields_reach_the_request(monkeypatch):
    import httpx
    from hunch import ChatDecider

    seen = {}

    def post(url, json, headers, timeout):
        seen.update(json)
        return _logprob_response(url, [("yes", 0.0)])

    monkeypatch.setattr(backends, "post", post)
    ChatDecider("m", "http://local/v1", extra={"reasoning_effort": "none", "max_tokens": 2}).decide("e", {"needs_prep": QUESTIONS["needs_prep"]})
    assert seen["reasoning_effort"] == "none" and seen["max_tokens"] == 2


def test_check_explains_what_is_wrong(monkeypatch):
    import httpx
    from hunch import ChatDecider

    def respond(status=200, tops=None, logprobs=True):
        def post(url, json, headers, timeout):
            if status != 200:
                return httpx.Response(status, text="model not found", request=httpx.Request("POST", url))
            choice = {"logprobs": {"content": [{"top_logprobs": [{"token": t, "logprob": -0.1} for t in tops]}]}} if logprobs else {"logprobs": None}
            return httpx.Response(200, json={"choices": [choice]}, request=httpx.Request("POST", url))
        return post

    decider = ChatDecider("m", "http://local/v1")
    monkeypatch.setattr(backends, "post", respond(tops=["yes", "no"]))
    assert decider.check() is decider
    monkeypatch.setattr(backends, "post", respond(tops=["<|channel>", "The"]))
    with pytest.raises(RuntimeError, match=r"top tokens were \['<\|channel>', 'The'\].*extra"):
        decider.check()
    monkeypatch.setattr(backends, "post", respond(logprobs=False))
    with pytest.raises(RuntimeError, match="no logprobs"):
        decider.check()
    monkeypatch.setattr(backends, "post", respond(status=404))
    with pytest.raises(RuntimeError, match="404.*model not found"):
        decider.check()

    def unreachable(url, json, headers, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(backends, "post", unreachable)
    with pytest.raises(RuntimeError, match="cannot reach http://local/v1/chat/completions"):
        decider.check()


def test_vllm_decider_and_env_option(monkeypatch):
    from hunch import ChatDecider, VLLMDecider, router_from_env

    local = VLLMDecider("Qwen/Qwen3-8B", extra={"top_p": 1})
    assert local.url == "http://localhost:8000/v1/chat/completions" and local.headers == {}
    assert local.extra == {"chat_template_kwargs": {"enable_thinking": False}, "top_p": 1}
    monkeypatch.setenv("HUNCH_SYSTEM1_MODEL", "m")
    monkeypatch.setenv("HUNCH_SYSTEM2", "none")
    with pytest.raises(ValueError, match="base_url"):
        router_from_env()  # no OpenAI (or any) endpoint is assumed
    monkeypatch.setenv("HUNCH_SYSTEM1_URL", "http://one/v1")
    assert type(router_from_env().system1) is ChatDecider
    monkeypatch.delenv("HUNCH_SYSTEM1_URL")
    monkeypatch.setenv("HUNCH_SYSTEM1", "vllm")
    assert isinstance(router_from_env().system1, VLLMDecider) and router_from_env().system1.url.startswith("http://localhost:8000")
    monkeypatch.setenv("HUNCH_SYSTEM1_URL", "http://gpu-box:8000/v1")
    assert router_from_env().system1.url == "http://gpu-box:8000/v1/chat/completions"
    monkeypatch.setenv("HUNCH_SYSTEM1", "bogus")
    with pytest.raises(ValueError):
        router_from_env()


def test_api_reports_an_unready_system1(monkeypatch):
    from fastapi.testclient import TestClient
    import hunch.api as api

    class Down(FakeDecider):
        def check(self):
            raise RuntimeError("cannot reach http://vllm:8000/v1/chat/completions")

    monkeypatch.setenv("HUNCH_API_KEY", "secret")
    monkeypatch.setattr(api, "_router", None)
    monkeypatch.setattr(api, "router_from_env", lambda: Router(Down()))
    response = TestClient(api.app).post("/decide", json={"state": "x", "questions": QUESTIONS}, headers={"Authorization": "Bearer secret"})
    assert response.status_code == 503 and "cannot reach http://vllm:8000" in response.json()["detail"]
    assert api._router is None


class Flaky:
    """System 1 or 2 that fails with the given errors before it works."""

    def __init__(self, inner, errors):
        self.inner, self.errors, self.calls = inner, list(errors), 0

    def _maybe_fail(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)

    def decide(self, state, questions, images=()):
        self._maybe_fail()
        return self.inner.decide(state, questions, images)

    def answer(self, state, questions, images=()):
        self._maybe_fail()
        return self.inner.answer(state, questions, images)


def _status_error(code):
    import httpx

    return httpx.HTTPStatusError("x", request=httpx.Request("POST", "http://m"), response=httpx.Response(code, request=httpx.Request("POST", "http://m")))


def test_transient_errors_are_retried():
    import httpx

    system1 = Flaky(FakeDecider(), [httpx.ConnectError("refused"), _status_error(503)])
    result = Router(system1, FakeClaude(), backoff=0).decide("event", QUESTIONS)
    assert system1.calls == 3 and result["degraded"] is None and result["escalated"] == ["importance"]
    gave_up = Flaky(FakeDecider(), [httpx.ConnectError("refused")] * 5)
    Router(gave_up, FakeClaude(), retries=1, backoff=0).decide("event", QUESTIONS)
    assert gave_up.calls == 2


def test_system1_outage_sends_everything_to_system2():
    import httpx

    claude = FakeClaude()
    router = Router(Flaky(FakeDecider(), [httpx.ConnectError("refused")] * 9), claude, retries=1, backoff=0)
    result = router.decide("event", QUESTIONS)
    assert result["degraded"][0]["system"] == 1 and "refused" in result["degraded"][0]["error"]
    assert claude.asked == list(QUESTIONS) and result["escalated"] == list(QUESTIONS) and result["unsure"] == list(QUESTIONS)
    assert all(a["source"] == "system2" and a["hunch"] is None for a in result["answers"].values())
    with pytest.raises(httpx.ConnectError):
        Router(Flaky(FakeDecider(), [httpx.ConnectError("refused")] * 9), claude, retries=1, backoff=0).decide("event", QUESTIONS, escalate=False)
    with pytest.raises(httpx.ConnectError):
        Router(Flaky(FakeDecider(), [httpx.ConnectError("refused")] * 9), None, retries=1, backoff=0).decide("event", QUESTIONS)


def test_system2_outage_keeps_system1_answers(tmp_path):
    import httpx
    from hunch import calibrate

    store = _calibration_store(tmp_path)
    router = Router(FakeDecider(), Flaky(FakeClaude(), [_status_error(500)] * 9), retries=0, backoff=0, log=store)
    result = router.decide("event", QUESTIONS)
    assert result["degraded"][-1]["system"] == 2 and result["escalated"] == [] and result["unsure"] == ["importance"]
    assert result["answers"]["importance"]["source"] == "hunch" and result["answers"]["importance"]["answer"] == 3
    # A System 1 outage logs answers that have no confidence; calibration skips them instead of crashing.
    down = Router(Flaky(FakeDecider(), [httpx.ConnectError("x")] * 9), FakeClaude(), retries=0, backoff=0, log=store).decide("event", QUESTIONS)
    assert down["degraded"][0]["system"] == 1 and calibrate(store, min_samples=1) == {}


def test_mistakes_are_not_hidden_by_falling_back():
    claude = FakeClaude()
    for code in (401, 404):
        with pytest.raises(Exception, match="x"):
            Router(Flaky(FakeDecider(), [_status_error(code)] * 9), claude, backoff=0).decide("event", QUESTIONS)
    assert claude.asked is None
    with pytest.raises(Exception, match="x"):
        Router(FakeDecider(), Flaky(FakeClaude(), [_status_error(401)] * 9), backoff=0).decide("event", QUESTIONS)


def _bag(texts):
    import numpy as np

    words = ["server", "invoice", "sale", "meeting"]
    return np.array([[t.lower().count(w) for w in words] for t in texts], dtype=float)


def test_local_decider_trains_on_the_log_and_answers_in_the_same_shape(tmp_path):
    from hunch import LocalDecider

    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"], "event_type": QUESTIONS["event_type"]}
    docs = [("server meeting", True, "meeting"), ("meeting about server", True, "meeting"),
            ("invoice sale", False, "interview"), ("sale invoice due", False, "interview")] * 5
    for text, prep, kind in docs:
        decision = Router(FakeDecider(), None, log=store).decide(text, q)
        store.label(decision["id"], "needs_prep", prep)
        store.label(decision["id"], "event_type", kind)
    local = LocalDecider.train(store.labeled(), embed=_bag)
    answers = local.decide("a server meeting today", q)
    assert answers["needs_prep"]["answer"] is True and answers["event_type"]["answer"] == "meeting"
    assert 0.5 < answers["event_type"]["confidence"] <= 1 and set(answers["event_type"]["probabilities"]) == {"meeting", "interview"}
    # It drops into a Router like any System 1, and round-trips through disk.
    local.save(tmp_path / "local.joblib")
    reloaded = LocalDecider.load(tmp_path / "local.joblib", embed=_bag)
    assert Router(reloaded, None).decide("sale invoice", q)["answers"]["needs_prep"]["answer"] is False
    assert local.decide("x", {"importance": QUESTIONS["importance"]}) == {}  # no head, so left for the next tier
    with pytest.raises(ValueError, match="images"):
        local.decide("x", q, images=["data:image/png;base64,AAAA"])
    with pytest.raises(ValueError, match="two different answers"):
        LocalDecider.train([{"state": "a", "key": "k", "truth": True}] * 3, embed=_bag)


class Tier:
    """A System 1 tier that answers the keys in `known` with fixed confidences, and leaves the rest out."""

    def __init__(self, known):
        self.known, self.asked = known, []

    def decide(self, state, questions, images=()):
        self.asked.append(list(questions))
        return {k: {"answer": self.known[k][0], "confidence": self.known[k][1], "probabilities": {}} for k in questions if k in self.known}


def test_chain_passes_only_unsure_questions_down():
    fast = Tier({"needs_prep": (True, 0.99), "importance": (1, 0.5), "event_type": ("meeting", 0.6)})
    llm = Tier({"importance": (3, 0.9), "event_type": ("interview", 0.7)})
    claude = FakeClaude()
    result = Router([fast, llm], claude, threshold=[0.9, 0.8]).decide("event", QUESTIONS)
    assert fast.asked == [["needs_prep", "importance", "event_type"]] and llm.asked == [["importance", "event_type"]]
    a = result["answers"]
    assert (a["needs_prep"]["source"], a["needs_prep"]["tier"]) == ("hunch", 0)
    assert (a["importance"]["source"], a["importance"]["tier"], a["importance"]["answer"]) == ("hunch", 1, 3)
    assert [t["tier"] for t in a["importance"]["tried"]] == [0, 1] and a["importance"]["tried"][0]["confidence"] == 0.5
    # event_type is still below the second tier's threshold, so only it reaches System 2, with the last tier's answer kept.
    assert result["unsure"] == ["event_type"] and claude.asked == ["event_type"]
    assert a["event_type"]["source"] == "system2" and a["event_type"]["hunch"]["tier"] == 1 and a["event_type"]["hunch"]["answer"] == "interview"
    assert len(result["latency_ms"]["tiers"]) == 2 and result["degraded"] is None


def test_a_tier_that_cannot_answer_a_question_passes_it_down():
    fast, llm = Tier({"needs_prep": (True, 0.99)}), Tier({"importance": (3, 0.95), "event_type": ("meeting", 0.95)})
    result = Router([fast, llm], None).decide("event", QUESTIONS)
    assert llm.asked == [["importance", "event_type"]] and result["unsure"] == []
    assert result["answers"]["importance"]["tier"] == 1 and result["answers"]["needs_prep"]["tier"] == 0
    with pytest.raises(ValueError, match="no System 1 tier answered 'importance'"):
        Router([fast], None).decide("event", QUESTIONS)
    assert Router([fast], FakeClaude()).decide("event", QUESTIONS)["answers"]["importance"]["hunch"] is None
    with pytest.raises(ValueError, match="2 tiers but 3 thresholds"):
        Router([fast, llm], None, threshold=[0.5, 0.5, 0.5])


def test_a_tier_outage_is_skipped():
    import httpx

    llm = Tier({"needs_prep": (True, 0.95), "importance": (2, 0.95), "event_type": ("meeting", 0.95)})
    router = Router([Flaky(FakeDecider(), [httpx.ConnectError("down")] * 9), llm], None, retries=0, backoff=0)
    result = router.decide("event", QUESTIONS)
    assert result["degraded"] == [{"system": 1, "tier": 0, "error": result["degraded"][0]["error"]}] and "down" in result["degraded"][0]["error"]
    assert all(a["tier"] == 1 for a in result["answers"].values())
    with pytest.raises(httpx.ConnectError):
        Router([Flaky(FakeDecider(), [httpx.ConnectError("down")] * 9)], None, retries=0, backoff=0).decide("event", QUESTIONS)


def test_each_tier_is_calibrated_on_its_own(tmp_path):
    from hunch import calibrate

    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    # Tier 0 is right only when it is very sure (0.95); at 0.7 it is always wrong. Tier 1 is right at 0.7.
    for i in range(100):
        sure = i % 2 == 0
        fast = Tier({"needs_prep": (True if sure else False, 0.95 if sure else 0.7)})
        llm = Tier({"needs_prep": (True, 0.7)})
        result = Router([fast, llm], None, threshold=[0.8, 0.5], log=store).decide("event", q)
        store.label(result["id"], "needs_prep", True)
    assert calibrate(store, 0.95, 30, tier=0)["needs_prep"]["threshold"] == 0.95
    assert calibrate(store, 0.95, 30, tier=1)["needs_prep"]["threshold"] == 0.7
    assert store.thresholds(0) == {"needs_prep": 0.95} and store.thresholds(1) == {"needs_prep": 0.7}
    # The router now trusts tier 0 only at 0.95 and tier 1 at 0.7, instead of its 0.8 and 0.5 defaults.
    tier0 = Tier({"needs_prep": (True, 0.9)})
    tier1 = Tier({"needs_prep": (True, 0.6)})
    result = Router([tier0, tier1], FakeClaude(), threshold=[0.8, 0.5], log=store).decide("event", q)
    assert tier1.asked and result["escalated"] == ["needs_prep"]


def test_golden_run_reports_who_answered(tmp_path):
    fast, llm = Tier({"needs_prep": (True, 0.99), "importance": (1, 0.5)}), Tier({"importance": (4, 0.95), "event_type": ("meeting", 0.95)})
    report = Router([fast, llm], None).test([{"state": "a", "expected": {"needs_prep": True, "importance": 4, "event_type": "meeting"}}], questions=QUESTIONS)
    assert report["answered_by"] == {"tier 0": pytest.approx(1 / 3), "tier 1": pytest.approx(2 / 3)} and report["overall"]["accuracy"] == 1.0


def test_stats_reports_who_answers_and_how_accurate_each_tier_is(tmp_path):
    import httpx
    from hunch import stats

    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"]}

    class Script:
        """A tier or System 2 that answers needs_prep from a list, one entry per call: (answer, confidence)."""

        def __init__(self, *calls):
            self.calls = list(calls)

        def decide(self, state, questions, images=()):
            answer, confidence = self.calls.pop(0)
            return {"needs_prep": {"answer": answer, "confidence": confidence, "probabilities": {}}}

        def answer(self, state, questions, images=()):
            return {"needs_prep": {"answer": self.calls.pop(0)[0], "reason": "r"}}

    # Six sure tier-0 answers (the audit says the last one is wrong), then four unsure there and sure in tier 1
    # (the audit says one of them is wrong). audit=1 grades every kept answer, so each is weighted 1.
    fast = Script(*[(True, 0.95)] * 6 + [(True, 0.5)] * 4)
    llm = Script(*[(True, 0.9)] * 4)
    audits = Script(*[(True,)] * 5 + [(False,)] + [(True,)] * 3 + [(False,)])
    router = Router([fast, llm], audits, threshold=[0.8, 0.8], log=store, audit=1.0)
    ids = [router.decide("event", q)["id"] for _ in range(10)]
    store.label(ids[0], "needs_prep", False)  # a human says the first answer was wrong, whatever the audit said
    report = stats(store)["questions"]["needs_prep"]
    assert report["asked"] == 10 and report["answered_by"] == {"tier 0": 0.6, "tier 1": 0.4}
    assert report["tier_accuracy"]["tier 0"] == {"graded": 6, "accuracy": pytest.approx(4 / 6)}  # 4 of 6 right: the audit's miss, and the human's
    assert report["tier_accuracy"]["tier 1"] == {"graded": 4, "accuracy": pytest.approx(3 / 4)}
    assert report["accuracy_vs_labels"] == {"graded": 1, "accuracy": 0.0}
    latency = stats(store)["latency_ms"]
    assert stats(store)["decisions"] == 10 and set(latency["tier_mean"]) == {"tier 0", "tier 1"} and latency["total"]["p95"] >= latency["total"]["p50"]
    assert latency["system2_called"] == 1.0 and stats(store, since=9e12)["decisions"] == 0
    # An outage shows up, and a System 2 answer is counted for who answered but not as a tier's accuracy.
    down = Router(Flaky(FakeDecider(), [httpx.ConnectError("x")] * 9), FakeClaude(), retries=0, backoff=0, log=store)
    down.decide("event", {"importance": QUESTIONS["importance"]})
    after = stats(store)
    assert after["degraded"] == 1 and after["questions"]["importance"]["answered_by"] == {"system2": 1.0}
    assert after["questions"]["importance"]["tier_accuracy"] == {}


def test_api_stats(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import hunch.api as api

    monkeypatch.setenv("HUNCH_API_KEY", "secret")
    monkeypatch.setattr(api, "_router", Router(FakeDecider(), FakeClaude(), log=_calibration_store(tmp_path)))
    client = TestClient(api.app)
    api._router.decide("event", QUESTIONS)
    assert client.get("/stats").status_code == 401
    body = client.get("/stats", headers={"Authorization": "Bearer secret"}).json()
    assert body["decisions"] == 1 and body["questions"]["importance"]["answered_by"] == {"system2": 1.0}


def test_review_queue_lists_disagreements_and_clears_when_graded(tmp_path):
    from hunch import review_queue, stats

    store = _calibration_store(tmp_path)

    class Say:
        """A tier or System 2 answering needs_prep: (answer, confidence), one per call."""

        def __init__(self, *calls):
            self.calls = list(calls)

        def decide(self, state, questions, images=()):
            answer, confidence = self.calls.pop(0)
            return {"needs_prep": {"answer": answer, "confidence": confidence, "probabilities": {}}}

        def answer(self, state, questions, images=()):
            return {"needs_prep": {"answer": self.calls.pop(0)[0], "reason": "because"}}

    q = {"needs_prep": QUESTIONS["needs_prep"]}
    # 1: unsure, System 2 disagrees -> queued.  2: unsure, System 2 agrees -> not queued.
    # 3: sure but the audit disagrees -> queued.  4: sure and the audit agrees -> not queued.
    # 5: unsure with escalate=False -> queued as unresolved.
    ids = []
    for state, tier, system2, escalate in [("one", (True, 0.5), (False,), True), ("two", (True, 0.5), (True,), True),
                                           ("three", (True, 0.95), (False,), True), ("four", (True, 0.95), (True,), True),
                                           ("five", (True, 0.5), (True,), False)]:
        router = Router(Say(tier), Say(system2), threshold=0.8, log=store, audit=1.0)
        ids.append(router.decide(state, q, escalate=escalate)["id"])
    queue = review_queue(store)
    assert [(item["state"], item["reason"]) for item in queue] == [("one", "disagreement"), ("three", "audit_disagreement"), ("five", "unresolved")]
    first = queue[0]
    assert first["id"] == ids[0] and first["answer"] is False and first["source"] == "system2" and first["question"] == q["needs_prep"]
    assert [c["who"] for c in first["candidates"]] == ["tier 0", "system2"] and first["candidates"][1]["reason"] == "because"
    assert [i["state"] for i in review_queue(store, reasons=("unresolved",))] == ["five"] and len(review_queue(store, limit=1)) == 1
    assert stats(store)["review_queue"] == {"disagreement": 1, "audit_disagreement": 1, "unresolved": 1}
    store.label(ids[0], "needs_prep", True)
    assert [i["state"] for i in review_queue(store)] == ["three", "five"]
    # A System 2 answer with no System 1 answer to compare against is not queued.
    import httpx
    down = Router(Flaky(FakeDecider(), [httpx.ConnectError("x")] * 9), FakeClaude(), retries=0, backoff=0, log=store)
    down.decide("six", {"importance": QUESTIONS["importance"]})
    assert [i["state"] for i in review_queue(store)] == ["three", "five"]


def test_api_review_queue(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import hunch.api as api

    monkeypatch.setenv("HUNCH_API_KEY", "secret")
    store = _calibration_store(tmp_path)
    monkeypatch.setattr(api, "_router", Router(FakeDecider(), FakeClaude(), log=store))
    client, auth = TestClient(api.app), {"Authorization": "Bearer secret"}
    decision = client.post("/decide", json={"state": "event", "questions": QUESTIONS}, headers=auth).json()
    assert client.get("/review").status_code == 401 and client.get("/review?reason=nope", headers=auth).status_code == 422
    queue = client.get("/review", headers=auth).json()
    # FakeDecider says importance 3, FakeClaude says 4: that disagreement is the only item.
    assert [(i["id"], i["key"], i["reason"]) for i in queue] == [(decision["id"], "importance", "disagreement")] and queue[0]["state"] == "event"
    assert client.post("/feedback", json={"id": decision["id"], "labels": {"importance": 3}}, headers=auth).status_code == 200
    assert client.get("/review", headers=auth).json() == []


class Counting:
    """A tier and System 2 in one that counts the questions it was asked, with an identity so it can be cached."""

    def __init__(self, identity="m1", confidence=0.6):
        self.identity, self.confidence, self.asked = identity, confidence, []

    def decide(self, state, questions, images=()):
        self.asked += list(questions)
        return {k: {"answer": True, "confidence": self.confidence, "probabilities": {}} for k in questions}

    def answer(self, state, questions, images=()):
        self.asked += list(questions)
        return {k: {"answer": True, "reason": "r"} for k in questions}


def test_cache_skips_repeated_questions_and_rescoring_is_free(tmp_path):
    from hunch import Cache

    q = {"needs_prep": QUESTIONS["needs_prep"]}
    cache = Cache(tmp_path / "cache.db")
    tier, claude = Counting(confidence=0.6), Counting("c1")
    router = Router(tier, claude, threshold=0.8, cache=cache)
    first = router.decide("event", q)
    assert first["cache"] == {"hits": 0, "misses": 2} and tier.asked == ["needs_prep"] and claude.asked == ["needs_prep"] and cache.size() == 2
    again = router.decide("event", q)
    assert again["cache"] == {"hits": 2, "misses": 0} and tier.asked == ["needs_prep"] and claude.asked == ["needs_prep"]
    assert again["answers"]["needs_prep"]["answer"] == first["answers"]["needs_prep"]["answer"] and again["escalated"] == ["needs_prep"]
    # The cache holds what the models said, not the decision: a new threshold is applied with no new calls.
    trusting = router.decide("event", q, thresholds={"needs_prep": 0.5})
    assert trusting["escalated"] == [] and trusting["answers"]["needs_prep"]["source"] == "hunch" and len(tier.asked) == 1
    # A different text, question definition, question name or image is a different question.
    router.decide("another event", q)
    router.decide("event", {"needs_prep": QUESTIONS["needs_prep"] | {"instructions": "Different?"}})
    router.decide("event", {"renamed": QUESTIONS["needs_prep"]})
    router.decide("event", q, images=["data:image/png;base64,AAAA"])
    assert len(tier.asked) == 5
    # Rescoring a golden set with another threshold calls nothing.
    calls = len(tier.asked) + len(claude.asked)
    cases = [{"state": "event", "expected": {"needs_prep": True}}]
    assert router.test(cases, questions=q, thresholds={"needs_prep": 0.5})["overall"]["accuracy"] == 1.0
    assert router.test(cases, questions=q, thresholds={"needs_prep": 0.9})["overall"]["accuracy"] == 1.0
    assert len(tier.asked) + len(claude.asked) == calls


def test_cache_is_per_model_and_only_for_models_with_an_identity(tmp_path):
    from hunch import Cache

    q = {"needs_prep": QUESTIONS["needs_prep"]}
    cache = Cache(tmp_path / "cache.db")
    one = Counting("model-1", 0.95)
    Router(one, None, cache=cache).decide("event", q)
    other = Counting("model-2", 0.95)
    Router(other, None, cache=cache).decide("event", q)
    assert one.asked == other.asked == ["needs_prep"]  # a different model is never served the first one's answer
    anonymous = Counting(None, 0.95)
    for _ in range(2):
        result = Router(anonymous, None, cache=cache).decide("event", q)
    assert anonymous.asked == ["needs_prep"] * 2 and result["cache"] == {"hits": 0, "misses": 0}
    assert Router(one, None).decide("event", q)["cache"] is None and one.asked == ["needs_prep", "needs_prep"]


def test_partial_hits_call_the_model_only_for_what_is_missing(tmp_path):
    from hunch import Cache

    cache = Cache(tmp_path / "cache.db")
    claude = Counting("c1")
    router = Router(Counting(confidence=0.0), claude, threshold=0.8, cache=cache)
    router.decide("event", {"needs_prep": QUESTIONS["needs_prep"]})
    router.decide("event", {"needs_prep": QUESTIONS["needs_prep"], "event_type": QUESTIONS["event_type"]})
    assert claude.asked == ["needs_prep", "event_type"]


def test_cache_ttl_clear_and_identities(tmp_path):
    from hunch import Cache, ChatDecider, ChatReasoner, ClaudeAPI, LocalDecider
    from hunch.cache import cache_key

    cache = Cache(tmp_path / "ttl.db", ttl=0.05)
    cache.put_many({"k": {"a": 1}})
    assert cache.get_many(["k", "missing"]) == {"k": {"a": 1}}
    import time
    time.sleep(0.1)
    assert cache.get_many(["k"]) == {}
    cache.put_many({"k": {"a": 2}})
    Cache(tmp_path / "ttl.db").clear()
    assert cache.size() == 0
    assert cache_key("m", "q", QUESTIONS["needs_prep"], "x", []) == cache_key("m", "q", QUESTIONS["needs_prep"], "x", ())
    a, b = ChatDecider("m", "http://one/v1", api_key="secret"), ChatDecider("m", "http://one/v1", api_key="other")
    assert a.identity == b.identity and "secret" not in a.identity  # a key is not part of the model
    assert a.identity != ChatDecider("m", "http://two/v1").identity != ChatDecider("m", "http://one/v1", extra={"x": 1}).identity
    assert ChatReasoner("m", "http://one/v1").identity != ChatReasoner("n", "http://one/v1").identity
    assert ClaudeAPI(client=object()).identity != ClaudeAPI("claude-opus-5-5", client=object()).identity
    rows = [{"state": s, "key": "k", "truth": t} for s, t in [("server meeting", True), ("invoice sale", False)] * 4]
    assert LocalDecider.train(rows, embed=_bag).identity == LocalDecider.train(rows, embed=_bag).identity
    assert LocalDecider.train(rows, embed=_bag).identity != LocalDecider.train(rows[:-2] + [{"state": "meeting", "key": "k", "truth": False}] * 2, embed=_bag).identity


def test_cache_hit_rate_in_stats_and_env_config(tmp_path, monkeypatch):
    from hunch import Cache, Store, router_from_env, stats

    q = {"needs_prep": QUESTIONS["needs_prep"]}
    store = _calibration_store(tmp_path)
    router = Router(Counting(confidence=0.95), None, cache=Cache(tmp_path / "c.db"), log=store)
    for _ in range(4):
        router.decide("event", q)
    assert stats(store)["cache"] == {"hits": 3, "misses": 1, "hit_rate": 0.75}
    assert stats(Store(tmp_path / "empty.db"))["cache"] is None
    monkeypatch.setenv("HUNCH_SYSTEM1_MODEL", "m")
    monkeypatch.setenv("HUNCH_SYSTEM1_URL", "http://one/v1")
    monkeypatch.setenv("HUNCH_SYSTEM2", "none")
    assert router_from_env().cache is None
    monkeypatch.setenv("HUNCH_CACHE", str(tmp_path / "env.db"))
    monkeypatch.setenv("HUNCH_CACHE_TTL", "60")
    assert router_from_env().cache.ttl == 60.0



def test_summarize_says_why_there_is_no_threshold():
    import random
    from hunch.calibrate import summarize

    few = summarize([(0.9, True, 1.0)] * 10, 0.9, 30)
    assert few["threshold"] is None and few["reason"] == "not enough data" and few["best_target"] is None
    assert summarize([], 0.9, 30)["reason"] == "not enough data"

    rng = random.Random(0)
    weak = [(0.5 + 0.5 * rng.random(), rng.random() < 0.7, 1.0) for _ in range(300)]  # 70% right at every confidence
    unreachable = summarize(weak, 0.9, 30)
    assert unreachable["threshold"] is None and unreachable["reason"] == "target unreachable" and unreachable["best_target"] < 0.75
    # best_target is exactly the highest target that finds a threshold on this data.
    assert summarize(weak, unreachable["best_target"] - 1e-9, 30)["threshold"] is not None
    assert summarize(weak, unreachable["best_target"] + 1e-3, 30)["threshold"] is None

    perfect = [(0.5 + i / 100, True, 1.0) for i in range(40)]  # right every time, but only 40 answers
    thin = summarize(perfect, 0.97, 30)
    assert thin["threshold"] is None and thin["reason"] == "not enough evidence" and 0.9 < thin["best_target"] < 0.97
    ok = summarize(perfect * 10, 0.97, 30)
    assert ok["threshold"] is not None and ok["reason"] is None and ok["best_target"] >= 0.97


def test_calibrate_logs_stats_flags_and_clears_an_unreachable_target(tmp_path, caplog):
    import logging
    from hunch import Store, calibrate, stats

    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    for i in range(60):  # 70% right whatever the confidence
        decision = Router(Tier({"needs_prep": (True, 0.6 + (i % 4) / 10)}), None, log=store).decide("event", q)
        store.label(decision["id"], "needs_prep", i % 10 < 7)
    with caplog.at_level(logging.WARNING, logger="hunch"):
        report = calibrate(store, 0.9, 30)["needs_prep"]
    assert report["reason"] == "target unreachable" and store.thresholds() == {"needs_prep": float("inf")}
    # Failing closed: the tier is no longer trusted for this question, so even a confident answer escalates.
    closed = Router(Tier({"needs_prep": (True, 0.99)}), FakeClaude(), log=store).decide("event", q)
    assert closed["escalated"] == ["needs_prep"] and closed["answers"]["needs_prep"]["source"] == "system2"
    open_store = Store(tmp_path / "open.db")
    for i in range(60):
        decision = Router(Tier({"needs_prep": (True, 0.6 + (i % 4) / 10)}), None, log=open_store).decide("event", q)
        open_store.label(decision["id"], "needs_prep", i % 10 < 7)
    calibrate(open_store, 0.9, 30, fail_closed=False)
    assert open_store.thresholds() == {}
    assert "no threshold for 'needs_prep' at tier 0: target unreachable" in caplog.text and "the best this data supports is 0." in caplog.text
    flagged, = stats(store)["uncalibrated"]
    assert (flagged["tier"], flagged["key"], flagged["reason"], flagged["target"]) == (0, "needs_prep", "target unreachable", 0.9)
    assert flagged["best_target"] == report["best_target"]
    # Calibrating at a target the data supports finds a threshold and clears the flag.
    assert calibrate(store, report["best_target"] - 0.01, 30)["needs_prep"]["threshold"] is not None
    assert stats(store)["uncalibrated"] == [] and "needs_prep" in store.thresholds()


def test_retrain_promotes_a_classifier_that_escalates_less_and_keeps_it_otherwise(tmp_path):
    from hunch import retrain

    # Decision ids are random, so the train/held-out split varies by run: use enough decisions that any split can pass.
    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"], "event_type": QUESTIONS["event_type"]}
    docs = [("server meeting", True, "meeting"), ("meeting about server", True, "meeting"),
            ("invoice sale", False, "interview"), ("sale invoice due", False, "interview")] * 40
    for text, prep, kind in docs:
        decision = Router(FakeDecider(), None, log=store).decide(text, q)
        store.label(decision["id"], "needs_prep", prep)
        store.label(decision["id"], "event_type", kind)
    first = retrain(store, target_accuracy=0.8, min_samples=5, embed=_bag)
    assert first["report"]["needs_prep"]["promoted"] and first["report"]["event_type"]["promoted"]
    assert store.thresholds(0).keys() == {"needs_prep", "event_type"}
    assert first["decider"].decide("a server meeting", q)["needs_prep"]["answer"] is True
    # The same data gives the same classifier, which escalates no less than the current one, so it is not promoted.
    second = retrain(store, current=first["decider"], target_accuracy=0.8, min_samples=5, embed=_bag)
    assert not any(r["promoted"] for r in second["report"].values()) and second["decider"] is first["decider"]
    # A question with a single answer in the log has nothing to learn from and is left out.
    only = Router(FakeDecider(), None, log=store).decide("server", {"importance": QUESTIONS["importance"]})
    store.label(only["id"], "importance", 3)
    assert "importance" not in retrain(store, target_accuracy=0.8, min_samples=5, embed=_bag)["report"]


def test_guard_vetoes_a_confident_answer_it_confidently_disagrees_with(tmp_path):
    fast = Tier({"needs_prep": (True, 0.99), "importance": (3, 0.99), "event_type": ("meeting", 0.99)})
    # Disagrees on needs_prep (sure), disagrees on event_type (not sure enough), agrees on importance.
    guard = Tier({"needs_prep": (False, 0.95), "importance": (3, 0.99), "event_type": ("interview", 0.6)})
    claude = FakeClaude()
    result = Router(fast, claude, guard=guard, guard_threshold=0.9, log=_calibration_store(tmp_path)).decide("event", QUESTIONS)
    a = result["answers"]
    assert result["unsure"] == ["needs_prep"] and result["escalated"] == ["needs_prep"] and claude.asked == ["needs_prep"]
    assert a["needs_prep"]["source"] == "system2" and a["needs_prep"]["hunch"]["answer"] is True and a["needs_prep"]["guard"]["answer"] is False
    assert a["importance"]["source"] == "hunch" and a["event_type"]["source"] == "hunch" and "guard" not in a["event_type"]
    assert guard.asked == [["needs_prep", "importance", "event_type"]]
    # Without a System 2 the vetoed answer stands, flagged as unsure.
    alone = Router(fast, None, guard=guard).decide("event", QUESTIONS)
    assert alone["unsure"] == ["needs_prep"] and alone["answers"]["needs_prep"]["answer"] is True and alone["answers"]["needs_prep"]["guard"]["answer"] is False
    # The guard only checks answers a tier was confident about; unsure ones already escalate.
    unsure = Tier({"needs_prep": (True, 0.5), "importance": (3, 0.99), "event_type": ("meeting", 0.99)})
    watcher = Tier({"needs_prep": (False, 0.99), "importance": (3, 0.99), "event_type": ("meeting", 0.99)})
    Router(unsure, FakeClaude(), guard=watcher).decide("event", QUESTIONS)
    assert watcher.asked == [["importance", "event_type"]]


def test_a_guard_that_is_down_vetoes_nothing_and_a_broken_one_raises():
    from hunch import Outage

    fast =Tier({"needs_prep": (True, 0.99), "importance": (3, 0.99), "event_type": ("meeting", 0.99)})
    claude = FakeClaude()
    result = Router(fast, claude, guard=Flaky(fast, [Outage("overloaded")] * 9), retries=0, backoff=0).decide("event", QUESTIONS)
    assert result["unsure"] == [] and claude.asked is None and all(a["source"] == "hunch" for a in result["answers"].values())
    assert result["degraded"] == [{"system": "guard", "tier": None, "error": result["degraded"][0]["error"]}] and "overloaded" in result["degraded"][0]["error"]
    with pytest.raises(RuntimeError, match="bad key"):
        Router(fast, claude, guard=Flaky(fast, [RuntimeError("bad key")] * 9), retries=0, backoff=0).decide("event", QUESTIONS)


def test_the_core_needs_no_anthropic_sdk():
    import subprocess
    import sys

    code = ("import sys; sys.modules['anthropic'] = None\n"
            "import hunch\n"
            "from hunch import Router, ChatReasoner\n"
            "try:\n    hunch.ClaudeAPI()\nexcept ImportError as e:\n    assert 'hunch[claude]' in str(e)\nelse:\n    raise SystemExit('ClaudeAPI built without the SDK')\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_any_adapter_can_mark_an_outage_and_claude_errors_become_one():
    import anthropic
    import httpx
    from hunch import ClaudeAPI, Outage

    # An Outage from any System 2 falls back like a network error; a mistake still raises.
    down = Router(FakeDecider(), Flaky(FakeClaude(), [Outage("overloaded")] * 9), retries=0, backoff=0).decide("event", QUESTIONS)
    assert down["degraded"][-1]["system"] == 2 and down["escalated"] == []
    with pytest.raises(RuntimeError, match="bad key"):
        Router(FakeDecider(), Flaky(FakeClaude(), [RuntimeError("bad key")] * 9), retries=0, backoff=0).decide("event", QUESTIONS)

    class Client:
        class beta:
            class messages:
                @staticmethod
                def create(**kwargs):
                    raise anthropic.APIConnectionError(request=httpx.Request("POST", "http://x"))

    with pytest.raises(Outage):
        ClaudeAPI(client=Client()).answer("event", QUESTIONS)


def test_providers_fill_in_the_url_and_key_but_nothing_is_assumed(monkeypatch):
    from hunch import PROVIDERS, ChatDecider, ChatReasoner, router_from_env

    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    groq = ChatDecider("m", provider="groq")
    assert groq.url == "https://api.groq.com/openai/v1/chat/completions" and groq.headers == {"Authorization": "Bearer from-env"}
    # An explicit URL or key wins over the provider's, and a local provider needs no key.
    assert ChatDecider("m", "http://x/v1", provider="groq", api_key="k").headers == {"Authorization": "Bearer k"}
    assert ChatDecider("m", "http://x/v1", provider="groq").url == "http://x/v1/chat/completions"
    assert ChatReasoner("m", provider="ollama").url == "http://localhost:11434/v1/chat/completions"
    with pytest.raises(ValueError, match="base_url"):
        ChatDecider("m")
    with pytest.raises(ValueError, match="unknown provider"):
        ChatDecider("m", provider="nope")
    assert {"openai", "xai", "mistral", "deepseek", "groq", "together", "fireworks", "openrouter", "ollama", "vllm"} <= set(PROVIDERS)
    # The environment can name a provider for either system, and that alone selects a chat System 2 under auto.
    monkeypatch.setenv("HUNCH_SYSTEM1_MODEL", "m")
    monkeypatch.setenv("HUNCH_SYSTEM1_PROVIDER", "ollama")
    monkeypatch.setenv("HUNCH_SYSTEM2_MODEL", "big")
    monkeypatch.setenv("HUNCH_SYSTEM2_PROVIDER", "openrouter")
    router = router_from_env()
    assert router.system1.url.startswith("http://localhost:11434") and isinstance(router.system2, ChatReasoner)
    assert router.system2.url.startswith("https://openrouter.ai")


def _gemini_logprobs(url, tops):
    import httpx

    step = {"candidates": [{"token": t, "logProbability": lp} for t, lp in tops]}
    body = {"candidates": [{"content": {"parts": [{"text": tops[0][0]}]}, "logprobsResult": {"topCandidates": [step]}}]}
    return httpx.Response(200, json=body, request=httpx.Request("POST", url))


def test_gemini_decider_reads_logprobs_from_its_own_api(monkeypatch):
    import math
    from hunch import GeminiDecider

    seen = {}

    def post(url, json, headers, timeout):
        seen.update(url=url, json=json, headers=headers)
        return _gemini_logprobs(url, [("yes", math.log(0.9)), ("no", math.log(0.1))])

    monkeypatch.setattr(backends, "post", post)
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    decider = GeminiDecider("gemini-2.5-flash", extra={"thinkingConfig": {"thinkingBudget": 0}})
    answer = decider.decide("event", {"needs_prep": QUESTIONS["needs_prep"]}, images=["data:image/png;base64,AAAA"])["needs_prep"]
    assert answer["answer"] is True and answer["confidence"] == pytest.approx(0.9)
    assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert seen["headers"] == {"x-goog-api-key": "g-key"}
    config = seen["json"]["generationConfig"]
    assert config["maxOutputTokens"] == 1 and config["responseLogprobs"] is True and config["logprobs"] == 20 and config["thinkingConfig"] == {"thinkingBudget": 0}
    assert seen["json"]["contents"][0]["parts"][0] == {"inlineData": {"mimeType": "image/png", "data": "AAAA"}}
    assert "<state>" in seen["json"]["contents"][0]["parts"][1]["text"]
    # A model that thinks first puts nothing on the options, and the error says what to try.
    monkeypatch.setattr(backends, "post", lambda url, json, headers, timeout: _gemini_logprobs(url, [("The", -0.1)]))
    with pytest.raises(RuntimeError, match="thinkingBudget"):
        decider.check()


def test_gemini_reasoner_uses_a_gemini_schema_and_checks_the_reply(monkeypatch):
    import json
    import httpx
    from hunch import GeminiReasoner

    seen = {}
    reply = {"needs_prep": {"answer": True, "reason": "r"}, "importance": {"answer": 3, "reason": "r"}, "event_type": {"answer": "interview", "reason": "r"}}

    def respond(payload):
        def post(url, json, headers, timeout):
            seen.update(json=json, headers=headers)
            return httpx.Response(200, json=payload, request=httpx.Request("POST", url))
        return post

    def candidate(text):
        return {"candidates": [{"content": {"parts": [{"text": text}]}}]}

    monkeypatch.setattr(backends, "post", respond(candidate(json.dumps(reply))))
    system2 = GeminiReasoner("gemini-2.5-pro", api_key="k")
    assert system2.answer("event", QUESTIONS) == reply
    schema = seen["json"]["generationConfig"]["responseSchema"]
    assert seen["headers"] == {"x-goog-api-key": "k"} and seen["json"]["generationConfig"]["responseMimeType"] == "application/json"
    assert "additionalProperties" not in json.dumps(schema)
    assert schema["properties"]["event_type"]["properties"]["answer"]["enum"] == ["meeting", "interview"]
    assert schema["properties"]["importance"]["properties"]["answer"]["type"] == "integer"
    bad = reply | {"importance": {"answer": 9, "reason": "r"}}
    monkeypatch.setattr(backends, "post", respond(candidate(json.dumps(bad))))
    with pytest.raises(RuntimeError, match="importance"):
        system2.answer("event", QUESTIONS)
    monkeypatch.setattr(backends, "post", respond({"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(RuntimeError, match="SAFETY"):
        system2.answer("event", QUESTIONS)


def test_vendor_classes_carry_their_endpoint_and_quirks(monkeypatch):
    import hunch
    from hunch import ChatDecider, ChatReasoner, DeepSeekReasoner, GroqReasoner, OllamaDecider, OpenAIDecider, OpenAIReasoner

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    decider = OpenAIDecider("gpt-4o-mini")
    assert isinstance(decider, ChatDecider) and decider.url == "https://api.openai.com/v1/chat/completions"
    assert decider.headers == {"Authorization": "Bearer sk-env"} and "OpenAI" in OpenAIDecider.__doc__
    assert OpenAIDecider("m", api_key="k", base_url="http://proxy/v1").headers == {"Authorization": "Bearer k"}
    assert OpenAIDecider("m", base_url="http://proxy/v1").url == "http://proxy/v1/chat/completions"
    assert OllamaDecider("llama3.1:8b").url == "http://localhost:11434/v1/chat/completions" and OllamaDecider("m").headers == {}
    assert isinstance(OpenAIReasoner("m"), ChatReasoner) and GroqReasoner("m", api_key="k").url.startswith("https://api.groq.com")
    assert not hasattr(hunch, "GroqDecider")
    assert OpenAIReasoner("m").identity != DeepSeekReasoner("m").identity
    # DeepSeek only offers JSON mode, so the schema goes in the prompt and the reply is checked here.
    seen = {}

    def post(url, json, headers, timeout):
        import httpx

        seen.update(json)
        content = '{"needs_prep": {"answer": "maybe", "reason": "r"}}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]}, request=httpx.Request("POST", url))

    monkeypatch.setattr(backends, "post", post)
    with pytest.raises(RuntimeError, match="needs_prep"):
        DeepSeekReasoner("deepseek-chat", api_key="k").answer("event", {"needs_prep": QUESTIONS["needs_prep"]})
    assert seen["response_format"] == {"type": "json_object"} and "JSON Schema" in seen["messages"][0]["content"]
    for name in ("OpenAIDecider", "MistralReasoner", "XAIDecider", "TogetherDecider", "FireworksReasoner", "OpenRouterDecider", "GeminiDecider", "GeminiReasoner"):
        assert name in hunch.__all__ and hasattr(hunch, name)
    with pytest.raises(ValueError, match="json_mode"):
        ChatReasoner("m", "http://x/v1", json_mode="bogus")


def test_env_can_select_gemini(monkeypatch):
    from hunch import GeminiDecider, GeminiReasoner, router_from_env

    monkeypatch.setenv("HUNCH_SYSTEM1", "gemini")
    monkeypatch.setenv("HUNCH_SYSTEM1_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("HUNCH_SYSTEM1_EXTRA", '{"thinkingConfig": {"thinkingBudget": 0}}')
    monkeypatch.setenv("HUNCH_SYSTEM2", "gemini")
    monkeypatch.setenv("HUNCH_SYSTEM2_MODEL", "gemini-2.5-pro")
    router = router_from_env()
    assert isinstance(router.system1, GeminiDecider) and router.system1.extra == {"thinkingConfig": {"thinkingBudget": 0}}
    assert isinstance(router.system2, GeminiReasoner) and router.system2.url.endswith("models/gemini-2.5-pro:generateContent")


def test_http_errors_carry_the_servers_explanation(monkeypatch):
    import httpx
    from hunch import GeminiDecider

    def post(url, json, headers, timeout):
        return httpx.Response(400, text='{"error": {"message": "Logprobs is not enabled for this model"}}', request=httpx.Request("POST", url))

    monkeypatch.setattr(backends, "post", post)
    with pytest.raises(httpx.HTTPStatusError, match="Logprobs is not enabled") as e:
        GeminiDecider("m", api_key="k").decide("event", {"needs_prep": QUESTIONS["needs_prep"]})
    assert e.value.response.status_code == 400


def test_stats_flags_drift_from_the_last_calibration(tmp_path):
    from hunch import Store, calibrate, stats

    q = {"needs_prep": QUESTIONS["needs_prep"]}

    class Says:
        """System 2 that always answers needs_prep the same way."""

        def __init__(self, answer):
            self.value = answer

        def answer(self, state, questions, images=()):
            return {"needs_prep": {"answer": self.value, "reason": "r"}}

    def calibrated(path):
        # Right at 0.95, wrong at 0.7, half each: the threshold is 0.95 and half the answers fall below it.
        store = Store(path)
        for i in range(100):
            sure = i % 2 == 0
            result = Router(Tier({"needs_prep": (sure, 0.95 if sure else 0.7)}), None, log=store).decide("event", q)
            store.label(result["id"], "needs_prep", True)
        assert calibrate(store, 0.95, 30)["needs_prep"]["threshold"] == 0.95
        return store

    store = calibrated(tmp_path / "a.db")
    drift = stats(store)["drift"]
    assert drift == [{"tier": 0, "key": "needs_prep", "calibrated_at": drift[0]["calibrated_at"], "threshold": 0.95, "target": 0.95,
                      "answers_since": 0, "escalation_rate": {"at_calibration": 0.5, "since": None}, "accuracy_since": None,
                      "flags": [], "unchecked": ["escalation_rate", "accuracy"]}]
    # The same mix as at calibration is not drift. Labels on unaudited answers are not a random sample, so even all
    # wrong they do not count toward accuracy.
    for i in range(60):
        result = Router(Tier({"needs_prep": (True, 0.95 if i % 2 else 0.7)}), None, log=store).decide("event", q)
        store.label(result["id"], "needs_prep", False)
    drift = stats(store)["drift"][0]
    assert drift["flags"] == [] and drift["unchecked"] == ["accuracy"] and drift["accuracy_since"] is None
    # Now the tier is unsure about everything: the share escalated moved.
    for _ in range(60):
        Router(Tier({"needs_prep": (True, 0.7)}), None, log=store).decide("event", q)
    drift = stats(store)["drift"][0]
    assert drift["flags"] == ["escalation_rate"] and drift["answers_since"] == 120 and drift["escalation_rate"]["since"] == pytest.approx(0.75)

    # Audits say the confident answers are wrong: accuracy fell below the target, once there are enough of them.
    wrong = calibrated(tmp_path / "b.db")
    audited = Router(Tier({"needs_prep": (True, 0.95)}), Says(False), log=wrong, audit=1.0)
    unsure = Router(Tier({"needs_prep": (True, 0.7)}), Says(False), log=wrong, audit=1.0)
    for _ in range(20):
        audited.decide("event", q), unsure.decide("event", q)
    drift = stats(wrong)["drift"][0]
    assert drift["flags"] == [] and drift["unchecked"] == ["accuracy"] and drift["accuracy_since"] == {"graded": 20, "accuracy": 0.0}
    for _ in range(20):
        audited.decide("event", q), unsure.decide("event", q)
    drift = stats(wrong)["drift"][0]
    assert drift["flags"] == ["accuracy"] and drift["unchecked"] == [] and drift["accuracy_since"]["graded"] == 40
    # A label overrides the audit's answer: the person says these were right after all.
    for decision_id, _, _, result in wrong.results(drift["calibrated_at"]):
        if "audit" in result["answers"]["needs_prep"]:
            wrong.label(decision_id, "needs_prep", True)
    assert stats(wrong)["drift"][0]["flags"] == [] and stats(wrong)["drift"][0]["accuracy_since"]["accuracy"] == 1.0

    # A recalibration that fails closed drops the snapshot, since the tier is no longer trusted for the question.
    for _ in range(200):
        result = Router(Tier({"needs_prep": (True, 0.95)}), None, log=wrong).decide("event", q)
        wrong.label(result["id"], "needs_prep", False)
    assert calibrate(wrong, 0.95, 30)["needs_prep"]["reason"] == "target unreachable" and stats(wrong)["drift"] == []
    # retrain promoting a new classifier drops the snapshot too: its threshold and confidences are new.
    store.save_calibrations({}, 0, ["needs_prep"])
    assert stats(store)["drift"] == []
