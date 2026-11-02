import json
from types import SimpleNamespace
import httpx
import pytest
import sureshot.adapters.base as backends
import sureshot.cli as cli
from sureshot import Store
from sureshot.adapters.anthropic import ClaudeAPI
from sureshot.adapters.chat import ChatReasoner
from sureshot.batch import decide_rows
from sureshot.importer import rows, write_rows
from sureshot.router import Router
from test_router import QUESTIONS, FakeClaude, FakeDecider

DATA = [{"text": f"event {i}", "id": i} for i in range(3)] + [{"text": "", "id": 3}]


class Batched(FakeClaude):
    """FakeClaude with a batch API: a batch is done on its second check; `drop` rows come back unanswered."""
    batches = True

    identity = "batched"

    def __init__(self, drop=(), stop=False, stop_submit=None, flaky=False):
        super().__init__()
        self.submitted, self.checks, self.direct, self.drop, self.stop, self.stop_submit, self.flaky = [], 0, 0, set(drop), stop, stop_submit, flaky

    def answer(self, state, questions, images=()):
        self.direct += 1
        return super().answer(state, questions, images)

    def batch_submit(self, items):
        if len(self.submitted) + 1 == self.stop_submit:
            raise KeyboardInterrupt
        self.submitted.append(items)
        return {"id": len(self.submitted)}

    def batch_results(self, handle, questions):
        self.checks += 1
        if self.stop:
            raise KeyboardInterrupt
        if self.checks == 1 and self.flaky:
            raise httpx.ConnectError("nodename nor servname provided")
        if self.checks % 2:
            return None
        return {cid: FakeClaude.answer(self, None, qs) for cid, qs in questions.items() if cid not in self.drop}


def _run(tmp_path, system2, data=DATA, cache=None, **kwargs):
    store = Store(tmp_path / "sureshot.db")
    router = Router(FakeDecider(), system2, log=store, questions={"events": QUESTIONS}, cache=cache)
    return store, decide_rows(router, "events", data, tmp_path / "out.progress", poll=0, **kwargs)


def test_batch_answers_are_cached_and_cached_questions_are_not_sent(tmp_path):
    from sureshot import Cache

    cache = Cache(tmp_path / "cache.db")
    _run(tmp_path, Batched(), cache=cache)
    (tmp_path / "again").mkdir()
    again = Batched()
    _, results = _run(tmp_path / "again", again, cache=cache)
    assert again.submitted == [] and again.direct == 0  # every escalation was answered from the cache
    assert all(r["answers"]["importance"]["source"] == "system2" for r in results[:3])
    # Live decisions read what the batch cached, too.
    live = Router(FakeDecider(), again, cache=cache).decide("event 0", QUESTIONS)
    assert live["answers"]["importance"]["source"] == "system2" and again.direct == 0


def test_rows_are_split_into_batches_and_a_stop_between_them_sends_only_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr("sureshot.batch.PER_BATCH", 2)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, Batched(stop_submit=2))
    again = Batched()
    store, results = _run(tmp_path, again)
    assert [[custom_id for custom_id, _, _ in items] for items in again.submitted] == [["row-2"]]
    assert again.direct == 0 and len(store.results()) == 3
    assert all(r["answers"]["importance"]["source"] == "system2" for r in results[:3])


def test_escalations_go_out_as_one_batch_and_decisions_are_logged_as_live(tmp_path):
    system2 = Batched()
    store, results = _run(tmp_path, system2)
    assert [items for items in system2.submitted] == [[(f"row-{i}", f"event {i}", {"importance": QUESTIONS["importance"]}) for i in range(3)]]
    assert system2.direct == 0
    assert results[3] == {"error": "no 'text'"}
    live = Router(FakeDecider(), FakeClaude()).decide("event 0", QUESTIONS)
    for result in results[:3]:
        assert result["escalated"] == ["importance"] and result["answers"]["importance"] | {"tried": None} == live["answers"]["importance"] | {"tried": None}
        assert result["latency_ms"]["system2"] is None
    assert sorted(store.states({r["id"] for r in results[:3]}).values()) == ["event 0", "event 1", "event 2"]
    assert len(store.results()) == 3


def test_a_stopped_run_resumes_without_asking_or_logging_again(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, Batched(stop=True))
    again = Batched()
    store, results = _run(tmp_path, again)
    assert again.submitted == [] and again.checks == 2  # the batch sent before is waited for, not sent again
    assert len(store.results()) == 3 and all(r["answers"]["importance"]["source"] == "system2" for r in results[:3])
    with pytest.raises(ValueError, match="other rows or questions"):
        _run(tmp_path, Batched(), data=DATA[:2])


def test_a_network_blip_while_waiting_does_not_end_the_run(tmp_path):
    system2 = Batched(flaky=True)
    router = Router(FakeDecider(), system2, log=Store(tmp_path / "sureshot.db"), questions={"events": QUESTIONS}, retries=0)
    said = []
    results = decide_rows(router, "events", DATA, tmp_path / "out.progress", poll=0, say=said.append)
    assert len(system2.submitted) == 1 and all(r["answers"]["importance"]["source"] == "system2" for r in results[:3])
    assert any("could not check a batch" in message for message in said)


def test_what_a_batch_did_not_answer_is_asked_directly(tmp_path):
    system2 = Batched(drop={"row-1"})
    _, results = _run(tmp_path, system2)
    assert system2.direct == 1 and all(r["answers"]["importance"]["source"] == "system2" for r in results[:3])


def test_a_system2_without_batches_is_asked_as_live(tmp_path):
    system2 = FakeClaude()
    store, results = _run(tmp_path, system2, workers=2)
    assert all(r["answers"]["importance"]["source"] == "system2" for r in results[:3]) and len(store.results()) == 3
    (tmp_path / "other").mkdir()
    system2 = Batched()
    _, results = _run(tmp_path / "other", system2, escalate=False)
    assert all(r["escalated"] == [] for r in results[:3]) and system2.submitted == [] and system2.direct == 0


def test_the_cli_writes_each_row_with_its_answers(tmp_path, capsys, monkeypatch):
    source = tmp_path / "events.csv"
    write_rows(source, DATA)
    router = Router(FakeDecider(), Batched(), log=Store(tmp_path / "sureshot.db"), questions={"events": QUESTIONS})
    monkeypatch.setattr(cli, "_router", lambda args: router)
    cli.main(["decide", "--batch", str(source), "--poll", "0"])
    report = json.loads(capsys.readouterr().out)
    assert report == {"rows": 4, "decided": 3, "errors": 1, "escalated": 3, "out": str(tmp_path / "events.decided.csv")}
    out = rows(tmp_path / "events.decided.csv")
    assert out[0]["id"] == "0" and out[0]["sureshot_importance"] == "4" and out[0]["sureshot_importance_source"] == "system2"
    assert out[0]["sureshot_needs_prep_source"] == "tier 0" and out[0]["sureshot_needs_prep_confidence"] == "0.966"
    assert out[3]["sureshot_error"] == "no 'text'" and not (tmp_path / "events.decided.csv.progress").exists()


def test_parquet_round_trip_and_through_the_cli(tmp_path, capsys, monkeypatch):
    write_rows(tmp_path / "a.parquet", [{"text": "x"}, {"text": "y", "extra": 1}])
    assert rows(tmp_path / "a.parquet") == [{"text": "x", "extra": None}, {"text": "y", "extra": 1}]
    router = Router(FakeDecider(), Batched(), log=Store(tmp_path / "sureshot.db"), questions={"events": QUESTIONS})
    monkeypatch.setattr(cli, "_router", lambda args: router)
    cli.main(["decide", "--batch", str(tmp_path / "a.parquet"), "--out", str(tmp_path / "b.parquet"), "--poll", "0"])
    out = rows(tmp_path / "b.parquet")
    assert json.loads(capsys.readouterr().out)["decided"] == 2
    assert out[1]["extra"] == 1 and out[1]["sureshot_importance"] == 4 and out[1]["sureshot_needs_prep"] is True


def test_openai_batches_upload_jsonl_and_read_the_output_file(monkeypatch):
    reasoner = ChatReasoner("gpt-5-mini", "https://api.openai.com/v1", "key")
    assert reasoner.batches and not ChatReasoner("m", "http://localhost:8000/v1").batches
    q = {"importance": QUESTIONS["importance"]}
    posted, status = [], {"status": "in_progress"}

    def post(url, **kwargs):
        posted.append((url, kwargs))
        return _response({"id": "file-1"} if url.endswith("/files") else {"id": "batch-1"})

    def get(url, **kwargs):
        if url.endswith("/batches/batch-1"):
            return _response(status)
        reply = {"choices": [{"message": {"content": json.dumps({"importance": {"answer": 3, "reason": "r"}})}}]}
        return _response(None, "\n".join(json.dumps(line) for line in [
            {"custom_id": "row-0", "response": {"status_code": 200, "body": reply}},
            {"custom_id": "row-1", "response": {"status_code": 500, "body": {}}}]))

    monkeypatch.setattr(backends, "post", post)
    monkeypatch.setattr(backends, "get", get)
    handle = reasoner.batch_submit([("row-0", "a", q), ("row-1", "b", q)])
    upload, create = posted
    lines = [json.loads(line) for line in upload[1]["files"]["file"][1].decode().splitlines()]
    assert handle == {"openai": "batch-1"} and upload[1]["data"] == {"purpose": "batch"}
    assert lines[0]["custom_id"] == "row-0" and lines[0]["url"] == "/v1/chat/completions" and lines[0]["body"]["model"] == "gpt-5-mini"
    assert create[1]["json"] == {"input_file_id": "file-1", "endpoint": "/v1/chat/completions", "completion_window": "24h"}
    assert reasoner.batch_results(handle, {"row-0": q, "row-1": q}) is None
    status.update(status="completed", output_file_id="file-2")
    assert reasoner.batch_results(handle, {"row-0": q, "row-1": q}) == {"row-0": {"importance": {"answer": 3, "reason": "r"}}}
    # An expired batch gives what it finished; a failed one, nothing, so its rows are asked directly.
    status.update(status="expired")
    assert reasoner.batch_results(handle, {"row-0": q, "row-1": q}) == {"row-0": {"importance": {"answer": 3, "reason": "r"}}}
    status.clear()
    status.update(status="failed", errors={"data": [{"message": "bad"}]})
    assert reasoner.batch_results(handle, {"row-0": q}) == {}


def test_claude_batches_drop_fallbacks_and_read_succeeded_results():
    text = SimpleNamespace(type="text", text=json.dumps({"importance": {"answer": 2, "reason": "r"}}))
    message = SimpleNamespace(stop_reason="end_turn", content=[text])
    created, state = [], {"status": "in_progress"}
    batches = SimpleNamespace(
        create=lambda requests: created.append(requests) or SimpleNamespace(id="msgbatch_1"),
        retrieve=lambda batch_id: SimpleNamespace(processing_status=state["status"]),
        results=lambda batch_id: [SimpleNamespace(custom_id="row-0", result=SimpleNamespace(type="succeeded", message=message)),
                                  SimpleNamespace(custom_id="row-1", result=SimpleNamespace(type="errored"))])
    claude = ClaudeAPI(client=SimpleNamespace(messages=SimpleNamespace(batches=batches)))
    q = {"importance": QUESTIONS["importance"]}
    handle = claude.batch_submit([("row-0", "a", q), ("row-1", "b", q)])
    params = created[0][0]["params"]
    assert handle == {"anthropic": "msgbatch_1"} and "fallbacks" not in params and "betas" not in params
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert claude.batch_results(handle, {"row-0": q, "row-1": q}) is None
    state["status"] = "ended"
    assert claude.batch_results(handle, {"row-0": q, "row-1": q}) == {"row-0": {"importance": {"answer": 2, "reason": "r"}}}


def _response(data, text=None):
    import httpx

    request = httpx.Request("GET", "https://api.openai.com")
    return httpx.Response(200, json=data, request=request) if text is None else httpx.Response(200, text=text, request=request)
