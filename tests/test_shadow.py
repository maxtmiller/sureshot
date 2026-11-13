import httpx
import json
import pytest
import sureshot.cli as cli
from sureshot import Store, calibrate, shadow
from sureshot.formats import anthropic_request, gemini_request, model_answer, openai_request
from sureshot.router import Router
from sureshot.shadow import report
from test_formats import GEMINI_BODY, SCHEMA, FirstOption, _chat
from test_router import QUESTIONS, FakeClaude


class Local:
    """Knows only event_type, and is sure only about the easy rows."""

    def decide(self, state, questions, images=()):
        if "event_type" not in questions:
            return {}
        easy = state.startswith("easy")
        return {"event_type": {"answer": "meeting" if easy else "interview", "confidence": 0.99 if easy else 0.5}}


class LLM:
    """Right and sure on easy and medium rows, wrong and unsure on hard ones."""

    def decide(self, state, questions, images=()):
        hard = state.startswith("hard")
        right = {"event_type": "meeting", "needs_prep": True, "importance": 2}
        wrong = {"event_type": "interview", "needs_prep": False, "importance": 4}
        return {k: {"answer": (wrong if hard else right)[k], "confidence": 0.6 if hard else 0.97} for k in questions}


def _rows():
    return [{"text": f"{kind} {i}", "event_type": "meeting", "needs_prep": "yes"} for kind, n in (("easy", 60), ("medium", 60), ("hard", 30)) for i in range(n)]


def test_shadow_replays_the_cascade_against_your_model():
    router = Router([Local(), LLM()], FakeClaude())
    report = shadow(router, QUESTIONS, _rows(), costs={"tier 0": 0.0, "tier 1": 0.001, "system2": 0.01})
    assert report["shadowed"] == 150 and report["skipped"] == {}
    event = report["questions"]["event_type"]
    assert event["tiers"][0]["threshold"] == 0.99 and event["tiers"][1]["threshold"] == 0.97
    assert event["kept"] == pytest.approx(0.8) and event["agreement"] == 1.0
    assert 0.95 < event["low"] < 1.0
    assert set(report["questions"]) == {"event_type", "needs_prep"}  # no row has an importance answer
    assert list(report["questions"]["needs_prep"]["tiers"]) == [1]
    assert report["kept"] == pytest.approx(0.8)
    # event_type: medium rows pay tier 1, hard rows pay tier 1 and your model; needs_prep never asks tier 0, which has no threshold
    assert report["cost"]["current"] == pytest.approx(3.0)
    assert report["cost"]["sureshot"] == pytest.approx(60 * 0.001 + 30 * 0.011 + 150 * 0.001 + 30 * 0.01)
    assert report["cost"]["savings"] == pytest.approx(1 - 0.84 / 3.0)


def test_shadow_skips_rows_it_cannot_replay():
    data = [{"text": "", "event_type": "meeting"}, {"text": "easy", "event_type": "party"}, {"text": "easy"}, {"text": "easy", "event_type": "meeting"}]
    report = shadow(Router([Local(), LLM()], FakeClaude()), QUESTIONS, data)
    assert report["shadowed"] == 1 and report["cost"] is None
    assert report["skipped"] == {"no text": 1, "invalid answer": 1, "no answers": 1}
    assert report["errors"] == ["row 2: 'party' is not an answer to this choice question"]
    assert report["questions"]["event_type"]["kept"] == 0.0 and report["agreement"] is None  # too few rows to certify a threshold


def test_cli_shadow(tmp_path, monkeypatch, capsys):
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"events": QUESTIONS}))
    path = tmp_path / "history.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in _rows()))
    monkeypatch.setattr(cli, "router_from_env", lambda: Router([Local(), LLM()], FakeClaude(), questions=str(questions)))
    cli.main(["shadow", str(path), "--questions", str(questions), "--costs", '{"tier 1": 0.001, "system2": 0.01}'])
    report = json.loads(capsys.readouterr().out)
    assert report["kept"] == pytest.approx(0.8) and report["cost"]["savings"] == pytest.approx(0.72)


# Live shadow mode: the drop-in routes forward everything and log what System 1 would have said.
ANSWER = {"team": "support", "urgent": True, "priority": 5, "size": 40}
FMT = {"type": "json_schema", "json_schema": {"name": "ticket", "schema": SCHEMA}}


@pytest.fixture
def live(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import sureshot.api as api

    monkeypatch.setenv("SURESHOT_API_KEY", "secret")
    monkeypatch.setenv("SURESHOT_SHADOW", "1")
    for fmt, url in (("OPENAI", "https://openai.test/v1"), ("ANTHROPIC", "https://anthropic.test"), ("GEMINI", "https://gemini.test")):
        monkeypatch.setenv(f"SURESHOT_{fmt}_PASSTHROUGH_URL", url)
    store = Store(tmp_path / "sureshot.db")
    monkeypatch.setattr(api, "_router", Router(FirstOption(), None, log=store))
    return TestClient(api.app, headers={"Authorization": "Bearer secret"}), store, api


def _upstream(monkeypatch, api, body, stream=False):
    def post(url, json, headers, timeout):
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    def post_stream(url, json, headers, timeout):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=httpx.ByteStream(body))

    monkeypatch.setattr(api, "post_stream" if stream else "post", post_stream if stream else post)


def _shadowed(store):
    return [result for _, _, _, result in store.results() if result.get("shadow")]


def test_shadow_mode_returns_your_model_s_response_and_logs_system_1_beside_it(live, monkeypatch):
    client, store, api = live
    upstream = {"choices": [{"message": {"role": "assistant", "content": json.dumps(ANSWER)}}], "usage": {"total_tokens": 9}}
    _upstream(monkeypatch, api, upstream)
    response = client.post("/v1/chat/completions", json=_chat(response_format=FMT))
    assert response.json() == upstream and "x-sureshot" not in response.headers
    client.post("/v1/chat/completions", json=_chat())  # free text: forwarded and counted, not logged
    [result] = _shadowed(store)
    team = result["answers"]["ticket.team"]
    assert team["answer"] == "support" and team["source"] == "system2"
    assert team["tried"] == [{"tier": 0, "answer": "billing", "confidence": 0.99, "confident": True, "threshold": 0.8}]
    assert result["answers"]["ticket.size"]["answer"] == 2 and result["escalated"] == list(result["answers"])
    assert store.traffic() == {"structured": 1, "other": 1}
    assert client.get("/stats").json()["decisions"] == 1


def test_shadow_mode_reads_streamed_answers_in_each_format(live, monkeypatch):
    client, store, api = live
    half = json.dumps(ANSWER)[:20], json.dumps(ANSWER)[20:]
    openai = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': part}}]})}\n\n" for part in half) + "data: [DONE]\n\n"
    _upstream(monkeypatch, api, openai.encode(), stream=True)
    assert client.post("/v1/chat/completions", json=_chat(response_format=FMT, stream=True)).text == openai

    tool = [{"name": "route", "input_schema": SCHEMA}]
    events = [{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
              {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Routing."}},
              {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "name": "route", "input": {}}},
              *[{"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": part}} for part in half]]
    _upstream(monkeypatch, api, "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode(), stream=True)
    client.post("/v1/messages", json={"model": "claude-haiku-5-5", "max_tokens": 10, "messages": [{"role": "user", "content": "I was charged twice"}],
                                      "tools": tool, "tool_choice": {"type": "tool", "name": "route"}, "stream": True})

    chunks = [{"candidates": [{"content": {"parts": [{"text": part}]}}]} for part in ('{"team": "support", "urgent": true,', ' "priority": 5}')]
    _upstream(monkeypatch, api, "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode(), stream=True)
    client.post("/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=secret", json=GEMINI_BODY)

    openai_result, anthropic_result, gemini_result = _shadowed(store)
    assert openai_result["answers"]["ticket.priority"]["answer"] == 4
    assert anthropic_result["answers"]["route.team"]["answer"] == "support"
    assert {k: a["answer"] for k, a in gemini_result["answers"].items()} == {"Ticket.team": "support", "Ticket.urgent": True, "Ticket.priority": 4}


def test_model_answers_are_read_or_left_out():
    request = openai_request(_chat(response_format=FMT))
    assert model_answer("OPENAI", b"not json", request) == {}
    partial = json.dumps({"choices": [{"message": {"content": json.dumps({"team": "sales", "urgent": False})}}]}).encode()
    assert model_answer("OPENAI", partial, request) == {"ticket.urgent": False}
    body = {"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}], "tools": [{"name": "route", "input_schema": SCHEMA}],
            "tool_choice": {"type": "any"}}
    message = {"content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "name": "route", "input": ANSWER}]}
    assert model_answer("ANTHROPIC", json.dumps(message).encode(), anthropic_request(body))["route.size"] == 2
    gemini = {"candidates": [{"content": {"parts": [{"text": json.dumps({"team": "billing", "urgent": False, "priority": 1})}]}}]}
    assert model_answer("GEMINI", json.dumps([gemini]).encode(), gemini_request(GEMINI_BODY), stream=True)["Ticket.priority"] == 0


def test_shadow_mode_samples_and_needs_a_log(live, monkeypatch):
    client, store, api = live
    _upstream(monkeypatch, api, {"choices": [{"message": {"content": json.dumps(ANSWER)}}]})
    monkeypatch.setenv("SURESHOT_SHADOW", "0.0001")
    monkeypatch.setattr(api.random, "random", lambda: 0.5)
    client.post("/v1/chat/completions", json=_chat(response_format=FMT))
    assert _shadowed(store) == [] and store.traffic() == {"structured": 1}
    monkeypatch.setattr(api, "_router", Router(FirstOption(), None))
    assert client.post("/v1/chat/completions", json=_chat(response_format=FMT)).status_code == 400


def test_the_shadow_log_reports_and_calibrates(live, monkeypatch):
    client, store, api = live
    for i in range(60):  # your model agrees with FirstOption on team for 50 of 60 requests
        answer = ANSWER | {"team": "billing" if i < 50 else "support"}
        _upstream(monkeypatch, api, {"choices": [{"message": {"content": json.dumps(answer)}}]})
        client.post("/v1/chat/completions", json=_chat(response_format=FMT))
    client.post("/v1/chat/completions", json=_chat())
    out = report(store, target=0.7, costs={"tier 0": 0.0001, "system2": 0.004})
    assert out["shadowed"] == 60 and out["traffic"]["coverage"] == pytest.approx(60 / 61)
    assert out["questions"]["ticket.team"]["agreement"] == pytest.approx(50 / 60) and out["questions"]["ticket.team"]["kept"] == 1.0
    assert out["questions"]["ticket.urgent"]["agreement"] == 1.0 and out["cost"]["savings"] > 0.9
    assert client.get("/shadow", params={"target": 0.7, "costs": json.dumps({"system2": 0.004})}).json()["kept"] == out["kept"]
    assert client.get("/shadow", params={"costs": "{}"}).status_code == 422
    # Going live needs no new data: calibrate learns thresholds from the shadow log.
    assert calibrate(store, 0.7, save=False)["ticket.team"]["threshold"] == 0.99
