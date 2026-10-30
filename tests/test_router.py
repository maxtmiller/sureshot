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


def test_chat_decider_numbers_the_options_past_26(monkeypatch):
    import math
    from hunch import ChatDecider

    intents = {f"intent_{i}": f"Intent {i}" for i in range(77)}
    sent = []

    def post(url, json, headers, timeout):
        sent.append(json["messages"][1]["content"][-1]["text"])
        return _logprob_response(url, [("52", math.log(0.9)), (" 62", math.log(0.09)), ("Z", math.log(0.01))])

    monkeypatch.setattr(backends, "post", post)
    answer = ChatDecider("m", "http://local/v1", api_key="k").decide("my card", {"intent": {"type": "choice", "criteria": intents}})["intent"]
    assert "1: Intent 0" in sent[0] and "77: Intent 76" in sent[0] and "A:" not in sent[0]
    # "Z" is not a label here, so the confidence is renormalized over the two numbers.
    assert answer["answer"] == "intent_51" and answer["confidence"] == pytest.approx(0.9 / 0.99)


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

    return Store(tmp_path / "hunch.db", seed=0)


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
    auth = {"Authorization": "Bearer secret"}
    body = client.get("/stats", headers=auth).json()
    assert body["decisions"] == 1 and body["questions"]["importance"]["answered_by"] == {"system2": 1.0} and body["cost"] is None
    # The stats options are query parameters; costs is JSON.
    priced = client.get("/stats", headers=auth, params={"costs": '{"tier 0": 0, "system2": 10}', "alpha": 0.05, "min_samples": 5}).json()
    assert priced["cost"]["always_system2"] == 30 and priced["cost"]["total"] == 10
    for params in ({"costs": "not json"}, {"costs": '{"tier 0": 1}'}, {"costs": '{"system2": "cheap"}'}, {"costs": "[1]"}, {"alpha": 2}):
        assert client.get("/stats", headers=auth, params=params).status_code == 422, params


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
    assert stats(store)["review_queue"] == {"sample": 0, "confident": 0, "disagreement": 1, "audit_disagreement": 1, "unresolved": 1}
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
                      "distribution_change": None, "change": {}, "flags": [],
                      "unchecked": ["escalation_rate", "distribution", "accuracy", "change_escalation_rate", "change_distribution", "change_accuracy"],
                      "baseline": None, "upstream": []}]
    # The same mix as at calibration is not drift. Labels on unaudited answers are not a random sample, so even all
    # wrong they do not count toward accuracy.
    for i in range(60):
        result = Router(Tier({"needs_prep": (True, 0.95 if i % 2 else 0.7)}), None, log=store).decide("event", q)
        store.label(result["id"], "needs_prep", False)
    drift = stats(store)["drift"][0]
    assert drift["flags"] == [] and drift["unchecked"] == ["accuracy", "change_accuracy"] and drift["accuracy_since"] is None
    # Now the tier is unsure about everything: the share escalated moved.
    for _ in range(240):
        Router(Tier({"needs_prep": (True, 0.7)}), None, log=store).decide("event", q)
    drift = stats(store)["drift"][0]
    assert drift["flags"] == ["escalation_rate", "distribution", "change_escalation_rate", "change_distribution"] and drift["answers_since"] == 300 and drift["escalation_rate"]["since"] == pytest.approx(0.9)

    # Audits say the confident answers are wrong: accuracy fell below the target, once there are enough of them.
    wrong = calibrated(tmp_path / "b.db")
    audited = Router(Tier({"needs_prep": (True, 0.95)}), Says(False), log=wrong, audit=1.0)
    unsure = Router(Tier({"needs_prep": (True, 0.7)}), Says(False), log=wrong, audit=1.0)
    for _ in range(20):
        audited.decide("event", q), unsure.decide("event", q)
    drift = stats(wrong)["drift"][0]
    assert drift["flags"] == [] and drift["unchecked"] == ["accuracy", "change_accuracy"] and drift["accuracy_since"] == {"graded": 20, "human": 0, "accuracy": 0.0}
    for _ in range(20):
        audited.decide("event", q), unsure.decide("event", q)
    drift = stats(wrong)["drift"][0]
    assert drift["flags"] == ["accuracy", "change_accuracy"] and drift["unchecked"] == [] and drift["accuracy_since"]["graded"] == 40
    # A label overrides the audit's answer: the person says these were right after all.
    for decision_id, _, _, result in wrong.results(drift["calibrated_at"]):
        if "audit" in result["answers"]["needs_prep"]:
            wrong.label(decision_id, "needs_prep", True)
    assert stats(wrong)["drift"][0]["flags"] == [] and stats(wrong)["drift"][0]["accuracy_since"] == {"graded": 40, "human": 40, "accuracy": 1.0}

    # A recalibration that fails closed drops the snapshot, since the tier is no longer trusted for the question.
    for _ in range(200):
        result = Router(Tier({"needs_prep": (True, 0.95)}), None, log=wrong).decide("event", q)
        wrong.label(result["id"], "needs_prep", False)
    assert calibrate(wrong, 0.95, 30)["needs_prep"]["reason"] == "target unreachable" and stats(wrong)["drift"] == []
    # retrain promoting a new classifier drops the snapshot too: its threshold and confidences are new.
    store.save_calibrations({}, 0, ["needs_prep"])
    assert stats(store)["drift"] == []


def test_change_detector_catches_a_late_change_the_cumulative_check_dilutes(tmp_path):
    from hunch import Store, calibrate, stats

    q = {"needs_prep": QUESTIONS["needs_prep"]}
    store = Store(tmp_path / "hunch.db")
    for i in range(100):
        sure = i % 2 == 0
        result = Router(Tier({"needs_prep": (sure, 0.95 if sure else 0.7)}), None, log=store).decide("event", q)
        store.label(result["id"], "needs_prep", True)
    calibrate(store, 0.95, 30)
    # 2000 answers like calibration, then the tier turns unsure about the last 300.
    for i in range(2000):
        Router(Tier({"needs_prep": (True, 0.95 if i % 2 else 0.7)}), None, log=store).decide("event", q)
    for _ in range(300):
        Router(Tier({"needs_prep": (True, 0.7)}), None, log=store).decide("event", q)
    drift = stats(store)["drift"][0]
    assert drift["escalation_rate"]["since"] == pytest.approx(1300 / 2300) and "escalation_rate" not in drift["flags"]
    assert "change_escalation_rate" in drift["flags"]
    times = [ts for _, ts, _, _ in store.results(drift["calibrated_at"])]
    assert times.index(drift["change"]["change_escalation_rate"]["started_at"]) >= 1950


def test_drift_sees_confidence_sliding_while_still_above_the_threshold(tmp_path):
    from hunch import Store, calibrate, stats

    q = {"needs_prep": QUESTIONS["needs_prep"]}
    store = Store(tmp_path / "hunch.db")
    # Right at 0.99 and 0.97, wrong at 0.7, a third each: the threshold is 0.97 and a third falls below it.
    for i in range(99):
        confidence = (0.99, 0.97, 0.7)[i % 3]
        result = Router(Tier({"needs_prep": (confidence > 0.9, confidence)}), None, log=store).decide("event", q)
        store.label(result["id"], "needs_prep", True)
    assert calibrate(store, 0.95, 30)["needs_prep"]["threshold"] == 0.97
    # The tier is less sure: what was 0.99 is now 0.97. Still above the threshold, so the share escalated holds.
    for i in range(90):
        Router(Tier({"needs_prep": (True, (0.97, 0.97, 0.7)[i % 3])}), None, log=store).decide("event", q)
    drift = stats(store)["drift"][0]
    assert drift["escalation_rate"] == {"at_calibration": pytest.approx(1 / 3), "since": pytest.approx(1 / 3)}
    assert drift["flags"] == ["distribution"]


def test_a_log_saved_before_the_distribution_check_still_opens(tmp_path):
    import sqlite3
    from hunch import Store, stats

    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE calibrations (tier INTEGER, key TEXT, ts REAL, threshold REAL, target REAL, escalation_rate REAL, answers INTEGER, PRIMARY KEY (tier, key))")
    db.execute("INSERT INTO calibrations VALUES (0, 'needs_prep', 0, 0.9, 0.95, 0.5, 100)")
    db.commit()
    db.close()
    store = Store(path)
    assert store.calibrations()[0]["histogram"] is None and Store(path).calibrations() == store.calibrations()
    for _ in range(40):
        Router(Tier({"needs_prep": (True, 0.95)}), None, log=store).decide("event", {"needs_prep": QUESTIONS["needs_prep"]})
    assert "distribution" in stats(store)["drift"][0]["unchecked"]


def test_drift_needs_a_change_big_enough_to_act_on(tmp_path):
    from hunch import Store, stats

    q = {"needs_prep": QUESTIONS["needs_prep"]}
    store = Store(tmp_path / "hunch.db")
    # A calibration on a large log: half the answers at 0.7, below the threshold of 0.95.
    store.save_calibrations({"needs_prep": {"threshold": 0.95, "target": 0.95, "escalation_rate": 0.5, "answers": 100000,
                                            "histogram": {"edges": [0.7, 0.95], "counts": [0, 50000, 50000]}}}, 0, ["needs_prep"])
    # Now 60% fall below it: too many answers for that to be chance, but smaller than these minimums ask for.
    for i in range(2000):
        Router(Tier({"needs_prep": (True, 0.7 if i % 50 < 30 else 0.95)}), None, log=store).decide("event", q)
    drift = stats(store)["drift"][0]
    assert drift["escalation_rate"]["since"] == pytest.approx(0.6) and drift["distribution_change"] == pytest.approx(0.1)
    assert stats(store, min_escalation_change=0.15, min_distribution_change=0.15)["drift"][0]["flags"] == []
    assert stats(store, min_escalation_change=0, min_distribution_change=0)["drift"][0]["flags"] == [
        "escalation_rate", "distribution", "change_escalation_rate", "change_distribution"]


def test_anytime_bounds_hold_over_every_look():
    import random
    from hunch.stats import _rate_moved, _sequence

    # A rate that never changes, checked after every answer for 1000 answers in 100 runs, at a 5% budget: a one-off
    # test checked this often would flag most runs; this flags at most about 5 of them.
    random.seed(0)
    false_alarms = 0
    for _ in range(100):
        before = sum(random.random() < 0.3 for _ in range(200)) / 200
        seen = 0
        for n in range(1, 1001):
            seen += random.random() < 0.3
            if n >= 30 and _rate_moved(seen / n, n, before, 200, 0.05):
                false_alarms += 1
                break
    assert false_alarms <= 8
    low, high = _sequence(30, 100, 0.01)
    assert low < 0.3 < high and high - low > _sequence(300, 1000, 0.01)[1] - _sequence(300, 1000, 0.01)[0]
    assert _sequence(0, 50, 0.01)[0] == 0.0 and _sequence(50, 50, 0.01)[1] == 1.0


def test_cusum_rarely_raises_a_false_alarm_and_dates_a_real_change():
    import math
    import random
    from hunch.stats import _cusum

    random.seed(1)
    h = math.log(20000)
    # No change: a flag within 2000 answers at most 2000 / 20000 of the time.
    assert sum(_cusum([random.random() < 0.2 for _ in range(2000)], 0.2, 0.25, h) is not None for _ in range(100)) <= 20
    # A change at answer 500 from 0.2 to 0.35 is flagged soon after, and dated near it.
    xs = [random.random() < 0.2 for _ in range(500)] + [random.random() < 0.35 for _ in range(2000)]
    start, flagged = _cusum(xs, 0.2, 0.25, h)
    assert start >= 400 and flagged < 1500


def test_drift_points_at_the_upstream_tier(tmp_path):
    from hunch import Store, calibrate, stats

    q = {"needs_prep": QUESTIONS["needs_prep"]}
    store = Store(tmp_path / "hunch.db")
    # Each tier is right when sure (0.95) and wrong when not (0.7), half each, so both calibrate to 0.95.
    for i in range(200):
        fast = Tier({"needs_prep": (True, 0.95) if i % 2 == 0 else (False, 0.7)})
        llm = Tier({"needs_prep": (True, 0.95) if i % 4 == 1 else (False, 0.7)})
        result = Router([fast, llm], None, threshold=[0.8, 0.8], log=store).decide("event", q)
        store.label(result["id"], "needs_prep", True)
    assert calibrate(store, 0.95, 30, tier=0)["needs_prep"]["threshold"] == 0.95
    assert calibrate(store, 0.95, 30, tier=1)["needs_prep"]["threshold"] == 0.95
    # Tier 0 now passes everything down, and tier 1 is unsure about what it gets: both drift, and tier 0 is why.
    for _ in range(100):
        Router([Tier({"needs_prep": (False, 0.7)}), Tier({"needs_prep": (False, 0.7)})], None, log=store).decide("event", q)
    tier0, tier1 = stats(store)["drift"]
    assert (tier0["tier"], tier1["tier"]) == (0, 1) and "escalation_rate" in tier0["flags"] and "escalation_rate" in tier1["flags"]
    assert tier0["upstream"] == [] and tier1["upstream"] == [{"tier": 0, "reason": "drift"}]
    # Recalibrating tier 0 changes what tier 1 is asked, until tier 1 is recalibrated too.
    calibrate(store, 0.95, 30, tier=0)
    tier0, tier1 = stats(store)["drift"]
    assert tier0["flags"] == [] and tier1["upstream"] == [{"tier": 0, "reason": "recalibrated"}]
    calibrate(store, 0.95, 30, tier=1)
    assert stats(store)["drift"][1]["upstream"] == []


def test_drift_watches_questions_without_a_calibration(tmp_path):
    from hunch import Store, calibrate, stats

    q = {"needs_prep": QUESTIONS["needs_prep"]}
    store = Store(tmp_path / "hunch.db")
    # Never calibrated: its first 30 answers are the baseline (all sure, so all kept at the default 0.8).
    for _ in range(30):
        Router(Tier({"needs_prep": (True, 0.95)}), None, log=store).decide("event", q)
    drift = stats(store)["drift"]
    assert len(drift) == 1 and drift[0]["baseline"] == {"answers": 30} and drift[0]["calibrated_at"] is None
    assert drift[0]["threshold"] is None and drift[0]["answers_since"] == 0
    for _ in range(40):
        Router(Tier({"needs_prep": (True, 0.95)}), None, log=store).decide("event", q)
    drift = stats(store)["drift"][0]
    assert drift["flags"] == [] and drift["unchecked"] == ["accuracy", "change_accuracy"] and drift["escalation_rate"] == {"at_calibration": 0.0, "since": 0.0}
    # Now it is unsure about everything: it passes more on, and its confidences moved.
    for _ in range(60):
        Router(Tier({"needs_prep": (True, 0.6)}), None, log=store).decide("event", q)
    assert stats(store)["drift"][0]["flags"] == ["escalation_rate", "distribution", "change_escalation_rate", "change_distribution"]

    # A calibration that fails closed starts a new baseline: the tier now passes everything on, which is not drift.
    for _ in range(100):
        result = Router(Tier({"needs_prep": (True, 0.95)}), None, log=store).decide("event", q)
        store.label(result["id"], "needs_prep", False)
    assert calibrate(store, 0.95, 30)["needs_prep"]["reason"] == "target unreachable" and stats(store)["drift"] == []
    for _ in range(70):
        Router(Tier({"needs_prep": (True, 0.95)}), None, log=store).decide("event", q)
    drift = stats(store)["drift"][0]
    assert drift["flags"] == [] and drift["escalation_rate"] == {"at_calibration": 1.0, "since": 1.0} and drift["answers_since"] == 40


def test_retrain_saves_a_drift_snapshot_for_what_it_promotes(tmp_path):
    from hunch import calibrate, retrain, stats

    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    for text, prep in [("server meeting", True), ("meeting about server", True), ("invoice sale", False), ("sale invoice due", False)] * 40:
        store.label(Router(FakeDecider(), None, log=store).decide(text, q)["id"], "needs_prep", prep)
    # The fake model is right half the time, so 0.4 is a target this data can certify.
    calibrate(store, 0.4, 30)
    assert [d["calibrated_at"] is not None for d in stats(store)["drift"]] == [True]
    assert retrain(store, target_accuracy=0.8, min_samples=5, embed=_bag)["report"]["needs_prep"]["promoted"]
    # The promoted classifier replaces the old snapshot with its own: its threshold, the target, and its confidences
    # on the logged decisions it was not trained on (the held-out ones), so drift checks its accuracy too.
    import zlib

    held = sum(zlib.crc32(i.encode()) % 100 < 30 for i, _, _, _ in store.results())
    (snapshot,) = store.calibrations()
    assert snapshot["target"] == 0.8 and snapshot["threshold"] == store.thresholds(0)["needs_prep"] and snapshot["answers"] == held
    assert store.resets() == {} and sum(snapshot["histogram"]["counts"]) == held
    drift = stats(store)["drift"]
    assert len(drift) == 1 and drift[0]["calibrated_at"] == snapshot["ts"] and drift[0]["baseline"] is None


def test_stats_prices_each_stage_and_what_it_saves(tmp_path):
    from hunch import stats

    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"], "importance": QUESTIONS["importance"]}
    # Tier 0 keeps needs_prep and passes importance on; tier 1 is unsure too, so System 2 answers it.
    fast, llm = Tier({"needs_prep": (True, 0.99), "importance": (1, 0.5)}), Tier({"importance": (2, 0.6)})
    for _ in range(10):
        Router([fast, llm], FakeClaude(), threshold=[0.8, 0.8], log=store).decide("event", q)
    costs = {"tier 0": 0.0, "tier 1": 1.0, "system2": 10.0}
    cost = stats(store, costs=costs)["cost"]
    assert cost["total"] == 110 and cost["per_decision"] == 11 and cost["always_system2"] == 200 and cost["savings"] == pytest.approx(0.45)
    # Tier 0 is free and saves a System 2 question every time; tier 1 is asked every time and never keeps an answer.
    assert cost["tiers"] == {"tier 0": {"cost": 0.0, "saves_vs_system2": 100.0, "net": 100.0},
                             "tier 1": {"cost": 10.0, "saves_vs_system2": 0.0, "net": -10.0}}
    assert stats(store, costs=costs | {"ms": 1.0})["cost"]["total"] > 110 and stats(store)["cost"] is None
    # Audits are System 2 questions too: both confident answers of this decision are audited.
    Router(Tier({"needs_prep": (True, 0.99), "importance": (1, 0.99)}), FakeClaude(), log=store, audit=1.0).decide("event", q)
    assert stats(store, costs=costs)["cost"]["total"] == 110 + 20
    with pytest.raises(ValueError, match="system2"):
        stats(store, costs={"tier 0": 1.0})


def test_a_random_labeling_sample_measures_the_router_against_its_target(tmp_path):
    from hunch import review_queue, stats

    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    router = Router(Tier({"needs_prep": (True, 0.99)}), FakeClaude(), log=store)
    for _ in range(300):
        router.decide("event", q)
    # Nothing disagrees, so without a sample nothing is queued; with one, a fixed random share is, first.
    assert review_queue(store) == []
    queue = review_queue(store, limit=None, sample=0.1)
    assert 15 <= len(queue) <= 45 and {i["reason"] for i in queue} == {"sample"}
    assert [i["id"] for i in review_queue(store, limit=None, sample=0.1)] == [i["id"] for i in queue]
    assert stats(store, sample=0.1)["review_queue"]["sample"] == len(queue)
    # A person says nine in ten of the sampled answers were wrong: the router is far below the target.
    store.save_issues({"needs_prep": {"reason": "target unreachable", "best_target": 0.5, "target": 0.95}}, 0, ["needs_prep"])
    for n, item in enumerate(queue):
        store.label(item["id"], "needs_prep", n % 10 == 0)
    on_sample = stats(store, sample=0.1)["questions"]["needs_prep"]["accuracy_on_sample"]
    assert on_sample["graded"] == len(queue) and on_sample["accuracy"] == pytest.approx(sum(n % 10 == 0 for n in range(len(queue))) / len(queue))
    assert on_sample["low"] < on_sample["accuracy"] < on_sample["high"] < 0.95 and on_sample["target"] == 0.95 and on_sample["below_target"] is True
    assert stats(store)["questions"]["needs_prep"]["accuracy_on_sample"] is None


def test_trust_uncalibrated_false_escalates_until_there_is_a_threshold(tmp_path):
    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    sure = Tier({"needs_prep": (True, 0.99)})
    cautious = Router(sure, FakeClaude(), log=store, trust_uncalibrated=False)
    result = cautious.decide("event", q)
    assert result["escalated"] == ["needs_prep"] and result["answers"]["needs_prep"]["hunch"]["confidence"] == 0.99
    # A threshold passed per call, or saved by calibrate or retrain, counts as calibrated.
    assert cautious.decide("event", q, thresholds={"needs_prep": 0.9})["escalated"] == []
    store.save_thresholds({"needs_prep": 0.9}, 0)
    assert cautious.decide("event", q)["escalated"] == []
    # The default keeps trusting the router's threshold, and without a log there is nothing to calibrate with.
    (tmp_path / "b").mkdir()
    assert Router(sure, FakeClaude(), log=_calibration_store(tmp_path / "b")).decide("event", q)["escalated"] == []
    assert Router(sure, FakeClaude(), trust_uncalibrated=False).decide("event", q)["escalated"] == []


def test_a_tier_that_failed_closed_is_only_probed(tmp_path):
    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"], "importance": QUESTIONS["importance"]}
    store.save_thresholds({"needs_prep": float("inf")}, 0)
    fast = Tier({"needs_prep": (True, 0.99), "importance": (3, 0.99)})
    result = Router(fast, FakeClaude(), log=store, probe=0.0).decide("event", q)
    # needs_prep skips the tier and goes to System 2; importance is still asked.
    assert fast.asked == [["importance"]] and result["escalated"] == ["needs_prep"] and result["answers"]["needs_prep"]["tried"] == []
    probed = Tier({"needs_prep": (True, 0.99), "importance": (3, 0.99)})
    assert Router(probed, FakeClaude(), log=store, probe=1.0).decide("event", q)["answers"]["needs_prep"]["tried"][0]["tier"] == 0
    # With nowhere to pass it on, the tier is asked and its answer stands as unsure.
    alone = Tier({"needs_prep": (True, 0.99), "importance": (3, 0.99)})
    result = Router(alone, None, log=store, probe=0.0).decide("event", q)
    assert alone.asked == [["needs_prep", "importance"]] and result["unsure"] == ["needs_prep"]
    # A tier asked about nothing is not called at all.
    idle = Tier({"needs_prep": (True, 0.99)})
    Router([idle, Tier({"needs_prep": (True, 0.5)})], FakeClaude(), log=store, probe=0.0).decide("event", {"needs_prep": QUESTIONS["needs_prep"]})
    assert idle.asked == []


def test_retrain_rechecks_the_classifier_it_keeps(tmp_path):
    from hunch import retrain, stats

    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    docs = [("server meeting", True), ("meeting about server", True), ("invoice sale", False), ("sale invoice due", False)] * 40
    ids = []
    for text, prep in docs:
        ids.append(Router(FakeDecider(), None, log=store).decide(text, q)["id"])
        store.label(ids[-1], "needs_prep", prep)
    first = retrain(store, target_accuracy=0.8, min_samples=5, embed=_bag)
    assert first["report"]["needs_prep"]["promoted"] and first["report"]["needs_prep"]["kept"] is None
    # Nothing new: the same classifier is kept, checked again and passes, and its snapshot is refreshed.
    snapshot = store.calibrations()[0]
    second = retrain(store, current=first["decider"], target_accuracy=0.8, min_samples=5, embed=_bag)
    assert second["decider"] is first["decider"] and second["report"]["needs_prep"]["kept"] == "rechecked"
    assert store.calibrations()[0]["ts"] > snapshot["ts"] and store.thresholds(0)["needs_prep"] < float("inf")
    # A target its held-out decisions are too few to show (0.99) but do not rule out leaves its threshold alone.
    before = store.thresholds(0)["needs_prep"]
    unproven = retrain(store, current=first["decider"], target_accuracy=0.99, min_samples=5, embed=_bag)
    assert unproven["report"]["needs_prep"]["kept"] == "unproven" and store.thresholds(0)["needs_prep"] == before
    # The world changes: every label is now the opposite, and the target is 0.99. The old classifier fails its
    # recheck, so it is no longer trusted; a new one has too few held-out decisions to certify 0.99 yet.
    for i, (_, prep) in zip(ids, docs):
        store.label(i, "needs_prep", not prep)
    third = retrain(store, current=first["decider"], target_accuracy=0.99, min_samples=5, embed=_bag)
    assert third["report"]["needs_prep"]["kept"] == "failed" and store.thresholds(0)["needs_prep"] == float("inf")
    assert (0, "needs_prep") in store.resets() and not third["report"]["needs_prep"]["promoted"] and stats(store)["drift"] == []


def test_drift_accuracy_counts_labels_on_the_random_sample(tmp_path):
    from hunch import Store, calibrate, review_queue, stats

    q = {"needs_prep": QUESTIONS["needs_prep"]}
    store = Store(tmp_path / "hunch.db")
    for i in range(100):
        sure = i % 2 == 0
        store.label(Router(Tier({"needs_prep": (sure, 0.95 if sure else 0.7)}), None, log=store).decide("event", q)["id"], "needs_prep", True)
    calibrate(store, 0.95, 30)
    # No audits, so without the sample the accuracy check has nothing to go on.
    for _ in range(400):
        Router(Tier({"needs_prep": (True, 0.95)}), FakeClaude(), log=store).decide("event", q)
    assert "accuracy" in stats(store)["drift"][0]["unchecked"]
    # People label the sample and find the kept answers mostly wrong.
    for n, item in enumerate(review_queue(store, limit=None, sample=0.2)):
        store.label(item["id"], "needs_prep", n % 5 == 0)
    drift = stats(store, sample=0.2)["drift"][0]
    assert drift["accuracy_since"]["graded"] >= 30 and drift["accuracy_since"]["human"] == drift["accuracy_since"]["graded"]
    assert "accuracy" in drift["flags"] and "accuracy" not in drift["unchecked"]


def test_calibration_checks_a_threshold_against_labels_on_the_random_sample(tmp_path):
    from hunch import calibrate, review_queue

    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    # The tier always agrees with System 2 (the audits), so graded by System 2 it looks perfect.
    router = Router(Tier({"needs_prep": (True, 0.99)}), FakeClaude(), log=store, audit=1.0)
    for _ in range(300):
        router.decide("event", q)
    report = calibrate(store, 0.9, 30, save=False)["needs_prep"]
    assert report["threshold"] == 0.99 and report["graded_by"] == {"human": 0, "system2": 300}
    # People label the random sample, and say a third of it was wrong. The rest is still graded by System 2, so
    # without the sample check the threshold passes; with it, calibration fails closed.
    sample = review_queue(store, limit=None, sample=0.1)
    for n, item in enumerate(sample):
        store.label(item["id"], "needs_prep", n % 3 != 0)
    assert calibrate(store, 0.9, 30, save=False)["needs_prep"]["threshold"] == 0.99
    checked = calibrate(store, 0.9, 30, sample=0.1)["needs_prep"]
    assert checked["reason"] == "below target on labels" and checked["threshold"] is None and store.thresholds() == {"needs_prep": float("inf")}
    assert checked["label_check"]["graded"] == len(sample) and checked["label_check"]["high"] < 0.9
    assert checked["graded_by"] == {"human": len(sample), "system2": 300 - len(sample)}


def test_prune_stops_asking_a_tier_that_costs_more_than_it_saves(tmp_path):
    from hunch import prune

    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    store.save_thresholds({"needs_prep": 0.9}, 0)
    # The tier is sure about one question in ten, so it keeps a tenth of its answers.
    for i in range(200):
        Router(Tier({"needs_prep": (True, 0.95 if i % 10 == 0 else 0.5)}), FakeClaude(), log=store).decide("event", q)
    # In money alone it pays (a tenth of 10 saved for 0.5); counting its latency it does not (2 a question).
    report = prune(store, {"tier 0": 0.5, "system2": 10})["needs_prep"]
    assert report == {"answers": 200, "keep_rate": 0.1, "net": pytest.approx(0.5), "skipped": False} and store.skipped(0) == set()
    assert prune(store, {"tier 0": 2.0, "system2": 10})["needs_prep"]["skipped"] and store.skipped(0) == {"needs_prep"}
    # A skipped tier is only probed; with probe=0 it is not asked, and System 2 answers.
    idle = Tier({"needs_prep": (True, 0.95)})
    result = Router(idle, FakeClaude(), log=store, probe=0.0).decide("event", q)
    assert idle.asked == [] and result["escalated"] == ["needs_prep"]
    # Once it pays again it is asked again; too few answers leaves it as it is.
    assert not prune(store, {"tier 0": 0.5, "system2": 10})["needs_prep"]["skipped"] and store.skipped(0) == set()
    assert prune(store, {"tier 0": 2.0, "system2": 10}, min_answers=1000) == {} and store.skipped(0) == set()
    with pytest.raises(ValueError, match="system2"):
        prune(store, {"tier 0": 1.0})


def test_recheck_fails_a_certified_tier_its_labeled_sample_rules_out(tmp_path):
    from hunch import calibrate, recheck, review_queue, stats

    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    # The tier always agrees with System 2 (the audits), so calibration certifies it.
    router = Router(Tier({"needs_prep": (True, 0.99)}), FakeClaude(), log=store, audit=1.0)
    for _ in range(300):
        router.decide("event", q)
    assert calibrate(store, 0.9, 30)["needs_prep"]["threshold"] == 0.99
    # Nothing labeled yet: nothing to go on, nothing changes.
    assert recheck(store, sample=0.1)["needs_prep"] == {"threshold": 0.99, "target": 0.9, "graded": 0, "accuracy": None, "low": None, "high": None, "failed": False, "confirmed": True}
    # A few wrong labels are not enough to rule the target out; many are.
    sample = review_queue(store, limit=None, sample=0.1)
    store.label(sample[0]["id"], "needs_prep", False)
    assert not recheck(store, sample=0.1)["needs_prep"]["failed"] and store.thresholds() == {"needs_prep": 0.99}
    for item in sample:
        store.label(item["id"], "needs_prep", False)
    report = recheck(store, sample=0.1)["needs_prep"]
    assert report["failed"] and report["graded"] == len(sample) and report["accuracy"] == 0.0
    assert store.thresholds() == {"needs_prep": float("inf")} and store.issues()[0]["reason"] == "below target on labels"
    # A tier that already failed closed is not rechecked.
    assert recheck(store, sample=0.1) == {}


def test_retrain_learns_from_questions_no_tier_answered(tmp_path):
    from hunch import LocalDecider, retrain

    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    # Every tier is skipped (a classifier with no heads answers nothing), so System 2 answers every question.
    router = Router(LocalDecider({}, _bag), FakeClaude(), log=store)
    for text, prep in [("server meeting", True), ("meeting about server", True), ("invoice sale", False), ("sale invoice due", False)] * 40:
        store.label(router.decide(text, q)["id"], "needs_prep", prep)
    assert store.labeled() == [] and len(store.graded()) == 160
    assert retrain(store, target_accuracy=0.8, min_samples=5, embed=_bag)["report"]["needs_prep"]["promoted"]


def test_a_seeded_store_logs_the_same_ids_and_labeling_sample(tmp_path):
    from hunch import Store
    from hunch.review import review_queue

    q = {"needs_prep": QUESTIONS["needs_prep"]}
    runs = []
    for name in ("a.db", "b.db"):
        router = Router(FakeDecider(), FakeClaude(), log=Store(tmp_path / name, seed=3))
        ids = [router.decide(f"meeting {i}", q)["id"] for i in range(50)]
        runs.append((ids, [item["id"] for item in review_queue(router.log, reasons=("sample",), sample=0.2)]))
    assert runs[0] == runs[1] and runs[0][1]
    assert Store(tmp_path / "c.db").log("x", q, {}) != Store(tmp_path / "d.db").log("x", q, {})


def test_recheck_counts_labels_on_a_share_of_the_confident_answers(tmp_path):
    from hunch import Store, recheck, review_queue

    store = Store(tmp_path / "s.db", seed=1)
    store.save_thresholds({"needs_prep": 0.9})
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    router = Router(Tier({"needs_prep": (True, 0.99)}), FakeClaude(), log=store)
    for _ in range(1000):
        router.decide("event", q)
    queue = review_queue(store, limit=None, sample=0.02, confident=0.3)
    assert {item["reason"] for item in queue} == {"sample", "confident"}
    # The tier is right 80% of the time against a 90% target.
    for i, item in enumerate(queue):
        store.label(item["id"], "needs_prep", i % 5 != 0)
    few = recheck(store, sample=0.02, target_accuracy=0.9)["needs_prep"]
    assert not few["failed"] and few["graded"] < 40
    many = recheck(store, sample=0.02, target_accuracy=0.9, confident=0.3)["needs_prep"]
    assert many["failed"] and many["graded"] == len(queue) and many["accuracy"] == pytest.approx(0.8, abs=0.01)


def test_labels_must_confirm_a_threshold_system2_certified(tmp_path):
    from hunch import calibrate, recheck, review_queue

    store = _calibration_store(tmp_path)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    # The tier always agrees with System 2 (the audits), so graded by System 2 it looks perfect.
    router = Router(Tier({"needs_prep": (True, 0.99)}), FakeClaude(), log=store, audit=1.0)
    for _ in range(300):
        router.decide("event", q)
    # No labels yet: the threshold is saved but not trusted, and the tier is still asked.
    report = calibrate(store, 0.9, 30, confident=0.1)["needs_prep"]
    assert report["threshold"] == 0.99 and report["reason"] == "not enough labels" and not report["label_check"]["confirmed"]
    assert store.thresholds() == {"needs_prep": 0.99} and store.unconfirmed() == {"needs_prep"}
    assert store.issues()[0]["reason"] == "not enough labels"
    result = router.decide("event", q)
    assert result["escalated"] == ["needs_prep"] and result["answers"]["needs_prep"]["tried"][0]["confident"]
    # Its confident answers are queued at random even though none was kept, and once the labels show the target,
    # recheck confirms the threshold and the router keeps the tier's answers.
    queue = review_queue(store, limit=None, confident=0.1)
    assert queue and {item["reason"] for item in queue} == {"confident"}
    for item in queue:
        store.label(item["id"], "needs_prep", True)
    checked = recheck(store, sample=0.0, confident=0.1)["needs_prep"]
    assert checked["confirmed"] and not checked["failed"] and checked["graded"] == len(queue)
    assert store.unconfirmed() == set() and store.issues() == []
    assert router.decide("event", q)["escalated"] == []
