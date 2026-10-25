import pytest
from hunch.backends import normalize
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
    from hunch.backends import HTTPDecider

    seen = {}

    def post(url, json, headers, timeout):
        seen.update(url=url, json=json, headers=headers)
        return httpx.Response(200, json={"out": RAW}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
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

    monkeypatch.setattr(httpx, "post", post)
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

    monkeypatch.setattr(httpx, "post", post)
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

    monkeypatch.setattr(httpx, "post", post)
    result = Router(FakeDecider(), ChatReasoner("m", "http://local/v1")).decide("event", QUESTIONS)
    assert seen["url"] == "http://local/v1/chat/completions" and seen["json"]["response_format"]["json_schema"]["strict"]
    assert result["answers"]["importance"]["answer"] == 4


def test_router_from_env(monkeypatch):
    from hunch import ChatDecider, ChatReasoner, router_from_env

    for k, v in {"HUNCH_SYSTEM1_MODEL": "a", "HUNCH_SYSTEM1_URL": "http://one/v1",
                 "HUNCH_SYSTEM2": "chat", "HUNCH_SYSTEM2_MODEL": "b", "HUNCH_THRESHOLD": "0.9"}.items():
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

    monkeypatch.setattr(httpx, "post", post)
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
    monkeypatch.setattr(httpx, "post", respond(tops=["yes", "no"]))
    assert decider.check() is decider
    monkeypatch.setattr(httpx, "post", respond(tops=["<|channel>", "The"]))
    with pytest.raises(RuntimeError, match=r"top tokens were \['<\|channel>', 'The'\].*extra"):
        decider.check()
    monkeypatch.setattr(httpx, "post", respond(logprobs=False))
    with pytest.raises(RuntimeError, match="no logprobs"):
        decider.check()
    monkeypatch.setattr(httpx, "post", respond(status=404))
    with pytest.raises(RuntimeError, match="404.*model not found"):
        decider.check()

    def unreachable(url, json, headers, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", unreachable)
    with pytest.raises(RuntimeError, match="cannot reach http://local/v1/chat/completions"):
        decider.check()


def test_vllm_decider_and_env_option(monkeypatch):
    from hunch import ChatDecider, VLLMDecider, router_from_env

    local = VLLMDecider("Qwen/Qwen3-8B", extra={"top_p": 1})
    assert local.url == "http://localhost:8000/v1/chat/completions" and local.headers == {}
    assert local.extra == {"chat_template_kwargs": {"enable_thinking": False}, "top_p": 1}
    monkeypatch.setenv("HUNCH_SYSTEM1_MODEL", "m")
    monkeypatch.setenv("HUNCH_SYSTEM2", "none")
    assert type(router_from_env().system1) is ChatDecider
    monkeypatch.setenv("HUNCH_SYSTEM1", "vllm")
    assert isinstance(router_from_env().system1, VLLMDecider) and router_from_env().system1.url.startswith("http://localhost:8000")
    monkeypatch.setenv("HUNCH_SYSTEM1_URL", "http://gpu-box:8000/v1")
    assert router_from_env().system1.url == "http://gpu-box:8000/v1/chat/completions"
    monkeypatch.setenv("HUNCH_SYSTEM1", "bogus")
    with pytest.raises(ValueError):
        router_from_env()
