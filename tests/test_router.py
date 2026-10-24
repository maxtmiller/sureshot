import pytest
from system1.backends import normalize
from system1.questions import answer_schema, validate
from system1.router import Router

QUESTIONS = {
    "needs_prep": {"type": "noul", "instructions": "Does this event need preparation?"},
    "importance": {"type": "score", "instructions": "How important?", "criteria": ["Optional", "Low", "Normal", "High", "Critical"]},
    "event_type": {"type": "choice", "instructions": "What kind of event?", "criteria": {"meeting": "A meeting", "interview": "An interview"}},
}

# A real pplx-decider response (reflex history), trimmed to two choice options.
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


def test_only_unsure_questions_escalate():
    claude = FakeClaude()
    result = Router(FakeDecider(), claude, threshold=0.8).decide("event", QUESTIONS)
    assert result["unsure"] == ["importance"] and result["escalated"] == ["importance"] and claude.asked == ["importance"]
    assert result["answers"]["importance"]["source"] == "system2" and result["answers"]["importance"]["answer"] == 4
    assert result["answers"]["importance"]["system1"]["answer"] == 3
    assert result["answers"]["event_type"]["source"] == "system1"


def test_per_question_threshold_and_no_escalation():
    claude = FakeClaude()
    result = Router(FakeDecider(), claude, threshold=0.8).decide("event", QUESTIONS, thresholds={"importance": 0.4, "event_type": 0.99})
    assert result["unsure"] == ["event_type"] and claude.asked == ["event_type"]
    kept = Router(FakeDecider(), claude, threshold=0.8).decide("event", QUESTIONS, escalate=False)
    assert kept["unsure"] == ["importance"] and kept["escalated"] == [] and kept["answers"]["importance"]["source"] == "system1"
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
    import system1.api as api

    monkeypatch.setenv("SYSTEM1_API_KEY", "secret")
    monkeypatch.setattr(api, "_router", Router(FakeDecider(), FakeClaude()))
    client = TestClient(api.app)
    body = {"state": "event", "questions": QUESTIONS}
    assert client.post("/decide", json=body).status_code == 401
    assert client.post("/decide", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = client.post("/decide", json=body, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200 and ok.json()["escalated"] == ["importance"]
    bad = client.post("/decide", json={"state": "x", "questions": {}}, headers={"Authorization": "Bearer secret"})
    assert bad.status_code == 422
