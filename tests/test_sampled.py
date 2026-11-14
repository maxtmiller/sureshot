import itertools
import threading
import pytest
from sureshot import Router, SampledDecider, router_from_env
from test_router import QUESTIONS, FakeClaude


class Scripted:
    """A System 2 that gives the next scripted reply on each call; an exception in the script is raised."""

    identity = "scripted"

    def __init__(self, replies):
        self.replies, self.lock = itertools.cycle(replies), threading.Lock()

    def answer(self, state, questions, images=()):
        with self.lock:
            reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return {k: {"answer": v, "reason": "r"} for k, v in reply.items() if k in questions}


def _reply(event, prep, importance):
    return {"event_type": event, "needs_prep": prep, "importance": importance}


def test_the_share_of_agreeing_samples_is_the_confidence():
    replies = [_reply("meeting", True, 3)] * 3 + [_reply("interview", True, 1), _reply("meeting", False, 3)]
    decided = SampledDecider(Scripted(replies), n=5).decide("Interview Friday", QUESTIONS)
    assert decided["event_type"] == {"answer": "meeting", "confidence": 0.8, "probabilities": {"meeting": 0.8, "interview": 0.2}}
    assert decided["needs_prep"]["answer"] is True and decided["needs_prep"]["probabilities"] == {"true": 0.8, "false": 0.2}
    assert decided["importance"]["answer"] == 3 and decided["importance"]["probabilities"]["1"] == 0.2
    assert decided["importance"]["score"] == pytest.approx((3 * 4 + 1) / 5)


def test_failed_samples_are_left_out_unless_all_fail():
    decided = SampledDecider(Scripted([_reply("meeting", True, 3), RuntimeError("declined")]), n=4).decide("x", QUESTIONS)
    assert decided["event_type"]["confidence"] == 1.0
    with pytest.raises(RuntimeError, match="declined"):
        SampledDecider(Scripted([RuntimeError("declined")]), n=3).decide("x", QUESTIONS)
    with pytest.raises(ValueError):
        SampledDecider(Scripted([]), n=1)
    assert SampledDecider(Scripted([]), n=3).identity == "sampled|3|scripted"


def test_a_sampled_tier_escalates_when_the_votes_split():
    split = Router(SampledDecider(Scripted([_reply("meeting", True, 3), _reply("interview", True, 3)]), n=4), FakeClaude(), threshold=0.8)
    result = split.decide("x", QUESTIONS)
    assert "event_type" in result["escalated"] and result["answers"]["needs_prep"]["source"] == "sureshot"


def test_env_samples_a_model_without_logprobs(monkeypatch):
    from sureshot import ChatReasoner

    monkeypatch.setenv("SURESHOT_SYSTEM1_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("SURESHOT_SYSTEM1_PROVIDER", "openai")
    monkeypatch.setenv("SURESHOT_SYSTEM1_API_KEY", "k")
    monkeypatch.setenv("SURESHOT_SYSTEM1_SAMPLES", "3")
    monkeypatch.setenv("SURESHOT_SYSTEM2", "none")
    system1 = router_from_env().system1
    assert isinstance(system1, SampledDecider) and isinstance(system1.reasoner, ChatReasoner) and system1.n == 3
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("SURESHOT_SYSTEM1", "claude")
    monkeypatch.setenv("SURESHOT_SYSTEM1_MODEL", "claude-haiku-5-5")
    monkeypatch.delenv("SURESHOT_SYSTEM1_SAMPLES")
    system1 = router_from_env().system1
    assert system1.n == 5 and system1.reasoner.model == "claude-haiku-5-5"
