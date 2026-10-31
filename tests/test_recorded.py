"""Replays real OpenAI exchanges recorded by record_responses.py, with no network. Fails if an adapter now sends a
different request than the one recorded (rerun the recorder after a deliberate change), or if it cannot parse what
the real API returned."""
from json import dumps, loads
from pathlib import Path

import httpx
import pytest

import sureshot.adapters.base as backends
from sureshot import ChatDecider, ChatReasoner
from record_responses import STATE
from test_router import QUESTIONS

RECORDINGS = Path(__file__).parent / "recordings"


@pytest.fixture
def replay(monkeypatch):
    def use(name):
        exchanges = {dumps(e["request"], sort_keys=True): e for e in loads((RECORDINGS / f"{name}.json").read_text())}

        def post(url, json, headers, timeout):
            key = dumps(json, sort_keys=True)
            assert key in exchanges, f"request differs from the {name} recording; rerun tests/record_responses.py if that is intended"
            e = exchanges[key]
            assert url == e["url"]
            return httpx.Response(e["status"], json=e["response"], request=httpx.Request("POST", url))

        monkeypatch.setattr(backends, "post", post)
    return use


def test_openai_decider_reads_real_logprobs(replay):
    replay("openai_decider")
    answers = ChatDecider("gpt-4o-mini", provider="openai", api_key="k").decide(STATE, QUESTIONS)
    assert isinstance(answers["needs_prep"]["answer"], bool)
    assert answers["importance"]["answer"] in range(5)
    assert answers["event_type"]["answer"] in QUESTIONS["event_type"]["criteria"]
    for a in answers.values():
        assert 0 < a["confidence"] <= 1 and sum(a["probabilities"].values()) == pytest.approx(1)


@pytest.mark.parametrize("name, json_mode", [("openai_reasoner", "schema"), ("openai_reasoner_object", "object")])
def test_openai_reasoner_parses_real_replies(replay, name, json_mode):
    replay(name)
    answers = ChatReasoner("gpt-4o-mini", provider="openai", api_key="k", json_mode=json_mode).answer(STATE, QUESTIONS)
    assert set(answers) == set(QUESTIONS) and all(a["reason"] for a in answers.values())


def test_a_real_error_keeps_the_servers_explanation(replay):
    replay("openai_deprecated_model")
    with pytest.raises(httpx.HTTPStatusError, match="deprecated"):
        ChatDecider("gpt-4.1-nano", provider="openai", api_key="k").decide(STATE, {"needs_prep": QUESTIONS["needs_prep"]})
