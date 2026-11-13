import json
import pytest
import sureshot.cli as cli
from sureshot import shadow
from sureshot.router import Router
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
