import io
import json
import pytest
import sureshot.cli as cli
from sureshot import Store, calibrate
from sureshot.router import Router
from test_router import QUESTIONS, FakeClaude, FakeDecider, Scripted, _bag


def run(capsys, *argv):
    cli.main(list(argv))
    return json.loads(capsys.readouterr().out)


def _labeled_store(tmp_path):
    store = Store(tmp_path / "sureshot.db", seed=0)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    for i in range(120):
        confidence = 0.95 if i % 2 else 0.6
        result = Router(Scripted(confidence), None, log=store).decide("event", q)
        store.label(result["id"], "needs_prep", i % 4 != 0 if confidence == 0.6 else True)
    return store


def test_store_commands_match_the_library(tmp_path, capsys, monkeypatch):
    store = _labeled_store(tmp_path)
    db = str(tmp_path / "sureshot.db")
    expected = calibrate(store, 0.95, 30, save=False)["needs_prep"]
    report = run(capsys, "calibrate", "--db", db, "--target", "0.95")["needs_prep"]
    assert {k: report[k] for k in ("threshold", "samples", "escalation_rate")} == {k: expected[k] for k in ("threshold", "samples", "escalation_rate")}
    assert store.thresholds() == {"needs_prep": 0.95}
    monkeypatch.setenv("SURESHOT_DB", db)
    assert run(capsys, "recheck")["needs_prep"]["threshold"] == 0.95
    assert run(capsys, "stats", "--costs", '{"tier 0": 0, "system2": 0.004}')["cost"] is not None
    assert isinstance(run(capsys, "review", "--limit", "5"), list)
    assert run(capsys, "prune", "--costs", '{"tier 0": 0.001, "system2": 0.004}', "--min-answers", "10")["needs_prep"]["skipped"] is False


def test_store_commands_need_an_existing_log(tmp_path, monkeypatch):
    monkeypatch.delenv("SURESHOT_DB", raising=False)
    with pytest.raises(SystemExit, match="SURESHOT_DB"):
        cli.main(["stats"])
    with pytest.raises(SystemExit, match="no log"):
        cli.main(["stats", "--db", str(tmp_path / "typo.db")])
    with pytest.raises(SystemExit, match="system2"):
        cli.main(["prune", "--db", str(_labeled_store(tmp_path).path), "--costs", '{"tier 0": 0}'])


def test_decide_reads_a_request_from_stdin(capsys, monkeypatch):
    monkeypatch.setattr(cli, "router_from_env", lambda: Router(FakeDecider(), FakeClaude()))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"state": "event", "questions": QUESTIONS})))
    result = run(capsys, "decide")
    assert result["answers"]["event_type"]["answer"] == "interview" and result["escalated"] == ["importance"]
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"questions": QUESTIONS})))
    with pytest.raises(SystemExit, match="state"):
        cli.main(["decide"])


def test_test_fails_below_min_accuracy(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "router_from_env", lambda: Router(FakeDecider(), FakeClaude()))
    golden, questions = tmp_path / "golden.jsonl", tmp_path / "questions.json"
    golden.write_text("\n".join(json.dumps(c) for c in [{"state": "a", "expected": {"importance": 4, "event_type": "interview"}},
                                                         {"state": "b", "expected": {"event_type": "meeting"}}]))
    questions.write_text(json.dumps(QUESTIONS))
    assert run(capsys, "test", str(golden), "--questions", str(questions), "--min-accuracy", "0.6")["overall"]["accuracy"] == pytest.approx(2 / 3)
    with pytest.raises(SystemExit) as exit:
        cli.main(["test", str(golden), "--questions", str(questions), "--min-accuracy", "0.9"])
    assert exit.value.code == 1 and json.loads(capsys.readouterr().out)["failures"]


def test_retrain_saves_the_classifier_only_when_it_changes(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sureshot.local.sentence_embedder", lambda name: _bag)
    store = Store(tmp_path / "sureshot.db", seed=0)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    for text, prep in [("server meeting", True), ("meeting about server", True), ("invoice sale", False), ("sale invoice due", False)] * 40:
        store.label(Router(FakeDecider(), None, log=store).decide(text, q)["id"], "needs_prep", prep)
    model = tmp_path / "local.joblib"
    args = ["retrain", "--db", str(tmp_path / "sureshot.db"), "--model", str(model), "--target", "0.8", "--min-samples", "5"]
    assert run(capsys, *args)["needs_prep"]["promoted"] and model.exists()
    saved = model.stat().st_mtime_ns
    assert not run(capsys, *args)["needs_prep"]["promoted"] and model.stat().st_mtime_ns == saved
