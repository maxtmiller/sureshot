import csv
import json
import os
import pytest
import sureshot.cli as cli
from sureshot import Store, calibrate, review_queue, stats
from sureshot.importer import import_rows, parse, rows
from sureshot.review import in_sample
from sureshot.router import Router
from test_router import QUESTIONS, FakeClaude, FakeDecider


def test_parse_reads_cells_as_a_csv_writes_them():
    noul, score, choice = QUESTIONS["needs_prep"], QUESTIONS["importance"], QUESTIONS["event_type"]
    assert [parse(noul, v) for v in ("yes", "FALSE", "1", True, "n")] == [True, False, True, True, False]
    assert [parse(score, v) for v in ("3", "critical", 0)] == [3, 4, 0]
    assert [parse(choice, v) for v in ("meeting", "Interview")] == ["meeting", "interview"]
    assert parse(choice, "") is None and parse(noul, None) is None
    for question, value in ((noul, "maybe"), (score, "9"), (choice, "meet")):
        with pytest.raises(ValueError, match="not an answer"):
            parse(question, value)


def _csv(path, records):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["text", "needs_prep", "importance", "event_type"])
        writer.writeheader()
        writer.writerows(records)
    return path


def test_imported_rows_are_logged_and_labeled(tmp_path):
    store = Store(tmp_path / "sureshot.db", seed=0)
    path = _csv(tmp_path / "labels.csv", [
        {"text": "Interview on Friday", "needs_prep": "yes", "importance": "4", "event_type": "interview"},
        {"text": "Team sync", "event_type": "meeting"},
        {"text": "Team sync", "event_type": "meeting"},
        {"text": "", "event_type": "meeting"},
        {"text": "Lunch", "importance": "huge"},
        {"text": "No labels here"},
    ])
    report = import_rows(Router(FakeDecider(), FakeClaude(), log=store), store, QUESTIONS, rows(path))
    assert {k: report[k] for k in ("rows", "imported", "already", "labels")} == {"rows": 6, "imported": 2, "already": 0, "labels": 4}
    assert report["skipped"] == {"duplicate text": 1, "no text": 1, "invalid label": 1, "no labels": 1}
    assert report["errors"] == ["row 5: 'huge' is not an answer to this score question"]
    labels = store.labels()
    assert len(labels) == 4 and all(i.startswith("import-") for i, _ in labels)
    with store._db() as db:
        assert {s for (s,) in db.execute("SELECT source FROM labels").fetchall()} == {"import"}
    graded = store.labeled()
    assert graded and all(row["kind"] == "human" for row in graded)
    with store._db() as db:
        assert all(not json.loads(r)["escalated"] for (r,) in db.execute("SELECT result FROM decisions").fetchall())  # System 1 only
    again = import_rows(Router(FakeDecider(), FakeClaude(), log=store), store, QUESTIONS, rows(path))
    assert again["imported"] == 0 and again["already"] == 2


def test_imported_decisions_are_not_live_traffic(tmp_path):
    store = Store(tmp_path / "sureshot.db", seed=0)
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps({"text": f"event {i}", "needs_prep": True, "importance": 3}) for i in range(40)))
    import_rows(Router(FakeDecider(), FakeClaude(), log=store), store, QUESTIONS, rows(path), escalate=True)
    live = Router(FakeDecider(), FakeClaude(), log=store).decide("live event", QUESTIONS)["id"]
    assert [i for i, _, _, _ in store.results()] == [live]
    assert {item["id"] for item in review_queue(store, None, sample=1.0)} == {live}
    assert stats(store)["decisions"] == 1
    assert not any(in_sample(i, 1.0) for i, _ in store.labels())
    # The labels grade answers, but none of them is a random review label, so none can confirm a threshold.
    report = calibrate(store, 0.5, min_samples=10, sample=1.0, save=False)
    assert report["needs_prep"]["label_check"]["graded"] == 0 and report["needs_prep"]["graded_by"]["human"] > 0


def test_cli_import(tmp_path, monkeypatch, capsys):
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"events": QUESTIONS}))
    db = str(tmp_path / "sureshot.db")
    monkeypatch.setattr(cli, "router_from_env", lambda: Router(FakeDecider(), FakeClaude(), log=Store(os.environ["SURESHOT_DB"]), questions=os.environ["SURESHOT_QUESTIONS"]))
    path = _csv(tmp_path / "labels.csv", [{"text": f"event {i}", "event_type": "meeting"} for i in range(5)])
    cli.main(["import", str(path), "--db", db, "--questions", str(questions), "--limit", "3"])
    assert json.loads(capsys.readouterr().out)["imported"] == 3
    monkeypatch.setattr(cli, "router_from_env", lambda: Router(FakeDecider(), FakeClaude()))
    monkeypatch.delenv("SURESHOT_DB")
    with pytest.raises(SystemExit, match="SURESHOT_DB"):
        cli.main(["import", str(path)])
