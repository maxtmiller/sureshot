"""The source is always a real SQLite file; the target is Postgres when SURESHOT_TEST_POSTGRES is set, else SQLite."""
import json
import sqlite3
import pytest
import sureshot.cli as cli
import sureshot.migrate as migrate_module
from sureshot import Cache, Store, calibrate
from sureshot.db import SQLite
from sureshot.migrate import migrate
from sureshot.router import Router
from test_router import QUESTIONS, Scripted


def _sqlite_log(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    with monkeypatch.context() as m:
        m.setattr("sureshot.store.database", SQLite)
        m.setattr("sureshot.cache.database", SQLite)
        store, cache = Store(path, seed=0), Cache(path)
    q = {"needs_prep": QUESTIONS["needs_prep"]}
    for i in range(80):
        confidence = 0.95 if i % 2 else 0.6
        result = Router(Scripted(confidence), None, log=store).decide(f"event {i}", q, ref=f"ticket-{i}")
        store.label(result["id"], "needs_prep", i % 4 != 0 if confidence == 0.6 else True)
    calibrate(store, 0.95, 30)
    store.save_skips(1, ["needs_prep"], [])
    store.save_unconfirmed(0, ["other"], [])
    store.mark_posted([(result["id"], "needs_prep")])
    cache.put_many({"k1": {"answer": True}, "k2": {"answer": False}})
    return path, store


def _snapshot(store):
    return (store.results(), store.labels(), store.thresholds(), store.skipped(1), store.unconfirmed(), store.posted(),
            store.calibrations(), store.issues(), store.find("ticket-7"))


def test_migrate_copies_every_table_and_is_idempotent(tmp_path, monkeypatch):
    path, source = _sqlite_log(tmp_path, monkeypatch)
    target = tmp_path / "new.db"
    monkeypatch.setattr(migrate_module, "CHUNK", 7)  # several chunks per table
    report = migrate(path, target)
    assert report["decisions"] == {"rows": 80, "copied": 80}
    assert report["labels"] == {"rows": 80, "copied": 80}
    assert report["cache"] == {"rows": 2, "copied": 2}
    assert _snapshot(Store(target)) == _snapshot(source)
    assert Cache(target).get_many(["k1", "k2"]) == {"k1": {"answer": True}, "k2": {"answer": False}}

    again = migrate(path, target)
    assert all(t["copied"] == 0 for t in again.values())
    assert _snapshot(Store(target)) == _snapshot(source)


def test_migrate_keeps_rows_the_target_already_has(tmp_path, monkeypatch):
    path, source = _sqlite_log(tmp_path, monkeypatch)
    target = tmp_path / "new.db"
    migrate(path, target)
    first = next(iter(source.labels()))
    Store(target).label(first[0], "needs_prep", not source.labels()[first])
    source.label(source.results()[-1][0], "needs_prep", False)
    report = migrate(path, target)
    assert report["labels"]["copied"] == 0
    assert Store(target).labels()[first] is not source.labels()[first]


def test_migrate_reads_a_log_from_an_older_sureshot(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE decisions (id TEXT PRIMARY KEY, ts REAL, state TEXT, questions TEXT, images INTEGER, result TEXT)")
        db.execute("CREATE TABLE labels (decision_id TEXT, key TEXT, answer TEXT, source TEXT, PRIMARY KEY (decision_id, key))")
        result = {"answers": {"ok": {"answer": True, "confidence": 0.9, "source": "sureshot"}}, "unsure": [], "audit_rate": 0.02}
        db.execute("INSERT INTO decisions VALUES ('d1', 1.5, 'text', ?, 0, ?)", (json.dumps({"ok": {"type": "noul"}}), json.dumps(result)))
        db.execute("INSERT INTO labels VALUES ('d1', 'ok', 'true', 'human')")
    db.close()
    report = migrate(path, tmp_path / "new.db")
    assert set(report) == {"decisions", "labels"}
    store = Store(tmp_path / "new.db")
    assert store.results() == [("d1", 1.5, {"ok": {"type": "noul"}}, result)]
    assert store.labels() == {("d1", "ok"): True}
    assert store.find("anything") is None


def test_migrate_needs_a_sureshot_sqlite_file(tmp_path):
    with pytest.raises(ValueError, match="no SQLite file"):
        migrate(tmp_path / "missing.db", tmp_path / "new.db")
    assert not (tmp_path / "missing.db").exists()
    sqlite3.connect(tmp_path / "other.db").execute("CREATE TABLE things (x)").connection.close()
    with pytest.raises(ValueError, match="no sureshot tables"):
        migrate(tmp_path / "other.db", tmp_path / "new.db")


def test_migrate_command(tmp_path, capsys, monkeypatch):
    path, _ = _sqlite_log(tmp_path, monkeypatch)
    cli.main(["migrate", "--from", str(path), "--to", str(tmp_path / "new.db")])
    assert json.loads(capsys.readouterr().out)["decisions"] == {"rows": 80, "copied": 80}
    with pytest.raises(SystemExit, match="--from is the SQLite file"):
        cli.main(["migrate", "--from", "postgresql://x/y", "--to", str(tmp_path / "new.db")])
