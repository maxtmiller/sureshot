import base64
import hashlib
import hmac
import json
import sqlite3
import pytest
from sureshot import Store, webhooks
from sureshot.router import Router
from test_router import QUESTIONS, FakeClaude, FakeDecider

SPEC = {"verify": "github", "secret": "HOOK_SECRET", "events": [
    {"when": {"action": "labeled"}, "ref": "issue.number",
     "labels": {"event_type": {"from": "label.name", "map": {"interview": "interview", "call": "meeting"}},
                "needs_prep": {"value": False}}},
    {"when": {"action": "prioritized"}, "id": "sureshot_id", "labels": {"importance": {"from": "priority"}}},
]}


def _decided(tmp_path, ref="42"):
    store = Store(tmp_path / "sureshot.db", seed=0)
    return store, Router(FakeDecider(), FakeClaude(), log=store).decide("event", QUESTIONS, ref=ref)["id"]


def test_events_label_the_decision_they_are_about(tmp_path):
    store, decision_id = _decided(tmp_path)
    assert store.find("42") == decision_id and store.find("7") is None
    report = webhooks.apply(store, "github", SPEC, {"action": "labeled", "issue": {"number": 42}, "label": {"name": "call"}})
    assert report == [{"id": decision_id, "labeled": {"event_type": "meeting", "needs_prep": False}, "skipped": {}}]
    assert store.labels() == {(decision_id, "event_type"): "meeting", (decision_id, "needs_prep"): False}
    assert webhooks.apply(store, "github", SPEC, {"action": "opened"}) == []
    with store._db() as db:
        assert {s for (s,) in db.execute("SELECT source FROM labels").fetchall()} == {"webhook:github"}


def test_events_that_cannot_label_are_skipped_not_errors(tmp_path):
    store, decision_id = _decided(tmp_path)
    unknown = webhooks.apply(store, "github", SPEC, {"action": "labeled", "issue": {"number": 7}, "label": {"name": "call"}})
    assert unknown == [{"id": None, "labeled": {}, "skipped": {"event_type": "no logged decision", "needs_prep": "no logged decision"}}]
    unmapped = webhooks.apply(store, "github", SPEC, {"action": "labeled", "issue": {"number": 42}, "label": {"name": "wontfix"}})
    assert unmapped[0]["skipped"] == {"event_type": "'wontfix' is not in the map"} and unmapped[0]["labeled"] == {"needs_prep": False}
    invalid = webhooks.apply(store, "github", SPEC, {"action": "prioritized", "sureshot_id": decision_id, "priority": 9})
    assert "not a valid answer" in invalid[0]["skipped"]["importance"]
    missing = webhooks.apply(store, "github", SPEC, {"action": "prioritized", "sureshot_id": decision_id})
    assert missing[0]["skipped"] == {"importance": "no priority in the event"}
    assert webhooks.apply(store, "github", SPEC, {"action": "prioritized", "sureshot_id": decision_id, "priority": 3})[0]["labeled"] == {"importance": 3}


def test_fields_and_maps():
    event = {"labels": [{"name": "a"}, {"name": "b"}], "n": 3, "flag": True}
    assert webhooks.field(event, "labels.1.name") == "b" and webhooks.field(event, "labels.5.name") is None
    assert webhooks.field(event, "n.x") is None and webhooks.field(event, "missing") is None


def test_json_maps_match_numbers_and_booleans(tmp_path):
    store, decision_id = _decided(tmp_path)
    spec = {"events": [{"id": "id", "labels": {"importance": {"from": "p", "map": {"1": 0, "2": 4}}, "needs_prep": {"from": "f", "map": {"true": True}}}}]}
    assert webhooks.apply(store, "x", spec, {"id": decision_id, "p": 2, "f": True})[0]["labeled"] == {"importance": 4, "needs_prep": True}


def test_load_checks_the_file(tmp_path, monkeypatch):
    path = tmp_path / "webhooks.json"
    path.write_text(json.dumps({"github": SPEC}))
    with pytest.raises(ValueError, match="HOOK_SECRET is not set"):
        webhooks.load(path)
    monkeypatch.setenv("HOOK_SECRET", "s")
    assert webhooks.load(path) == {"github": SPEC}
    bad = {"a": {"verify": "gitlab", "events": [{"ref": "x", "id": "y", "labels": {"k": {"from": "f", "value": 1}}}]}, "b": {"verify": "linear", "events": []}}
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError) as e:
        webhooks.load(path)
    for problem in ("a: verify must be one of", "needs either ref or id", "label k: needs either from", "b: linear needs secret", "b: events must be"):
        assert problem in str(e.value)


def _mac(secret, message):
    return hmac.new(secret.encode(), message, hashlib.sha256)


@pytest.mark.parametrize("kind, headers", [
    ("github", lambda body: {"X-Hub-Signature-256": "sha256=" + _mac("s", body).hexdigest()}),
    ("jira", lambda body: {"X-Hub-Signature": "sha256=" + _mac("s", body).hexdigest()}),
    ("linear", lambda body: {"Linear-Signature": _mac("s", body).hexdigest()}),
    ("zendesk", lambda body: {"X-Zendesk-Webhook-Signature-Timestamp": "2026-11-01T00:00:00Z",
                              "X-Zendesk-Webhook-Signature": base64.b64encode(_mac("s", b"2026-11-01T00:00:00Z" + body).digest()).decode()}),
    ("bearer", lambda body: {"Authorization": "Bearer s"}),
])
def test_api_verifies_each_source(tmp_path, monkeypatch, kind, headers):
    from fastapi.testclient import TestClient
    import sureshot.api as api

    store, decision_id = _decided(tmp_path)
    monkeypatch.setenv("HOOK_SECRET", "s")
    monkeypatch.setattr(api, "_router", Router(FakeDecider(), FakeClaude(), log=store))
    monkeypatch.setattr(api, "_webhooks", {"src": SPEC | {"verify": kind}})
    client = TestClient(api.app)
    body = json.dumps({"action": "prioritized", "sureshot_id": decision_id, "priority": 4}).encode()
    response = client.post("/feedback/webhook/src", content=body, headers=headers(body))
    assert response.status_code == 200 and response.json()["events"][0]["labeled"] == {"importance": 4}
    assert client.post("/feedback/webhook/src", content=body + b" ", headers=headers(body)).status_code == (200 if kind == "bearer" else 401)
    assert client.post("/feedback/webhook/src", content=body).status_code == 401
    assert client.post("/feedback/webhook/other", content=body, headers=headers(body)).status_code == 404
    assert client.post("/feedback/webhook/src", content=b"not json", headers=headers(b"not json")).status_code == 400


def test_api_decide_takes_a_ref(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import sureshot.api as api

    store = Store(tmp_path / "sureshot.db")
    monkeypatch.setenv("SURESHOT_API_KEY", "secret")
    monkeypatch.setattr(api, "_router", Router(FakeDecider(), FakeClaude(), log=store))
    decision = TestClient(api.app).post("/decide", json={"state": "event", "questions": QUESTIONS, "ref": "ticket-1"},
                                        headers={"Authorization": "Bearer secret"}).json()
    assert store.find("ticket-1") == decision["id"]


@pytest.mark.sqlite
def test_a_log_from_before_refs_gets_the_column(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE decisions (id TEXT PRIMARY KEY, ts REAL, state TEXT, questions TEXT, images INTEGER, result TEXT)")
        db.execute("INSERT INTO decisions VALUES ('old', 0, 's', '{}', 0, '{}')")
    store = Store(path)
    assert store.find("anything") is None
    store.log("s", {}, {}, ref="r")
    assert store.find("r") is not None


def test_the_example_file_loads(monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("ZENDESK_WEBHOOK_SECRET", "z")
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "g")
    assert set(webhooks.load(Path(__file__).parents[1] / "examples" / "webhooks.yaml")) == {"zendesk", "github"}
