import hashlib
import hmac
import json
import time
from urllib.parse import urlencode
import httpx
import pytest
import sureshot.cli as cli
from sureshot import Store, review_queue, slack
from sureshot.router import Router
from test_router import QUESTIONS, FakeClaude, FakeDecider

MANY = {"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": f"Option {i}" for i in range(8)}}


@pytest.fixture
def sent(monkeypatch):
    """The messages posted to Slack, with Slack answering ok."""
    messages = []

    def fake(url, json=None, **kwargs):
        messages.append((url, json, kwargs))
        return httpx.Response(200, json={"ok": True}, request=httpx.Request("POST", url))

    monkeypatch.setattr(slack, "post", fake)
    monkeypatch.setenv("SURESHOT_SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SURESHOT_SLACK_CHANNEL", "C1")
    return messages


def _store(tmp_path, n=2):
    store = Store(tmp_path / "sureshot.db", seed=0)
    for i in range(n):
        Router(FakeDecider(), FakeClaude(), log=store).decide(f"event {i} <!channel>", QUESTIONS)
    return store


def test_blocks_show_the_content_and_an_answer_per_option(tmp_path):
    items = review_queue(_store(tmp_path, 1), sample=1.0)
    out = slack.blocks(items)
    assert out[0]["text"] == {"type": "plain_text", "text": "event 0 <!channel>"}  # plain text: no mentions
    actions = {b["block_id"].split(":", 2)[2]: b["elements"] for b in out if b["type"] == "actions"}
    assert [e["text"]["text"] for e in actions["needs_prep"]] == ["Yes", "No"]
    assert [json.loads(e["value"]) for e in actions["importance"]] == [0, 1, 2, 3, 4]
    assert [json.loads(e["value"]) for e in actions["event_type"]] == ["meeting", "interview"]
    item = items[0] | {"key": "many", "question": MANY, "candidates": []}
    menu = slack.blocks([item])[-1]["elements"][0]
    assert menu["type"] == "static_select" and len(menu["options"]) == 8
    huge = item | {"question": MANY | {"criteria": {f"o{i}": "" for i in range(101)}}}
    assert not slack.answerable(huge) and slack.answerable(item)


def test_post_queue_sends_one_message_per_decision_once(tmp_path, sent):
    store = _store(tmp_path)
    items = review_queue(store, sample=1.0)
    assert slack.post_queue(store, items) == {"messages": 2, "questions": 6, "unanswerable": 0}
    url, body, kwargs = sent[0]
    assert url == "https://slack.com/api/chat.postMessage" and body["channel"] == "C1" and kwargs["headers"]["Authorization"] == "Bearer xoxb-test"
    assert slack.post_queue(store, items) == {"messages": 0, "questions": 0, "unanswerable": 0} and len(sent) == 2
    assert review_queue(store, sample=1.0, skip=store.posted()) == []


def test_a_decision_with_many_questions_is_split_across_messages(tmp_path, monkeypatch, sent):
    store = _store(tmp_path, 1)
    monkeypatch.setattr(slack, "PER_MESSAGE", 2)
    assert slack.post_queue(store, review_queue(store, sample=1.0)) == {"messages": 2, "questions": 3, "unanswerable": 0}
    assert all(len(body["blocks"]) <= 1 + 3 * 2 for _, body, _ in sent)


def test_post_queue_errors(tmp_path, monkeypatch, sent):
    store = _store(tmp_path, 1)
    monkeypatch.setattr(slack, "post", lambda url, **kw: httpx.Response(200, json={"ok": False, "error": "not_in_channel"}, request=httpx.Request("POST", url)))
    with pytest.raises(RuntimeError, match="not_in_channel"):
        slack.post_queue(store, review_queue(store, sample=1.0))
    assert store.posted() == set()
    monkeypatch.delenv("SURESHOT_SLACK_CHANNEL")
    with pytest.raises(ValueError, match="SURESHOT_SLACK_CHANNEL"):
        slack.post_queue(store, [])


def _sign(body, secret="shh", timestamp=None):
    timestamp = str(int(timestamp or time.time()))
    return {"X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": "v0=" + hmac.new(secret.encode(), f"v0:{timestamp}:".encode() + body, hashlib.sha256).hexdigest()}


def test_verify_checks_the_signature_and_its_age():
    body = b"payload=x"
    assert slack.verify({k.lower(): v for k, v in _sign(body).items()}, body, "shh")
    assert not slack.verify({k.lower(): v for k, v in _sign(body, "other").items()}, body, "shh")
    assert not slack.verify({k.lower(): v for k, v in _sign(body, timestamp=time.time() - 600).items()}, body, "shh")
    assert not slack.verify({}, body, "shh")


def _click(store, key, element=0, user="U1"):
    item = next(i for i in review_queue(store, sample=1.0) if i["key"] == key)
    message = {"text": "sureshot", "blocks": slack.blocks([i for i in review_queue(store, sample=1.0) if i["id"] == item["id"]])}
    block = next(b for b in message["blocks"] if b.get("block_id") == f"sureshot:{item['id']}:{key}")
    action = dict(block["elements"][element], block_id=block["block_id"])
    return item["id"], {"type": "block_actions", "user": {"id": user}, "actions": [action], "message": message, "response_url": "https://hooks.slack.test/r"}


def test_a_click_labels_and_marks_who_answered(tmp_path):
    store = _store(tmp_path, 1)
    decision_id, payload = _click(store, "event_type", 1)
    reply = slack.handle(store, payload)
    assert store.labels() == {(decision_id, "event_type"): "interview"}
    with store._db() as db:
        assert db.execute("SELECT source FROM labels").fetchall() == [("slack:U1",)]
    marked = [b for b in reply["blocks"] if b["type"] == "context" and "Labeled" in b["elements"][0]["text"]]
    assert reply["replace_original"] and marked[0]["elements"][0]["text"] == "Labeled *interview* by <@U1>"
    assert sum(b["type"] == "actions" for b in reply["blocks"]) == 2  # the other questions can still be answered


def test_a_menu_choice_labels(tmp_path):
    store = _store(tmp_path, 1)
    decision_id, payload = _click(store, "needs_prep")
    action = payload["actions"][0]
    payload["actions"] = [{"type": "static_select", "block_id": action["block_id"], "action_id": "answer",
                           "selected_option": {"text": {"type": "plain_text", "text": "No"}, "value": "false"}}]
    assert "Labeled *No*" in json.dumps(slack.handle(store, payload)) and store.labels() == {(decision_id, "needs_prep"): False}


def test_clicks_that_cannot_label(tmp_path):
    store = _store(tmp_path, 1)
    _, payload = _click(store, "importance")
    payload["actions"][0]["value"] = "9"
    assert slack.handle(store, payload)["response_type"] == "ephemeral" and store.labels() == {}
    assert slack.handle(store, {"type": "block_actions", "actions": [{"block_id": "other"}]}) is None
    assert slack.handle(store, {"type": "view_submission"}) is None


def test_api_saves_the_click_and_updates_the_message(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import sureshot.api as api

    store = _store(tmp_path, 1)
    replies = []
    monkeypatch.setattr(api, "post", lambda url, **kw: replies.append((url, kw["json"])))
    monkeypatch.setattr(api, "_router", Router(FakeDecider(), FakeClaude(), log=store))
    client = TestClient(api.app)
    decision_id, payload = _click(store, "event_type")
    body = urlencode({"payload": json.dumps(payload)}).encode()
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    assert client.post("/slack/actions", content=body, headers=headers | _sign(body)).status_code == 404
    monkeypatch.setenv("SURESHOT_SLACK_SIGNING_SECRET", "shh")
    assert client.post("/slack/actions", content=body, headers=headers | _sign(body, "other")).status_code == 401
    assert client.post("/slack/actions", content=b"x=1", headers=headers | _sign(b"x=1")).status_code == 400
    assert client.post("/slack/actions", content=body, headers=headers | _sign(body)).status_code == 200
    assert store.labels() == {(decision_id, "event_type"): "meeting"}
    assert replies[0][0] == "https://hooks.slack.test/r" and replies[0][1]["replace_original"]


def test_cli_review_slack(tmp_path, capsys, sent):
    _store(tmp_path)
    db = str(tmp_path / "sureshot.db")
    cli.main(["review", "--db", db, "--sample", "1", "--slack", "--limit", "4"])
    assert json.loads(capsys.readouterr().out) == {"messages": 2, "questions": 4, "unanswerable": 0}
    cli.main(["review", "--db", db, "--sample", "1", "--slack", "--channel", "C2"])
    assert json.loads(capsys.readouterr().out) == {"messages": 1, "questions": 2, "unanswerable": 0} and sent[-1][1]["channel"] == "C2"
