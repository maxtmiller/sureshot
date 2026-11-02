"""Label the review queue from Slack: `sureshot review --slack` posts queue items to a channel, one message per
decision with a button per option (a menu past five), and a click on one, sent by Slack to POST /slack/actions,
labels the question and marks the message with who answered.

It needs a Slack app (examples/slack-manifest.yaml) with the chat:write scope, invited to the channel, and its
Interactivity request URL set to <sureshot>/slack/actions. SURESHOT_SLACK_BOT_TOKEN posts, SURESHOT_SLACK_CHANNEL is
where, and SURESHOT_SLACK_SIGNING_SECRET checks the clicks. Anyone in the channel can label."""
import hashlib
import hmac
import json
import os
import time
from sureshot.adapters.base import ok, post
from sureshot.questions import show

BUTTONS, MENU = 5, 100  # past MENU options a question can't be answered in Slack (a menu's limit)
TEXT = 2900  # a section's text is limited to 3000 characters
PER_MESSAGE = 16


def _cut(text, limit=TEXT):
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _options(question):
    """[(label, answer)] for each answer to a question."""
    if question["type"] == "noul":
        return [("Yes", True), ("No", False)]
    if question["type"] == "score":
        return [(str(level), i) for i, level in enumerate(question["criteria"])]
    return [(key, key) for key in question["criteria"]]


def answerable(item):
    return len(_options(item["question"])) <= MENU


def blocks(items):
    """Block Kit blocks for one decision's queue items (all with the same id): its content as plain text, so nothing
    in it can mention anyone, then each question with what the models said and its answers."""
    out = [{"type": "section", "text": {"type": "plain_text", "text": _cut(items[0]["state"]) or " "}}]
    for item in items:
        question = item["question"]
        out.append({"type": "section", "text": {"type": "mrkdwn", "text": _cut(f"*{_escape(item['key'])}* ({item['reason']}): {_escape(question.get('instructions', ''))}")}})
        said = [f"{c['who']}: {show(question, c['answer'])}" + (f" ({c['confidence']:.2f})" if c.get("confidence") is not None else "") for c in item["candidates"]]
        if said:
            out.append({"type": "context", "elements": [{"type": "mrkdwn", "text": _cut(_escape(" · ".join(said)), 1900)}]})
        options = _options(question)
        block_id = f"sureshot:{item['id']}:{item['key']}"
        if len(options) <= BUTTONS:
            elements = [{"type": "button", "action_id": f"answer:{i}", "text": {"type": "plain_text", "text": _cut(label, 75)}, "value": json.dumps(answer)}
                        for i, (label, answer) in enumerate(options)]
        else:
            elements = [{"type": "static_select", "action_id": "answer", "placeholder": {"type": "plain_text", "text": "Answer"},
                         "options": [{"text": {"type": "plain_text", "text": _cut(label, 75)}, "value": json.dumps(answer)} for label, answer in options]}]
        out.append({"type": "actions", "block_id": block_id, "elements": elements})
    return out


def post_queue(store, items, channel=None, token=None):
    """Posts the queue items not posted before, one message per decision, and records them as posted so the next run
    sends new ones. Returns {messages, questions, unanswerable}: items with more options than a menu holds are left
    for `sureshot review --label`."""
    channel, token = channel or os.environ.get("SURESHOT_SLACK_CHANNEL"), token or os.environ.get("SURESHOT_SLACK_BOT_TOKEN")
    if not channel or not token:
        raise ValueError("set SURESHOT_SLACK_BOT_TOKEN and SURESHOT_SLACK_CHANNEL (or pass a channel)")
    posted = store.posted()
    fresh = [item for item in items if (item["id"], item["key"]) not in posted]
    unanswerable = sum(not answerable(item) for item in fresh)
    by_decision = {}
    for item in fresh:
        if answerable(item):
            by_decision.setdefault(item["id"], []).append(item)
    # A message holds at most 50 blocks: the content and three per question.
    groups = [items[i:i + PER_MESSAGE] for items in by_decision.values() for i in range(0, len(items), PER_MESSAGE)]
    for group in groups:
        response = post("https://slack.com/api/chat.postMessage", headers={"Authorization": f"Bearer {token}"}, timeout=30,
                        json={"channel": channel, "text": f"sureshot: {len(group)} question(s) to label", "blocks": blocks(group)})
        ok(response)
        if not response.json().get("ok"):
            raise RuntimeError(f"Slack refused the message: {response.json().get('error')}")
        store.mark_posted([(item["id"], item["key"]) for item in group])
    return {"messages": len(groups), "questions": sum(map(len, groups)), "unanswerable": unanswerable}


def verify(headers, body, secret=None, now=None):
    """Whether a request (its headers, case-insensitive, and raw body bytes) is from Slack, signed in the last five
    minutes, as Slack asks, so a captured request can't be replayed later."""
    secret = secret or os.environ.get("SURESHOT_SLACK_SIGNING_SECRET")
    timestamp = headers.get("x-slack-request-timestamp", "")
    if not secret or not timestamp.isdigit() or abs((now or time.time()) - int(timestamp)) > 300:
        return False
    expected = "v0=" + hmac.new(secret.encode(), f"v0:{timestamp}:".encode() + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(headers.get("x-slack-signature", "").encode(), expected.encode())


def handle(store, payload):
    """Labels the question a click answered. Returns the message to send to the click's response_url: the original
    with that question's answers replaced by who labeled it, or a note only the clicker sees when it can't be saved.
    None for an interaction that isn't a sureshot answer."""
    action = (payload.get("actions") or [{}])[0]
    if payload.get("type") != "block_actions" or not action.get("block_id", "").startswith("sureshot:"):
        return None
    _, decision_id, key = action["block_id"].split(":", 2)
    answer = json.loads(action["selected_option"]["value"] if "selected_option" in action else action["value"])
    user = payload.get("user", {}).get("id", "unknown")
    try:
        store.label(decision_id, key, answer, source=f"slack:{user}")
    except (KeyError, ValueError) as e:
        return {"response_type": "ephemeral", "replace_original": False, "text": f"Not saved: {e.args[0]}"}
    shown = action.get("selected_option", {}).get("text", {}).get("text") or action.get("text", {}).get("text") or json.dumps(answer)
    done = {"type": "context", "elements": [{"type": "mrkdwn", "text": f"Labeled *{_escape(shown)}* by <@{user}>"}]}
    message = payload.get("message", {})
    return {"replace_original": True, "text": message.get("text", "sureshot"),
            "blocks": [done if block.get("block_id") == action["block_id"] else block for block in message.get("blocks", [])]}
