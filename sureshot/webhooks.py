"""Labels from the tools where people already act: a webhook from a ticket tracker or an automation turns an event
(a ticket moved to Billing, an issue labeled bug) into a label on the decision about that item.

A webhooks file (JSON or YAML, SURESHOT_WEBHOOKS for the API) names each source, how its requests are verified, and
which events set which labels:

    github:                               # POST /feedback/webhook/github
      verify: github                      # github, jira, linear, zendesk, or bearer (the default)
      secret: GITHUB_WEBHOOK_SECRET       # the environment variable holding the signing secret
      events:
        - when: {action: labeled}         # only events whose fields equal these
          ref: issue.html_url             # the ref given to decide; or id: a field holding the decision id
          labels:
            kind: {from: label.name, map: {bug: bug, enhancement: feature}}
            needs_reply: {value: false}

Fields are dotted paths into the event's JSON, with list positions as numbers (labels.0.name). A label takes its
answer from a field (`from`, translated by `map` when given) or is a fixed `value`. bearer checks an
`Authorization: Bearer` header against the secret, SURESHOT_API_KEY by default, for automations that can set one."""
import base64
import hashlib
import hmac
import json
import os
from sureshot.questions import read

VERIFY = ("bearer", "github", "jira", "linear", "zendesk")


def load(path):
    """The webhooks file, checked: {source: spec}. Raises ValueError on a malformed file or a secret that is not set."""
    data = read(path)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path} must be an object of named webhook sources")
    problems = []
    for name, spec in data.items():
        if not isinstance(spec, dict):
            problems.append(f"{name}: must be an object")
            continue
        if extra := set(spec) - {"verify", "secret", "events"}:
            problems.append(f"{name}: unsupported fields {sorted(extra)}")
        kind = spec.get("verify", "bearer")
        if kind not in VERIFY:
            problems.append(f"{name}: verify must be one of {list(VERIFY)}")
        secret = spec.get("secret", "SURESHOT_API_KEY" if kind == "bearer" else None)
        if not isinstance(secret, str):
            problems.append(f"{name}: {kind} needs secret, the environment variable holding its signing secret")
        elif not os.environ.get(secret):
            problems.append(f"{name}: {secret} is not set")
        events = spec.get("events")
        if not isinstance(events, list) or not events:
            problems.append(f"{name}: events must be a non-empty list")
            continue
        for i, event in enumerate(events):
            problems += [f"{name}, event {i}: {p}" for p in _event_problems(event)]
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return data


def _event_problems(event):
    if not isinstance(event, dict):
        return ["must be an object"]
    problems = []
    if extra := set(event) - {"when", "ref", "id", "labels"}:
        problems.append(f"unsupported fields {sorted(extra)}")
    if ("ref" in event) == ("id" in event) or not isinstance(event.get("ref", event.get("id")), str):
        problems.append("needs either ref or id, a field path")
    if not isinstance(event.get("when", {}), dict):
        problems.append("when must be an object of field paths and values")
    labels = event.get("labels")
    if not isinstance(labels, dict) or not labels:
        return problems + ["labels must be a non-empty object"]
    for key, label in labels.items():
        if not isinstance(label, dict) or ("from" in label) == ("value" in label) or set(label) - {"from", "map", "value"}:
            problems.append(f"label {key}: needs either from (a field path, with an optional map) or value")
        elif "map" in label and ("from" not in label or not isinstance(label["map"], dict)):
            problems.append(f"label {key}: map must be an object, used with from")
    return problems


def verify(spec, headers, body):
    """Whether a request (its headers, case-insensitive, and raw body bytes) comes from the source."""
    kind = spec.get("verify", "bearer")
    secret = os.environ.get(spec.get("secret", "SURESHOT_API_KEY")) or ""
    if not secret:
        return False

    def mac(message):
        return hmac.new(secret.encode(), message, hashlib.sha256)

    if kind == "bearer":
        given, expected = headers.get("authorization", ""), f"Bearer {secret}"
    elif kind == "github":
        given, expected = headers.get("x-hub-signature-256", ""), "sha256=" + mac(body).hexdigest()
    elif kind == "jira":
        given, expected = headers.get("x-hub-signature", ""), "sha256=" + mac(body).hexdigest()
    elif kind == "linear":
        given, expected = headers.get("linear-signature", ""), mac(body).hexdigest()
    else:
        timestamp = headers.get("x-zendesk-webhook-signature-timestamp", "")
        given, expected = headers.get("x-zendesk-webhook-signature", ""), base64.b64encode(mac(timestamp.encode() + body).digest()).decode()
    return hmac.compare_digest(given.encode(), expected.encode())


def field(event, path):
    """The value at a dotted path in the event, or None where it is missing."""
    value = event
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def apply(store, name, spec, event):
    """Labels the decisions an event is about, by every rule in spec["events"] it matches. Returns one report per
    matching rule: {id (None when no logged decision matches), labeled ({key: answer}), skipped ({key: why})}. Labels
    are recorded with the source "webhook:<name>", and a repeated event sets the same labels again."""
    reports = []
    for rule in spec["events"]:
        if any(field(event, path) != value for path, value in rule.get("when", {}).items()):
            continue
        if "id" in rule:
            decision_id = field(event, rule["id"])
        else:
            ref = field(event, rule["ref"])
            decision_id = store.find(str(ref)) if ref is not None else None
        report = {"id": decision_id, "labeled": {}, "skipped": {}}
        reports.append(report)
        for key, label in rule["labels"].items():
            if decision_id is None:
                report["skipped"][key] = "no logged decision"
                continue
            if "value" in label:
                answer = label["value"]
            else:
                answer = field(event, label["from"])
                if answer is None:
                    report["skipped"][key] = f"no {label['from']} in the event"
                    continue
                if "map" in label:
                    # JSON files can only have string keys, so a number or boolean in the event also matches its JSON text.
                    mapping, text = label["map"], answer if isinstance(answer, str) else json.dumps(answer)
                    if answer in mapping:
                        answer = mapping[answer]
                    elif text in mapping:
                        answer = mapping[text]
                    else:
                        report["skipped"][key] = f"{answer!r} is not in the map"
                        continue
            try:
                store.label(str(decision_id), key, answer, source=f"webhook:{name}")
            except (KeyError, ValueError) as e:
                report["skipped"][key] = str(e.args[0])
                continue
            report["labeled"][key] = answer
    return reports
