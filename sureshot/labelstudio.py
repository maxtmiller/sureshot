"""Label the review queue in Label Studio: `sureshot label-studio push` sends queue items of one question set to a
project ("sureshot: <set>", created with a labeling interface built from the questions), and `sureshot label-studio
pull` saves the finished annotations as labels.

SURESHOT_LABEL_STUDIO_URL is the server and SURESHOT_LABEL_STUDIO_TOKEN an access token from its Account & Settings
page: a personal access token, or a legacy token. A task is one decision; where several annotators answered a
question, it is labeled only when they agree."""
import json
import os
import re
from xml.sax.saxutils import quoteattr
import httpx
from sureshot.questions import show


def _name(key):
    return "q_" + re.sub(r"\W", "_", key)


def names(questions):
    """{control name: key}: each question's name in the labeling interface."""
    out = {}
    for key in questions:
        if _name(key) in out:
            raise ValueError(f"questions {out[_name(key)]!r} and {key!r} have the same name in Label Studio")
        out[_name(key)] = key
    return out


def _choices(question):
    """[(what the annotator sees, the answer, a hint)] for each answer to a question."""
    if question["type"] == "noul":
        criteria = question.get("criteria") or {}
        return [("Yes", True, criteria.get("true", "")), ("No", False, criteria.get("false", ""))]
    if question["type"] == "score":
        return [(show(question, i), i, "") for i in range(len(question["criteria"]))]
    return [(key, key, description) for key, description in question["criteria"].items()]


def label_config(questions):
    """The labeling interface: the text, then each question as a single choice whose alias is the answer's JSON."""
    names(questions)
    parts = ['<View>', '  <Text name="text" value="$text"/>']
    for key, question in questions.items():
        parts.append(f"  <Header value={quoteattr(f'{key}: ' + question.get('instructions', ''))}/>")
        parts.append(f'  <Choices name="{_name(key)}" toName="text" choice="single-radio">')
        parts += [f"    <Choice value={quoteattr(str(value))} alias={quoteattr(json.dumps(answer))}" + (f" hint={quoteattr(hint)}" if hint else "") + "/>"
                  for value, answer, hint in _choices(question)]
        parts.append("  </Choices>")
    return "\n".join(parts + ["</View>"])


class LabelStudio:
    def __init__(self, url=None, token=None, transport=None):
        url, self.token = url or os.environ.get("SURESHOT_LABEL_STUDIO_URL"), token or os.environ.get("SURESHOT_LABEL_STUDIO_TOKEN")
        if not url or not self.token:
            raise ValueError("set SURESHOT_LABEL_STUDIO_URL and SURESHOT_LABEL_STUDIO_TOKEN")
        self.client, self.access = httpx.Client(base_url=url.rstrip("/"), timeout=60, transport=transport), None

    def _auth(self, fresh=False):
        # A personal access token is a JWT, traded for a short-lived access token; a legacy token is sent as it is.
        if "." not in self.token:
            return {"Authorization": f"Token {self.token}"}
        if fresh or self.access is None:
            response = self.client.post("/api/token/refresh", json={"refresh": self.token})
            response.raise_for_status()
            self.access = response.json()["access"]
        return {"Authorization": f"Bearer {self.access}"}

    def request(self, method, path, **kwargs):
        response = self.client.request(method, path, headers=self._auth(), **kwargs)
        if response.status_code == 401 and "." in self.token:  # the access token expired
            response = self.client.request(method, path, headers=self._auth(fresh=True), **kwargs)
        if response.is_error:
            raise RuntimeError(f"Label Studio answered {response.status_code} to {method} {path}: {response.text[:300]}")
        return response.json() if response.content else None

    def project(self, title, config, create=True):
        """The id of the project with this title, its labeling interface set to config; created when missing (or None
        when create is false)."""
        projects = self.request("GET", "/api/projects", params={"page_size": 1000, "title": title})
        match = next((p for p in (projects["results"] if isinstance(projects, dict) else projects) if p["title"] == title), None)
        if match is None:
            return self.request("POST", "/api/projects", json={"title": title, "label_config": config})["id"] if create else None
        if match.get("label_config") != config:
            self.request("PATCH", f"/api/projects/{match['id']}", json={"label_config": config})
        return match["id"]

    def import_tasks(self, project, tasks):
        self.request("POST", f"/api/projects/{project}/import", json=tasks)

    def export(self, project):
        """Every task with an annotation, as Label Studio's JSON export gives them."""
        return self.request("GET", f"/api/projects/{project}/export", params={"exportType": "JSON"})


def _answer(question, chosen):
    """(answer, None) for a chosen option, saved as its alias or, by older interfaces, its visible value; else (None,
    why)."""
    for value, answer, _ in _choices(question):
        if chosen in (json.dumps(answer), str(value)):
            return answer, None
    return None, "not an answer from sureshot's labeling interface"


def _title(name):
    return f"sureshot: {name}"


def push(store, studio, name, questions, items):
    """Sends the queue items that belong to the question set (same key and question) and were not sent before, one
    task per decision, and records them as sent. Returns {project, tasks, questions}."""
    project = studio.project(_title(name), label_config(questions))
    sent = store.posted()
    fresh = [item for item in items if questions.get(item["key"]) == item["question"] and (item["id"], item["key"]) not in sent]
    by_decision = {}
    for item in fresh:
        by_decision.setdefault(item["id"], []).append(item)
    if by_decision:
        studio.import_tasks(project, [{"data": {"text": group[0]["state"], "sureshot_id": decision_id, "queued": [i["key"] for i in group]}}
                                      for decision_id, group in by_decision.items()])
        store.mark_posted([(item["id"], item["key"]) for item in fresh])
    return {"project": project, "tasks": len(by_decision), "questions": len(fresh)}


def pull(store, studio, name, questions):
    """Labels every question the project's annotators answered. Returns {project, labeled, unchanged, skipped}, skipped
    being {reason: count}: annotators who disagree, or an answer the decision can't take."""
    project = studio.project(_title(name), label_config(questions), create=False)
    if project is None:
        raise ValueError(f"no Label Studio project {_title(name)!r}: push to it first")
    keys, existing = names(questions), store.labels()
    report = {"project": project, "labeled": 0, "unchanged": 0, "skipped": {}}
    for task in studio.export(project):
        decision_id = task.get("data", {}).get("sureshot_id")
        answers = {}
        for annotation in task.get("annotations", []):
            if annotation.get("was_cancelled"):
                continue
            for result in annotation.get("result", []):
                key, chosen = keys.get(result.get("from_name")), result.get("value", {}).get("choices") or []
                if key is not None and len(chosen) == 1:
                    answers.setdefault(key, []).append(chosen[0])
        for key, given in answers.items():
            reason = None
            if len(set(given)) > 1:
                reason = "annotators disagree"
            else:
                answer, reason = _answer(questions[key], given[0])
            if reason is None and (decision_id, key) in existing and existing[(decision_id, key)] == answer and type(existing[(decision_id, key)]) is type(answer):
                report["unchanged"] += 1
                continue
            if reason is None:
                try:
                    store.label(decision_id, key, answer, source="label-studio")
                except KeyError:
                    reason = "no logged decision"
                except ValueError:
                    reason = "not a valid answer"
            if reason:
                report["skipped"][reason] = report["skipped"].get(reason, 0) + 1
            else:
                report["labeled"] += 1
    return report
