import json
import time
import httpx
import pytest
import sureshot.cli as cli
from sureshot import Store, review_queue
from sureshot import labelstudio
from sureshot.labelstudio import LabelStudio, label_config, pull, push
from sureshot.router import Router
from test_router import QUESTIONS, FakeClaude, FakeDecider


class FakeServer:
    """The parts of Label Studio's API that sureshot uses, in memory. Accepts a legacy token "legacy", or the personal
    access token "a.b.c", which refreshes to an access token that expires after `expires` requests."""

    def __init__(self, expires=None):
        self.projects, self.tasks, self.refreshes, self.expires, self.used = {}, {}, 0, expires, 0

    def __call__(self, request):
        auth, path = request.headers.get("authorization", ""), request.url.path
        if path == "/api/token/refresh":
            if json.loads(request.content) != {"refresh": "a.b.c"}:
                return httpx.Response(401)
            self.refreshes, self.used = self.refreshes + 1, 0
            return httpx.Response(200, json={"access": f"access-{self.refreshes}"})
        self.used += 1
        if auth not in ("Token legacy", f"Bearer access-{self.refreshes}") or (self.expires and self.used > self.expires):
            return httpx.Response(401, json={"detail": "bad token"})
        body = json.loads(request.content) if request.content else None
        if path == "/api/projects" and request.method == "GET":
            return httpx.Response(200, json={"results": list(self.projects.values())})
        if path == "/api/projects" and request.method == "POST":
            project = {"id": len(self.projects) + 1, **body}
            self.projects[project["id"]] = project
            return httpx.Response(201, json=project)
        project = int(path.split("/")[3])
        if request.method == "PATCH":
            self.projects[project].update(body)
            return httpx.Response(200, json=self.projects[project])
        if path.endswith("/import"):
            self.tasks.setdefault(project, []).extend({"id": len(self.tasks.get(project, [])) + 1, "annotations": [], **t} for t in body)
            return httpx.Response(201, json={"task_count": len(body)})
        return httpx.Response(200, json=[t for t in self.tasks.get(project, []) if t["annotations"]])

    def annotate(self, project, task, **answers):
        """Adds one annotator's answers ({key: what Label Studio saved}) to a task."""
        result = [{"from_name": f"q_{key}", "to_name": "text", "type": "choices", "value": {"choices": [chosen]}} for key, chosen in answers.items()]
        self.tasks[project][task]["annotations"].append({"result": result, "was_cancelled": False})


def _store(tmp_path, n=2):
    store = Store(tmp_path / "sureshot.db", seed=0)
    for i in range(n):
        Router(FakeDecider(), FakeClaude(), log=store).decide(f"event {i}", QUESTIONS)
    return store


def test_the_interface_has_a_single_choice_per_question():
    config = label_config(QUESTIONS)
    assert '<Text name="text" value="$text"/>' in config and config.count("<Choices ") == 3
    assert '<Choice value="Yes" alias="answer:true"/>' in config and '<Choice value="2 Normal" alias="answer:2"/>' in config
    assert '<Choice value="meeting" alias=\'answer:"meeting"\' hint="A meeting"/>' in config
    with pytest.raises(ValueError, match="same name"):
        label_config({"a.b": QUESTIONS["needs_prep"], "a_b": QUESTIONS["needs_prep"]})


def test_push_once_and_pull_the_annotations(tmp_path):
    server, store = FakeServer(), _store(tmp_path)
    studio = LabelStudio("http://ls.test", "legacy", transport=httpx.MockTransport(server))
    items = review_queue(store, sample=1.0, keys=set(QUESTIONS))
    assert push(store, studio, "events", QUESTIONS, items) == {"project": 1, "tasks": 2, "questions": 6}
    assert push(store, studio, "events", QUESTIONS, items) == {"project": 1, "tasks": 0, "questions": 0}
    assert server.projects[1]["title"] == "sureshot: events" and server.projects[1]["label_config"] == label_config(QUESTIONS)
    first, second = server.tasks[1]
    server.annotate(1, 0, event_type='answer:"interview"', importance="2 Normal")  # the visible value also maps back
    server.annotate(1, 1, importance="answer:4")
    server.annotate(1, 1, importance="answer:3", needs_prep="answer:false")
    assert pull(store, studio, "events", QUESTIONS) == {"project": 1, "labeled": 3, "unchanged": 0, "skipped": {"annotators disagree": 1}}
    one, two = first["data"]["sureshot_id"], second["data"]["sureshot_id"]
    assert store.labels() == {(one, "event_type"): "interview", (one, "importance"): 2, (two, "needs_prep"): False}
    assert pull(store, studio, "events", QUESTIONS)["unchanged"] == 3


def test_pull_skips_what_it_cannot_label(tmp_path):
    server, store = FakeServer(), _store(tmp_path, 1)
    studio = LabelStudio("http://ls.test", "legacy", transport=httpx.MockTransport(server))
    with pytest.raises(ValueError, match="push to it first"):
        pull(store, studio, "events", QUESTIONS)
    push(store, studio, "events", QUESTIONS, review_queue(store, sample=1.0))
    server.annotate(1, 0, importance="ten")
    server.tasks[1].append({"id": 2, "data": {"text": "x", "sureshot_id": "gone"}, "annotations": []})
    server.annotate(1, 1, importance="answer:1")
    assert pull(store, studio, "events", QUESTIONS)["skipped"] == {"not an answer from sureshot's labeling interface": 1, "no logged decision": 1}


def test_a_changed_question_set_updates_the_interface(tmp_path):
    server, store = FakeServer(), _store(tmp_path, 1)
    studio = LabelStudio("http://ls.test", "legacy", transport=httpx.MockTransport(server))
    push(store, studio, "events", QUESTIONS, [])
    fewer = {"needs_prep": QUESTIONS["needs_prep"]}
    push(store, studio, "events", fewer, review_queue(store, sample=1.0))
    assert server.projects[1]["label_config"] == label_config(fewer) and len(server.tasks[1]) == 1


def test_a_personal_access_token_is_refreshed_when_it_expires(tmp_path):
    server = FakeServer(expires=2)
    studio = LabelStudio("http://ls.test", "a.b.c", transport=httpx.MockTransport(server))
    for _ in range(5):
        studio.project("sureshot: x", "<View/>")
    assert server.refreshes == 3 and len(server.projects) == 1
    with pytest.raises(RuntimeError, match="401"):
        LabelStudio("http://ls.test", "wrong", transport=httpx.MockTransport(server)).project("sureshot: x", "<View/>")


def test_the_same_label_again_keeps_its_place(tmp_path):
    store = _store(tmp_path, 1)
    decision_id = review_queue(store, sample=1.0)[0]["id"]
    store.label(decision_id, "needs_prep", True)
    first = store.labeled()[0]["labeled_at"]
    time.sleep(0.01)
    store.label(decision_id, "needs_prep", True, source="label-studio")
    assert store.labeled()[0]["labeled_at"] == first
    store.label(decision_id, "needs_prep", False)
    assert store.labeled()[0]["labeled_at"] > first and store.labels()[(decision_id, "needs_prep")] is False


def test_cli(tmp_path, monkeypatch, capsys):
    server = FakeServer()
    real = labelstudio.LabelStudio
    monkeypatch.setattr(labelstudio, "LabelStudio", lambda: real("http://ls.test", "legacy", transport=httpx.MockTransport(server)))
    _store(tmp_path)
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"events": QUESTIONS}))
    db = str(tmp_path / "sureshot.db")
    cli.main(["label-studio", "push", "--db", db, "--questions", str(questions), "--sample", "1", "--limit", "4"])
    assert json.loads(capsys.readouterr().out) == {"project": 1, "tasks": 2, "questions": 4}
    server.annotate(1, 0, needs_prep="answer:true")
    cli.main(["label-studio", "pull", "--db", db, "--questions", str(questions)])
    assert json.loads(capsys.readouterr().out)["labeled"] == 1
