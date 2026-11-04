"""push and pull against an in-memory stand-in for Argilla, and with SURESHOT_TEST_ARGILLA set to a server's URL (its
owner's key in SURESHOT_TEST_ARGILLA_KEY, default "argilla.apikey"), against that server through the argilla client."""
import json
import os
import uuid
import pytest
import sureshot.cli as cli
from sureshot import Store, review_queue
from sureshot import argilla as sa
from sureshot.argilla import labels, pull, push
from sureshot.router import Router
from test_router import QUESTIONS, FakeClaude, FakeDecider


class FakeServer:
    """The three calls push and pull make, in memory. respond() adds an annotator's submitted answers."""

    def __init__(self):
        self.datasets = {}

    def dataset(self, name, questions, create=True):
        if name not in self.datasets:
            if not create:
                return None
            self.datasets[name] = {"labels": {k: list(labels(q)) for k, q in questions.items()}, "records": {}}
        elif self.datasets[name]["labels"] != {k: list(labels(q)) for k, q in questions.items()}:
            raise ValueError("asks other questions")
        return name

    def log(self, dataset, records):
        for decision_id, text in records:
            self.datasets[dataset]["records"].setdefault(decision_id, {"text": text, "responses": []})["text"] = text

    def responses(self, dataset):
        return [(i, [pair for response in r["responses"] for pair in response]) for i, r in self.datasets[dataset]["records"].items()]

    def respond(self, dataset, decision_id, **answers):
        self.datasets[dataset]["records"][decision_id]["responses"].append([(k, json.dumps(a)) for k, a in answers.items()])


def _store(tmp_path, n=2):
    store = Store(tmp_path / "sureshot.db", seed=0)
    for i in range(n):
        Router(FakeDecider(), FakeClaude(), log=store).decide(f"event {i}", QUESTIONS)
    return store


def test_labels_carry_each_answer_as_json():
    assert labels(QUESTIONS["needs_prep"]) == {"true": "Yes", "false": "No"}
    assert list(labels(QUESTIONS["importance"])) == [str(i) for i in range(len(QUESTIONS["importance"]["criteria"]))]
    assert list(labels(QUESTIONS["event_type"])) == [json.dumps(k) for k in QUESTIONS["event_type"]["criteria"]]


def test_push_once_and_pull_the_responses(tmp_path):
    store, server = _store(tmp_path), FakeServer()
    items = review_queue(store, limit=None)
    report = push(store, server, "events", QUESTIONS, items)
    assert report == {"dataset": "sureshot-events", "records": 2, "questions": len(items)}
    assert push(store, server, "events", QUESTIONS, review_queue(store, limit=None, skip=store.posted()))["records"] == 0
    first, second = sorted(server.datasets["sureshot-events"]["records"])
    server.respond("sureshot-events", first, needs_prep=False, importance=1, event_type="interview")
    server.respond("sureshot-events", second, needs_prep=True, importance=2, event_type="meeting")
    server.respond("sureshot-events", second, needs_prep=False, importance=2, event_type="meeting")
    report = pull(store, server, "events", QUESTIONS)
    assert report == {"dataset": "sureshot-events", "labeled": 5, "unchanged": 0, "skipped": {"annotators disagree": 1}}
    assert store.labels()[(first, "needs_prep")] is False and store.labels()[(first, "event_type")] == "interview"
    assert (second, "needs_prep") not in store.labels() and store.labels()[(second, "importance")] == 2
    assert pull(store, server, "events", QUESTIONS)["unchanged"] == 5


def test_pull_skips_what_it_cannot_label(tmp_path):
    store, server = _store(tmp_path), FakeServer()
    push(store, server, "events", QUESTIONS, review_queue(store, limit=None))
    server.log("sureshot-events", [("gone", "x")])
    server.respond("sureshot-events", "gone", needs_prep=True)
    decision = next(iter(server.datasets["sureshot-events"]["records"]))
    server.respond("sureshot-events", decision, importance=99, unknown=True)
    assert pull(store, server, "events", QUESTIONS)["skipped"] == {"no logged decision": 1, "not a valid answer": 1}
    with pytest.raises(ValueError, match="push to it first"):
        pull(store, server, "other", QUESTIONS)


def test_cli(tmp_path, monkeypatch, capsys):
    store, server = _store(tmp_path), FakeServer()
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"events": QUESTIONS}))
    monkeypatch.setattr(sa, "Argilla", lambda: server)
    db = str(tmp_path / "sureshot.db")
    cli.main(["argilla", "push", "--db", db, "--questions", str(questions)])
    assert json.loads(capsys.readouterr().out)["records"] == 2
    decision = next(iter(server.datasets["sureshot-events"]["records"]))
    server.respond("sureshot-events", decision, needs_prep=False)
    cli.main(["argilla", "pull", "--db", db, "--questions", str(questions)])
    assert json.loads(capsys.readouterr().out)["labeled"] == 1


def test_cli_needs_the_server(tmp_path, monkeypatch):
    pytest.importorskip("argilla")
    _store(tmp_path)
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"events": QUESTIONS}))
    monkeypatch.delenv("SURESHOT_ARGILLA_URL", raising=False)
    with pytest.raises(SystemExit, match="set SURESHOT_ARGILLA_URL"):
        cli.main(["argilla", "push", "--db", str(tmp_path / "sureshot.db"), "--questions", str(questions)])


URL = os.environ.get("SURESHOT_TEST_ARGILLA")


@pytest.mark.skipif(not URL, reason="set SURESHOT_TEST_ARGILLA to an Argilla server's URL")
def test_a_real_server(tmp_path):
    import argilla as rg

    key = os.environ.get("SURESHOT_TEST_ARGILLA_KEY", "argilla.apikey")
    client = rg.Argilla(api_url=URL, api_key=key)
    if not list(client.workspaces):
        rg.Workspace(name="sureshot-test", client=client).create().add_user(client.me.username)
    name = f"t{uuid.uuid4().hex[:8]}"
    store = _store(tmp_path, 3)
    server = sa.Argilla(URL, key)
    report = push(store, server, name, QUESTIONS, review_queue(store, limit=None))
    assert report["records"] == 3
    assert push(store, server, name, QUESTIONS, review_queue(store, limit=None, skip=store.posted()))["records"] == 0
    dataset = client.datasets(name=f"sureshot-{name}", workspace=server.workspace)
    server.log(dataset, [(r.id, r.fields["text"]) for r in dataset.records])  # the same ids again: replaced, not added
    assert len(list(dataset.records)) == 3
    other = rg.User(username=f"a{uuid.uuid4().hex[:8]}", password="annotator123", role="annotator", client=client).create()
    other.add_to_workspace(server.workspace)
    records = sorted(dataset.records(with_responses=True), key=lambda r: r.id)
    for record, (user, needs_prep) in zip(records, [(client.me, False), (client.me, True), (other, False)]):
        for k, answer in {"needs_prep": needs_prep, "importance": 3, "event_type": "interview"}.items():
            record.responses.add(rg.Response(question_name=k, value=json.dumps(answer), user_id=user.id, status="submitted"))
    records[1].responses.add(rg.Response(question_name="needs_prep", value="false", user_id=other.id, status="submitted"))
    records[1].responses.add(rg.Response(question_name="importance", value="3", user_id=other.id, status="submitted"))
    records[1].responses.add(rg.Response(question_name="event_type", value='"interview"', user_id=other.id, status="submitted"))
    dataset.records.log(records)
    report = pull(store, server, name, QUESTIONS)
    assert report["labeled"] == 8 and report["skipped"] == {"annotators disagree": 1}
    assert store.labels()[(records[0].id, "needs_prep")] is False and store.labels()[(records[2].id, "event_type")] == "interview"
    with pytest.raises(ValueError, match="asks other questions"):
        push(store, server, name, {"needs_prep": QUESTIONS["needs_prep"]}, [])
    dataset.delete()
    other.delete()
