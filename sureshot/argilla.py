"""Label the review queue in Argilla: `sureshot argilla push` sends queue items of one question set to a dataset
("sureshot-<set>", created with a label question per sureshot question), and `sureshot argilla pull` saves the
submitted responses as labels.

SURESHOT_ARGILLA_URL is the server, SURESHOT_ARGILLA_API_KEY a key from its "My settings" page, and
SURESHOT_ARGILLA_WORKSPACE the workspace (default: the first one the key can use). A record is one decision, its id
the decision's id; every question is required, so a submitted record answers them all. Where several annotators
answered a question, it is labeled only when they agree."""
import functools
import json
import os
from sureshot.labelstudio import _choices


def _errors(method):
    """Raises the client's errors (a bad key, a server that is down) as RuntimeError, which the CLI reports."""
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        try:
            from argilla._exceptions import ArgillaError
        except ImportError:
            ArgillaError = ()
        try:
            return method(*args, **kwargs)
        except ArgillaError as e:
            raise RuntimeError(f"Argilla: {e}") from e
    return wrapper


def labels(question):
    """{label value: what the annotator sees}: the value is the answer as JSON, so it comes back as it was."""
    return {json.dumps(answer): shown for shown, answer, _ in _choices(question)}


def _description(question):
    hints = [f"{shown}: {hint}" for shown, _, hint in _choices(question) if hint]
    return "\n".join([question.get("instructions", "")] + hints).strip() or None


class Argilla:
    """The parts of an Argilla server that push and pull use, through the argilla client."""

    @_errors
    def __init__(self, url=None, api_key=None, workspace=None):
        try:
            import argilla as rg
        except ImportError as e:
            raise ImportError('Argilla needs its client: pip install "sureshot[argilla]"') from e
        url, api_key = url or os.environ.get("SURESHOT_ARGILLA_URL"), api_key or os.environ.get("SURESHOT_ARGILLA_API_KEY")
        if not url or not api_key:
            raise ValueError("set SURESHOT_ARGILLA_URL and SURESHOT_ARGILLA_API_KEY")
        self.rg, self.client = rg, rg.Argilla(api_url=url, api_key=api_key)
        name = workspace or os.environ.get("SURESHOT_ARGILLA_WORKSPACE")
        self.workspace = self.client.workspaces(name) if name else next(iter(self.client.workspaces), None)
        if self.workspace is None:
            raise ValueError(f"no Argilla workspace {name!r}" if name else "the Argilla key can use no workspace: create one, or set SURESHOT_ARGILLA_WORKSPACE")

    @_errors
    def dataset(self, name, questions, create=True):
        """The dataset with this name, asking these questions; created when missing (or None when create is false)."""
        rg = self.rg
        found = self.client.datasets(name=name, workspace=self.workspace)
        if found is None:
            if not create:
                return None
            settings = rg.Settings(fields=[rg.TextField(name="text", title="Text")],
                                   questions=[rg.LabelQuestion(name=key, title=key, labels=labels(q), description=_description(q), required=True)
                                              for key, q in questions.items()])
            return rg.Dataset(name=name, workspace=self.workspace, settings=settings, client=self.client).create()
        asked = {q.name: list(q.labels) for q in found.settings.questions}
        if asked != {key: list(labels(q)) for key, q in questions.items()}:
            raise ValueError(f"Argilla dataset {name!r} asks other questions than the set: delete or rename it in Argilla to push the set as it is now")
        return found

    @_errors
    def log(self, dataset, records):
        """Adds records [(decision id, text)], replacing any with the same id."""
        # A batch larger than the records makes the client log a warning to stdout, ahead of the command's JSON.
        dataset.records.log([self.rg.Record(id=decision_id, fields={"text": text}) for decision_id, text in records], batch_size=min(256, len(records)))

    @_errors
    def responses(self, dataset):
        """[(decision id, [(question, value)])] for each record, with every submitted response's answers."""
        return [(record.id, [(r.question_name, r.value) for r in record.responses if str(getattr(r.status, "value", r.status)) == "submitted"])
                for record in dataset.records(with_responses=True, with_suggestions=False)]


def _name(name):
    return f"sureshot-{name}"


def push(store, server, name, questions, items):
    """Sends the queue items that belong to the question set (same key and question) and were not sent before, one
    record per decision, and records them as sent. Returns {dataset, records, questions}."""
    dataset = server.dataset(_name(name), questions)
    sent = store.posted()
    fresh = [item for item in items if questions.get(item["key"]) == item["question"] and (item["id"], item["key"]) not in sent]
    texts = {item["id"]: item["state"] for item in fresh}
    if texts:
        server.log(dataset, list(texts.items()))
        store.mark_posted([(item["id"], item["key"]) for item in fresh])
    return {"dataset": _name(name), "records": len(texts), "questions": len(fresh)}


def pull(store, server, name, questions):
    """Labels every question the dataset's annotators submitted. Returns {dataset, labeled, unchanged, skipped},
    skipped being {reason: count}: annotators who disagree, or an answer the decision can't take."""
    dataset = server.dataset(_name(name), questions, create=False)
    if dataset is None:
        raise ValueError(f"no Argilla dataset {_name(name)!r}: push to it first")
    existing = store.labels()
    report = {"dataset": _name(name), "labeled": 0, "unchanged": 0, "skipped": {}}
    for decision_id, responses in server.responses(dataset):
        answers = {}
        for key, value in responses:
            if key in questions:
                answers.setdefault(key, []).append(value)
        for key, given in answers.items():
            reason = "annotators disagree" if len(set(given)) > 1 else None
            if reason is None:
                answer = json.loads(given[0])
                if (decision_id, key) in existing and existing[(decision_id, key)] == answer and type(existing[(decision_id, key)]) is type(answer):
                    report["unchanged"] += 1
                    continue
                try:
                    store.label(decision_id, key, answer, source="argilla")
                except KeyError:
                    reason = "no logged decision"
                except ValueError:
                    reason = "not a valid answer"
            if reason:
                report["skipped"][reason] = report["skipped"].get(reason, 0) + 1
            else:
                report["labeled"] += 1
    return report
