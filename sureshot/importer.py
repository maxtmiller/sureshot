"""Import labeled examples you already have: each row is decided by the router (System 1 only by default) and logged
with its labels, so calibrate and retrain learn from it on day one.

Imported decisions are kept apart from live traffic: their ids start with "import-", so they are never in the random
review sample (labels on them grade answers and train classifiers, but never confirm a threshold, since old data
need not look like today's traffic), and Store.results() leaves them out of stats, drift, prune and the review queue."""
import copy
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def rows(path):
    """The rows of a CSV file (with a header), a JSONL file or a Parquet file (needing pyarrow), as dicts."""
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    if path.suffix == ".parquet":
        return _pyarrow().parquet.read_table(path).to_pylist()
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path, data):
    """Writes dicts as CSV (columns in the order first seen), JSONL or Parquet, by the file's suffix."""
    path = Path(path)
    columns = list(dict.fromkeys(key for row in data for key in row))
    if path.suffix == ".csv":
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, columns)
            writer.writeheader()
            writer.writerows(data)
    elif path.suffix == ".parquet":
        pa = _pyarrow()
        # from_pylist takes its columns from the first row, so every row gets them all
        pa.parquet.write_table(pa.Table.from_pylist([{c: row.get(c) for c in columns} for row in data]), path)
    else:
        path.write_text("".join(json.dumps(row) + "\n" for row in data))


def _pyarrow():
    try:
        import pyarrow
        import pyarrow.parquet  # noqa: F401
    except ImportError as e:
        raise ImportError("Parquet files need pyarrow: pip install pyarrow") from e
    return pyarrow


def parse(question, value):
    """The answer a cell gives to a question, or None for an empty cell. Raises ValueError for anything else. Takes
    JSON values as they are and text as a CSV writes it: yes/no or true/false or 1/0, a score level's position or
    name, a choice's key (any case)."""
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if question["type"] == "noul":
        if isinstance(value, bool):
            return value
        answer = {"true": True, "yes": True, "y": True, "1": True, "false": False, "no": False, "n": False, "0": False}.get(text)
    elif question["type"] == "score":
        levels = [str(level).lower() for level in question["criteria"]]
        answer = int(text) if text.isdigit() and int(text) < len(levels) else levels.index(text) if text in levels else None
    else:
        answer = value if value in question["criteria"] else next((key for key in question["criteria"] if key.lower() == text), None)
    if answer is None:
        raise ValueError(f"{value!r} is not an answer to this {question['type']} question")
    return answer


def import_rows(router, store, questions, data, text="text", escalate=False, workers=4):
    """Decides and logs each row of `data` (dicts with a `text` field and a field per question to label) and labels
    it. A row already imported (same text) is skipped, so an import can be run again after it stops. Returns {rows,
    imported, already, labels, skipped ({reason: rows}), errors (the first few, with row numbers)}."""
    plain = copy.copy(router)
    plain.record = False  # the log entry is written below, marked as imported
    report = {"rows": len(data), "imported": 0, "already": 0, "labels": 0, "skipped": {}, "errors": []}

    def skip(n, reason, detail=None):
        report["skipped"][reason] = report["skipped"].get(reason, 0) + 1
        if detail and len(report["errors"]) < 10:
            report["errors"].append(f"row {n}: {detail}")

    todo, seen = [], set()
    for n, row in enumerate(data, 1):
        content = row.get(text)
        if not isinstance(content, str) or not content.strip():
            skip(n, "no text")
            continue
        try:
            labels = {key: answer for key in questions if (answer := parse(questions[key], row.get(key))) is not None}
        except ValueError as e:
            skip(n, "invalid label", str(e))
            continue
        if not labels:
            skip(n, "no labels")
            continue
        ref = "import:" + hashlib.sha256(content.encode()).hexdigest()[:24]
        if ref in seen:
            skip(n, "duplicate text")
            continue
        seen.add(ref)
        if store.find(ref):
            report["already"] += 1
            continue
        todo.append((n, content, labels, ref))

    def run(job):
        n, content, labels, ref = job
        try:
            return job, plain.decide(content, questions, escalate=escalate), None
        except ValueError as e:
            return job, None, str(e)

    with ThreadPoolExecutor(max(1, workers)) as pool:
        for (n, content, labels, ref), result, error in pool.map(run, todo):
            if result is None:
                skip(n, "no model answered", error)
                continue
            decision_id = store.log(content, questions, result | {"audit_rate": router.audit}, ref=ref, imported=True)
            for key, answer in labels.items():
                store.label(decision_id, key, answer, source="import")
            report["imported"] += 1
            report["labels"] += len(labels)
    return report
