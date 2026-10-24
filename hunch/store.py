"""SQLite log of every decision, the labels that grade them, and the thresholds calibration learned."""
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from hunch.questions import check_answer

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, ts REAL, state TEXT, questions TEXT, images INTEGER, result TEXT);
CREATE TABLE IF NOT EXISTS labels (decision_id TEXT, key TEXT, answer TEXT, source TEXT, PRIMARY KEY (decision_id, key));
CREATE TABLE IF NOT EXISTS thresholds (key TEXT PRIMARY KEY, value REAL);
"""


class Store:
    def __init__(self, path):
        self.path = str(path)
        with self._db() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path)
        try:
            with db:
                yield db
        finally:
            db.close()

    def log(self, state, questions, result, images=0):
        """Records a decision (the Router's result, with its audit_rate) and returns its id."""
        decision_id = uuid.uuid4().hex
        with self._db() as db:
            db.execute("INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?)",
                       (decision_id, time.time(), state, json.dumps(questions), images, json.dumps(result)))
        return decision_id

    def label(self, decision_id, key, answer, source="human"):
        """Sets the true answer to one question of a logged decision, replacing any earlier label."""
        with self._db() as db:
            row = db.execute("SELECT questions FROM decisions WHERE id = ?", (decision_id,)).fetchone()
            if row is None:
                raise KeyError(f"no decision {decision_id}")
            questions = json.loads(row[0])
            if key not in questions:
                raise ValueError(f"decision {decision_id} has no question {key!r}")
            check_answer(questions[key], answer)
            db.execute("INSERT OR REPLACE INTO labels VALUES (?, ?, ?, ?)", (decision_id, key, json.dumps(answer), source))

    def thresholds(self):
        with self._db() as db:
            return dict(db.execute("SELECT key, value FROM thresholds").fetchall())

    def save_thresholds(self, thresholds):
        with self._db() as db:
            db.executemany("INSERT OR REPLACE INTO thresholds VALUES (?, ?)", thresholds.items())

    def labeled(self):
        """One row per graded question: System 1's answer and confidence, the truth, and a weight.

        The truth is a human label ("human"), else System 2's answer ("proxy"). System 2 saw every escalated question
        (weight 1) and a random audit_rate share of the confident ones (weight 1/audit_rate), so weighting by the
        inverse of that chance makes the rows represent all of System 1's answers, not only the unsure ones."""
        with self._db() as db:
            decisions = db.execute("SELECT id, state, questions, images, result FROM decisions").fetchall()
            labels = {(i, k): json.loads(a) for i, k, a in db.execute("SELECT decision_id, key, answer FROM labels")}
        rows = []
        for decision_id, state, questions, images, result in decisions:
            questions, result = json.loads(questions), json.loads(result)
            for key, answer in result["answers"].items():
                escalated = answer["source"] == "system2"
                fast = answer["hunch"] if escalated else answer
                proxy = answer["answer"] if escalated else answer.get("audit", {}).get("answer")
                if (decision_id, key) in labels:
                    truth, kind, weight = labels[(decision_id, key)], "human", 1.0
                elif escalated or "audit" in answer:
                    truth, kind, weight = proxy, "proxy", 1.0 if escalated else 1 / result["audit_rate"]
                else:
                    continue
                rows.append({"id": decision_id, "key": key, "state": state, "question": questions[key], "images": images,
                             "confidence": fast["confidence"], "answer": fast["answer"], "truth": truth, "kind": kind, "weight": weight})
        return rows
