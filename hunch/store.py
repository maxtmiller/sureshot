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
CREATE TABLE IF NOT EXISTS tier_thresholds (tier INTEGER, key TEXT, value REAL, PRIMARY KEY (tier, key));
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

    def thresholds(self, tier=0):
        """The calibrated thresholds of one tier, {key: value}."""
        with self._db() as db:
            return dict(db.execute("SELECT key, value FROM tier_thresholds WHERE tier = ?", (tier,)).fetchall())

    def save_thresholds(self, thresholds, tier=0):
        with self._db() as db:
            db.executemany("INSERT OR REPLACE INTO tier_thresholds VALUES (?, ?, ?)", [(tier, key, value) for key, value in thresholds.items()])

    def results(self, since=0.0):
        """[(id, ts, questions, result)] of every logged decision since a unix time, oldest first."""
        with self._db() as db:
            rows = db.execute("SELECT id, ts, questions, result FROM decisions WHERE ts >= ? ORDER BY ts", (since,)).fetchall()
        return [(decision_id, ts, json.loads(questions), json.loads(result)) for decision_id, ts, questions, result in rows]

    def labels(self):
        """{(decision id, key): the true answer} for every graded question."""
        with self._db() as db:
            return {(i, k): json.loads(a) for i, k, a in db.execute("SELECT decision_id, key, answer FROM labels")}

    def labeled(self):
        """One row per graded question and tier that answered it: that tier's answer and confidence, the truth, and a weight.

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
                # Decisions logged before tiers existed have no "tried": their only System 1 answer is the one itself.
                tried = answer.get("tried") or ([{"tier": 0, **{k: (answer["hunch"] if escalated else answer)[k] for k in ("answer", "confidence")}}]
                                                if (answer["hunch"] if escalated else answer) else [])
                proxy = answer["answer"] if escalated else answer.get("audit", {}).get("answer")
                if (decision_id, key) in labels:
                    truth, kind, weight = labels[(decision_id, key)], "human", 1.0
                elif escalated or "audit" in answer:
                    truth, kind, weight = proxy, "proxy", 1.0 if escalated else 1 / result["audit_rate"]
                else:
                    continue
                for entry in tried:
                    rows.append({"id": decision_id, "key": key, "state": state, "question": questions[key], "images": images, "tier": entry["tier"],
                                 "confidence": entry["confidence"], "answer": entry["answer"], "truth": truth, "kind": kind, "weight": weight})
        return rows
