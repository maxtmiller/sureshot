"""SQLite log of every decision, the labels that grade them, and the thresholds calibration learned."""
import json
import random
import sqlite3
import time
import uuid
from contextlib import contextmanager
from hunch.questions import check_answer

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, ts REAL, state TEXT, questions TEXT, images INTEGER, result TEXT);
CREATE TABLE IF NOT EXISTS labels (decision_id TEXT, key TEXT, answer TEXT, source TEXT, PRIMARY KEY (decision_id, key));
CREATE TABLE IF NOT EXISTS calibration_issues (tier INTEGER, key TEXT, reason TEXT, best_target REAL, target REAL, PRIMARY KEY (tier, key));
CREATE TABLE IF NOT EXISTS tier_thresholds (tier INTEGER, key TEXT, value REAL, PRIMARY KEY (tier, key));
CREATE TABLE IF NOT EXISTS skips (tier INTEGER, key TEXT, PRIMARY KEY (tier, key));
CREATE TABLE IF NOT EXISTS unconfirmed (tier INTEGER, key TEXT, PRIMARY KEY (tier, key));
CREATE TABLE IF NOT EXISTS calibrations (tier INTEGER, key TEXT, ts REAL, threshold REAL, target REAL, escalation_rate REAL, answers INTEGER, histogram TEXT, PRIMARY KEY (tier, key));
"""


class Store:
    def __init__(self, path, seed=None):
        """seed fixes the decision ids, which pick the labeling sample: a run that logs the same decisions in the
        same order gets the same ids. Without it ids are random uuids."""
        self.path = str(path)
        self._ids = random.Random(seed) if seed is not None else None
        with self._db() as db:
            db.executescript(SCHEMA)
            if "histogram" not in {row[1] for row in db.execute("PRAGMA table_info(calibrations)")}:
                try:  # a log made before drift compared the whole confidence distribution
                    db.execute("ALTER TABLE calibrations ADD COLUMN histogram TEXT")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e):  # another process added it first
                        raise
            db.execute("PRAGMA journal_mode=WAL")  # readers (stats, review) stop blocking the writers that log decisions

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def log(self, state, questions, result, images=0):
        """Records a decision (the Router's result, with its audit_rate) and returns its id."""
        decision_id = uuid.UUID(int=self._ids.getrandbits(128)).hex if self._ids else uuid.uuid4().hex
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

    def skipped(self, tier=0):
        """The questions a tier is only probed on, since asking it costs more than it saves (see hunch.prune)."""
        with self._db() as db:
            return {key for (key,) in db.execute("SELECT key FROM skips WHERE tier = ?", (tier,))}

    def save_skips(self, tier, skip, ask):
        """Marks the questions in `skip` as skipped by a tier and the ones in `ask` as asked again."""
        with self._db() as db:
            db.executemany("INSERT OR IGNORE INTO skips VALUES (?, ?)", [(tier, key) for key in skip])
            db.executemany("DELETE FROM skips WHERE tier = ? AND key = ?", [(tier, key) for key in ask])

    def unconfirmed(self, tier=0):
        """The questions whose saved threshold for a tier your labels have not confirmed yet (see hunch.calibrate): the
        tier is asked, but its answers are not kept."""
        with self._db() as db:
            return {key for (key,) in db.execute("SELECT key FROM unconfirmed WHERE tier = ?", (tier,))}

    def save_unconfirmed(self, tier, unconfirmed, confirmed):
        """Marks the questions in `unconfirmed` as waiting for your labels to confirm a tier's threshold, and the ones in
        `confirmed` as no longer waiting."""
        with self._db() as db:
            db.executemany("INSERT OR IGNORE INTO unconfirmed VALUES (?, ?)", [(tier, key) for key in unconfirmed])
            db.executemany("DELETE FROM unconfirmed WHERE tier = ? AND key = ?", [(tier, key) for key in confirmed if key not in unconfirmed])

    def save_thresholds(self, thresholds, tier=0):
        with self._db() as db:
            db.executemany("INSERT OR REPLACE INTO tier_thresholds VALUES (?, ?, ?)", [(tier, key, value) for key, value in thresholds.items()])

    def save_issues(self, issues, tier, calibrated):
        """Records why a calibration found no threshold ({key: {reason, best_target, target}}), and clears the questions
        in `calibrated` that now have one."""
        with self._db() as db:
            db.executemany("DELETE FROM calibration_issues WHERE tier = ? AND key = ?", [(tier, key) for key in calibrated])
            db.executemany("INSERT INTO calibration_issues VALUES (?, ?, ?, ?, ?)",
                           [(tier, key, i["reason"], i["best_target"], i["target"]) for key, i in issues.items()])

    def save_calibrations(self, snapshots, tier, calibrated):
        """Records what a calibration found ({key: {threshold, target, escalation_rate, answers, histogram}}), so stats
        can tell when later answers drift from it. The questions in `calibrated` left without one (it failed closed,
        or retrain had no logged answers to take one from) get a reset instead: drift learns a new baseline from their
        answers after it."""
        now = time.time()
        with self._db() as db:
            db.executemany("DELETE FROM calibrations WHERE tier = ? AND key = ?", [(tier, key) for key in calibrated])
            db.executemany("INSERT INTO calibrations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           [(tier, key, now, s["threshold"], s["target"], s["escalation_rate"], s["answers"], s["histogram"] and json.dumps(s["histogram"]))
                            for key, s in snapshots.items()])
            db.executemany("INSERT INTO calibrations (tier, key, ts) VALUES (?, ?, ?)", [(tier, key, now) for key in calibrated if key not in snapshots])

    def calibrations(self):
        """[{tier, key, ts, threshold, target, escalation_rate, answers, histogram}]: the latest calibration of each question
        that has a threshold. histogram is {edges, counts} of the tier's confidences then, or None for older snapshots."""
        with self._db() as db:
            rows = db.execute("SELECT tier, key, ts, threshold, target, escalation_rate, answers, histogram FROM calibrations WHERE threshold IS NOT NULL ORDER BY tier, key").fetchall()
        fields = ("tier", "key", "ts", "threshold", "target", "escalation_rate", "answers")
        return [dict(zip(fields, row[:-1]), histogram=row[-1] and json.loads(row[-1])) for row in rows]

    def resets(self):
        """{(tier, key): unix time} of the questions whose snapshot was dropped and not yet replaced by a calibration."""
        with self._db() as db:
            return {(tier, key): ts for tier, key, ts in db.execute("SELECT tier, key, ts FROM calibrations WHERE threshold IS NULL")}

    def issues(self):
        """[{tier, key, reason, best_target, target}] for questions whose latest calibration found no threshold."""
        with self._db() as db:
            rows = db.execute("SELECT tier, key, reason, best_target, target FROM calibration_issues ORDER BY tier, key").fetchall()
        return [dict(zip(("tier", "key", "reason", "best_target", "target"), row)) for row in rows]

    def results(self, since=0.0):
        """[(id, ts, questions, result)] of every logged decision since a unix time, oldest first."""
        with self._db() as db:
            rows = db.execute("SELECT id, ts, questions, result FROM decisions WHERE ts >= ? ORDER BY ts", (since,)).fetchall()
        return [(decision_id, ts, json.loads(questions), json.loads(result)) for decision_id, ts, questions, result in rows]

    def states(self, decision_ids):
        """{id: the state text} of some logged decisions."""
        ids = list(decision_ids)
        with self._db() as db:
            return dict(db.execute(f"SELECT id, state FROM decisions WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()) if ids else {}

    def labels(self):
        """{(decision id, key): the true answer} for every graded question."""
        with self._db() as db:
            return {(i, k): json.loads(a) for i, k, a in db.execute("SELECT decision_id, key, answer FROM labels")}

    def _graded(self):
        """(row, answer, result) for each graded question of each logged decision; row has id, key, state, question,
        images, truth, kind and weight.

        The truth is a human label ("human"), else System 2's answer ("proxy"). System 2 saw every escalated question
        (weight 1) and a random audit_rate share of the confident ones (weight 1/audit_rate), so weighting by the
        inverse of that chance makes the rows represent all of System 1's answers, not only the unsure ones."""
        with self._db() as db:
            decisions = db.execute("SELECT id, state, questions, images, result FROM decisions").fetchall()
            labels = {(i, k): json.loads(a) for i, k, a in db.execute("SELECT decision_id, key, answer FROM labels")}
        for decision_id, state, questions, images, result in decisions:
            questions, result = json.loads(questions), json.loads(result)
            for key, answer in result["answers"].items():
                escalated = answer["source"] == "system2"
                if (decision_id, key) in labels:
                    truth, kind, weight = labels[(decision_id, key)], "human", 1.0
                elif escalated or "audit" in answer:
                    truth, kind = answer["answer"] if escalated else answer["audit"]["answer"], "proxy"
                    weight = 1.0 if escalated else 1 / result["audit_rate"]
                else:
                    continue
                row = {"id": decision_id, "key": key, "state": state, "question": questions[key], "images": images, "truth": truth, "kind": kind, "weight": weight}
                yield row, answer, result

    def graded(self):
        """One row per graded question of each logged decision, whether or not a System 1 tier answered it (see
        _graded): what a classifier can learn from, which includes every question System 2 answered alone."""
        return [row for row, _, _ in self._graded()]

    def labeled(self):
        """One row per graded question and tier that answered it: the graded row (see _graded) with that tier's answer
        and confidence, and whether it was confident (at or above the threshold it was held to, kept or not). A question
        no tier answered has no row here; graded() has it."""
        rows = []
        for row, answer, result in self._graded():
            escalated = answer["source"] == "system2"
            # Decisions logged before tiers existed have no "tried": their only System 1 answer is the one itself.
            tried = answer.get("tried") or ([{"tier": 0, **{k: (answer["hunch"] if escalated else answer)[k] for k in ("answer", "confidence")}}]
                                            if (answer["hunch"] if escalated else answer) else [])
            for entry in tried:
                # Logs from before the router recorded "confident" have only whether the answer stood.
                confident = entry.get("confident", not escalated and answer.get("tier", 0) == entry["tier"] and row["key"] not in result["unsure"])
                rows.append(row | {"tier": entry["tier"], "confidence": entry["confidence"], "answer": entry["answer"], "confident": confident})
        return rows
