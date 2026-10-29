"""Retrain the cheap tier on the log's graded decisions, and keep the new classifier only where it escalates less."""
import zlib
from hunch.calibrate import summarize
from hunch.local import DEFAULT_EMBEDDING, LocalDecider


def _samples(decider, rows):
    """{key: [(confidence, correct, weight)]} for a decider's answers to graded rows; keys it has no head for are absent."""
    samples = {}
    for row in rows:
        answered = decider.decide(row["state"], {row["key"]: row["question"]}).get(row["key"])
        if answered:
            samples.setdefault(row["key"], []).append((answered["confidence"], answered["answer"] == row["truth"], row["weight"]))
    return samples


def retrain(store, current=None, tier=0, target_accuracy=0.95, min_samples=30, holdout=0.3, embed=None):
    """Trains a LocalDecider on the log's graded decisions and promotes it, question by question, only if on held-out
    decisions it reaches target_accuracy and passes on fewer questions than `current` (a LocalDecider, or None) does.
    Returns {"decider": current with the promoted questions replaced (current itself if none was), "report": {key:
    {"promoted", "threshold", "escalation_rate", "previous_escalation_rate"}}}. The thresholds of promoted questions
    are saved to the store for `tier`, so a Router using it applies them.

    Decisions are split by id into train and held-out (the same split every time, so repeated runs compare like with
    like). Held-out rows keep their weights, so audited-confident and escalated answers are weighted as in calibrate().
    `current` may have trained on the held-out decisions, which only makes it harder to beat."""
    pairs = {}
    for row in store.labeled():
        pairs.setdefault((row["id"], row["key"]), row)
    train, held = [], []
    for row in pairs.values():
        (held if zlib.crc32(row["id"].encode()) % 100 < holdout * 100 else train).append(row)
    answers = {}
    for row in train:
        answers.setdefault(row["key"], set()).add(row["truth"])
    train = [row for row in train if len(answers[row["key"]]) > 1]
    embedding = current.embedding if current else DEFAULT_EMBEDDING
    candidate = LocalDecider.train(train, embed=embed, embedding=embedding) if train else LocalDecider({}, embed, embedding)
    new = {key: summarize(s, target_accuracy, min_samples) for key, s in _samples(candidate, held).items()}
    old = {key: summarize(s, target_accuracy, min_samples) for key, s in _samples(current, held).items()} if current else {}
    heads, thresholds, report = dict(current.heads) if current else {}, {}, {}
    for key, r in new.items():
        before = old.get(key, {}).get("escalation_rate") if old.get(key, {}).get("threshold") is not None else None
        promoted = r["threshold"] is not None and (before is None or r["escalation_rate"] < before)
        if promoted:
            heads[key], thresholds[key] = candidate.heads[key], r["threshold"]
        report[key] = {"promoted": promoted, "threshold": r["threshold"], "escalation_rate": r["escalation_rate"], "previous_escalation_rate": before}
    store.save_thresholds(thresholds, tier)
    store.save_calibrations({}, tier, list(thresholds))  # the old snapshot describes the old classifier; calibrate() makes a new one
    decider = LocalDecider(heads, embed or (current and current._embed), embedding) if thresholds else current
    return {"decider": decider, "report": report}
