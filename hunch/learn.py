"""Retrain the cheap tier on the log's graded decisions, and keep the new classifier only where it escalates less."""
import zlib
from hunch.calibrate import histogram, summarize
from hunch.local import DEFAULT_EMBEDDING, LocalDecider


def _samples(decider, rows):
    """{key: [(confidence, correct, weight)]} for a decider's answers to graded rows; keys it has no head for are absent."""
    samples = {}
    for row in rows:
        answered = decider.decide(row["state"], {row["key"]: row["question"]}).get(row["key"])
        if answered:
            samples.setdefault(row["key"], []).append((answered["confidence"], answered["answer"] == row["truth"], row["weight"]))
    return samples


def _snapshots(store, decider, thresholds, target, trained):
    """The drift snapshot of each question in `thresholds`: the decider's confidence (its top class probability) on
    every logged decision that asked it, except the ones it was trained on (`trained`, ids), where it is more sure
    than on new traffic. A question with nothing left gets none, so drift learns a baseline instead."""
    asked = {}
    for decision_id, _, questions, _ in store.results():
        if decision_id in trained:
            continue
        for key in thresholds:
            if key in questions:
                asked.setdefault(key, []).append(decision_id)
    states = store.states({i for ids in asked.values() for i in ids})
    snapshots = {}
    for key, threshold in thresholds.items():
        texts = [states[i] for i in asked.get(key, [])]
        if texts:
            confidences = decider.heads[key].predict_proba(decider.embed(texts)).max(axis=1).tolist()
            snapshots[key] = {"threshold": threshold, "target": target, "answers": len(confidences),
                              "escalation_rate": sum(c < threshold for c in confidences) / len(confidences), "histogram": histogram(confidences)}
    return snapshots


def retrain(store, current=None, tier=0, target_accuracy=0.95, min_samples=30, holdout=0.3, embed=None):
    """Trains a LocalDecider on the log's graded decisions (all of them, including the ones System 2 answered with no
    tier asked, as when every tier before it is skipped) and promotes it, question by question, only if on held-out
    decisions it reaches target_accuracy and passes on fewer questions than `current` (a LocalDecider, or None) does.
    Returns {"decider": current with the promoted questions replaced (current itself if none was), "report": {key:
    {"promoted", "threshold", "escalation_rate", "previous_escalation_rate", "kept"}}}. The thresholds are saved to
    the store for `tier`, so a Router using it applies them.

    Where `current` is kept it is checked again on the same held-out decisions, which include everything graded since
    it was trained: "kept" is "rechecked" when it still meets the target there (its threshold is updated to what they
    support), "failed" when they rule the target out (it is not trusted, as when calibrate fails closed, until a later
    retrain certifies it or a new one), "unproven" when they can no longer show the target but do not rule it out
    (its threshold, trusted or not, is left as it was), and None when there is no current classifier for the
    question. The "unproven" case keeps a classifier near the target from being switched off and on night after
    night as one check passes and the next does not. So running retrain every night also recalibrates the
    classifier, and clears a drift flag on it once it has passed again.

    Decisions are split by id into train and held-out (the same split every time, so repeated runs compare like with
    like, and `current` never trained on the held-out ones if it came from an earlier retrain). Held-out rows keep
    their weights, so audited-confident and escalated answers are weighted as in calibrate().

    Each question given a threshold also gets a drift snapshot, as calibrate() does: the classifier's confidences on
    the logged decisions it was not trained on, with the threshold and target, so stats(store)["drift"] checks its
    later answers (accuracy included) against them."""
    pairs = {}
    for row in store.graded():
        pairs.setdefault((row["id"], row["key"]), row)
    train, held = [], []
    for row in pairs.values():
        (held if zlib.crc32(row["id"].encode()) % 100 < holdout * 100 else train).append(row)
    trained = {row["id"] for row in train}
    answers = {}
    for row in train:
        answers.setdefault(row["key"], set()).add(row["truth"])
    train = [row for row in train if len(answers[row["key"]]) > 1]
    embedding = current.embedding if current else DEFAULT_EMBEDDING
    candidate = LocalDecider.train(train, embed=embed, embedding=embedding) if train else LocalDecider({}, embed, embedding)
    new = {key: summarize(s, target_accuracy, min_samples) for key, s in _samples(candidate, held).items()}
    old = {key: summarize(s, target_accuracy, min_samples) for key, s in _samples(current, held).items()} if current else {}
    heads, promoted_at, kept_at, closed, report = dict(current.heads) if current else {}, {}, {}, [], {}
    for key in new.keys() | old.keys():
        r, o = new.get(key), old.get(key, {})
        before = o.get("escalation_rate") if o.get("threshold") is not None else None
        promoted = r is not None and r["threshold"] is not None and (before is None or r["escalation_rate"] < before)
        kept = None
        if promoted:
            heads[key], promoted_at[key] = candidate.heads[key], r["threshold"]
        elif o.get("threshold") is not None:
            kept, kept_at[key] = "rechecked", o["threshold"]
        elif o.get("reason") == "target unreachable":
            kept = "failed"
            closed.append(key)
        elif o:
            kept = "unproven"
        report[key] = {"promoted": promoted, "threshold": r and r["threshold"], "escalation_rate": r and r["escalation_rate"],
                       "previous_escalation_rate": before, "kept": kept}
    store.save_thresholds(promoted_at | kept_at | {key: float("inf") for key in closed}, tier)
    snapshots = _snapshots(store, candidate, promoted_at, target_accuracy, trained)
    if kept_at:
        snapshots |= _snapshots(store, current, kept_at, target_accuracy, trained)
    store.save_calibrations(snapshots, tier, list(promoted_at) + list(kept_at) + closed)
    decider = LocalDecider(heads, embed or (current and current._embed), embedding) if promoted_at else current
    return {"decider": decider, "report": report}
