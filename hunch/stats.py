"""A usage report from the decision log: who answers what, how accurate each tier is, and what each stage costs."""
import math
import statistics
from hunch.calibrate import tier_confidences, wilson_lower
from hunch.review import REASONS, pending

# How far the share escalated must move, in standard errors, to count as drift: about a 1% false alarm rate per check.
DRIFT_Z = 2.58


def _percentile(values, p):
    values = sorted(values)
    return values[math.ceil(p / 100 * len(values)) - 1]


def _timing(values):
    return {"p50": _percentile(values, 50), "p95": _percentile(values, 95), "mean": statistics.mean(values)} if values else None


def _drift(store, decisions, labels, min_samples):
    report = []
    for c in store.calibrations():
        after = [(decision_id, ts, questions, result) for decision_id, ts, questions, result in decisions if ts > c["ts"]]
        confidences = tier_confidences(after, c["tier"], c["key"])
        weight = right = weight_sq = 0.0
        graded = 0
        for decision_id, _, _, result in after:
            answer = result["answers"].get(c["key"])
            if not answer or answer["source"] == "system2" or answer.get("tier", 0) != c["tier"] or answer["confidence"] < c["threshold"]:
                continue
            human = labels.get((decision_id, c["key"]))
            truth = human if human is not None else answer.get("audit", {}).get("answer")
            if truth is None:
                continue
            w = 1.0 if human is not None else 1 / result["audit_rate"]
            graded, weight, right, weight_sq = graded + 1, weight + w, right + w * (answer["answer"] == truth), weight_sq + w * w
        flags = []
        now = sum(x < c["threshold"] for x in confidences) / len(confidences) if confidences else None
        expected, before = c["escalation_rate"], c["answers"]
        if now is not None and expected is not None and len(confidences) >= min_samples:
            pooled = (now * len(confidences) + expected * before) / (len(confidences) + before)
            spread = math.sqrt(pooled * (1 - pooled) * (1 / len(confidences) + 1 / before))
            if spread and abs(now - expected) / spread >= DRIFT_Z:
                flags.append("escalation_rate")
        accuracy = right / weight if weight else None
        if accuracy is not None and graded >= min_samples and 1 - wilson_lower(1 - accuracy, weight * weight / weight_sq) < c["target"]:
            flags.append("accuracy")
        report.append({"tier": c["tier"], "key": c["key"], "calibrated_at": c["ts"], "threshold": c["threshold"], "target": c["target"],
                       "answers_since": len(confidences), "escalation_rate": {"at_calibration": expected, "since": now},
                       "accuracy_since": {"graded": graded, "accuracy": accuracy} if graded else None, "flags": flags})
    return report


def stats(store, since=0.0, min_samples=30):
    """Summarizes the log since a unix time. Returns:

    decisions, degraded (how many had a model down), cache (model answers served from the cache against asked of
    the models, and the hit rate; None if no cache was used), and latency_ms: timing of the whole decision, the mean time each
    tier took, and how often System 2 was called (and its mean time when it was).

    uncalibrated: the questions whose latest calibration found no threshold, with the reason and the best target
    their data supports (see hunch.calibrate); until one is found, a tier whose data could not certify the target is not trusted for that question (it escalates), and a tier with too little data keeps its default threshold.

    drift: for each question and tier with a calibrated threshold, how its answers since that calibration (and since
    `since`) compare with it, and "flags" naming what has moved: "escalation_rate" when the share of the tier's
    answers below the threshold differs from the share at calibration by more than chance would explain (the
    confidence distribution moved, say the inputs changed), and "accuracy" when the graded answers it kept (your
    labels, else audits, weighted as below) are below the target even at the optimistic end of a 90% interval.
    Either needs min_samples answers since the calibration. A flag means: grade a fresh sample and calibrate again.

    review_queue: how many ungraded questions are waiting for a person, by reason (see hunch.review_queue).

    questions: for each question, how many times it was asked, answered_by (the share each tier and System 2 gave),
    and accuracy against your labels. Per tier, accuracy is of the answers that tier gave and kept (not the ones it
    passed down), against your label or else System 2's audit answer, with audited answers weighted up for the
    audit rate; that is what to look at to decide whether a tier is earning its place."""
    queue = pending(store, since)
    labels = store.labels()
    decisions = store.results(since)
    latency = {"total": [], "tiers": {}, "system2": []}
    questions, degraded, hits, misses = {}, 0, 0, 0
    for decision_id, _, _, result in decisions:
        degraded += bool(result.get("degraded"))
        hits += (result.get("cache") or {}).get("hits", 0)
        misses += (result.get("cache") or {}).get("misses", 0)
        latency["total"].append(result["latency_ms"]["total"])
        for i, ms in enumerate(result["latency_ms"].get("tiers", [result["latency_ms"]["system1"]])):
            latency["tiers"].setdefault(i, []).append(ms)
        if result["latency_ms"]["system2"] is not None:
            latency["system2"].append(result["latency_ms"]["system2"])
        for key, answer in result["answers"].items():
            q = questions.setdefault(key, {"asked": 0, "answered_by": {}, "human": [0, 0], "tiers": {}})
            q["asked"] += 1
            who = "system2" if answer["source"] == "system2" else f"tier {answer.get('tier', 0)}"
            q["answered_by"][who] = q["answered_by"].get(who, 0) + 1
            human = labels.get((decision_id, key))
            if human is not None:
                q["human"][0] += 1
                q["human"][1] += answer["answer"] == human
            if who == "system2":
                continue
            truth = human if human is not None else answer.get("audit", {}).get("answer")
            if truth is not None:
                weight = 1.0 if human is not None else 1 / result["audit_rate"]
                t = q["tiers"].setdefault(who, [0, 0.0, 0.0])
                t[0] += 1
                t[1] += weight * (answer["answer"] == truth)
                t[2] += weight
    for key, q in questions.items():
        q["answered_by"] = {who: n / q["asked"] for who, n in sorted(q["answered_by"].items())}
        q["accuracy_vs_labels"] = {"graded": q["human"][0], "accuracy": q["human"][1] / q["human"][0]} if q["human"][0] else None
        q["tier_accuracy"] = {who: {"graded": n, "accuracy": right / weight} for who, (n, right, weight) in sorted(q["tiers"].items())}
        del q["human"], q["tiers"]
    return {"decisions": len(decisions), "degraded": degraded,
            "cache": {"hits": hits, "misses": misses, "hit_rate": hits / (hits + misses)} if hits + misses else None,
            "latency_ms": {"total": _timing(latency["total"]), "tier_mean": {f"tier {i}": statistics.mean(ms) for i, ms in sorted(latency["tiers"].items())},
                           "system2_called": len(latency["system2"]) / len(decisions) if decisions else None,
                           "system2_mean": statistics.mean(latency["system2"]) if latency["system2"] else None},
            "uncalibrated": store.issues(), "drift": _drift(store, decisions, labels, min_samples), "review_queue": {reason: sum(item["reason"] == reason for item in queue) for reason in REASONS}, "questions": questions}
