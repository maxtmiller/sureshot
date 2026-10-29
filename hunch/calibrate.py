"""Pick, per question, the lowest confidence threshold at which System 1 meets an accuracy target."""
import logging
from bisect import bisect_right
from math import ceil, sqrt

log = logging.getLogger("hunch")


Z = 1.28


def wilson_lower(p, n, z=Z):
    """A conservative accuracy: the lower end of a one-sided 90% interval, so small samples don't pass by luck."""
    if n <= 0:
        return 0.0
    centre = p + z * z / (2 * n)
    spread = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - spread) / (1 + z * z / n)


def _scan(samples, target_accuracy, min_samples):
    """One fixed-sequence pass over samples sorted by descending confidence: (the last certified slice as {threshold,
    trusted_accuracy, escalation_rate} or None, the best optimistic accuracy of any slice keeping a tenth of the answers)."""
    total = sum(w for _, _, w in samples)
    weight = right = weight_sq = 0.0
    best_optimistic, certified, stopped = None, None, False
    # A slice shorter than `perfect` could not reach the target even with no errors, and testing it would end the sequence at once.
    perfect = ceil(Z * Z * target_accuracy / (1 - target_accuracy)) if target_accuracy < 1 else float("inf")
    start = max(min_samples, ceil(0.1 * len(samples)), perfect)
    for i, (confidence, ok, w) in enumerate(samples):
        weight, right, weight_sq = weight + w, right + w * ok, weight_sq + w * w
        if i + 1 < len(samples) and samples[i + 1][0] == confidence:
            continue
        accuracy = right / weight
        if i + 1 < max(min_samples // 2, ceil(0.1 * len(samples))):
            continue
        # A slice too thin to certify (under min_samples answers) still counts toward "could the target be reached", as
        # long as it keeps a tenth of the answers: a threshold that escalates nearly everything is not a way to reach it.
        best_optimistic = max(best_optimistic or 0.0, 1 - wilson_lower(1 - accuracy, weight * weight / weight_sq))
        if i + 1 < start or stopped:
            continue
        if wilson_lower(accuracy, weight * weight / weight_sq) >= target_accuracy:
            certified = {"threshold": confidence, "trusted_accuracy": accuracy, "escalation_rate": 1 - weight / total}
        else:
            stopped = True
    return certified, best_optimistic


def summarize(samples, target_accuracy=0.95, min_samples=30):
    """samples are (confidence, correct, weight). Returns the threshold (None if no confidence level has enough
    evidence), System 1's accuracy overall and above that threshold, and the share of answers below it (what escalates).

    The threshold comes from a fixed-sequence test: it starts at the most confident answers (the first slice with at
    least min_samples answers, a tenth of the total, and enough that a slice with no errors could pass), moves to ever
    less confident slices, certifies each with the one-sided bound above, and stops at the first slice that fails. The
    threshold is the last slice certified. Testing in an order fixed by the confidences alone is why the bound holds
    for the threshold that was picked; taking the best of many separate tests claims the same bound but misses the
    target up to twice as often.

    When there is no threshold, "reason" says why: "not enough data" (fewer than min_samples graded answers),
    "not enough evidence" (the target is not ruled out: at some threshold the optimistic end of the accuracy estimate
    reaches it, so more graded answers may find one), or "target unreachable" (even the optimistic end is below it,
    at every threshold that keeps a tenth of the answers). "best_target" is the highest target that
    finds a threshold on this data, judged as conservatively as the threshold is (None if there were too
    few answers to judge any): try a target at or below it."""
    samples = sorted(samples, key=lambda s: -s[0])
    report = {"samples": len(samples), "threshold": None, "accuracy": None, "trusted_accuracy": None, "escalation_rate": None,
              "reason": "not enough data", "best_target": None}
    if not samples:
        return report
    report["accuracy"] = sum(w for _, ok, w in samples if ok) / sum(w for _, _, w in samples)
    certified, best_optimistic = _scan(samples, target_accuracy, min_samples)
    if len(samples) >= min_samples:
        low, high = (target_accuracy, 1.0) if certified else (0.0, target_accuracy)  # low finds a threshold, high does not
        while high - low > 1e-4:
            mid = (low + high) / 2
            low, high = (mid, high) if _scan(samples, mid, min_samples)[0] else (low, mid)
        report["best_target"] = low
    if certified:
        report.update(certified, reason=None)
    elif best_optimistic is not None and len(samples) >= min_samples:
        report["reason"] = "not enough evidence" if best_optimistic >= target_accuracy else "target unreachable"
    return report


def tier_confidences(results, tier, key):
    """Every confidence one tier gave one question in some logged results, whether or not the answer was kept."""
    return [entry["confidence"] for _, _, _, result in results
            for entry in result["answers"].get(key, {}).get("tried", []) if entry["tier"] == tier]


def histogram(confidences, bins=5):
    """Quantile bin edges of some confidences and how many fall in each bin (fewer bins where values tie)."""
    ordered = sorted(confidences)
    edges = sorted({ordered[len(ordered) * i // bins] for i in range(1, bins)})
    counts = [0] * (len(edges) + 1)
    for confidence in ordered:
        counts[bisect_right(edges, confidence)] += 1
    return {"edges": edges, "counts": counts}


def calibrate(store, target_accuracy=0.95, min_samples=30, save=True, tier=0, fail_closed=True):
    """Calibrates every question for one tier (the first by default; call it once per tier of a chain) from its graded
    answers in the log and, with save, stores the thresholds so a Router using this store applies them. Returns
    {key: summarize(...)}.

    A key with threshold None keeps its current one, except when fail_closed is true and the data could not certify the
    target ("not enough evidence" or "target unreachable"): the tier is then never trusted for that question, which
    always passes to the next tier or System 2, until a later calibration finds a threshold. "not enough data" keeps
    the default, so a question with few graded answers still works.

    With save, each threshold found is also recorded with the share of all of this tier's logged answers (graded or
    not) that fall below it and how their confidences are spread, which is what stats(store)["drift"] compares later
    answers against."""
    by_key = {}
    for row in store.labeled():
        if row["tier"] != tier:
            continue
        by_key.setdefault(row["key"], []).append((row["confidence"], row["answer"] == row["truth"], row["weight"]))
    report = {key: summarize(samples, target_accuracy, min_samples) for key, samples in by_key.items()}
    for key, r in report.items():
        if r["threshold"] is None:
            best = f"; the best this data supports is {r['best_target']:.2f}" if r["best_target"] is not None else ""
            log.warning("no threshold for %r at tier %d: %s (target %.2f%s)", key, tier, r["reason"], target_accuracy, best)
    if save:
        closed = {key: float("inf") for key, r in report.items() if fail_closed and r["reason"] in ("not enough evidence", "target unreachable")}
        found = {key: r["threshold"] for key, r in report.items() if r["threshold"] is not None}
        store.save_thresholds(closed | found, tier)
        store.save_issues({key: {"reason": r["reason"], "best_target": r["best_target"], "target": target_accuracy} for key, r in report.items() if r["threshold"] is None},
                          tier, list(report))
        results = store.results() if found else []
        snapshots = {}
        for key, threshold in found.items():
            confidences = tier_confidences(results, tier, key)
            snapshots[key] = {"threshold": threshold, "target": target_accuracy, "answers": len(confidences),
                              "escalation_rate": sum(c < threshold for c in confidences) / len(confidences) if confidences else None,
                              "histogram": histogram(confidences) if confidences else None}
        store.save_calibrations(snapshots, tier, list(found) + list(closed))
    return report
