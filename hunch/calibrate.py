"""Pick, per question, the lowest confidence threshold at which System 1 meets an accuracy target."""
import logging
from math import ceil, sqrt

log = logging.getLogger("hunch")


def wilson_lower(p, n, z=1.28):
    """A conservative accuracy: the lower end of a one-sided 90% interval, so small samples don't pass by luck."""
    if n <= 0:
        return 0.0
    centre = p + z * z / (2 * n)
    spread = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - spread) / (1 + z * z / n)


def summarize(samples, target_accuracy=0.95, min_samples=30):
    """samples are (confidence, correct, weight). Returns the threshold (None if no confidence level has enough
    evidence), System 1's accuracy overall and above that threshold, and the share of answers below it (what escalates).

    When there is no threshold, "reason" says why: "not enough data" (fewer than min_samples graded answers),
    "not enough evidence" (the target is not ruled out: at some threshold the optimistic end of the accuracy estimate
    reaches it, so more graded answers may find one), or "target unreachable" (even the optimistic end is below it,
    at every threshold that keeps a tenth of the answers). "best_target" is the highest target that
    would have found a threshold on this data, judged as conservatively as the threshold is (None if there were too
    few answers to judge any): try a target at or below it."""
    samples = sorted(samples, key=lambda s: -s[0])
    total = sum(w for _, _, w in samples)
    report = {"samples": len(samples), "threshold": None, "accuracy": None, "trusted_accuracy": None, "escalation_rate": None,
              "reason": "not enough data", "best_target": None}
    if not samples:
        return report
    report["accuracy"] = sum(w for _, ok, w in samples if ok) / total
    weight = right = weight_sq = 0.0
    best_optimistic = None
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
        if i + 1 < min_samples:
            continue
        lower = wilson_lower(accuracy, weight * weight / weight_sq)
        report["best_target"] = max(report["best_target"] or 0.0, lower)
        if lower >= target_accuracy:
            report.update(threshold=confidence, trusted_accuracy=accuracy, escalation_rate=1 - weight / total)
    if report["threshold"] is not None:
        report["reason"] = None
    elif best_optimistic is not None and len(samples) >= min_samples:
        report["reason"] = "not enough evidence" if best_optimistic >= target_accuracy else "target unreachable"
    return report


def calibrate(store, target_accuracy=0.95, min_samples=30, save=True, tier=0):
    """Calibrates every question for one tier (the first by default; call it once per tier of a chain) from its graded
    answers in the log and, with save, stores the thresholds so a Router using this store applies them. Returns
    {key: summarize(...)}; a key with threshold None keeps its current one."""
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
        store.save_thresholds({key: r["threshold"] for key, r in report.items() if r["threshold"] is not None}, tier)
        store.save_issues({key: {"reason": r["reason"], "best_target": r["best_target"], "target": target_accuracy} for key, r in report.items() if r["threshold"] is None},
                          tier, list(report))
    return report
