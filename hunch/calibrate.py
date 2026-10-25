"""Pick, per question, the lowest confidence threshold at which System 1 meets an accuracy target."""
from math import sqrt


def wilson_lower(p, n, z=1.28):
    """A conservative accuracy: the lower end of a one-sided 90% interval, so small samples don't pass by luck."""
    if n <= 0:
        return 0.0
    centre = p + z * z / (2 * n)
    spread = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - spread) / (1 + z * z / n)


def summarize(samples, target_accuracy=0.95, min_samples=30):
    """samples are (confidence, correct, weight). Returns the threshold (None if no confidence level has enough
    evidence), System 1's accuracy overall and above that threshold, and the share of answers below it (what escalates)."""
    samples = sorted(samples, key=lambda s: -s[0])
    total = sum(w for _, _, w in samples)
    report = {"samples": len(samples), "threshold": None, "accuracy": None, "trusted_accuracy": None, "escalation_rate": None}
    if not samples:
        return report
    report["accuracy"] = sum(w for _, ok, w in samples if ok) / total
    weight = right = weight_sq = 0.0
    for i, (confidence, ok, w) in enumerate(samples):
        weight, right, weight_sq = weight + w, right + w * ok, weight_sq + w * w
        if i + 1 < len(samples) and samples[i + 1][0] == confidence:
            continue
        accuracy = right / weight
        if i + 1 >= min_samples and wilson_lower(accuracy, weight * weight / weight_sq) >= target_accuracy:
            report.update(threshold=confidence, trusted_accuracy=accuracy, escalation_rate=1 - weight / total)
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
    if save:
        store.save_thresholds({key: r["threshold"] for key, r in report.items() if r["threshold"] is not None}, tier)
    return report
