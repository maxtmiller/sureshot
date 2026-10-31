"""Pick, per question, the lowest confidence threshold at which System 1 meets an accuracy target."""
import logging
from bisect import bisect_right
from math import ceil, lgamma, log as ln, sqrt
from statistics import NormalDist
from hunch.review import in_sample

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


def confidence_sequence(successes, n, alpha):
    """(low, high): a confidence sequence for a rate seen as `successes` out of n, which contains the true rate at
    every n at once except alpha of the time, so it stays valid however often it is checked as answers come in.
    It is Robbins' mixture of the likelihood over a uniform prior, bounded with Ville's inequality; unlike a bound
    that ignores the spread (Hoeffding's), it is tight for rates near 0 or 1. n may be an effective count."""
    mix = lgamma(1 + successes) + lgamma(1 + n - successes) - lgamma(2 + n)
    bound = ln(1 / alpha)

    def outside(p):
        return mix - successes * ln(p) - (n - successes) * ln(1 - p) >= bound

    def edge(inside, out):
        for _ in range(60):
            mid = (inside + out) / 2
            inside, out = (inside, mid) if outside(mid) else (mid, out)
        return inside

    best = successes / n
    low = 0.0 if successes <= 0 or not outside(1e-12) else edge(best, 1e-12)
    high = 1.0 if successes >= n or not outside(1 - 1e-12) else edge(best, 1 - 1e-12)
    return low, high


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


def _label_check(rows, threshold, target, sample, confident):
    """Your labels on the random samples at or above a threshold (rows are store.labeled() rows of one tier and
    question): {graded, accuracy, low, confirmed, next_check, graded_all, high, ruled_out}.

    Confirming asks whether the tier was right when it said it was sure: it counts the labels on the answers the tier
    flagged as confident while held to this threshold or a lower one (see _flagged), which were all picked at the
    same rate, so each counts fully. graded and accuracy are theirs. The check runs every night, so it has to hold
    however often it is run: it is only tested when the labels reach fixed counts (see _confirmations), each test on
    the first that many labels in the order they were given, with the 10% of a one-sided 90% bound split across the
    tests. low is the bound at the last count reached, and next_check the count of the next test.

    Ruling out also looks at every random label at or above the threshold, flagged or not, each weighted by the
    inverse of its chance of being picked (the `sample` share, or the `confident` share where the tier was
    confident): graded_all. high is the lower of the tops of two confidence sequences, one on each set of labels, at
    2.5% each, so checking every night adds no false rejections; ruled_out is high below the target."""
    labels = []
    for row in rows:
        if row["kind"] != "human" or row["confidence"] < threshold:
            continue
        rate = max(sample, confident) if row["confident"] else sample
        if in_sample(row["id"], rate):
            labels.append((row["answer"] == row["truth"], 1 / rate, _flagged(row["confident"], row["held_to"], threshold), row["labeled_at"]))
    return _check(labels, target)


def _flagged(confident, held_to, threshold):
    """Whether an answer's label can confirm `threshold`: the tier was confident, held to that threshold or a lower
    one. A label from an answer held to a higher threshold says nothing about the answers between the two, so when a
    threshold is lowered that band needs labels of its own."""
    return bool(confident) and held_to is not None and held_to <= threshold


def _hold(threshold, saved, check_at):
    """(threshold, its label check): the new threshold, unless it is lower than the saved one and your labels don't
    confirm it, in which case the saved one is kept. Labels only confirm a threshold from answers held to it or a
    lower one, so a threshold lowered every night would never gather them; held still, it does. check_at(t) is the
    label check at threshold t."""
    check = check_at(threshold)
    if saved is not None and saved != float("inf") and threshold < saved and not check["confirmed"]:
        return saved, check_at(saved)
    return threshold, check


def _confirmations(target, alpha=0.1):
    """The tests that confirm a threshold, [(labels, z)]: the j-th uses the first `labels` labels and a one-sided bound
    at alpha / 2 ** (j + 1), so all of them together are wrong at most alpha of the time however long they run. The
    first is at the fewest labels that could pass with none wrong; each later one doubles it."""
    tests, n = [], 1
    for j in range(20):
        z = NormalDist().inv_cdf(1 - alpha / 2 ** (j + 1))
        n = max(n * 2 if tests else 1, ceil(z * z * target / (1 - target)) if target < 1 else 10 ** 9)
        tests.append((n, z))
    return tests


def _check(labels, target):
    """_label_check on labels already picked: [(right, weight, flagged, labeled_at)]."""
    flagged = [right for right, _, f, at in sorted(labels, key=lambda label: (label[3] is None, label[3] or 0)) if f]
    accuracy = sum(flagged) / len(flagged) if flagged else None
    low, confirmed, next_check = None, False, None
    for n, z in _confirmations(target) if target is not None else []:
        if n > len(flagged):
            next_check = n
            break
        low = wilson_lower(sum(flagged[:n]) / n, n, z)
        confirmed = confirmed or low >= target
    # Either set of labels can rule the target out; each gets half of the 5%.
    highs = [confidence_sequence(sum(flagged), len(flagged), 0.025)[1]] if flagged else []
    if labels:
        weight, weight_sq = sum(w for _, w, _, _ in labels), sum(w * w for _, w, _, _ in labels)
        # The effective count of the weighted labels: as much evidence as that many equally weighted ones.
        effective = weight * weight / weight_sq
        highs.append(confidence_sequence(sum(w for right, w, _, _ in labels if right) / weight * effective, effective, 0.025)[1])
    high = min(highs) if highs else None
    return {"graded": len(flagged), "accuracy": accuracy, "low": low, "confirmed": confirmed, "next_check": next_check,
            "graded_all": len(labels), "high": high, "ruled_out": target is not None and high is not None and high < target}


def calibrate(store, target_accuracy=0.95, min_samples=30, save=True, tier=0, fail_closed=True, sample=0.0, confident=0.0):
    """Calibrates every question for one tier (the first by default; call it once per tier of a chain) from its graded
    answers in the log and, with save, stores the thresholds so a Router using this store applies them. Returns
    {key: summarize(...)}.

    A key with threshold None keeps its current one, except when fail_closed is true and the data could not certify the
    target ("not enough evidence" or "target unreachable"): the tier is then never trusted for that question, which
    always passes to the next tier or System 2, until a later calibration finds a threshold. "not enough data" keeps
    the default, so a question with few graded answers still works.

    Where nobody labeled, an answer is graded by System 2's, so agreeing with System 2 counts as right: a threshold
    certifies agreement with System 2, which overstates the tier when System 2 is often wrong. "graded_by" says how
    many of the grades at or above the threshold came from your labels ("human") and how many from System 2.

    So with `sample` or `confident` (the rates given to review_queue), your labels on those random samples must
    confirm a threshold before it is trusted. If they rule the target out at or above it (a confidence sequence at
    5%, so calibrating every night adds no false rejections), the reason is "below target on labels" and, with
    fail_closed, the tier is not trusted. If they don't yet show the target on the bound calibration certifies with,
    the threshold is saved but unconfirmed, with the reason "not enough labels": the router keeps asking the tier
    but not keeping its answers, so labels on what it is confident about keep coming, and recheck confirms it once
    they show the target. A threshold lower than the saved one that the labels don't confirm is not saved: the saved
    one is kept ("held_at"), so labels can gather at it. Confirming counts the labels on the answers the tier flagged
    as confident; ruling out
    counts every random label (see _label_check). "label_check" gives {graded, accuracy, low, confirmed, graded_all,
    high, ruled_out}. Without either
    rate nothing needs confirming.

    With save, each threshold found is also recorded with the share of all of this tier's logged answers (graded or
    not) that fall below it and how their confidences are spread, which is what stats(store)["drift"] compares later
    answers against."""
    by_key, rows = {}, {}
    for row in store.labeled():
        if row["tier"] != tier:
            continue
        by_key.setdefault(row["key"], []).append((row["confidence"], row["answer"] == row["truth"], row["weight"]))
        rows.setdefault(row["key"], []).append(row)
    report = {key: summarize(samples, target_accuracy, min_samples) for key, samples in by_key.items()}
    saved = store.thresholds(tier)
    for key, r in report.items():
        kept = [row for row in rows[key] if r["threshold"] is None or row["confidence"] >= r["threshold"]]
        r["graded_by"] = {"human": sum(row["kind"] == "human" for row in kept), "system2": sum(row["kind"] == "proxy" for row in kept)}
        if (sample or confident) and r["threshold"] is not None:
            threshold, check = _hold(r["threshold"], saved.get(key), lambda t: _label_check(rows[key], t, target_accuracy, sample, confident))
            if threshold != r["threshold"]:
                r.update(threshold=threshold, held_at=threshold)
            r["label_check"] = check
            if check["ruled_out"]:
                r.update(threshold=None, trusted_accuracy=None, escalation_rate=None, reason="below target on labels")
            elif not check["confirmed"]:
                r["reason"] = "not enough labels"
    for key, r in report.items():
        if r["threshold"] is None:
            best = f"; the best this data supports is {r['best_target']:.2f}" if r["best_target"] is not None else ""
            log.warning("no threshold for %r at tier %d: %s (target %.2f%s)", key, tier, r["reason"], target_accuracy, best)
    if save:
        closed = {key: float("inf") for key, r in report.items()
                  if fail_closed and r["reason"] in ("not enough evidence", "target unreachable", "below target on labels")}
        found = {key: r["threshold"] for key, r in report.items() if r["threshold"] is not None}
        store.save_thresholds(closed | found, tier)
        store.save_issues({key: {"reason": r["reason"], "best_target": r["best_target"], "target": target_accuracy} for key, r in report.items() if r["reason"]},
                          tier, list(report))
        store.save_unconfirmed(tier, [key for key, r in report.items() if r["reason"] == "not enough labels"], list(report))
        results = store.results() if found else []
        snapshots = {}
        for key, threshold in found.items():
            confidences = tier_confidences(results, tier, key)
            snapshots[key] = {"threshold": threshold, "target": target_accuracy, "answers": len(confidences),
                              "escalation_rate": sum(c < threshold for c in confidences) / len(confidences) if confidences else None,
                              "histogram": histogram(confidences) if confidences else None}
        store.save_calibrations(snapshots, tier, list(found) + list(closed))
    return report


def prune(store, costs, tier=0, min_answers=100):
    """Stops asking a tier the questions it does not pay for, apart from the Router's probe sample, and asks it again
    the ones it does. costs is {"tier 0": ..., "system2": ...}, what one question costs at each stage (as for stats); to
    count waiting, add its price to each stage's cost (say $0.01 a second: 0.0057 more for a tier that takes 570 ms).

    A tier pays for a question when keep_rate * costs["system2"] > costs[f"tier {tier}"], where keep_rate is the share of
    its logged answers at or above its saved threshold: each answer it keeps saves a System 2 question, and each one
    it is asked costs its own. Comparing with System 2 overstates what an earlier tier saves when a later tier would
    have answered instead, so this errs toward asking. Questions with fewer than min_answers logged answers from the
    tier, or no saved threshold, are left as they are. While a tier is skipped its probe answers keep keep_rate up to
    date, so it is asked again if it starts paying. Returns {key: {answers, keep_rate, net, skipped}}, net being what
    asking it saves per question."""
    if "system2" not in costs:
        raise ValueError('costs needs a "system2" cost per question')
    thresholds = store.thresholds(tier)
    confidences = {}
    for _, _, _, result in store.results():
        for key, answer in result["answers"].items():
            if key in thresholds:
                confidences.setdefault(key, []).extend(e["confidence"] for e in answer.get("tried", []) if e["tier"] == tier)
    report, skip, ask = {}, [], []
    for key, seen in confidences.items():
        if len(seen) < min_answers:
            continue
        keep_rate = sum(c >= thresholds[key] for c in seen) / len(seen)
        net = keep_rate * costs["system2"] - costs.get(f"tier {tier}", 0.0)
        (skip if net < 0 else ask).append(key)
        report[key] = {"answers": len(seen), "keep_rate": keep_rate, "net": net, "skipped": net < 0}
    store.save_skips(tier, skip, ask)
    return report


def recheck(store, tier=0, sample=0.02, target_accuracy=None, confident=0.0):
    """Checks each saved threshold of one tier against your labels on the random samples (the `sample` and
    `confident` rates given to review_queue), without recalibrating. Run it on a schedule: a tier that is certified is
    rarely calibrated again, and where nobody labeled, calibrate grades a tier by System 2's answers, so a threshold
    can certify agreement with System 2 rather than accuracy.

    A question whose labels at or above its threshold rule the target out (a confidence sequence at 5%, so checking
    every night adds no false rejections) fails closed, as in calibrate, with the reason "below target on labels". An
    unconfirmed threshold (see calibrate) is confirmed once its labels show the target, and the tier's answers are
    then kept. Anything else is left as it is, drift snapshot included.

    The target is the one the threshold was calibrated for, else target_accuracy. Returns {key: {threshold, target,
    graded, accuracy, low, high, failed, confirmed}} for the questions with a finite threshold; graded, accuracy and
    low are of the labels on its confident answers (see _label_check)."""
    targets = {c["key"]: c["target"] for c in store.calibrations() if c["tier"] == tier}
    thresholds = {key: t for key, t in store.thresholds(tier).items() if t != float("inf")}
    unconfirmed = store.unconfirmed(tier)
    rows = {key: [] for key in thresholds}
    for row in store.labeled():
        if row["tier"] == tier and row["key"] in thresholds:
            rows[row["key"]].append(row)
    report, failed, confirmed = {}, {}, []
    for key, threshold in thresholds.items():
        target = targets.get(key, target_accuracy)
        check = _label_check(rows[key], threshold, target, sample, confident)
        report[key] = {"threshold": threshold, "target": target, **{k: check[k] for k in ("graded", "accuracy", "low", "high")},
                       "failed": check["ruled_out"], "confirmed": key not in unconfirmed or check["confirmed"]}
        if check["ruled_out"]:
            failed[key] = target
            log.warning("tier %d no longer trusted for %r: %.0f%% of %d labeled answers right (target %.2f)", tier, key, 100 * check["accuracy"], check["graded"], target)
        elif key in unconfirmed and check["confirmed"]:
            confirmed.append(key)
    if failed:
        store.save_thresholds({key: float("inf") for key in failed}, tier)
        store.save_issues({key: {"reason": "below target on labels", "best_target": None, "target": target} for key, target in failed.items()}, tier, list(failed))
        store.save_calibrations({}, tier, list(failed))
    if failed or confirmed:
        store.save_unconfirmed(tier, [], list(failed) + confirmed)
    if confirmed:
        store.save_issues({}, tier, confirmed)
    return report
