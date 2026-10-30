"""A usage report from the decision log: who answers what, how accurate each tier is, and what each stage costs."""
import math
import statistics
from bisect import bisect_right
from hunch.calibrate import confidence_sequence as _sequence, histogram, tier_confidences, wilson_lower
from hunch.review import REASONS, in_sample, pending


def _percentile(values, p):
    values = sorted(values)
    return values[math.ceil(p / 100 * len(values)) - 1]


def _timing(values):
    return {"p50": _percentile(values, 50), "p95": _percentile(values, 95), "mean": statistics.mean(values)} if values else None


def _normal_p(z):
    """The chance a standard normal is at least z."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def _z(tail):
    """The z a standard normal exceeds `tail` of the time."""
    low, high = 0.0, 40.0
    for _ in range(80):
        mid = (low + high) / 2
        low, high = (mid, high) if _normal_p(mid) > tail else (low, mid)
    return high


def _rate_moved(now, n, before, n_before, alpha):
    """Whether a rate seen over n answers since calibration and one seen over n_before at calibration cannot be the
    same rate. The calibration's answers are a fixed sample, so a one-off (Wilson) interval holds for them; the
    answers since keep coming, so theirs is a confidence sequence. Each gets half of alpha, and the rate moved when
    the two do not overlap."""
    z = _z(alpha / 4)
    low_before, high_before = wilson_lower(before, n_before, z), 1 - wilson_lower(1 - before, n_before, z)
    low, high = _sequence(now * n, n, alpha / 2)
    return high < low_before or high_before < low


def _cusum(xs, p0, p1, h):
    """Page's CUSUM for a rate moving from p0 to p1 in xs (True or False, oldest first): (where the change started,
    where it was flagged) as indexes, or None. Each answer adds its log-likelihood ratio and the score never drops
    below 0, so normal traffic keeps it near 0 however long it runs, and a change makes it climb. A flag needs the
    score to reach h. With no change, the chance of one within n answers is at most n / e^h: the score reaches h only
    if the likelihood ratio from some start point does, each of those is a martingale (Ville's inequality bounds it
    by e^-h), and there are n start points. The score depends only on the answers, so how often it is looked at does
    not matter."""
    up, down = math.log(p1 / p0), math.log((1 - p1) / (1 - p0))
    score, start = 0.0, 0
    for i, x in enumerate(xs):
        score = max(0.0, score + (up if x else down))
        if score == 0.0:
            start = i + 1
        elif score >= h:
            return start, i
    return None


def _measure(c, rows, labels):
    """What the drift checks look at in some logged decisions: the tier's confidences, whether each was passed on
    (below the threshold, or by what the tier did when there is none), and its audited answers' accuracy."""
    confidences = tier_confidences(rows, c["tier"], c["key"])
    answered_at = [ts for _, ts, _, result in rows for e in result["answers"].get(c["key"], {}).get("tried", []) if e["tier"] == c["tier"]]
    weight = right = weight_sq = 0.0
    graded = humans = 0
    correct, graded_at = [], []
    for decision_id, ts, _, result in rows:
        answer = result["answers"].get(c["key"])
        # Only audited answers: the audit is a random sample of the confident ones, while a label on an unaudited
        # answer was picked some other way (often because it looked wrong). A label still overrides the audit's answer.
        if not answer or answer["source"] == "system2" or answer.get("tier", 0) != c["tier"] or "audit" not in answer:
            continue
        human = labels.get((decision_id, c["key"]))
        truth = human if human is not None else answer["audit"]["answer"]
        w = 1 / result["audit_rate"]
        humans += human is not None
        correct.append(answer["answer"] == truth)
        graded_at.append(ts)
        graded, weight, right, weight_sq = graded + 1, weight + w, right + w * (answer["answer"] == truth), weight_sq + w * w
    shares = [x < c["threshold"] for x in confidences] if c["threshold"] is not None else _passed_down(rows, c["tier"], c["key"])
    bins = bin_of = None
    if c["histogram"] and confidences:
        bin_of = [bisect_right(c["histogram"]["edges"], confidence) for confidence in confidences]
        bins = [bin_of.count(b) for b in range(len(c["histogram"]["counts"]))]
    return {"confidences": confidences, "shares": shares, "bins": bins, "bin_of": bin_of, "answered_at": answered_at,
            "graded": graded, "humans": humans, "correct": correct, "graded_at": graded_at,
            "weight": weight, "right": right, "weight_sq": weight_sq}


def _summary(m):
    now = sum(m["shares"]) / len(m["shares"]) if m["shares"] else None
    accuracy = m["right"] / m["weight"] if m["weight"] else None
    return now, accuracy, ({"graded": m["graded"], "human": m["humans"], "accuracy": accuracy} if m["graded"] else None)


def _change(c, m):
    counts = c["histogram"]["counts"]
    n, k = sum(counts), len(m["confidences"])
    return sum(abs(before / n - after / k) for before, after in zip(counts, m["bins"])) / 2


def _cumulative(c, m, min_samples, budget, min_escalation_change, min_distribution_change):
    """The checks on everything since the calibration: (flags, unchecked). Each gets `budget` and its bound holds at
    every sample size, so checking often does not add false alarms."""
    flags, unchecked = [], []
    now, accuracy, _ = _summary(m)
    n = len(m["confidences"])
    if now is not None and c["escalation_rate"] is not None and n >= min_samples:
        if (abs(now - c["escalation_rate"]) >= min_escalation_change
                and _rate_moved(now, len(m["shares"]), c["escalation_rate"], c["answers"], budget)):
            flags.append("escalation_rate")
    else:
        unchecked.append("escalation_rate")
    if m["bins"] is not None and n >= min_samples:
        counts, total = c["histogram"]["counts"], sum(c["histogram"]["counts"])
        # Each bin is a rate of its own, and they share the check's budget.
        moved = any(_rate_moved(after / n, n, before / total, total, budget / len(counts)) for before, after in zip(counts, m["bins"]))
        if moved and _change(c, m) >= min_distribution_change:
            flags.append("distribution")
    else:
        unchecked.append("distribution")
    if c["target"] is not None and m["graded"] >= min_samples:
        # Below the target is already worth acting on, so there is no minimum change.
        effective = m["weight"] ** 2 / m["weight_sq"]  # audit weights vary only with the audit rate, so this is close
        if _sequence(accuracy * effective, effective, budget)[1] < c["target"]:
            flags.append("accuracy")
    else:
        unchecked.append("accuracy")
    return flags, unchecked


def _detectors(c, m, min_samples, z, min_escalation_change, min_distribution_change):
    """The change detectors that can run, as [(check, answers oldest first, p0, p1, their times, the rate at
    calibration, the minimum change)], and the checks
    that cannot. A rate is watched both ways, each starting from the edge of the calibration's interval nearest
    the change, so noise in the calibration's own answers is not taken for a change."""
    specs, unchecked = [], []
    n = len(m["confidences"])

    def both_ways(name, xs, rate, n_before, change, times):
        low, high = wilson_lower(rate, n_before, z), 1 - wilson_lower(1 - rate, n_before, z)
        if high + max(change, 0.02) < 1:
            specs.append((name, xs, high, high + max(change, 0.02), times, rate, change))
        if low - max(change, 0.02) > 0:
            specs.append((name, xs, low, low - max(change, 0.02), times, rate, change))

    if c["escalation_rate"] is not None and m["shares"] and n >= min_samples:
        both_ways("escalation_rate", m["shares"], c["escalation_rate"], c["answers"], min_escalation_change, m["answered_at"])
    else:
        unchecked.append("escalation_rate")
    if m["bins"] is not None and n >= min_samples:
        counts = c["histogram"]["counts"]
        for b, count in enumerate(counts):
            both_ways("distribution", [x == b for x in m["bin_of"]], count / sum(counts), sum(counts), min_distribution_change, m["answered_at"])
    else:
        unchecked.append("distribution")
    if c["target"] is not None and m["graded"] >= min_samples:
        # Watching for the error rate to double: from the target to as far below it as the target is below 1.
        if 0 < 2 * c["target"] - 1:
            specs.append(("accuracy", m["correct"], c["target"], 2 * c["target"] - 1, m["graded_at"], c["target"], 0.0))
    else:
        unchecked.append("accuracy")
    return specs, unchecked


def _passed_down(rows, tier, key):
    """For each decision in rows where `tier` answered `key`: whether it passed the question on (a later tier or
    System 2 answered, or nobody was sure) instead of its answer being kept."""
    shares = []
    for _, _, _, result in rows:
        answer = result["answers"].get(key)
        if answer and any(e["tier"] == tier for e in answer.get("tried", [])):
            shares.append(not (answer["source"] == "hunch" and answer.get("tier", 0) == tier and key not in result["unsure"]))
    return shares


def _answered(decisions, tier, key, after=0.0):
    """The decisions after a unix time in which `tier` answered `key`."""
    return [d for d in decisions if d[1] > after and any(e["tier"] == tier for e in d[3]["answers"].get(key, {}).get("tried", []))]


def _baselines(store, decisions, min_samples):
    """A snapshot for each question and tier that calibrate has none for (never calibrated, run on a per-call
    threshold, or reset when calibrate failed closed), learned from its first
    min_samples answers since the reset: [(snapshot, the answers after the baseline)]. It has no threshold or target,
    and no escalation rate or bins until it has min_samples answers."""
    calibrated = {(c["tier"], c["key"]) for c in store.calibrations()}
    resets = store.resets()
    pairs = {(e["tier"], key) for _, _, _, result in decisions for key, answer in result["answers"].items() for e in answer.get("tried", [])}
    out = []
    for tier, key in sorted(pairs - calibrated):
        rows = _answered(decisions, tier, key, resets.get((tier, key), 0.0))
        if not rows:
            continue
        base = rows[:min_samples]
        full = 0 < len(base) == min_samples
        shares = _passed_down(base, tier, key)
        out.append(({"tier": tier, "key": key, "ts": None, "threshold": None, "target": None, "answers": len(base),
                     "escalation_rate": sum(shares) / len(shares) if full else None,
                     "histogram": histogram(tier_confidences(base, tier, key)) if full else None}, rows[min_samples:]))
    return out


def _drift(store, decisions, labels, min_samples, alpha, false_alarm_every, min_escalation_change, min_distribution_change):
    snapshots = [(c, _answered(decisions, c["tier"], c["key"], c["ts"])) for c in store.calibrations()] + _baselines(store, decisions, min_samples)
    budget = alpha / max(1, 3 * len(snapshots))
    z = _z(alpha / 4)
    entries, watched = [], []
    for c, after in snapshots:
        m = _measure(c, after, labels)
        now, _, accuracy = _summary(m)
        flags, unchecked = _cumulative(c, m, min_samples, budget, min_escalation_change, min_distribution_change)
        specs, not_run = _detectors(c, m, min_samples, z, min_escalation_change, min_distribution_change)
        entry = {"tier": c["tier"], "key": c["key"], "calibrated_at": c["ts"], "threshold": c["threshold"], "target": c["target"],
                 "baseline": {"answers": c["answers"]} if c["ts"] is None else None,
                 "answers_since": len(m["confidences"]), "escalation_rate": {"at_calibration": c["escalation_rate"], "since": now},
                 "distribution_change": _change(c, m) if m["bins"] is not None else None, "accuracy_since": accuracy,
                 "change": {}, "flags": flags, "unchecked": unchecked + [f"change_{name}" for name in not_run]}
        entries.append(entry)
        watched += [(entry, spec) for spec in specs]
    # One line for every detector in the report, set so that together they raise a false flag within n answers at
    # most n / false_alarm_every of the time.
    h = math.log(false_alarm_every * max(1, len(watched)))
    for entry, (name, xs, p0, p1, times, rate, change) in watched:
        hit = _cusum(xs, p0, p1, h)
        # A CUSUM fires on any move past about halfway to p1, so the rate since the change started must also have
        # moved at least the minimum, as for the checks above.
        if hit and abs(sum(xs[hit[0]:]) / len(xs[hit[0]:]) - rate) < change:
            hit = None
        if hit and f"change_{name}" not in entry["change"]:
            entry["change"][f"change_{name}"] = {"started_at": times[hit[0]], "flagged_at": times[hit[1]]}
            entry["flags"].append(f"change_{name}")
    entries.sort(key=lambda entry: (entry["tier"], entry["key"]))
    # A tier only sees what the tiers before it passed down, so drift or a new threshold there changes its inputs.
    for entry in entries:
        before = [other for other in entries if other["key"] == entry["key"] and other["tier"] < entry["tier"]]
        entry["upstream"] = ([{"tier": other["tier"], "reason": "drift"} for other in before if other["flags"]]
                             + [{"tier": other["tier"], "reason": "recalibrated"} for other in before
                                if entry["calibrated_at"] is not None and other["calibrated_at"] is not None and other["calibrated_at"] > entry["calibrated_at"]])
    return entries


def _costs(costs, n_decisions, questions_asked, asked, system2_asked, kept, total_ms, system2_ms):
    if "system2" not in costs:
        raise ValueError('costs needs a "system2" cost per question')
    price = lambda who: costs.get(who, 0.0)
    per_ms = costs.get("ms", 0.0)
    spent = {who: n * price(who) for who, n in asked.items()}
    total = sum(spent.values()) + system2_asked * price("system2") + per_ms * total_ms
    always = questions_asked * price("system2")
    if per_ms:
        # Without System 2 latency measured there is nothing to price the baseline's waiting with.
        always = always + per_ms * system2_ms * n_decisions if system2_ms is not None else None
    tiers = {who: {"cost": cost, "saves_vs_system2": kept.get(who, 0) * price("system2"), "net": kept.get(who, 0) * price("system2") - cost}
             for who, cost in sorted(spent.items())}
    return {"total": total, "per_decision": total / n_decisions if n_decisions else None, "always_system2": always,
            "savings": 1 - total / always if always else None, "tiers": tiers}


def _on_sample(graded, right, target, z=1.645):
    """The router's accuracy on the labeled random sample, with a 90% interval, and whether even its top is below the
    target (the latest calibration's, for this question)."""
    if not graded:
        return {"graded": 0, "accuracy": None, "low": None, "high": None, "target": target, "below_target": None}
    accuracy = right / graded
    high = 1 - wilson_lower(1 - accuracy, graded, z)
    return {"graded": graded, "accuracy": accuracy, "low": wilson_lower(accuracy, graded, z), "high": high, "target": target,
            "below_target": target is not None and high < target}


def stats(store, since=0.0, min_samples=30, alpha=0.01, false_alarm_every=100_000, min_escalation_change=0.05, min_distribution_change=0.1, costs=None,
          sample=0.0):
    """Summarizes the log since a unix time. Returns:

    decisions, degraded (how many had a model down), cache (model answers served from the cache against asked of
    the models, and the hit rate; None if no cache was used), and latency_ms: timing of the whole decision, the mean time each
    tier took, and how often System 2 was called (and its mean time when it was).

    uncalibrated: the questions whose latest calibration found no threshold, with the reason and the best target
    their data supports (see hunch.calibrate); until one is found, a tier whose data could not certify the target is not trusted for that question (it escalates), and a tier with too little data keeps its default threshold.

    drift: for each question and tier with a calibrated threshold, how its answers since that calibration (and since
    `since`) compare with it, and "flags" naming what has moved: "escalation_rate" (the share of the tier's answers
    below the threshold), "distribution" (how its confidences spread over five quantile bins, which catches a slide
    that stays above the threshold) and "accuracy" (its audited answers, graded by your label where there is one,
    else by the audit, are below the target). Labels on unaudited answers are left out, since how they were picked
    is unknown. A flag needs both evidence and size. The checks on everything since the calibration share alpha,
    split evenly, and use confidence sequences (and a one-off interval for the calibration's own fixed sample), which
    hold at every sample size at once: however often you call stats, they raise a false flag at most that often,
    ever. The escalation rate must move at least min_escalation_change, and the bins must differ by at least
    min_distribution_change (the share of answers that would have to change bins). Those checks pool everything
    since the calibration, so a change that starts long after it is diluted by the normal traffic before; the
    change detectors ("change_escalation_rate", "change_distribution" and "change_accuracy") catch that. Each is a
    CUSUM that watches for the rate to move by the same minimums (accuracy: for the error rate to double), starting
    from the edge of the calibration's interval; "change" gives when the change started and when it was flagged.
    Together they raise a false flag within n answers at most n / false_alarm_every of the time, however often you
    call stats; a flag stays until the next calibration. Each check needs min_samples answers (audited ones, for
    accuracy) since the calibration; "unchecked" names the checks that could not run yet, so no flag is not mistaken
    for no drift. Snapshots from before the distribution check have no bins, so it is unchecked until the next
    calibration. A flag means: grade a fresh sample and calibrate
    again. "upstream" lists the earlier tiers of the same question that flagged drift in this report, or were
    recalibrated after this tier was: a tier only sees what they pass down, so its flags may be theirs. Fix those
    first. In each accuracy, "human" is how many of the graded answers have your label; the rest are graded by
    System 2's audit answer, so the accuracy is partly agreement with System 2. A question and tier with no snapshot from calibrate (never calibrated, on a per-call threshold, or reset
    when calibrate failed closed) is checked against a baseline of its first
    min_samples answers since then, in this report's range ("baseline" instead of "calibrated_at"): the escalation
    rate is then the share the tier passed on, and accuracy is unchecked, since there is no target.

    cost (None unless `costs` is given): what the decisions cost and what they saved. costs is {"tier 0": ...,
    "tier 1": ..., "system2": ...}, a cost per question asked of each (a tier left out costs nothing, as a local
    classifier nearly does), plus optionally "ms", a cost per millisecond of waiting. Returns the total and
    per_decision cost (audits count as System 2 questions), always_system2 (what sending every question to System 2
    would have cost; None if "ms" is given and System 2 latency was never measured), savings (the share saved
    against that), and per tier its cost on every question it saw, saves_vs_system2 (the System 2 questions its kept
    answers avoided) and net. A negative net means the tier costs more than it saves. Questions served from the
    cache are counted as asked, so with a cache this is an upper bound.

    review_queue: how many ungraded questions are waiting for a person, by reason (see hunch.review_queue), with
    `sample` the share of decisions in the random labeling sample, as given to review_queue.

    In questions, accuracy_on_sample (None unless `sample` is given) is the accuracy of the final answers on the
    labeled decisions of that random sample, which unlike your other labels says how often the router is right
    overall: {graded, accuracy, low, high (a 90% interval), target (the latest calibration's), below_target (even the
    top of the interval is below the target: no threshold fixes that, since System 2 or the labels are the limit)}.

    questions: for each question, how many times it was asked, answered_by (the share each tier and System 2 gave),
    and accuracy against your labels. Per tier, accuracy is of the answers that tier gave and kept (not the ones it
    passed down), against your label or else System 2's audit answer, with audited answers weighted up for the
    audit rate; that is what to look at to decide whether a tier is earning its place."""
    labels = store.labels()
    decisions = store.results(since)
    queue = pending(store, since, decisions, labels, sample)
    latency = {"total": [], "tiers": {}, "system2": []}
    questions, degraded, hits, misses = {}, 0, 0, 0
    asked, kept, system2_asked = {}, {}, 0
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
            q = questions.setdefault(key, {"asked": 0, "answered_by": {}, "human": [0, 0], "sampled": [0, 0], "tiers": {}})
            q["asked"] += 1
            who = "system2" if answer["source"] == "system2" else f"tier {answer.get('tier', 0)}"
            for entry in answer.get("tried", []):
                asked[f"tier {entry['tier']}"] = asked.get(f"tier {entry['tier']}", 0) + 1
            system2_asked += who == "system2" or "audit" in answer
            if who != "system2":
                kept[who] = kept.get(who, 0) + 1
            q["answered_by"][who] = q["answered_by"].get(who, 0) + 1
            human = labels.get((decision_id, key))
            if human is not None:
                q["human"][0] += 1
                q["human"][1] += answer["answer"] == human
                if in_sample(decision_id, sample):
                    q["sampled"][0] += 1
                    q["sampled"][1] += answer["answer"] == human
            if who == "system2":
                continue
            truth = human if human is not None else answer.get("audit", {}).get("answer")
            if truth is not None:
                weight = 1.0 if human is not None else 1 / result["audit_rate"]
                t = q["tiers"].setdefault(who, [0, 0.0, 0.0])
                t[0] += 1
                t[1] += weight * (answer["answer"] == truth)
                t[2] += weight
    targets = {}
    for c in store.calibrations() + store.issues():
        targets[c["key"]] = max(targets.get(c["key"], 0.0), c["target"])
    for key, q in questions.items():
        q["answered_by"] = {who: n / q["asked"] for who, n in sorted(q["answered_by"].items())}
        q["accuracy_vs_labels"] = {"graded": q["human"][0], "accuracy": q["human"][1] / q["human"][0]} if q["human"][0] else None
        q["tier_accuracy"] = {who: {"graded": n, "accuracy": right / weight} for who, (n, right, weight) in sorted(q["tiers"].items())}
        q["accuracy_on_sample"] = _on_sample(*q["sampled"], targets.get(key)) if sample else None
        del q["human"], q["sampled"], q["tiers"]
    cost = None
    if costs is not None:
        cost = _costs(costs, len(decisions), sum(q["asked"] for q in questions.values()), asked, system2_asked, kept,
                      sum(latency["total"]), statistics.mean(latency["system2"]) if latency["system2"] else None)
    return {"decisions": len(decisions), "degraded": degraded, "cost": cost,
            "cache": {"hits": hits, "misses": misses, "hit_rate": hits / (hits + misses)} if hits + misses else None,
            "latency_ms": {"total": _timing(latency["total"]), "tier_mean": {f"tier {i}": statistics.mean(ms) for i, ms in sorted(latency["tiers"].items())},
                           "system2_called": len(latency["system2"]) / len(decisions) if decisions else None,
                           "system2_mean": statistics.mean(latency["system2"]) if latency["system2"] else None},
            "uncalibrated": store.issues(), "drift": _drift(store, decisions, labels, min_samples, alpha, false_alarm_every, min_escalation_change, min_distribution_change), "review_queue": {reason: sum(item["reason"] == reason for item in queue) for reason in REASONS}, "questions": questions}
