"""A usage report from the decision log: who answers what, how accurate each tier is, and what each stage costs."""
import math
import statistics
from bisect import bisect_right
from hunch.calibrate import histogram, tier_confidences
from hunch.review import REASONS, pending


def _percentile(values, p):
    values = sorted(values)
    return values[math.ceil(p / 100 * len(values)) - 1]


def _timing(values):
    return {"p50": _percentile(values, 50), "p95": _percentile(values, 95), "mean": statistics.mean(values)} if values else None


def _normal_p(z):
    """The chance a standard normal is at least z."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def _chi2_p(statistic, df):
    """The chance a chi-square with df degrees of freedom is at least statistic (the Wilson-Hilferty approximation)."""
    return _normal_p(((statistic / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df)))


def _holm(p_values, alpha):
    """Which tests to reject so that, together, they raise any false alarm at most alpha of the time (Holm's method,
    which holds however the tests depend on each other)."""
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    rejected = [False] * len(p_values)
    for rank, i in enumerate(order):
        if p_values[i] > alpha / (len(p_values) - rank):
            break
        rejected[i] = True
    return rejected


def _distribution(histogram, confidences):
    """(p-value, change) of some confidences against the calibration bins: a two-sample chi-square test, and the share
    of answers that would have to change bins to match (the total variation distance)."""
    now = [0] * len(histogram["counts"])
    for confidence in confidences:
        now[bisect_right(histogram["edges"], confidence)] += 1
    n, m = sum(histogram["counts"]), len(confidences)
    statistic, bins = 0.0, 0
    for before, after in zip(histogram["counts"], now):
        if before + after:
            bins += 1
            for observed, size in ((before, n), (after, m)):
                expected = (before + after) * size / (n + m)
                statistic += (observed - expected) ** 2 / expected
    change = sum(abs(before / n - after / m) for before, after in zip(histogram["counts"], now)) / 2
    return (_chi2_p(statistic, bins - 1) if bins > 1 else 1.0), change


def _checks(c, rows, labels, min_samples, min_escalation_change, min_distribution_change):
    """The drift checks of one calibration snapshot over some logged decisions: what they measured, the tests that
    could run as (name, p-value, whether the change is big enough to act on), and the names of those that could not."""
    confidences = tier_confidences(rows, c["tier"], c["key"])
    weight = right = weight_sq = 0.0
    graded = 0
    for decision_id, _, _, result in rows:
        answer = result["answers"].get(c["key"])
        # Only audited answers: the audit is a random sample of the confident ones, while a label on an unaudited
        # answer was picked some other way (often because it looked wrong). A label still overrides the audit's answer.
        if not answer or answer["source"] == "system2" or answer.get("tier", 0) != c["tier"] or "audit" not in answer:
            continue
        human = labels.get((decision_id, c["key"]))
        truth = human if human is not None else answer["audit"]["answer"]
        w = 1 / result["audit_rate"]
        graded, weight, right, weight_sq = graded + 1, weight + w, right + w * (answer["answer"] == truth), weight_sq + w * w
    tests, unchecked = [], []
    shares = [x < c["threshold"] for x in confidences] if c["threshold"] is not None else _passed_down(rows, c["tier"], c["key"])
    now = sum(shares) / len(shares) if shares else None
    expected, before = c["escalation_rate"], c["answers"]
    if now is not None and expected is not None and len(confidences) >= min_samples:
        pooled = (now * len(confidences) + expected * before) / (len(confidences) + before)
        spread = math.sqrt(pooled * (1 - pooled) * (1 / len(confidences) + 1 / before))
        p = 2 * _normal_p(abs(now - expected) / spread) if spread else 1.0
        tests.append(("escalation_rate", p, abs(now - expected) >= min_escalation_change))
    else:
        unchecked.append("escalation_rate")
    change = None
    if c["histogram"] and len(confidences) >= min_samples:
        p, change = _distribution(c["histogram"], confidences)
        tests.append(("distribution", p, change >= min_distribution_change))
    else:
        unchecked.append("distribution")
    accuracy = right / weight if weight else None
    if c["target"] is not None and graded >= min_samples:
        spread = math.sqrt(c["target"] * (1 - c["target"]) * weight_sq) / weight
        p = _normal_p((c["target"] - accuracy) / spread) if spread else float(accuracy >= c["target"])
        tests.append(("accuracy", p, True))  # below the target is already worth acting on, so there is no minimum
    else:
        unchecked.append("accuracy")
    return {"answers": len(confidences), "escalation_rate": now, "distribution_change": change,
            "accuracy": {"graded": graded, "accuracy": accuracy} if graded else None, "tests": tests, "unchecked": unchecked}


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
    threshold, or reset when calibrate failed closed or retrain replaced the classifier), learned from its first
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


def _drift(store, decisions, labels, min_samples, window, alpha, min_escalation_change, min_distribution_change):
    entries, tests = [], []
    snapshots = [(c, _answered(decisions, c["tier"], c["key"], c["ts"])) for c in store.calibrations()] + _baselines(store, decisions, min_samples)
    for c, after in snapshots:
        total = _checks(c, after, labels, min_samples, min_escalation_change, min_distribution_change)
        # The same checks on the latest answers alone, so a change long after calibrating is not diluted by the
        # normal traffic before it. Only once there are more answers than the window, or it would repeat the above.
        recent = _checks(c, after[-window:], labels, min_samples, min_escalation_change, min_distribution_change) if window and len(after) > window else None
        entry = {"tier": c["tier"], "key": c["key"], "calibrated_at": c["ts"], "threshold": c["threshold"], "target": c["target"],
                 "baseline": {"answers": c["answers"]} if c["ts"] is None else None,
                 "answers_since": total["answers"], "escalation_rate": {"at_calibration": c["escalation_rate"], "since": total["escalation_rate"]},
                 "distribution_change": total["distribution_change"], "accuracy_since": total["accuracy"],
                 "recent": recent and {"window": window, "escalation_rate": recent["escalation_rate"],
                                       "distribution_change": recent["distribution_change"], "accuracy": recent["accuracy"]},
                 "flags": [], "unchecked": total["unchecked"] + [f"recent_{name}" for name in (recent or {}).get("unchecked", [])]}
        entries.append(entry)
        tests += [(entry, name, p, big) for name, p, big in total["tests"]]
        tests += [(entry, f"recent_{name}", p, big) for name, p, big in (recent or {}).get("tests", [])]
    # One false alarm budget for the whole report, however many questions, tiers and checks it covers.
    for (entry, name, _, big), rejected in zip(tests, _holm([p for _, _, p, _ in tests], alpha)):
        if rejected and big:
            entry["flags"].append(name)
    entries.sort(key=lambda entry: (entry["tier"], entry["key"]))
    # A tier only sees what the tiers before it passed down, so drift or a new threshold there changes its inputs.
    for entry in entries:
        before = [other for other in entries if other["key"] == entry["key"] and other["tier"] < entry["tier"]]
        entry["upstream"] = ([{"tier": other["tier"], "reason": "drift"} for other in before if other["flags"]]
                             + [{"tier": other["tier"], "reason": "recalibrated"} for other in before
                                if entry["calibrated_at"] is not None and other["calibrated_at"] is not None and other["calibrated_at"] > entry["calibrated_at"]])
    return entries


def stats(store, since=0.0, min_samples=30, window=500, alpha=0.01, min_escalation_change=0.05, min_distribution_change=0.1):
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
    is unknown. A flag needs both evidence and size: every check in the report shares one false alarm budget, alpha
    (split with Holm's method, so the whole report has about that chance of any false flag), the escalation rate
    must move at least min_escalation_change, and the bins must differ by at least min_distribution_change (the
    share of answers that would have to change bins). Each check needs min_samples answers (audited ones, for
    accuracy) since the calibration; "unchecked" names the checks that could not run yet, so no flag is not mistaken
    for no drift. Once a question has more than `window` answers since the calibration, the checks also run on the
    latest `window` alone ("recent", flagged as "recent_escalation_rate" and so on), so a change that starts long
    after calibrating is not hidden by the normal traffic before it. Snapshots from before the distribution check
    have no bins, so it is unchecked until the next calibration. A flag means: grade a fresh sample and calibrate
    again. "upstream" lists the earlier tiers of the same question that flagged drift in this report, or were
    recalibrated after this tier was: a tier only sees what they pass down, so its flags may be theirs. Fix those
    first. A question and tier with no snapshot from calibrate (never calibrated, on a per-call threshold, or reset
    when calibrate failed closed or retrain replaced the classifier) is checked against a baseline of its first
    min_samples answers since then, in this report's range ("baseline" instead of "calibrated_at"): the escalation
    rate is then the share the tier passed on, and accuracy is unchecked, since there is no target.

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
            "uncalibrated": store.issues(), "drift": _drift(store, decisions, labels, min_samples, window, alpha, min_escalation_change, min_distribution_change), "review_queue": {reason: sum(item["reason"] == reason for item in queue) for reason in REASONS}, "questions": questions}
