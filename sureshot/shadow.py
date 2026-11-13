"""What sureshot would do on traffic your current model answers. shadow() replays rows it already answered through the
System 1 tiers; record() logs one live request (see the API's SURESHOT_SHADOW), and report() reads those back. Each
says how much the tiers would keep, how often they would agree with your model, and what it would cost."""
import time
from concurrent.futures import ThreadPoolExecutor
from sureshot.calibrate import summarize, wilson_lower
from sureshot.importer import parse
from sureshot.questions import validate


def _share(right, n):
    return {"agreement": right / n if n else None, "low": wilson_lower(right / n, n) if n else None}


def _replay(replayed, keys, tiers, target, costs, min_samples):
    """replayed is (current, said): your model's {key: answer}, and per tier {key: {answer, confidence}} for what it said.
    Picks each tier's threshold for each question and replays the cascade with them (see shadow)."""
    summaries = {key: {} for key in keys}
    for key in keys:
        for i in range(tiers):
            samples = [(said[i][key]["confidence"], said[i][key]["answer"] == current[key], 1.0) for current, said in replayed if key in current and key in said[i]]
            if samples:
                summaries[key][i] = summarize(samples, target, min_samples)

    price = (costs or {}).get
    counts = {key: {"answered": 0, "kept": 0, "right": 0} for key in keys}
    spent = 0.0
    for current, said in replayed:
        for key, answer in current.items():
            c = counts[key]
            c["answered"] += 1
            for i in range(tiers):
                threshold = summaries[key].get(i, {}).get("threshold")
                if threshold is None:
                    continue
                spent += price(f"tier {i}", 0.0)
                if key in said[i] and said[i][key]["confidence"] >= threshold:
                    c["kept"] += 1
                    c["right"] += said[i][key]["answer"] == answer
                    break
            else:
                spent += price("system2", 0.0)

    report = {"questions": {key: {"answered": c["answered"], "kept": c["kept"] / c["answered"]} | _share(c["right"], c["kept"]) | {"tiers": summaries[key]}
                            for key, c in counts.items() if c["answered"]}}
    answered, kept, right = (sum(c[f] for c in counts.values()) for f in ("answered", "kept", "right"))
    report |= {"kept": kept / answered if answered else None} | _share(right, kept)
    current_cost = answered * price("system2", 0.0) if costs else None
    report["cost"] = {"current": current_cost, "sureshot": spent, "savings": 1 - spent / current_cost if current_cost else None} if costs else None
    return report


def shadow(router, questions, data, text="text", target=0.95, costs=None, min_samples=30, workers=4):
    """data is dicts with a `text` field and a field per question holding your current model's answer (read as
    sureshot.importer.parse reads labels). Every tier is asked every question a row has an answer for, and each tier's
    threshold for a question is the lowest that certifies `target` agreement with your model (see
    calibrate.summarize). The cascade is then replayed with those thresholds: a question stops at the first tier at
    or above its threshold, else your model keeps it. A tier with no threshold for a question is not asked, as a live
    router that failed closed would skip it. Nothing is logged and your model is not called.

    Returns {rows, shadowed, skipped ({reason: rows}), errors, kept, agreement, low, questions: {key: {answered, kept,
    agreement, low, tiers: {tier: summarize(...)}}}, cost}. kept is the share of answers the tiers would keep,
    agreement how often those match your model, and low its 90% lower bound. Agreement says whether switching would
    change your outputs, not whether either model is right: label a random sample for that. The thresholds are picked
    on the rows they are scored on, so a live calibration should confirm them before you rely on them.

    costs is the price per question of each tier ("tier 0", "tier 1", ...) and of your model ("system2"); cost is then
    {current, sureshot, savings}, else None."""
    questions = validate(questions)
    report = {"rows": len(data), "shadowed": 0, "skipped": {}, "errors": []}

    def skip(n, reason, detail=None):
        report["skipped"][reason] = report["skipped"].get(reason, 0) + 1
        if detail and len(report["errors"]) < 10:
            report["errors"].append(f"row {n}: {detail}")

    todo = []
    for n, row in enumerate(data, 1):
        content = row.get(text)
        if not isinstance(content, str) or not content.strip():
            skip(n, "no text")
            continue
        try:
            current = {key: answer for key in questions if (answer := parse(questions[key], row.get(key))) is not None}
        except ValueError as e:
            skip(n, "invalid answer", str(e))
            continue
        if not current:
            skip(n, "no answers")
            continue
        todo.append((n, content, current))

    def run(job):
        n, content, current = job
        asked = {key: questions[key] for key in current}
        try:
            return job, [router._cached(tier, "decide", content, asked, (), {"hits": 0, "misses": 0}) for tier in router.tiers], None
        except ValueError as e:
            return job, None, str(e)

    replayed = []
    with ThreadPoolExecutor(max(1, workers)) as pool:
        for (n, _, current), said, error in pool.map(run, todo):
            if said is None:
                skip(n, "no model answered", error)
            else:
                replayed.append((current, said))
    report["shadowed"] = len(replayed)
    return report | _replay(replayed, questions, len(router.tiers), target, costs, min_samples)


def record(router, state, questions, current, images=()):
    """Logs a request your current model answered ({key: answer}) as a decision System 2 answered in full, with what
    each System 1 tier said under "tried" (held to the threshold the router would use), so calibrate, stats,
    review_queue and report() read it like any other. Returns the decision id."""
    asked = {key: questions[key] for key in current}
    tried, latest, degraded, tiers_ms, tally = {key: [] for key in asked}, {}, [], [], {"hits": 0, "misses": 0}
    start = time.perf_counter()
    for i, tier in enumerate(router.tiers):
        tier_start, saved = time.perf_counter(), router.log.thresholds(i)
        try:
            said = router._cached(tier, "decide", state, asked, images, tally)
        except Exception as e:
            degraded.append({"system": 1, "tier": i, "error": repr(e)})
            said = {}
        tiers_ms.append(round((time.perf_counter() - tier_start) * 1000, 1))
        for key, answer in said.items():
            if key not in tried:
                continue
            limit = saved.get(key, router.tier_thresholds[i])
            tried[key].append({"tier": i, "answer": answer["answer"], "confidence": answer["confidence"], "confident": answer["confidence"] >= limit, "threshold": limit})
            latest[key] = answer | {"tier": i}
    answers = {key: {"answer": current[key], "reason": None, "source": "system2", "sureshot": latest.get(key), "tried": tried[key]} for key in asked}
    result = {"answers": answers, "unsure": list(asked), "escalated": list(asked), "degraded": degraded or None, "cache": tally if router.cache else None, "shadow": True,
              "latency_ms": {"system1": round(sum(tiers_ms), 1), "system2": None, "total": round((time.perf_counter() - start) * 1000, 1), "tiers": tiers_ms}}
    return router.log.log(state, asked, result | {"audit_rate": router.audit}, len(images))


def report(store, target=0.95, costs=None, min_samples=30, since=0.0):
    """shadow()'s report on the requests record() logged since a unix time, plus traffic: {structured, other,
    coverage}, how many requests shadow mode saw that sureshot could answer and could not (structured includes the
    ones a sample rate left out), and the share it could."""
    replayed, keys, tiers = [], set(), 0
    for _, _, _, result in store.results(since):
        if not result.get("shadow"):
            continue
        current, said = {}, {}
        for key, answer in result["answers"].items():
            current[key] = answer["answer"]
            for entry in answer["tried"]:
                said.setdefault(entry["tier"], {})[key] = entry
                tiers = max(tiers, entry["tier"] + 1)
        keys |= set(current)
        replayed.append((current, said))
    replayed = [(current, [said.get(i, {}) for i in range(tiers)]) for current, said in replayed]
    seen = store.traffic(since)
    structured, other = seen.get("structured", 0), seen.get("other", 0)
    traffic = {"structured": structured, "other": other, "coverage": structured / (structured + other) if structured + other else None}
    return {"shadowed": len(replayed), "traffic": traffic} | _replay(replayed, sorted(keys), tiers, target, costs, min_samples)
