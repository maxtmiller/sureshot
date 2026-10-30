"""Case study: hunch routing a bank's support messages for ten simulated days.

banking77 (PolyAI/banking77) is about 13,000 real customer messages, each labeled by a person with one of 77 intents.
The router is a local classifier (untrained at first), then gpt-4o-mini, then gpt-4.1 as System 2, with a 95%
accuracy target. Nothing is trusted until it is calibrated, and a tier that failed closed is only asked about 5% of
messages. Each day 300 messages arrive. Each night a person labels 30 items from the review queue, a random 2% of
messages first (here the dataset's own labels stand in for them), the classifier is retrained, gpt-4o-mini is
recalibrated whenever it has no certified threshold or drift flags it, checking its threshold against the labeled
random sample. The nightly retrain rechecks the classifier, recheck tests gpt-4o-mini's threshold against the
labeled sample, and prune stops asking gpt-4o-mini when it costs more than it saves (with --per-second, waiting is
priced in: that many dollars per second of latency). From day 6 half the messages come from eight
card-security intents, a shift drift monitoring should notice.

    python examples/case_study_banking77.py baseline   # asks both models about every message (the paid part)
    python examples/case_study_banking77.py simulate   # runs the ten days, reusing those answers from the cache

Add --smoke to run 20 messages a day for 2 days first, and --seed N to fix which answers get audited and probed. Needs an OpenAI key in OPENAI_API_KEY. Working files go to
data/case_study_banking77/ (or data/case_study_banking77_smoke/); the results are summary_<seed>.json and days_<seed>.jsonl there.
"""
import argparse
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset

import hunch.adapters.base as base
from hunch import Cache, ChatDecider, ChatReasoner, LocalDecider, Router, Store, calibrate, prune, recheck, retrain, review_queue, stats

PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4.1": (2.00, 8.00)}  # $ per million input and output tokens; check yours
TARGET, AUDIT, PROBE, SAMPLE, LABELS_PER_NIGHT, WORKERS = 0.95, 0.05, 0.05, 0.02, 30, 8
LOCAL_MS = 60  # the classifier's typical time per message on one laptop; measuring it live breaks when runs share a CPU
SHIFTED = ["compromised_card", "lost_or_stolen_card", "card_payment_not_recognised", "cash_withdrawal_not_recognised",
           "direct_debit_payment_not_recognised", "transaction_charged_twice", "card_swallowed", "lost_or_stolen_phone"]


def traffic(days, per_day, shift_day):
    """The messages for each day: a random mix, then from shift_day on, half from the SHIFTED intents."""
    data = load_dataset("PolyAI/banking77")
    names = data["train"].features["label"].names
    pool, seen = [], set()
    for split in ("train", "test"):
        for row in data[split]:
            if row["text"] not in seen:  # a repeated message would be served from the cache
                seen.add(row["text"])
                pool.append({"text": row["text"], "gold": names[row["label"]]})
    rng = random.Random(0)
    rng.shuffle(pool)
    shifted = [m for m in pool if m["gold"] in SHIFTED]
    rest = [m for m in pool if m["gold"] not in SHIFTED]
    out = []
    for day in range(1, days + 1):
        if day < shift_day:
            today = [rest.pop() if rng.random() < len(rest) / (len(rest) + len(shifted)) else shifted.pop() for _ in range(per_day)]
        else:
            today = [shifted.pop() for _ in range(per_day // 2)] + [rest.pop() for _ in range(per_day - per_day // 2)]
            rng.shuffle(today)
        out.append(today)
    return names, out


def question(names):
    return {"intent": {"type": "choice", "instructions": "What is this bank customer asking about?",
                       "criteria": {n: n.replace("_", " ").strip("?").capitalize() for n in names}}}


def models():
    key = os.environ["OPENAI_API_KEY"]
    return ChatDecider("gpt-4o-mini", provider="openai", api_key=key), ChatReasoner("gpt-4.1", provider="openai", api_key=key)


def baseline(work, days):
    """Asks gpt-4o-mini and gpt-4.1 about every message once, through the cache the simulation reads, and records
    their answers, latency and token use."""
    names, schedule = days
    q = question(names)
    mini, big = models()
    # Every message escalates (no confidence reaches 1.01), so both models answer it.
    router = Router([mini], big, threshold=[1.01], cache=Cache(work / "cache.db"), retries=4, backoff=2.0)
    usage = {}
    real_post = base.post

    def post(url, json, **kwargs):
        response = real_post(url, json=json, **kwargs)
        if response.is_success:
            used = response.json().get("usage", {})
            u = usage.setdefault(json["model"], {"calls": 0, "input": 0, "output": 0})
            u["calls"], u["input"], u["output"] = u["calls"] + 1, u["input"] + used.get("prompt_tokens", 0), u["output"] + used.get("completion_tokens", 0)
        return response

    base.post = post
    done_path = work / "baseline.jsonl"
    done = {json.loads(line)["text"] for line in done_path.read_text().splitlines()} if done_path.exists() else set()
    todo = [m for day in schedule for m in day if m["text"] not in done]

    def one(m):
        result = router.decide(m["text"], q)
        a = result["answers"]["intent"]
        return {"text": m["text"], "gold": m["gold"], "mini": a["hunch"]["answer"], "mini_confidence": a["hunch"]["confidence"],
                "big": a["answer"], "mini_ms": result["latency_ms"]["tiers"][0], "big_ms": result["latency_ms"]["system2"],
                "cached": result["cache"]["hits"] > 0}

    try:
        with ThreadPoolExecutor(WORKERS) as pool, done_path.open("a") as f:
            for i, row in enumerate(pool.map(one, todo), 1):
                f.write(json.dumps(row) + "\n")
                if i % 200 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
    finally:
        base.post = real_post
        old = json.loads((work / "usage.json").read_text()) if (work / "usage.json").exists() else {}
        for model, u in usage.items():
            o = old.setdefault(model, {"calls": 0, "input": 0, "output": 0})
            for k in u:
                o[k] += u[k]
        (work / "usage.json").write_text(json.dumps(old, indent=1))
    print("baseline done:", len(todo), "messages asked; usage", json.loads((work / "usage.json").read_text()))


def per_call(work):
    """Mean cost ($) and latency (ms) of one call to each model, from the baseline."""
    usage = json.loads((work / "usage.json").read_text())
    rows = [json.loads(line) for line in (work / "baseline.jsonl").read_text().splitlines()]
    live = [r for r in rows if not r["cached"]]
    cost = {m: (u["input"] * PRICES[m][0] + u["output"] * PRICES[m][1]) / 1e6 / u["calls"] for m, u in usage.items()}
    ms = {"gpt-4o-mini": statistics.mean(r["mini_ms"] for r in live), "gpt-4.1": statistics.mean(r["big_ms"] for r in live)}
    return cost, ms, rows


def simulate(work, days, seed=0, per_second=0.0):
    names, schedule = days
    random.seed(seed)  # the router picks audits and probes at random
    q = question(names)
    mini, big = models()
    cost, ms, rows = per_call(work)
    costs = {"tier 0": 0.0, "tier 1": cost["gpt-4o-mini"], "system2": cost["gpt-4.1"]}
    # What prune weighs: the price of each call plus the price of waiting for it.
    worth = {"tier 1": cost["gpt-4o-mini"] + per_second * ms["gpt-4o-mini"] / 1000, "system2": cost["gpt-4.1"] + per_second * ms["gpt-4.1"] / 1000}
    tag = f"{seed}" + (f"_wait{per_second:g}" if per_second else "")
    sim = work / f"sim_{tag}.db"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{sim}{suffix}").unlink(missing_ok=True)
    store, cache = Store(sim), Cache(work / "cache.db")
    gold, local, recalibrated, report = {}, None, [], []
    for day, messages in enumerate(schedule, 1):
        # Tier 0 is always the classifier (empty until retrain first promotes one), so gpt-4o-mini stays tier 1.
        router = Router([local or LocalDecider({}), mini], big, threshold=[0.95, 0.8], log=store, audit=AUDIT, cache=cache, retries=4, backoff=2.0,
                        trust_uncalibrated=False, probe=PROBE)
        start = time.time()

        def one(m):
            result = router.decide(m["text"], q)
            return m, result

        with ThreadPoolExecutor(WORKERS) as pool:
            decided = list(pool.map(one, messages))
        right = {"tier 0": [0, 0], "tier 1": [0, 0], "system2": [0, 0]}
        latency = []
        for m, result in decided:
            gold[result["id"]] = m["gold"]
            a = result["answers"]["intent"]
            who = "system2" if a["source"] == "system2" else f"tier {a.get('tier', 0)}"
            right[who][0] += 1
            right[who][1] += a["answer"] == m["gold"]
            tiers = {t["tier"] for t in a["tried"]}
            # The cache makes the logged latency meaningless, so price each decision with typical times instead.
            latency.append((LOCAL_MS if 0 in tiers else 0) + (ms["gpt-4o-mini"] if 1 in tiers else 0)
                           + (ms["gpt-4.1"] if result["latency_ms"]["system2"] is not None else 0))
        # Night: a person labels what the review queue asks for most.
        queue = review_queue(store, limit=LABELS_PER_NIGHT, sample=SAMPLE)
        for item in queue:
            store.label(item["id"], item["key"], gold[item["id"]], source="human")
        trained = retrain(store, current=local, tier=0, target_accuracy=TARGET, min_samples=30)
        local = trained["decider"]
        s = stats(store, since=start, costs=costs)  # today's cost
        whole = stats(store, sample=SAMPLE)  # drift compares everything since each calibration, not just today
        drift = {d["tier"]: d for d in whole["drift"]}
        # retrain rechecks the classifier every night; calibrate grades only its answers since promotion, too few.
        due = {d["tier"] for d in whole["drift"] if d["flags"] and d["tier"] == 1} | ({1} if store.thresholds(1).get("intent") in (None, float("inf")) else set())
        cals = {t: calibrate(store, TARGET, 30, tier=t, sample=SAMPLE)["intent"] for t in sorted(due)}
        cal = cals.get(1)
        if due:
            recalibrated.append([day, sorted(due)])
        rechecked = recheck(store, tier=1, sample=SAMPLE).get("intent")
        pruned = prune(store, worth, tier=1).get("intent")
        n = len(messages)
        day_report = {
            "day": day, "messages": n,
            "accuracy": sum(r[1] for r in right.values()) / n,
            "answered_by": {who: r[0] / n for who, r in right.items()},
            "accuracy_by": {who: (r[1] / r[0] if r[0] else None) for who, r in right.items()},
            "cost": s["cost"]["total"], "always_system2_cost": s["cost"]["always_system2"], "savings": s["cost"]["savings"],
            "latency_ms_mean": statistics.mean(latency),
            "labels_tonight": len(queue), "review_reasons": {r: sum(i["reason"] == r for i in queue) for r in ("sample", "disagreement", "audit_disagreement", "unresolved")},
            "accuracy_on_sample": whole["questions"]["intent"]["accuracy_on_sample"],
            "retrain": trained["report"].get("intent"), "recheck": rechecked, "prune": pruned, "calibrated": {t: {k: c.get(k) for k in ("threshold", "trusted_accuracy", "escalation_rate", "reason", "best_target", "graded_by", "sample_check")}
                           for t, c in cals.items()},
            "tier1_threshold": store.thresholds(1).get("intent"), "tier0_threshold": store.thresholds(0).get("intent"),
            "drift": {t: {"flags": d["flags"], "unchecked": d["unchecked"], "escalation_rate": d["escalation_rate"], "baseline": d["baseline"]} for t, d in drift.items()},
        }
        report.append(day_report)
        print(f"day {day:>2}: accuracy {day_report['accuracy']:.3f}  answered " +
              "  ".join(f"{w} {v:.0%}" for w, v in day_report["answered_by"].items()) +
              f"  saved {day_report['savings']:.0%}  drift {[f for d in whole['drift'] for f in d['flags']]}  recalibrated {cal is not None}", flush=True)
    everyone = [m for day in schedule for m in day]
    by_text = {r["text"]: r for r in rows}
    summary = {
        "days": report,
        "router": {"accuracy": sum(d["accuracy"] * d["messages"] for d in report) / len(everyone),
                   "cost": sum(d["cost"] for d in report), "latency_ms_mean": statistics.mean(d["latency_ms_mean"] for d in report),
                   "labels": sum(d["labels_tonight"] for d in report), "recalibrated_on": recalibrated},
        "always_gpt-4o-mini": {"accuracy": statistics.mean(by_text[m["text"]]["mini"] == m["gold"] for m in everyone),
                               "cost": cost["gpt-4o-mini"] * len(everyone), "latency_ms_mean": ms["gpt-4o-mini"]},
        "always_gpt-4.1": {"accuracy": statistics.mean(by_text[m["text"]]["big"] == m["gold"] for m in everyone),
                           "cost": cost["gpt-4.1"] * len(everyone), "latency_ms_mean": ms["gpt-4.1"]},
        "per_call": {"cost": cost, "latency_ms": ms},
    }
    (work / f"days_{tag}.jsonl").write_text("".join(json.dumps(d) + "\n" for d in report))
    (work / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "days"}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["baseline", "simulate"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-second", type=float, default=0.0)
    args = parser.parse_args()
    work = Path("data/case_study_banking77" + ("_smoke" if args.smoke else ""))
    work.mkdir(parents=True, exist_ok=True)
    days = traffic(2, 20, 2) if args.smoke else traffic(10, 300, 6)
    baseline(work, days) if args.phase == "baseline" else simulate(work, days, args.seed, args.per_second)
