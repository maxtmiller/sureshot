"""Run a golden dataset through a Router and report accuracy, escalation and latency."""
import json
import math
from pathlib import Path
from hunch.questions import check_answer


def load(path):
    """Cases from a JSONL file: one {"state", "expected", and optionally "questions" and "images"} per line."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def percentiles(values):
    values = sorted(values)
    return {p: values[math.ceil(p / 100 * len(values)) - 1] for p in (50, 95)} if values else None


def run(router, cases, questions=None, thresholds=None):
    """Each case is {"state", "expected": {key: answer}} plus its own "questions", or the shared `questions`. Only the
    questions with an expected answer are asked. Returns accuracy of the final answers and of System 1 alone, the
    escalation rate (each overall and per question), the share of answers each tier and System 2 gave, p50/p95 latency
    in ms, and every wrong final answer."""
    cases = load(cases) if isinstance(cases, (str, Path)) else cases
    tally, failures, answered_by, latency = {}, [], {}, {"system1": [], "system2": [], "total": []}
    for i, case in enumerate(cases):
        all_questions = case.get("questions") or questions
        for key, answer in case["expected"].items():
            try:
                check_answer(all_questions[key], answer)
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"case {i}, question {key!r}: bad expected answer or missing question ({e})") from e
        result = router.decide(case["state"], {key: all_questions[key] for key in case["expected"]}, case.get("images", ()), thresholds)
        for name in latency:
            if result["latency_ms"][name] is not None:
                latency[name].append(result["latency_ms"][name])
        for key, expected in case["expected"].items():
            answer = result["answers"][key]
            fast = answer["hunch"] if answer["source"] == "system2" else answer
            fast_answer = fast and fast["answer"]
            t = tally.setdefault(key, {"n": 0, "final": 0, "system1": 0, "escalated": 0})
            t["n"] += 1
            t["final"] += answer["answer"] == expected
            t["system1"] += fast is not None and fast_answer == expected
            t["escalated"] += answer["source"] == "system2"
            who = "system2" if answer["source"] == "system2" else f"tier {answer['tier']}"
            answered_by[who] = answered_by.get(who, 0) + 1
            if answer["answer"] != expected:
                failures.append({"case": i, "key": key, "expected": expected, "got": answer["answer"],
                                 "source": answer["source"], "confidence": fast and fast["confidence"]})
    total = {field: sum(t[field] for t in tally.values()) for field in ("n", "final", "system1", "escalated")}
    rates = lambda t: {"n": t["n"], "accuracy": t["final"] / t["n"], "system1_accuracy": t["system1"] / t["n"], "escalation_rate": t["escalated"] / t["n"]}
    return {"cases": len(cases), "overall": rates(total) if total["n"] else None,
            "answered_by": {who: n / total["n"] for who, n in sorted(answered_by.items())}, "questions": {key: rates(t) for key, t in tally.items()},
            "latency_ms": {name: percentiles(values) for name, values in latency.items()}, "failures": failures}
