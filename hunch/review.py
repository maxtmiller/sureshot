"""The questions worth a person's answer: where the models disagree, or nobody was sure."""

REASONS = ("disagreement", "audit_disagreement", "unresolved")


def review_queue(store, limit=50, reasons=REASONS, since=0.0):
    """Ungraded questions in the log that a person's label would help most, as a list of items, most useful first
    (by reason, then oldest first). An item is {id, key, reason, state, question, answer, source, candidates, ts}; give
    the true answer with store.label(id, key, answer) (or POST /feedback) and it leaves the queue.

    reasons:
    - "disagreement": a tier was unsure, System 2 answered, and the two differ.
    - "audit_disagreement": a confident answer was audited by System 2, which disagreed.
    - "unresolved": no tier was confident and System 2 was not asked (escalate=False, or it was down).

    candidates lists what each tier and System 2 said. Questions answered by System 2 with no System 1 answer to
    compare (System 1 was down) are not queued."""
    items = sorted((item for item in pending(store, since) if item["reason"] in reasons), key=lambda item: (REASONS.index(item["reason"]), item["ts"]))
    items = items[:limit] if limit is not None else items
    states = store.states({item["id"] for item in items})
    return [item | {"state": states[item["id"]]} for item in items]


def pending(store, since=0.0, decisions=None, labels=None):
    """Every queued item, without its state text (so counting is cheap). `decisions` and `labels` may be passed
    already loaded (as store.results(since) and store.labels() return them), so a caller that has them does not
    read and decode the log again."""
    labels = store.labels() if labels is None else labels
    items = []
    for decision_id, ts, questions, result in store.results(since) if decisions is None else decisions:
        for key, answer in result["answers"].items():
            if (decision_id, key) in labels:
                continue
            candidates = [{"who": f"tier {t['tier']}", "answer": t["answer"], "confidence": t["confidence"]} for t in answer.get("tried", [])]
            if answer["source"] == "system2":
                if answer.get("hunch") is None or answer["hunch"]["answer"] == answer["answer"]:
                    continue
                reason = "disagreement"
                candidates.append({"who": "system2", "answer": answer["answer"], "reason": answer["reason"]})
            elif "audit" in answer and answer["audit"]["answer"] != answer["answer"]:
                reason = "audit_disagreement"
                candidates.append({"who": "system2", "answer": answer["audit"]["answer"], "reason": answer["audit"]["reason"]})
            elif key in result["unsure"]:
                reason = "unresolved"
            else:
                continue
            items.append({"id": decision_id, "key": key, "reason": reason, "question": questions[key], "answer": answer["answer"],
                          "source": answer["source"], "candidates": candidates, "ts": ts})
    return items
