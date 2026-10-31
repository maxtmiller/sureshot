"""The questions worth a person's answer: a random sample, where the models disagree, or where nobody was sure."""
import zlib

REASONS = ("sample", "confident", "disagreement", "audit_disagreement", "unresolved")


def in_sample(decision_id, rate):
    """Whether a decision is in the random labeling sample: a fixed share of decisions, picked by id, so the same
    ones stay picked however often the queue is read."""
    return rate > 0 and zlib.crc32(decision_id.encode()) % 10_000 < rate * 10_000


def review_queue(store, limit=50, reasons=REASONS, since=0.0, sample=0.0, confident=0.0):
    """Ungraded questions in the log that a person's label would help most, as a list of items, most useful first
    (by reason, then oldest first). An item is {id, key, reason, state, question, answer, source, candidates, ts}; give
    the true answer with store.label(id, key, answer) (or POST /feedback) and it leaves the queue.

    reasons:
    - "sample": the decision is in a random `sample` share of all decisions, whatever the models said. Labels on
      these are what stats(store, sample=...) measures the router's accuracy with: the other reasons pick the hard
      cases, so their labels say nothing about how often the router is right overall. They come first.
    - "confident": a System 1 tier was confident (at or above its threshold, whether or not its answer was kept), and
      the decision is in a random `confident` share of those. Labels on these are what calibrate and recheck confirm a
      tier's threshold with: a small `sample` gives few labels on what a tier is confident about. Like "sample", it
      takes precedence over the reasons below, so these labels stay a random pick.
    - "disagreement": a tier was unsure, System 2 answered, and the two differ.
    - "audit_disagreement": a confident answer was audited by System 2, which disagreed.
    - "unresolved": no tier was confident and System 2 was not asked (escalate=False, or it was down).

    candidates lists what each tier and System 2 said. Questions answered by System 2 with no System 1 answer to
    compare (System 1 was down) are not queued."""
    items = sorted((item for item in pending(store, since, sample=sample, confident=confident) if item["reason"] in reasons), key=lambda item: (REASONS.index(item["reason"]), item["ts"]))
    items = items[:limit] if limit is not None else items
    states = store.states({item["id"] for item in items})
    return [item | {"state": states[item["id"]]} for item in items]


def pending(store, since=0.0, decisions=None, labels=None, sample=0.0, confident=0.0):
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
            reason = None
            if answer["source"] == "system2":
                candidates.append({"who": "system2", "answer": answer["answer"], "reason": answer["reason"]})
                if answer.get("sureshot") is not None and answer["sureshot"]["answer"] != answer["answer"]:
                    reason = "disagreement"
            elif "audit" in answer:
                candidates.append({"who": "system2", "answer": answer["audit"]["answer"], "reason": answer["audit"]["reason"]})
                if answer["audit"]["answer"] != answer["answer"]:
                    reason = "audit_disagreement"
            elif key in result["unsure"]:
                reason = "unresolved"
            if in_sample(decision_id, sample):
                reason = "sample"
            elif any(t.get("confident") for t in answer.get("tried", [])) and in_sample(decision_id, confident):
                reason = "confident"
            if reason is None:
                continue
            items.append({"id": decision_id, "key": key, "reason": reason, "question": questions[key], "answer": answer["answer"],
                          "source": answer["source"], "candidates": candidates, "ts": ts})
    return items
