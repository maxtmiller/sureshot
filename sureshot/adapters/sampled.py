"""A System 1 from a model that gives no logprobs: ask it the same questions several times and take the share of
answers that agree as the confidence."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


def _options(question):
    if question["type"] == "noul":
        return [True, False]
    if question["type"] == "score":
        return list(range(len(question["criteria"])))
    return list(question["criteria"])


class SampledDecider:
    """reasoner is any System 2 class (ClaudeAPI, ChatReasoner, GeminiReasoner, HTTPReasoner...). It is asked n times
    in parallel, each call answering every question; a question's answer is the one most samples gave, and its
    confidence the share that gave it.

    The samples have to vary for that share to mean anything: keep the model's temperature above 0 (the default for
    Claude, OpenAI and Gemini), or every answer is unanimous and calibration finds no threshold. A vote share is coarse
    (with n=5 it is 0.2, 0.4 ... 1.0), so a calibrated threshold is a number of votes; a larger n gives finer ones at
    n times the cost. A sample that fails is left out; the call fails only when every sample does."""

    def __init__(self, reasoner, n=5):
        if n < 2:
            raise ValueError("n must be at least 2")
        self.reasoner, self.n = reasoner, n
        identity = getattr(reasoner, "identity", None)
        self.identity = identity and f"sampled|{n}|{identity}"

    def decide(self, state, questions, images=()):
        with ThreadPoolExecutor(self.n) as pool:
            futures = [pool.submit(self.reasoner.answer, state, questions, images) for _ in range(self.n)]
        samples, errors = [], []
        for future in futures:
            try:
                samples.append(future.result())
            except Exception as e:
                errors.append(e)
        if not samples:
            raise errors[0]
        decided = {}
        for key, question in questions.items():
            votes = Counter(s[key]["answer"] for s in samples if key in s)
            if not votes:
                continue
            total = sum(votes.values())
            answer, count = votes.most_common(1)[0]
            probabilities = {str(o).lower() if question["type"] == "noul" else str(o): votes[o] / total for o in _options(question)}
            decided[key] = {"answer": answer, "confidence": count / total, "probabilities": probabilities}
            if question["type"] == "score":
                decided[key]["score"] = sum(level * n for level, n in votes.items()) / total
        return decided
