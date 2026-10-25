"""Answer with System 1, and send only the questions it is unsure about to System 2."""
import random
import subprocess
import time
import anthropic
import httpx
from hunch.questions import validate

RETRY_STATUS = (408, 429, 500, 502, 503, 504)


def _retriable(error):
    """A network failure or a status that usually clears on its own, from the HTTP backends."""
    return isinstance(error, httpx.TransportError) or (isinstance(error, httpx.HTTPStatusError) and error.response.status_code in RETRY_STATUS)


def _transient(error):
    """An outage rather than a mistake (bad key, wrong model...), so it is safe to fall back to the other model."""
    return _retriable(error) or isinstance(error, (subprocess.TimeoutExpired, anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError))


class Router:
    def __init__(self, system1, system2=None, threshold=0.8, log=None, audit=0.0, retries=2, backoff=0.5):
        """log is a Store that records every decision and supplies calibrated thresholds. audit is the share of confident
        questions that also go to System 2, only to be graded: it keeps the log's labels unbiased for calibration.
        A failed call to either model is retried `retries` times, waiting backoff, 2x backoff... seconds between tries."""
        self.system1, self.system2, self.threshold, self.log, self.audit = system1, system2, threshold, log, audit
        self.retries, self.backoff = retries, backoff

    def _call(self, fn, *args):
        for attempt in range(self.retries + 1):
            try:
                return fn(*args)
            except Exception as e:
                if attempt == self.retries or not _retriable(e):
                    raise
                time.sleep(self.backoff * 2 ** attempt)

    def decide(self, state, questions, images=(), thresholds=None, escalate=True):
        """Returns {"answers": {key: {answer, confidence, source, ...}}, "unsure": [...], "escalated": [...],
        "degraded", "latency_ms": {"system1", "system2" (None if not called), "total"}}, plus "id" when a log is set (pass
        it to the log's label() to grade the answer).

        "degraded" is None unless a model was down (after retries): {"system": 1 or 2, "error": ...}. If System 1 is
        down, System 2 answers every question (each answer has "hunch": None). If System 2 is down, System 1's answers
        stand, "escalated" is empty and the unsure keys are still listed in "unsure". Only outages fall back this way: a
        bad key or an unknown model raises, so a misconfiguration is not hidden by silently using the other model.

        A question is unsure when System 1's confidence is below its threshold: thresholds[key], else the log's
        calibrated one, else the router's. Unsure questions go to System 2 when escalate is true and a System 2 is set;
        their answer then comes from it, with System 1's answer kept under "hunch". Otherwise System 1's answer stands
        and the key is only listed in "unsure"."""
        validate(questions)
        thresholds = (self.log.thresholds() if self.log else {}) | (thresholds or {})
        start = time.perf_counter()
        degraded = None
        try:
            fast = self._call(self.system1.decide, state, questions, images)
        except Exception as e:
            if not (escalate and self.system2 and _transient(e)):
                raise
            fast, degraded = None, {"system": 1, "error": repr(e)}
        system1_ms = (time.perf_counter() - start) * 1000
        system2_ms = None
        if fast is None:
            slow_start = time.perf_counter()
            slow = self._call(self.system2.answer, state, questions, images)
            system2_ms = (time.perf_counter() - slow_start) * 1000
            answers = {key: {"answer": slow[key]["answer"], "reason": slow[key]["reason"], "source": "system2", "hunch": None} for key in questions}
            unsure = escalated = list(questions)
        else:
            answers = {key: answer | {"source": "hunch"} for key, answer in fast.items()}
            unsure = [key for key, answer in fast.items() if answer["confidence"] < thresholds.get(key, self.threshold)]
            escalated = unsure if escalate and self.system2 and unsure else []
            audited = [key for key in fast if key not in unsure and escalate and self.system2 and random.random() < self.audit]
            if escalated or audited:
                slow_start = time.perf_counter()
                try:
                    slow = self._call(self.system2.answer, state, {key: questions[key] for key in escalated + audited}, images)
                except Exception as e:
                    if not _transient(e):
                        raise
                    slow = None
                    if escalated:
                        degraded = {"system": 2, "error": repr(e)}
                system2_ms = (time.perf_counter() - slow_start) * 1000
                if slow is None:
                    escalated = []
                else:
                    for key in escalated:
                        answers[key] = {"answer": slow[key]["answer"], "reason": slow[key]["reason"], "source": "system2", "hunch": fast[key]}
                    for key in audited:
                        answers[key]["audit"] = slow[key]
        latency = {"system1": system1_ms, "system2": system2_ms, "total": (time.perf_counter() - start) * 1000}
        result = {"answers": answers, "unsure": unsure, "escalated": escalated, "degraded": degraded, "latency_ms": {k: v and round(v, 1) for k, v in latency.items()}}
        if self.log:
            result["id"] = self.log.log(state, questions, result | {"audit_rate": self.audit}, len(images))
        return result

    def test(self, cases, questions=None, thresholds=None):
        """Scores this router on a golden dataset (a list of cases or the path of a JSONL file; see hunch.golden.run)
        without writing to the log. The log's calibrated thresholds still apply."""
        from hunch.golden import run

        plain = Router(self.system1, self.system2, self.threshold, retries=self.retries, backoff=self.backoff)
        return run(plain, cases, questions, (self.log.thresholds() if self.log else {}) | (thresholds or {}))
