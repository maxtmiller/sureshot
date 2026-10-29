"""Answer with System 1 (one decider, or a chain of them), and send only the questions it is unsure about to System 2."""
import copy
import random
import threading
from contextlib import contextmanager
import subprocess
import time
import httpx
from hunch.adapters.base import Outage
from hunch.cache import cache_key
from hunch.questions import validate

RETRY_STATUS = (408, 429, 500, 502, 503, 504)


def _retriable(error):
    """A network failure or a status that usually clears on its own, from the HTTP backends."""
    return isinstance(error, httpx.TransportError) or (isinstance(error, httpx.HTTPStatusError) and error.response.status_code in RETRY_STATUS)


def _transient(error):
    """An outage rather than a mistake (bad key, wrong model...), so it is safe to fall back to the other model."""
    return _retriable(error) or isinstance(error, (subprocess.TimeoutExpired, Outage))


class Router:
    def __init__(self, system1, system2=None, threshold=0.8, log=None, audit=0.0, retries=2, backoff=0.5, cache=None, guard=None, guard_threshold=0.9):
        """system1 is one decider, or a list of them tried in order (say a trained classifier, then an LLM): a question
        stops at the first tier that is confident about it. A tier may answer only some of the questions it is given;
        the rest pass down. threshold is a number, or a list with one number per tier, since a classifier's 0.9 and
        an LLM's 0.9 mean different things.

        log is a Store that records every decision and supplies calibrated thresholds. audit is the share of confident
        questions that also go to System 2, only to be graded: it keeps the log's labels unbiased for calibration.
        A failed call to either model is retried `retries` times, waiting backoff, 2x backoff... seconds between tries.

        cache is a hunch.Cache: a model's answer to a question about a text is reused instead of asking again. It caches
        what the models said, not the decision, so changing a threshold or recalibrating needs no cache clearing. Only
        models with an `identity` are cached.

        guard is a second System 1 (say a LocalDecider trained on your labels) that checks the answers the tiers were
        confident about: where it disagrees with probability at least guard_threshold, the answer is treated as unsure
        and escalates. A model's confidence says nothing about errors it is confidently wrong about; a model trained
        on your own labels often knows them. It runs in-process with the router, and its errors are not caught."""
        self.system1, self.system2, self.threshold, self.log, self.audit = system1, system2, threshold, log, audit
        self.guard, self.guard_threshold = guard, guard_threshold
        self.tiers = list(system1) if isinstance(system1, (list, tuple)) else [system1]
        self.tier_thresholds = list(threshold) if isinstance(threshold, (list, tuple)) else [threshold] * len(self.tiers)
        if len(self.tier_thresholds) != len(self.tiers):
            raise ValueError(f"{len(self.tiers)} tiers but {len(self.tier_thresholds)} thresholds")
        self.retries, self.backoff, self.record, self.cache = retries, backoff, True, cache
        self._flights, self._flights_lock = {}, threading.Lock()

    def _call(self, fn, *args):
        for attempt in range(self.retries + 1):
            try:
                return fn(*args)
            except Exception as e:
                if attempt == self.retries or not _retriable(e):
                    raise
                time.sleep(self.backoff * 2 ** attempt)

    @contextmanager
    def _single_flight(self, hashes):
        """Holds a lock per cache key, so concurrent requests for the same question wait for the first instead of all
        asking the model. Locks are taken in sorted order (no deadlock) and dropped when nobody is using them."""
        ordered = sorted(set(hashes))
        with self._flights_lock:
            flights = [self._flights.setdefault(h, [threading.Lock(), 0]) for h in ordered]
            for flight in flights:
                flight[1] += 1
        held = []
        try:
            for flight in flights:
                flight[0].acquire()
                held.append(flight)
            yield
        finally:
            for flight in reversed(held):
                flight[0].release()
            with self._flights_lock:
                for h, flight in zip(ordered, flights):
                    flight[1] -= 1
                    if not flight[1]:
                        del self._flights[h]

    def _cached(self, model, method, state, questions, images, tally):
        """model.method(state, questions, images), taking from the cache what it has and calling the model for the rest."""
        identity = getattr(model, "identity", None) if self.cache else None
        call = getattr(model, method)
        if identity is None:
            return self._call(call, state, questions, images)
        hashes = {key: cache_key(identity, key, question, state, images) for key, question in questions.items()}
        found = self.cache.get_many(hashes.values())
        hits = {key: found[h] for key, h in hashes.items() if h in found}
        missing = {key: question for key, question in questions.items() if key not in hits}
        fresh = {}
        if missing:
            with self._single_flight(hashes[key] for key in missing):
                # Another request may have asked the model while this one waited.
                found = self.cache.get_many(hashes[key] for key in missing)
                hits |= {key: found[hashes[key]] for key in missing if hashes[key] in found}
                missing = {key: question for key, question in missing.items() if key not in hits}
                fresh = self._call(call, state, missing, images) if missing else {}
                self.cache.put_many({hashes[key]: value for key, value in fresh.items() if key in hashes})
        tally["hits"] += len(hits)
        tally["misses"] += len(missing)
        return hits | fresh

    def decide(self, state, questions, images=(), thresholds=None, escalate=True):
        """Returns {"answers": {key: {answer, confidence, source, ...}}, "unsure": [...], "escalated": [...],
        "degraded", "cache" ({"hits", "misses"}: how many model answers came from the cache; None without one), "latency_ms": {"system1" (all tiers), "tiers" (each), "system2" (None if not called), "total"}},
        plus "id" when a log is set (pass it to the log's label() to grade the answer).

        Each question goes to the first tier; if its confidence is below that tier's threshold it goes to the next.
        A tier's threshold is thresholds[key] (the same for every tier), else the log's calibrated one for that tier
        and question, else the router's. Questions no tier is confident about are "unsure". They go to System 2 when
        escalate is true and a System 2 is set; their answer then comes from it, with the last tier's answer kept under
        "hunch". Otherwise the last tier's answer stands and the key is only listed in "unsure". An answer from a tier
        has "source": "hunch" and "tier"; every answer lists what each tier said under "tried". A confident answer the
        guard disagreed with is unsure, and carries the guard's answer under "guard".

        "degraded" is None unless a model was down (after retries): a list of {"system": 1 or 2, "tier", "error"}. A
        tier that is down is skipped. If every tier is down, System 2 answers every question (each answer has
        "hunch": None). If System 2 is down, the last tier's answers stand and "escalated" is empty. Only outages fall
        back this way: a bad key or an unknown model raises, so a misconfiguration is not hidden by silently using
        another model."""
        validate(questions)
        thresholds = thresholds or {}
        start = time.perf_counter()
        tally = {"hits": 0, "misses": 0}
        degraded, tier_ms, tried, latest, answers, remaining = [], [], {key: [] for key in questions}, {}, {}, list(questions)
        for i, tier in enumerate(self.tiers):
            if not remaining:
                break
            tier_start = time.perf_counter()
            try:
                fast = self._cached(tier, "decide", state, {key: questions[key] for key in remaining}, images, tally)
            except Exception as e:
                if not (_transient(e) and (i + 1 < len(self.tiers) or (escalate and self.system2))):
                    raise
                degraded.append({"system": 1, "tier": i, "error": repr(e)})
                fast = {}
            tier_ms.append((time.perf_counter() - tier_start) * 1000)
            saved = self.log.thresholds(i) if self.log else {}
            unsure = []
            for key in remaining:
                if key not in fast:
                    unsure.append(key)
                    continue
                tried[key].append({"tier": i, "answer": fast[key]["answer"], "confidence": fast[key]["confidence"]})
                latest[key] = fast[key] | {"tier": i}
                if fast[key]["confidence"] < thresholds.get(key, saved.get(key, self.tier_thresholds[i])):
                    unsure.append(key)
                else:
                    answers[key] = fast[key] | {"source": "hunch", "tier": i}
            remaining = unsure
        vetoed = {}
        if self.guard and answers:
            checks = self._cached(self.guard, "decide", state, {key: questions[key] for key in answers}, images, tally)
            vetoed = {key: check for key, check in checks.items()
                      if check["answer"] != answers[key]["answer"] and check["confidence"] >= self.guard_threshold}
            for key in vetoed:
                del answers[key]
            remaining = remaining + list(vetoed)
        system2_ms = None
        escalated = remaining if escalate and self.system2 and remaining else []
        audited = [key for key in answers if escalate and self.system2 and random.random() < self.audit]
        if escalated or audited:
            slow_start = time.perf_counter()
            try:
                slow = self._cached(self.system2, "answer", state, {key: questions[key] for key in escalated + audited}, images, tally)
            except Exception as e:
                if not _transient(e) or any(key not in latest for key in escalated):
                    raise
                slow = None
                if escalated:
                    degraded.append({"system": 2, "tier": None, "error": repr(e)})
            system2_ms = (time.perf_counter() - slow_start) * 1000
            if slow is None:
                escalated = []
            else:
                for key in escalated:
                    answers[key] = {"answer": slow[key]["answer"], "reason": slow[key]["reason"], "source": "system2", "hunch": latest.get(key)}
                for key in audited:
                    answers[key]["audit"] = slow[key]
        for key in remaining:
            if key not in escalated:
                if key not in latest:
                    raise ValueError(f"no System 1 tier answered {key!r}, and there is no System 2 to escalate it to")
                answers[key] = latest[key] | {"source": "hunch"}
        answers = {key: answers[key] | {"tried": tried[key]} | ({"guard": vetoed[key]} if key in vetoed else {}) for key in questions}
        latency = {"system1": sum(tier_ms), "system2": system2_ms, "total": (time.perf_counter() - start) * 1000}
        result = {"answers": answers, "unsure": remaining, "escalated": escalated, "degraded": degraded or None, "cache": tally if self.cache else None,
                  "latency_ms": {k: v and round(v, 1) for k, v in latency.items()} | {"tiers": [round(ms, 1) for ms in tier_ms]}}
        if self.log and self.record:
            result["id"] = self.log.log(state, questions, result | {"audit_rate": self.audit}, len(images))
        return result

    def test(self, cases, questions=None, thresholds=None):
        """Scores this router on a golden dataset (a list of cases or the path of a JSONL file; see hunch.golden.run)
        without writing to the log. The log's calibrated thresholds still apply."""
        from hunch.golden import run

        plain = copy.copy(self)
        plain.record = False
        return run(plain, cases, questions, thresholds)
