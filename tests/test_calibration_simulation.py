"""Does calibration do what it claims? Simulate models whose accuracy at each confidence is known, then check what
sureshot estimates and picks against the truth. Seeds are fixed, so these are deterministic."""
import random
import pytest
from sureshot import Router, Store, calibrate
from sureshot.calibrate import summarize

CONF = lambda rng: 0.5 + 0.5 * rng.random() ** 0.5  # confidence, skewed high
Q = {"q": {"type": "noul", "instructions": "?"}}


def true_kept_accuracy(curve, threshold, n=60_000):
    """The real accuracy among answers at or above a confidence, from a large sample."""
    rng = random.Random(99)
    kept = [curve(c) for c in (CONF(rng) for _ in range(n)) if c >= threshold]
    return sum(kept) / len(kept)


def calibrate_once(curve, n, target, rng):
    samples = []
    for _ in range(n):
        c = CONF(rng)
        samples.append((c, rng.random() < curve(c), 1.0))
    return summarize(samples, target, min_samples=30)["threshold"]


def test_threshold_meets_the_accuracy_target_about_as_often_as_claimed():
    # A "calibrated" model (right with probability equal to its confidence), 1000 graded answers, 90% target. The bound
    # is one-sided at 90%, and the fixed-sequence test keeps it honest for the threshold it picks: about 10% of
    # calibrations or fewer land under the target, and only by a hair.
    rng, curve = random.Random(1), lambda c: c
    cache, misses, shortfalls = {}, 0, []
    thresholds = [calibrate_once(curve, 1000, 0.9, rng) for _ in range(200)]
    assert sum(t is None for t in thresholds) <= 10
    for t in filter(None, thresholds):
        key = round(t, 3)
        cache.setdefault(key, true_kept_accuracy(curve, key))
        misses += cache[key] < 0.9
        shortfalls.append(max(0, 0.9 - cache[key]))
    ran = len(shortfalls)
    assert misses / ran <= 0.12
    assert sum(shortfalls) / ran < 0.01 and max(shortfalls) < 0.04


def test_a_strong_model_gets_a_threshold_at_a_hard_target_with_few_answers():
    # The sequence starts at the first slice that could pass with no errors, so a model that can reach 95% is certified
    # in more than half of calibrations on 300 answers, and about as rarely too optimistic as the bound claims.
    rng, curve = random.Random(6), lambda c: min(1.0, 0.4 + 1.2 * (c - 0.5))
    thresholds = [t for t in (calibrate_once(curve, 300, 0.95, rng) for _ in range(200)) if t is not None]
    assert len(thresholds) >= 100
    cache = {}
    misses = sum(cache.setdefault(round(t, 3), true_kept_accuracy(curve, round(t, 3))) < 0.95 for t in thresholds)
    assert misses / len(thresholds) <= 0.12


def test_it_abstains_when_the_target_cannot_be_reached():
    # This model never gets above 80% right, so no threshold can reach 90%. It should almost never pick one.
    rng, curve = random.Random(2), lambda c: 0.5 + 0.6 * (c - 0.5)
    picked = sum(calibrate_once(curve, 300, 0.9, rng) is not None for _ in range(200))
    assert picked <= 6


class Tier:
    """A tier that answers with a drawn confidence, right with probability curve(confidence)."""

    def __init__(self, rng, truth, curve):
        self.rng, self.truth, self.curve, self.drawn = rng, truth, curve, {}

    def decide(self, state, questions, images=()):
        c = CONF(self.rng)
        right = self.rng.random() < self.curve(c)
        self.drawn[state] = right
        return {"q": {"answer": self.truth[state] == right, "confidence": c, "probabilities": {}}}


class Oracle:
    def __init__(self, truth):
        self.truth = truth

    def answer(self, state, questions, images=()):
        return {"q": {"answer": self.truth[state], "reason": "r"}}


def simulate(tmp_path, curves, thresholds, audit, n=1500, seed=0):
    random.seed(seed)  # the router's audit sampling uses the global generator
    rng = random.Random(seed)
    truth = {f"s{i}": rng.random() < 0.5 for i in range(n)}
    tiers = [Tier(rng, truth, curve) for curve in curves]
    store = Store(tmp_path / f"sim{seed}.db")
    router = Router(tiers, Oracle(truth), threshold=thresholds, log=store, audit=audit)
    for state in truth:
        router.decide(state, Q)
    rows = store.labeled()
    result = []
    for i, tier in enumerate(tiers):
        mine = [r for r in rows if r["tier"] == i]
        true_accuracy = sum(tier.drawn.values()) / len(tier.drawn)
        weighted = sum(r["weight"] * (r["answer"] == r["truth"]) for r in mine) / sum(r["weight"] for r in mine)
        unweighted = sum(r["answer"] == r["truth"] for r in mine) / len(mine)
        result.append((true_accuracy, weighted, unweighted))
    return result, store


def test_audit_weights_recover_the_true_accuracy(tmp_path):
    # Escalated answers are all graded and confident ones are graded at the audit rate, so the log over-represents
    # the unsure. Weighting by the inverse of that chance fixes it; averaging the graded rows does not.
    (true, weighted, unweighted), = simulate(tmp_path, [lambda c: min(1.0, 0.4 + 1.2 * (c - 0.5))], 0.8, audit=0.2)[0]
    assert abs(weighted - true) < 0.04
    assert true - unweighted > 0.05


def test_weights_hold_for_every_tier_of_a_chain(tmp_path):
    # Tier 1 only sees what tier 0 was unsure about, and its answers are graded when System 2 was asked or audited.
    curves = [lambda c: min(1.0, 0.3 + 1.1 * (c - 0.5)), lambda c: min(1.0, 0.55 + 0.9 * (c - 0.5))]
    result, _ = simulate(tmp_path, curves, [0.85, 0.8], audit=0.2)
    for true, weighted, _ in result:
        assert abs(weighted - true) < 0.05
    # Tier 0 sees every item, so averaging its graded rows is only slightly off. Tier 1 sees a filtered population and is
    # far off without the weights.
    true, weighted, unweighted = result[1]
    assert true - unweighted > 0.05 > abs(weighted - true)


def test_a_threshold_picked_from_audited_logs_meets_the_target(tmp_path):
    _, store = simulate(tmp_path, [lambda c: min(1.0, 0.4 + 1.2 * (c - 0.5))], 0.8, audit=0.2, n=2000, seed=3)
    threshold = calibrate(store, 0.9, 30, save=False)["q"]["threshold"]
    assert threshold is not None
    assert true_kept_accuracy(lambda c: min(1.0, 0.4 + 1.2 * (c - 0.5)), threshold) >= 0.89


def test_reasons_match_what_the_simulated_model_can_do():
    from sureshot.calibrate import summarize

    rng = random.Random(4)
    # Never above 80% right: at a 90% target, 300 answers are enough to see it is out of reach.
    weak = [(c, rng.random() < 0.5 + 0.6 * (c - 0.5), 1.0) for c in (CONF(rng) for _ in range(300))]
    report = summarize(weak, 0.9, 30)
    assert report["threshold"] is None and report["reason"] == "target unreachable" and report["best_target"] < 0.85


def test_unreachable_is_only_claimed_when_the_data_rules_the_target_out():
    from sureshot.calibrate import summarize

    def reasons(curve, n, target, trials=200):
        rng, out = random.Random(5), []
        for _ in range(trials):
            samples = [(c, rng.random() < curve(c), 1.0) for c in (CONF(rng) for _ in range(n))]
            out.append(summarize(samples, target, 30)["reason"])
        return out

    # A model that can reach 95%. With little data the numbers may not show it yet, but the data must rarely say the
    # target is out of reach: about 15% of the time at 40 answers, 1% at 100, never by 300.
    curve = lambda c: min(1.0, 0.4 + 1.2 * (c - 0.5))
    assert reasons(curve, 40, 0.95).count("target unreachable") <= 50
    assert reasons(curve, 100, 0.95).count("target unreachable") <= 8
    assert reasons(curve, 300, 0.95).count("target unreachable") == 0
    # A model capped near 80% right: the data says so reliably only with plenty of answers, and before that it says
    # "not enough evidence" (which is true) and never finds a threshold.
    capped = lambda c: 0.5 + 0.6 * (c - 0.5)
    assert reasons(capped, 1000, 0.9, trials=100).count("target unreachable") >= 80
    few = reasons(capped, 100, 0.9)
    assert few.count(None) <= 4 and "target unreachable" in few
