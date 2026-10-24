"""Answer with System 1, and send only the questions it is unsure about to System 2."""
import random
from hunch.questions import validate


class Router:
    def __init__(self, system1, system2=None, threshold=0.8, log=None, audit=0.0):
        """log is a Store that records every decision and supplies calibrated thresholds. audit is the share of confident
        questions that also go to System 2, only to be graded: it keeps the log's labels unbiased for calibration."""
        self.system1, self.system2, self.threshold, self.log, self.audit = system1, system2, threshold, log, audit

    def decide(self, state, questions, images=(), thresholds=None, escalate=True):
        """Returns {"answers": {key: {answer, confidence, source, ...}}, "unsure": [...], "escalated": [...]}, plus "id"
        when a log is set (pass it to the log's label() to grade the answer).

        A question is unsure when System 1's confidence is below its threshold: thresholds[key], else the log's
        calibrated one, else the router's. Unsure questions go to System 2 when escalate is true and a System 2 is set;
        their answer then comes from it, with System 1's answer kept under "hunch". Otherwise System 1's answer stands
        and the key is only listed in "unsure"."""
        validate(questions)
        thresholds = (self.log.thresholds() if self.log else {}) | (thresholds or {})
        fast = self.system1.decide(state, questions, images)
        answers = {key: answer | {"source": "hunch"} for key, answer in fast.items()}
        unsure = [key for key, answer in fast.items() if answer["confidence"] < thresholds.get(key, self.threshold)]
        escalated = unsure if escalate and self.system2 and unsure else []
        audited = [key for key in fast if key not in unsure and escalate and self.system2 and random.random() < self.audit]
        if escalated or audited:
            asked = escalated + audited
            slow = self.system2.answer(state, {key: questions[key] for key in asked}, images)
            for key in escalated:
                answers[key] = {"answer": slow[key]["answer"], "reason": slow[key]["reason"], "source": "system2", "hunch": fast[key]}
            for key in audited:
                answers[key]["audit"] = slow[key]
        result = {"answers": answers, "unsure": unsure, "escalated": escalated}
        if self.log:
            result["id"] = self.log.log(state, questions, result | {"audit_rate": self.audit}, len(images))
        return result
