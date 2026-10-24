"""Answer with System 1, and send only the questions it is unsure about to System 2."""
from hunch.questions import validate


class Router:
    def __init__(self, system1, system2=None, threshold=0.8):
        self.system1, self.system2, self.threshold = system1, system2, threshold

    def decide(self, state, questions, images=(), thresholds=None, escalate=True):
        """Returns {"answers": {key: {answer, confidence, source, ...}}, "unsure": [...], "escalated": [...]}.

        A question is unsure when System 1's confidence is below its threshold (thresholds[key], else the router's).
        Unsure questions go to System 2 when escalate is true and a System 2 is set; their answer then comes from it,
        with System 1's answer kept under "hunch". Otherwise System 1's answer stands and the key is only listed
        in "unsure"."""
        validate(questions)
        thresholds = thresholds or {}
        fast = self.system1.decide(state, questions, images)
        answers = {key: answer | {"source": "hunch"} for key, answer in fast.items()}
        unsure = [key for key, answer in fast.items() if answer["confidence"] < thresholds.get(key, self.threshold)]
        escalated = unsure if escalate and self.system2 and unsure else []
        if escalated:
            slow = self.system2.answer(state, {key: questions[key] for key in escalated}, images)
            for key in escalated:
                answers[key] = {"answer": slow[key]["answer"], "reason": slow[key]["reason"], "source": "system2", "hunch": fast[key]}
        return {"answers": answers, "unsure": unsure, "escalated": escalated}
