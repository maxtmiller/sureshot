"""Adapters for any model behind an HTTP API that you describe with two functions."""
from hunch.adapters import base
from hunch.adapters.base import normalize

class HTTPDecider:
    """Any System 1 model behind an HTTP API. You supply two functions:

    request(state, questions, images) -> the JSON body to POST.
    parse(response_json, questions) -> {key: raw}, where raw is {"noul": p} for a yes/no question, or
    {"probabilities": {option: p}} for a choice or score (optionally with "choice", "confidence", "score").

    HTTPDecider POSTs, then normalizes every answer to {answer, confidence, probabilities}."""

    def __init__(self, url, request, parse, api_key=None, headers=None, timeout=600, identity=None):
        """identity names this model and your request/parse functions for the response cache (see hunch.Cache); without
        one, its answers are not cached. Change it whenever the model or the functions change."""
        self.url, self.request, self.parse, self.timeout, self.identity = url, request, parse, timeout, identity
        self.headers = ({"Authorization": f"Bearer {api_key}"} if api_key else {}) | (headers or {})

    def decide(self, state, questions, images=()):
        response = base.post(
            self.url,
            json=self.request(state, questions, list(images)),
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        raw = self.parse(response.json(), questions)
        return {key: normalize(questions[key], raw[key]) for key in questions}


class HTTPReasoner:
    """Any System 2 model behind an HTTP API. You supply two functions:

    request(state, questions, images) -> the JSON body to POST. `prompt`, `SYSTEM_PROMPT` and `answer_schema`
    build the pieces most LLM APIs need.
    parse(response_json, questions) -> {key: {"answer": ..., "reason": ...}}, answer being an option key, a level
    index or a boolean."""

    def __init__(self, url, request, parse, api_key=None, headers=None, timeout=600, identity=None):
        """identity names this model and your request/parse functions for the response cache (see hunch.Cache); without
        one, its answers are not cached. Change it whenever the model or the functions change."""
        self.url, self.request, self.parse, self.timeout, self.identity = url, request, parse, timeout, identity
        self.headers = ({"Authorization": f"Bearer {api_key}"} if api_key else {}) | (headers or {})

    def answer(self, state, questions, images=()):
        response = base.post(self.url, json=self.request(state, questions, list(images)), headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        return self.parse(response.json(), questions)
