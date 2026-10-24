# hunch

Fast, calibrated decisions from a System 1 model, with escalation to a System 2 model when it isn't sure.

System 1 is [pplx-decider](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b), served on Modal by
[reflex](https://github.com/maxtmiller/reflex): one forward pass returns a probability for every option of every
question. System 2 is Claude. The router asks System 1 first and sends only the questions below their confidence
threshold to Claude, so most decisions cost one cheap forward pass and the hard ones get reasoning.

```
state + questions ─▶ System 1 (pplx-decider) ─▶ confident? ── yes ──▶ answer (source: system1)
                                                   │
                                                   no ──▶ System 2 (Claude) ──▶ answer + reason (source: system2)
```

## Questions

The same format pplx-decider takes: an object of named questions, each `choice` (named options), `score` (an
ordered list of levels) or `noul` (yes/no).

```json
{
  "category": {"type": "choice", "instructions": "What kind of email is this?",
               "criteria": {"work": "From a real person", "promo": "Marketing"}},
  "urgency": {"type": "score", "instructions": "How soon does this need attention?",
              "criteria": ["Never", "This week", "Today"]},
  "needs_reply": {"type": "noul", "instructions": "Does this email need a personal reply?"}
}
```

Every answer comes back in one shape: `answer` (an option key, a level index, or a boolean), `confidence`
(0 to 1; for `noul`, the distance from a coin flip), `probabilities`, and `source`.

## Library

```python
from hunch import Router, Decider, ClaudeCLI, ClaudeAPI

router = Router(Decider(), ClaudeAPI(), threshold=0.8)
result = router.decide(email_text, questions, thresholds={"urgency": 0.6})
result["answers"]["urgency"]   # {"answer": 0, "reason": "...", "source": "system2", "hunch": {...}}
result["escalated"]            # ["urgency"]
```

- `Decider(url, api_key)`: the pplx-decider endpoint (`PPLX_DECIDER_URL`, `PPLX_DECIDER_API_KEY`).
- `ClaudeAPI(model="claude-opus-5-5", effort="low")`: Claude through the Anthropic API with structured outputs,
  constrained to the question schema. Refusals are rerun on a fallback model the API picks (server-side
  fallbacks).
- `ClaudeCLI(model=None)`: Claude through your local `claude` CLI login, with no tools. Text only.
- `default_system2()`: `ClaudeAPI` when an Anthropic credential is set, else `ClaudeCLI` if installed.
- `decide(..., escalate=False)` keeps System 1's answers and only lists the unsure keys.

## HTTP API

```bash
pip install -e ".[api]"
cp .env.example .env    # fill in the keys
uvicorn hunch.api:app
curl -s localhost:8000/decide -H "Authorization: Bearer $HUNCH_API_KEY" \
  -d '{"state": "...", "questions": {...}, "thresholds": {"urgency": 0.6}}'
```

`POST /decide` takes `state`, `questions`, and optionally `images` (data URIs), `thresholds` and `escalate`.
Every request needs `Authorization: Bearer $HUNCH_API_KEY`.

On Modal: create a secret named `hunch` with `PPLX_DECIDER_URL`, `PPLX_DECIDER_API_KEY`, `ANTHROPIC_API_KEY`
and `HUNCH_API_KEY`, then `modal deploy deploy/modal_app.py`.

## Tests

```bash
pip install -e ".[dev]" && pytest
```
