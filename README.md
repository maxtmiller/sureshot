# sureshot

[![PyPI](https://img.shields.io/pypi/v/sureshot)](https://pypi.org/project/sureshot/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/sureshot/)
[![CI](https://github.com/maxtmiller/sureshot/actions/workflows/ci.yml/badge.svg)](https://github.com/maxtmiller/sureshot/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/maxtmiller/sureshot)](https://github.com/maxtmiller/sureshot/blob/main/LICENSE)

Fast, calibrated answers from a cheap model, with escalation to a stronger model only when the cheap one isn't sure.

You ask structured questions about a piece of content (an email, a ticket, an image). **System 1** is a fast model
that gives a probability for each option: a small LLM with logprobs, or a classifier trained on your own labels.
**System 2** is a stronger model that answers with a reason. The router asks System 1 first and sends only the
questions it is unsure about to System 2, so most decisions cost one cheap call.

```
content + questions ─▶ System 1 ─▶ confident? ── yes ──▶ answer (source: sureshot)
                                      │
                                      no ──▶ System 2 ──▶ answer + reason (source: system2)
```

As you label answers, sureshot learns how far to trust System 1 on each question, trains a cheaper tier, and warns
you when things change.

## Install

```bash
pip install sureshot                # any OpenAI-compatible or HTTP model, as System 1 or System 2
pip install "sureshot[claude]"      # adds Claude as System 2
pip install "sureshot[api]"         # adds the HTTP API
```

Every other extra (Postgres, the local classifier, MCP, Parquet, OpenTelemetry, Argilla) is listed in the
[guide](https://github.com/maxtmiller/sureshot/blob/main/docs/guide.md#install-extras).

## Quick start

With `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` set:

```python
from sureshot import Router, OpenAIDecider, ClaudeAPI

questions = {
    "category": {"type": "choice", "instructions": "What kind of email is this?",
                 "criteria": {"work": "From a real person", "promo": "Marketing"}},
    "needs_reply": {"type": "noul", "instructions": "Does this email need a personal reply?"},
}

router = Router(OpenAIDecider("gpt-4o-mini"), ClaudeAPI(), threshold=0.8)
result = router.decide(email_text, questions)

result["answers"]["category"]   # {"answer": "work", "confidence": 0.97, "source": "sureshot", ...}
result["escalated"]             # the questions System 2 answered, e.g. ["needs_reply"]
```

A question is a `choice` (one of named options), a `score` (a level from an ordered list) or a `noul` (yes or no).
Any model works on either side: Mistral, DeepSeek, Together, OpenRouter, Ollama, vLLM and others have built-in
classes, and anything else plugs in over HTTP ([models](https://github.com/maxtmiller/sureshot/blob/main/docs/guide.md#models)).
System 1 must return logprobs, so `.check()` a decider before using it.

## Learn thresholds from your labels

A fixed threshold of 0.8 is only a starting point: a confidence of 0.8 rarely means 80% right. Log decisions, label
some of them, and let sureshot find the threshold that meets your accuracy target:

```python
from sureshot import Store, calibrate, review_queue

store = Store("sureshot.db")
router = Router(OpenAIDecider("gpt-4o-mini"), ClaudeAPI(), log=store, audit=0.05)

for item in review_queue(store, limit=30, sample=0.02, confident=0.1):   # the answers worth labeling
    store.label(item["id"], item["key"], true_answer)

calibrate(store, target_accuracy=0.95, sample=0.02, confident=0.1)       # the router uses these thresholds
```

The [feedback loop](https://github.com/maxtmiller/sureshot/blob/main/docs/guide.md#the-feedback-loop) covers the rest:
training a local classifier as a cheaper first tier, rechecking thresholds, and spotting drift.

## HTTP API and configuration

Run sureshot as a service, configured from environment variables:

```bash
docker run -p 8000:8000 -v sureshot-data:/data \
  -e SURESHOT_API_KEY=... -e SURESHOT_SYSTEM1_MODEL=gpt-4o-mini -e SURESHOT_SYSTEM1_PROVIDER=openai \
  -e OPENAI_API_KEY=... -e ANTHROPIC_API_KEY=... \
  ghcr.io/maxtmiller/sureshot

curl -s localhost:8000/decide -H "Authorization: Bearer $SURESHOT_API_KEY" -H "Content-Type: application/json" \
  -d '{"state": "...", "questions": {...}}'
```

Or `pip install "sureshot[api]"` and `sureshot serve`. The routes and settings are in the
[guide](https://github.com/maxtmiller/sureshot/blob/main/docs/guide.md#http-api-and-configuration).

## Where next

- [Guide](https://github.com/maxtmiller/sureshot/blob/main/docs/guide.md): the full reference, including the
  [drop-in for OpenAI, Anthropic and Gemini apps](https://github.com/maxtmiller/sureshot/blob/main/docs/guide.md#drop-in-for-openai-anthropic-and-gemini-apps),
  the [MCP server](https://github.com/maxtmiller/sureshot/blob/main/docs/guide.md#mcp-for-agents) and the
  [command line](https://github.com/maxtmiller/sureshot/blob/main/docs/guide.md#command-line)
- [Recipes](https://github.com/maxtmiller/sureshot/blob/main/examples/recipes.md): cron, CI, queues, LangChain,
  LlamaIndex, n8n, Zapier, Make, Slack and Label Studio
- [The banking77 case study](https://github.com/maxtmiller/sureshot/blob/main/examples/case_study_banking77.md): the
  whole loop on real data
- [ARCHITECTURE.md](https://github.com/maxtmiller/sureshot/blob/main/ARCHITECTURE.md): how each part works and why

## Development

```bash
pip install -e ".[dev]" && pytest
```
