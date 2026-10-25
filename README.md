# hunch

Fast, calibrated decisions from a System 1 model, with escalation to a System 2 model when it isn't sure.

System 1 is any fast model that can give a probability for each option, such as a small LLM served with vLLM or
Ollama, or a hosted API that returns logprobs. System 2 is any stronger model, Claude by default. The router asks
System 1 first and sends only the questions below their confidence threshold to System 2, so most decisions cost one
cheap call and the hard ones get reasoning.

```
state + questions ─▶ System 1 ─▶ confident? ── yes ──▶ answer (source: hunch)
                                    │
                                    no ──▶ System 2 ──▶ answer + reason (source: system2)
```

## Questions

An object of named questions, each `choice` (named options), `score` (an
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
from hunch import Router, ChatDecider, ClaudeAPI

router = Router(ChatDecider("Qwen/Qwen3-8B", "http://localhost:8000/v1"), ClaudeAPI(), threshold=0.8)
result = router.decide(email_text, questions, thresholds={"urgency": 0.6})
result["answers"]["urgency"]   # {"answer": 0, "reason": "...", "source": "system2", "hunch": {...}}
result["escalated"]            # ["urgency"]
```

- `ChatDecider(model, base_url, api_key)`: any OpenAI-compatible endpoint that returns logprobs (OpenAI, vLLM,
  Ollama, Together, Fireworks, Groq), hosted or self-served, with no adapter code. Each question is asked with
  `max_tokens=1`, and the model's probability over the option labels is the calibrated confidence. Choices are
  limited to 26 options. Point it at your own server with `base_url="http://localhost:8000/v1"`.
- `VLLMDecider(model, base_url="http://localhost:8000/v1")`: a model you serve yourself with vLLM. It is a
  `ChatDecider` with a local default URL that turns off thinking for models that support it (see
  [Self-hosting System 1](#self-hosting-system-1-with-vllm)).
- `ChatDecider(...).check()` asks one yes/no question and raises an error saying what is wrong (unreachable, unknown
  model, no logprobs, or a model that starts thinking and needs `extra`). The HTTP API runs it before its first request.
- `HTTPDecider(url, request, parse, api_key)`: any System 1 model behind a non-OpenAI HTTP API. `request(state, questions,
  images)` returns the JSON body to POST, and `parse(response, questions)` returns `{key: raw}` where raw is
  `{"noul": p}` or `{"probabilities": {option: p}}`. hunch normalizes the rest.

  ```python
  system1 = HTTPDecider(
      "https://my-model.example.com/classify",
      request=lambda state, questions, images: {"text": state, "questions": questions},
      parse=lambda data, questions: {k: {"probabilities": v} for k, v in data["scores"].items()},
      api_key="...",
  )
  ```

  For a model behind a different protocol, any object with `decide(state, questions, images)` returning
  `{key: {answer, confidence, probabilities}}` works as System 1.
- `ClaudeAPI(model="claude-haiku-5-5", effort="low")`: Claude through the Anthropic API with structured outputs,
  constrained to the question schema. Refusals are rerun on a fallback model the API picks (server-side
  fallbacks).
- `HTTPReasoner(url, request, parse, api_key)`: any System 2 model behind an HTTP API. `request(state, questions,
  images)` returns the JSON body, and `parse(response, questions)` returns `{key: {"answer", "reason"}}`. Build
  prompts with `prompt(state, questions)`, `SYSTEM_PROMPT` and `answer_schema(questions)`.

  ```python
  system2 = HTTPReasoner(
      "https://llm.example.com/v1/answer",
      request=lambda state, questions, images: {"system": SYSTEM_PROMPT, "input": prompt(state, questions),
                                                "schema": answer_schema(questions)},
      parse=lambda data, questions: data["answers"],
      api_key="...",
  )
  ```
- `ChatReasoner(model, base_url, api_key)`: any OpenAI-compatible endpoint as System 2, constrained to the
  question schema with structured outputs.
- `ChatDecider` and `ChatReasoner` also take `extra={...}`, merged into every request body. A model that would
  spend its one token starting to think (such as Gemma on Ollama) needs `extra={"reasoning_effort": "none"}` to put
  probability on the options.
- `ClaudeCLI(model=None)`: Claude through your local `claude` CLI login, with no tools. Text only.
- `default_system2()`: `ClaudeAPI` when an Anthropic credential is set, else `ClaudeCLI` if installed.
- `decide(..., escalate=False)` keeps System 1's answers and only lists the unsure keys.

## Self-hosting System 1 with vLLM

A hosted API costs at least ~500 ms per call in network and queueing. For a faster System 1, serve an open model
with [vLLM](https://docs.vllm.ai) on a GPU on the same machine or network as hunch, so there is no internet hop.
Prefix caching lets the per-question requests for one state share its prefill.

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-prefix-caching --max-logprobs 20 --max-model-len 4096
```

```python
from hunch import Router, VLLMDecider, ClaudeAPI

router = Router(VLLMDecider("Qwen/Qwen2.5-7B-Instruct").check(), ClaudeAPI())
```

Or set `HUNCH_SYSTEM1=vllm`, `HUNCH_SYSTEM1_MODEL` and, for a remote box, `HUNCH_SYSTEM1_URL`. `serve/docker-compose.yml`
runs vLLM and the hunch API together on one GPU machine (Docker with the NVIDIA container toolkit):
`cp .env.example .env`, set `HUNCH_SYSTEM1_MODEL`, `HUNCH_API_KEY` and a System 2 key, then
`docker compose -f serve/docker-compose.yml up --build`. The compose file itself has not been run (it needs Docker with an NVIDIA GPU). Its pieces were: the official
`vllm/vllm-openai` image started with the compose flags on an NVIDIA L4 and served Qwen2.5-7B-Instruct, `serve/Dockerfile`
built and started the API, and `VLLMDecider`, `check()`, `router.test`, `/decide` and `/feedback` ran against that server.
When System 1 is unreachable, `/decide` answers 503 with the reason.

A 7B model fits a 24 GB GPU. On the L4 it answered a three-question email in about 150 ms (p50 and p95), against about
500 to 600 ms for hosted APIs; a faster GPU should lower that. Zero-shot it was less accurate than `gpt-4o-mini` on a
small labeled set (71% against 84%), so log decisions, run `calibrate`, and keep a System 2 behind it.
`--max-logprobs 20` matches the 20 top logprobs hunch requests.

If you install vLLM with pip instead of using the official image, it may need `nvcc` for FlashInfer; set
`VLLM_USE_FLASHINFER_SAMPLER=0` or use `vllm/vllm-openai`.

## Testing against a golden dataset

`router.test(cases)` scores the router on cases you have already answered, so you can compare models or catch a
regression before shipping. A case is `{"state": ..., "expected": {key: answer}}`, plus its own `"questions"` or a
shared `questions=` argument. `cases` is a list or the path of a JSONL file with one case per line. Only the
questions with an expected answer are asked, and nothing is written to the log.

```python
report = router.test("golden.jsonl", questions=questions)
report["overall"]     # {"n": 120, "accuracy": 0.93, "system1_accuracy": 0.81, "escalation_rate": 0.24}
report["questions"]   # the same per question
report["latency_ms"]  # {"system1": {50: 310.2, 95: 642.9}, "system2": {50: 1480.0, 95: 2210.5}, "total": {...}}
report["failures"]    # [{"case": 7, "key": "urgency", "expected": 2, "got": 1, "source": "hunch", "confidence": 0.97}, ...]
```

`accuracy` is the final answer and `system1_accuracy` is System 1 alone, so the gap is what escalation buys.
Every `decide` result also has `latency_ms` (`system1`, `system2` or `None`, `total`), and it is stored in the log.
Cases run one at a time, so the latencies are those of an unloaded system.

## Calibrating from your own data

A confidence of 0.8 does not mean 80% right, especially from raw logprobs. Log decisions, grade some, and hunch
learns the threshold for each question that meets an accuracy target.

```python
from hunch import Router, Store, calibrate, evaluate

store = Store("hunch.db")
router = Router(system1, system2, log=store, audit=0.1)
result = router.decide(email_text, questions)          # result["id"] identifies the logged decision
store.label(result["id"], "urgency", 2)                 # the true answer, from a human or a downstream outcome

calibrate(store, target_accuracy=0.95)                  # saves per-question thresholds the router now applies
# {"urgency": {"threshold": 0.71, "accuracy": 0.9, "trusted_accuracy": 0.96, "escalation_rate": 0.22, "samples": 410}}

evaluate(store, {"qwen-8b": ChatDecider("Qwen/Qwen3-8B", "http://localhost:8000/v1"),
                 "llama-8b": ChatDecider("llama3.1:8b", "http://localhost:11434/v1")})
```

- Truth is a human label when there is one, else System 2's answer. Escalated questions always have System 2's
  answer, but they are all low confidence. `audit=0.1` also sends a random 10% of confident questions to System 2
  (their answers still come from System 1), and calibration weights those 10x, so the estimate covers every
  confidence level. Without audit or human labels the calibration only sees the unsure questions.
- `threshold` is the lowest confidence at which System 1's accuracy, judged conservatively (the lower end of a 90%
  interval), meets the target. It is `None` until there are enough graded answers (`min_samples`, default 30), and
  that question keeps its current threshold. `escalation_rate` is the share of System 1's answers that would go to
  System 2 at that threshold, which is the cost of the target.
- `evaluate` replays the graded decisions through each candidate System 1 and returns the same report, so you can
  see which model meets the target with the least escalation. Decisions with images are skipped.
- Thresholds passed to `decide` beat the calibrated ones, which beat `threshold`.

## Configuring models with env vars

`router_from_env()` builds the Router from the environment, and the HTTP API uses it, so a deployment
switches models without code changes.

| Variable | Values |
| --- | --- |
| `HUNCH_SYSTEM1` | `chat` (default, a `ChatDecider`) or `vllm` (a `VLLMDecider`, URL defaulting to `http://localhost:8000/v1`) |
| `HUNCH_SYSTEM1_MODEL`, `_URL`, `_API_KEY` | System 1: model name (required), base URL (default OpenAI), key (optional for local servers) |
| `HUNCH_SYSTEM2` | `auto` (default: Claude API, else CLI), `claude`, `claude-cli`, `chat`, `none` |
| `HUNCH_SYSTEM2_MODEL`, `_URL`, `_API_KEY` | model for `claude` or `chat`; URL and key for `chat` |
| `HUNCH_SYSTEM1_EXTRA`, `HUNCH_SYSTEM2_EXTRA` | JSON object merged into every request body, e.g. `{"reasoning_effort": "none"}` |
| `HUNCH_THRESHOLD` | confidence below which a question escalates (default 0.8) |
| `HUNCH_DB` | path of the SQLite log; set it to enable logging, `/feedback` and `/calibrate` |
| `HUNCH_AUDIT` | share of confident questions also checked by System 2 for calibration (default 0) |

```bash
# a self-served model on vLLM as System 1, Claude as System 2
HUNCH_SYSTEM1_MODEL=Qwen/Qwen3-8B HUNCH_SYSTEM1_URL=http://localhost:8000/v1 uvicorn hunch.api:app
```

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

With `HUNCH_DB` set, `/decide` also returns an `id`. `POST /feedback` with `{"id": ..., "labels": {"urgency": 2}}`
records the true answers, and `POST /calibrate` with `{"target_accuracy": 0.95}` computes and saves thresholds. A
workflow tool can send feedback from wherever a person corrects the result.

## Tests

```bash
pip install -e ".[dev]" && pytest
```
