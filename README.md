# sureshot

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

Around that handoff, sureshot logs every decision and uses your labels to learn how far to trust System 1, to train
a cheaper tier, and to warn you when things change. [ARCHITECTURE.md](https://github.com/maxtmiller/sureshot/blob/main/ARCHITECTURE.md) explains how each part works
and why. [The banking77 case study](https://github.com/maxtmiller/sureshot/blob/main/examples/case_study_banking77.md) runs the whole loop on real data.
[Recipes](https://github.com/maxtmiller/sureshot/blob/main/examples/recipes.md) plug it into cron, CI, LangChain,
LlamaIndex, n8n, Zapier and Make, with email-triage and ticket-routing examples.

- [Install](#install)
- [Quick start](#quick-start)
- [The pieces](#the-pieces)
- [Questions](#questions)
- [Models](#models)
- [The router](#the-router)
- [The feedback loop](#the-feedback-loop): log, label, calibrate, retrain, recheck, prune, monitor
- [HTTP API and configuration](#http-api-and-configuration)
- [Drop-in for OpenAI, Anthropic and Gemini apps](#drop-in-for-openai-anthropic-and-gemini-apps)
- [Command line](#command-line)
- [Performance](#performance)
- [Tests](#tests)

## Install

```bash
pip install sureshot                # core: any OpenAI-compatible or HTTP model, as System 1 or System 2
pip install "sureshot[claude]"      # ClaudeAPI, Claude as System 2 through the Anthropic SDK
pip install "sureshot[local]"       # LocalDecider, the in-process classifier
pip install "sureshot[api]"         # the HTTP API
pip install "sureshot[yaml]"        # YAML questions files (included in [api])
```

The extras combine, e.g. `pip install "sureshot[claude,api]"`. Only `httpx` is required: the Quick start uses Claude
as System 2, but `ChatReasoner` or a [vendor class](#vendors) works without the `[claude]` extra.

## Quick start

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

A fixed threshold of 0.8 is only a starting point: a confidence of 0.8 rarely means 80% right. To learn real
thresholds from your data, add a log and [calibrate](#calibrate-learn-each-questions-threshold).

## The pieces

| Piece | What it does |
| --- | --- |
| [Questions](#questions) | What you ask: `choice`, `score` or `noul` (yes/no) |
| [System 1 deciders](#system-1) | Fast models with a confidence: `ChatDecider`, vendor classes, `VLLMDecider`, `HTTPDecider`, `LocalDecider` |
| [System 2 reasoners](#system-2) | Stronger models that answer with a reason: `ClaudeAPI`, `ChatReasoner`, vendor classes, `HTTPReasoner` |
| [`Router`](#the-router) | Tries the tiers in order, escalates what nobody is sure about, retries, and falls back when a model is down |
| [`Store`](#log-and-label) | SQLite log of decisions, your labels, and the thresholds sureshot learns |
| [`Cache`](#cache) | Reuses a model's answer to the same question about the same text |
| [`review_queue`](#review-queue-what-to-label-next) | The questions where a person's label helps most |
| [`calibrate`](#calibrate-learn-each-questions-threshold) | Learns per-question thresholds that meet an accuracy target |
| [`retrain`](#retrain-a-cheaper-first-tier) | Trains a `LocalDecider` from the log and promotes it only if it helps |
| [`recheck`](#recheck-and-prune) | Tests a trusted tier against your labels on a schedule |
| [`prune`](#recheck-and-prune) | Stops asking a tier the questions it doesn't pay for |
| [`stats`](#stats-and-drift) | Usage, accuracy, cost, latency and drift report |
| [`router.test`, `evaluate`](#testing-and-comparing-models) | Score a golden dataset; compare candidate models on your log |
| [HTTP API](#http-api-and-configuration) | `/decide`, `/feedback`, `/calibrate`, `/stats`, `/review`, configured from env vars |
| [Drop-in formats](#drop-in-for-openai-anthropic-and-gemini-apps) | Answers an app's OpenAI, Anthropic or Gemini structured-output requests with no code change but its base URL |
| [`sureshot` command](#command-line) | The API, a single decision and every feedback-loop step from a shell, cron or CI |

## Questions

An object of named questions, each one of three types:

```json
{
  "category": {"type": "choice", "instructions": "What kind of email is this?",
               "criteria": {"work": "From a real person", "promo": "Marketing"}},
  "urgency": {"type": "score", "instructions": "How soon does this need attention?",
              "criteria": ["Never", "This week", "Today"]},
  "needs_reply": {"type": "noul", "instructions": "Does this email need a personal reply?"}
}
```

- `choice`: one of named options. `score`: a level from an ordered list. `noul`: yes or no.
- Questions can live in a JSON or YAML file that a team versions and reviews, holding one set or several named ones:

  ```yaml
  # questions.yaml
  email_triage:
    needs_reply: {type: noul, instructions: Does this email need a personal reply?}
  ticket_routing:
    team:
      type: choice
      instructions: Which team should handle this ticket?
      criteria: {billing: Payments and invoices, support: Everything else}
  ```

  `Router(..., questions="questions.yaml")` (or `SURESHOT_QUESTIONS`) loads and validates it, and `decide` then takes a
  set's name, `router.decide(text, "email_triage")`, or nothing when the file has one set. Inline questions still work.
- Every answer has the same shape: `answer` (an option key, a level index, or a boolean), `confidence` (0 to 1;
  for `noul`, the distance from a coin flip), `probabilities` and `source` (`sureshot` or `system2`).

## Models

### System 1

A System 1 must return a probability for each option. Each question is asked on its own with `max_tokens=1`, and
the probability on the option labels is the confidence.

| Class | Use it for |
| --- | --- |
| `ChatDecider(model, base_url, api_key)` | Any OpenAI-compatible endpoint that returns logprobs. `provider="groq"` (one of `sureshot.PROVIDERS`) fills in the URL and key variable |
| `OpenAIDecider`, `MistralDecider`, ... | The same with a vendor's endpoint built in (table below) |
| `VLLMDecider(model)` | A model you serve with vLLM, at `http://localhost:8000/v1` by default (see [self-hosting](#self-hosting-system-1-with-vllm)) |
| `HTTPDecider(url, request, parse)` | Any other HTTP API: you write the request body and parse the probabilities |
| `LocalDecider` | An in-process classifier trained on your labels, about 10 ms (see [retrain](#retrain-a-cheaper-first-tier)) |

Run `.check()` on a decider before using it: it asks one yes/no question and says what is wrong (unreachable,
unknown model, no logprobs, or a model that starts thinking). A model that thinks before answering needs thinking
turned off through `extra=`, e.g. `extra={"reasoning_effort": "none"}`.

### System 2

| Class | Use it for |
| --- | --- |
| `ClaudeAPI(model="claude-haiku-5-5", effort="low")` | Claude through the Anthropic API, constrained to the question schema |
| `ClaudeCLI()` | Claude through your local `claude` CLI login. Text only |
| `ChatReasoner(model, base_url, api_key)` | Any OpenAI-compatible endpoint with structured outputs (`json_mode="object"` for JSON mode only) |
| `OpenAIReasoner`, `GroqReasoner`, ... | The same with a vendor's endpoint built in |
| `HTTPReasoner(url, request, parse)` | Any other HTTP API. Build prompts with `prompt`, `SYSTEM_PROMPT` and `answer_schema` |

`default_system2()` picks the first one configured: a `ChatReasoner` if `SURESHOT_SYSTEM2_URL` is set, else `ClaudeAPI`
if keyed, else `ClaudeCLI` if installed.

### Vendors

| Vendor | System 1 | System 2 |
| --- | --- | --- |
| OpenAI | `OpenAIDecider` | `OpenAIReasoner` |
| Gemini | `GeminiDecider` (no current model returns logprobs) | `GeminiReasoner` |
| Mistral | `MistralDecider` | `MistralReasoner` |
| DeepSeek | `DeepSeekDecider` | `DeepSeekReasoner` (JSON mode) |
| xAI | `XAIDecider` | `XAIReasoner` |
| Groq | none (no logprobs) | `GroqReasoner` |
| Together AI | `TogetherDecider` | `TogetherReasoner` |
| Fireworks AI | `FireworksDecider` | `FireworksReasoner` |
| OpenRouter | `OpenRouterDecider` | `OpenRouterReasoner` |
| Ollama (local) | `OllamaDecider` | `OllamaReasoner` |
| Claude | none (no logprobs) | `ClaudeAPI`, `ClaudeCLI` |

Each vendor class is the generic `Chat*` class with that vendor's URL and key variable; `api_key=` and `base_url=`
override them. What each model supports is up to the vendor, so `.check()` first. Which of these were checked live
is in [ARCHITECTURE.md](https://github.com/maxtmiller/sureshot/blob/main/ARCHITECTURE.md#vendors).

### Your own models

Any object with `decide(state, questions, images)` returning `{key: {answer, confidence, probabilities}}` works as
System 1, and any object with `answer(state, questions, images)` returning `{key: {answer, reason}}` as System 2.
For an HTTP API, `HTTPDecider` and `HTTPReasoner` take two functions:

```python
system1 = HTTPDecider(
    "https://my-model.example.com/classify",
    request=lambda state, questions, images: {"text": state, "questions": questions},
    parse=lambda data, questions: {k: {"probabilities": v} for k, v in data["scores"].items()},
    api_key="...",
    identity="my-classifier-v1",   # enables caching; change it when the model changes
)
```

Raise `sureshot.Outage` when your model is down or overloaded, so the router retries and falls back. Any other
exception is treated as a bug and raised.

### Self-hosting System 1 with vLLM

A hosted API costs at least ~500 ms per call. A model served next to sureshot answered a three-question email in about
150 ms on an NVIDIA L4.

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-prefix-caching --max-logprobs 20 --max-model-len 4096
```

```python
router = Router(VLLMDecider("Qwen/Qwen2.5-7B-Instruct").check(), ClaudeAPI())
```

`serve/docker-compose.yml` runs vLLM and the sureshot API together on one GPU machine: `cp .env.example .env`, set
`SURESHOT_SYSTEM1_MODEL`, `SURESHOT_API_KEY` and a System 2 key, then `docker compose -f serve/docker-compose.yml up`
(add `--build` to build sureshot from your checkout instead of pulling the published image). A zero-shot 7B model is less accurate than `gpt-4o-mini`, so log decisions, calibrate, and keep a System 2
behind it.

## The router

```python
router = Router(
    [LocalDecider.load("local.joblib"), OpenAIDecider("gpt-4o-mini")],   # tiers, tried in order
    ClaudeAPI(),                                                         # System 2
    threshold=[0.95, 0.8],      # default threshold per tier, until calibration saves better ones
    log=Store("sureshot.db"),      # log every decision, and apply the thresholds it has learned
    audit=0.05,                 # also send 5% of confident answers to System 2, to grade them
    cache=Cache("cache.db"),
)
result = router.decide(text, questions, thresholds={"urgency": 0.6})
```

A question stops at the first tier that is confident about it; what no tier is sure about goes to System 2. The
options:

| Option | What it does |
| --- | --- |
| `system1` | One decider or a list of tiers. A tier may answer only some questions (a `LocalDecider` answers the ones it was trained on); the rest pass down |
| `threshold` | Default confidence needed to keep an answer, one number or one per tier. Thresholds passed to `decide` beat calibrated ones, which beat this |
| `log` | A `Store`: logs decisions and supplies calibrated thresholds |
| `audit` | Share of confident answers also sent to System 2, only to grade them. This is what lets calibration see every confidence level |
| `trust_uncalibrated=False` | Trust a tier on a question only once it has a calibrated threshold. Until then everything escalates, which gives calibration an unbiased start |
| `guard`, `guard_threshold` | A second System 1 that can veto confident answers it disagrees with (see [catching confident errors](#catching-confident-errors)) |
| `probe` | Share of questions still asked of a tier that is not trusted on them or was pruned (default 0.05), so data keeps coming |
| `cache` | A `Cache` of model answers |
| `retries`, `backoff` | Retries on network errors and HTTP 408, 429 and 5xx (default 2, waiting 0.5 s then 1 s) |

`decide(..., escalate=False)` skips System 2 and lists the unsure questions in `unsure`.

The result:

```python
result["answers"]["urgency"]
# {"answer": 2, "reason": "...", "source": "system2",
#  "sureshot": {"answer": 1, "confidence": 0.62, "tier": 1},              # the last tier's answer, when escalated
#  "tried": [{"tier": 0, "answer": 1, "confidence": 0.41}, {"tier": 1, "answer": 1, "confidence": 0.62}]}
result["escalated"]    # questions System 2 answered
result["unsure"]       # questions no tier was confident about
result["degraded"]     # None, or which models were down
result["latency_ms"]   # {"system1", "tiers", "system2", "total"}
result["id"]           # the logged decision, to attach a label to
```

### When a model is down

After retries, the router falls back and says so in `degraded`:

- **A tier is down:** it is skipped. If every tier is down, System 2 answers everything.
- **System 2 is down:** the last tier's answers stand, and the unsure questions stay in `unsure`.
- **The guard is down:** nothing is vetoed.
- **Everything is down, or there is nothing to fall back to:** the error is raised.

Only outages fall back. A bad key or an unknown model raises, so a misconfiguration isn't hidden behind answers
from the other model.

### Cache

`Router(cache=Cache("cache.db", ttl=7 * 24 * 3600))` reuses a model's answer to the same question about the same
text. It caches what the models said, not the decision, so changing thresholds or recalibrating needs no clearing.
It helps when items are seen more than once, and above all when rescoring a golden set or trying thresholds, which
then costs nothing. On a stream of unique items it saves nothing; check `stats(store)["cache"]`.

A vendor can change a model behind the same name, which the cache can't see. Pin a dated model name, set a `ttl`, or
`clear()` the cache when you change models. Details of the key are in [ARCHITECTURE.md](https://github.com/maxtmiller/sureshot/blob/main/ARCHITECTURE.md#the-cache-cachepy).

### Catching confident errors

A model's confidence can't reveal the mistakes it is sure about. `Router(..., guard=g, guard_threshold=0.9)` shows
each confident answer to a second System 1, and when it disagrees with probability at least 0.9, the answer escalates.
The guard can be a `LocalDecider` trained on your labels (runs in-process) or a second LLM. On public datasets a
second LLM caught about twice as many confident errors as the local guard, for a call per confident answer; the
numbers are in [ARCHITECTURE.md](https://github.com/maxtmiller/sureshot/blob/main/ARCHITECTURE.md#the-router-routerpy).

## The feedback loop

With a log, sureshot improves as you label. A typical setup runs the router all day and a short job each night:

```python
from sureshot import Router, Store, calibrate, prune, recheck, retrain, review_queue, stats

store = Store("sureshot.db")
router = Router([local, llm], system2, log=store, audit=0.05, trust_uncalibrated=False)

# During the day
result = router.decide(text, questions)

# Each night
for item in review_queue(store, limit=30, sample=0.02, confident=0.1):     # a person labels these
    store.label(item["id"], item["key"], answer_from_a_person)
local = retrain(store, current=local, sample=0.02, confident=0.1)["decider"]  # a better classifier, if it helps
calibrate(store, target_accuracy=0.95, tier=1, sample=0.02, confident=0.1)  # thresholds for the LLM tier
recheck(store, tier=1, sample=0.02, confident=0.1)                        # confirmed? still meeting the target?
prune(store, {"tier 1": 0.00008, "system2": 0.0034}, tier=1)              # stop asking it where it doesn't pay
report = stats(store, sample=0.02, costs={"tier 1": 0.00008, "system2": 0.0034})
```

The rest of this section explains each step.

### Log and label

`Router(log=Store("sureshot.db"))` logs every decision and returns its `id`. Attach the true answer with
`store.label(id, key, answer)` (or `POST /feedback`), from a person or a downstream outcome.

Where nobody has labeled, sureshot grades a tier by System 2's answer. That measures agreement with System 2, which is
not the truth when System 2 is often wrong, so label where you can and measure with a random sample (below).

### Review queue: what to label next

`review_queue(store)` (or `GET /review`) lists the ungraded questions where a label helps most, in this order:

| Reason | Why it's queued |
| --- | --- |
| `sample` | In a fixed random share of all decisions (`sample=0.02`). The only labels that say how often the router is right overall |
| `confident` | In a fixed random share of the answers a tier was confident about, kept or not (`confident=0.1`). What confirms and rechecks a tier's threshold |
| `disagreement` | A tier was unsure, System 2 answered, and they differ |
| `audit_disagreement` | A confident answer was audited and System 2 disagreed: how confident mistakes get found |
| `unresolved` | No tier was confident and System 2 wasn't asked |

```python
for item in review_queue(store, limit=20, sample=0.02):
    item["reason"], item["state"], item["question"], item["answer"]
    item["candidates"]   # what each tier and System 2 said, with System 2's reason
    store.label(item["id"], item["key"], true_answer)   # the item leaves the queue
```

### Calibrate: learn each question's threshold

```python
calibrate(store, target_accuracy=0.95, tier=0)
# {"urgency": {"threshold": 0.71, "accuracy": 0.9, "trusted_accuracy": 0.96, "escalation_rate": 0.22, "samples": 410}}
```

For each question, `calibrate` finds the lowest confidence at which the tier's accuracy, judged conservatively,
meets the target, and saves it so the router applies it. `escalation_rate` is the share that would go to System 2:
the cost of the target. Call it once per tier. It needs graded answers at every confidence level, which is what
`audit` provides.

Where nobody labeled, an answer is graded by System 2's, so on its own a threshold only shows agreement with
System 2. When two models make the same mistakes, that can look perfect: on the trec case study `gpt-4o-mini` was
certified at 95% this way and was 81% to 87% right. So pass `sample=` and `confident=` (the rates you give
`review_queue`), and your labels on those random samples must **confirm** the threshold: until they show the target,
it is saved but unconfirmed (`reason` is `not enough labels`), and the router keeps asking the tier without keeping
its answers. `recheck` confirms it once the labels arrive.

When it finds no threshold, `reason` says why, and `best_target` is the highest target this data would support:

| `reason` | What to do |
| --- | --- |
| `not enough data` | Fewer than `min_samples` (30) graded answers. Grade more; the threshold is unchanged |
| `not enough evidence` | The target isn't ruled out but can't be shown yet. Grade more |
| `target unreachable` | The tier can't reach the target. Lower it, or change or drop the tier |
| `below target on labels` | A threshold was found, but your labels on the random samples rule the target out |
| `not enough labels` | A threshold was found but your labels don't confirm it yet. It is saved and not trusted; label more |

In `not enough evidence`, `target unreachable` and `below target on labels` the tier **fails closed**: it is no longer trusted on that question, which passes to the
next tier or System 2, and the tier is only asked on a `probe` share. Calibrate again later; with more data it may
pass. `fail_closed=False` keeps the old threshold instead. `stats(store)["uncalibrated"]` lists these questions.

How the statistics work, and how often they are wrong in simulation: [ARCHITECTURE.md](https://github.com/maxtmiller/sureshot/blob/main/ARCHITECTURE.md#calibration-calibratepy).

### Retrain: a cheaper first tier

`LocalDecider` (`pip install "sureshot[local]"`) embeds the text with a small local model and trains one logistic
regression per question on your graded decisions. It answers in about 10 ms with no server, so it makes a good
tier 0 in front of an LLM. It is strong on topical questions and weak on subtle ones such as tone; calibration
decides per question.

```python
result = retrain(store, current=local, tier=0)   # train on the log, promote only what helps
local = result["decider"]
local.save("local.joblib")                        # LocalDecider.load("local.joblib") in production
```

`retrain` promotes the new classifier question by question, only if on held-out decisions it meets the target and
passes on fewer questions than `current`. Where it keeps `current`, it checks it again, so a nightly `retrain` also
recalibrates the classifier. Use it rather than `calibrate(tier=0)` for the classifier. It needs roughly 100 to 300
labels per question.

The classifier learns mostly from System 2's answers and is graded mostly by them, so it can share System 2's
mistakes and still look right. Pass `sample=` and `confident=`, as for `calibrate`, and your labels on those random
samples must confirm it: retrain runs the classifier on every held-out decision a person labeled at random, and
until those labels show the target its threshold is saved but not trusted.

Confirming counts the labels on the answers a tier flagged as confident, and 95% takes at least 52 of them, all
right (25 for 90%; with a wrong one, the next test at twice the count). Size `confident=` so that many arrive in a reasonable time: in the case studies, 30
labels a night with `confident=0.3` gave the classifier 2 to 48 of them in ten days.

### Recheck and prune

- **`recheck(store, tier=1, sample=0.02, confident=0.1)`** tests each saved threshold against your labels on the
  random samples: it confirms an unconfirmed threshold once they show the target, and fails a tier closed when they
  rule the target out. Run it on a schedule: a tier that is certified is
  rarely calibrated again, and a threshold certified against System 2 can overstate the tier.
- **`prune(store, costs, tier=1)`** stops asking a tier the questions where the answers it keeps save less than it
  costs to ask. It is then only probed, and asked again once it pays. To count waiting as a cost, add its price to
  each stage's cost (say $0.01 a second).

### Stats and drift

`stats(store)` (or `GET /stats`) summarizes the log:

```python
report = stats(store, sample=0.02, costs={"tier 0": 0, "tier 1": 0.0002, "system2": 0.004})
report["questions"]["topic"]["answered_by"]         # {"tier 0": 0.74, "tier 1": 0.26}
report["questions"]["topic"]["tier_accuracy"]       # accuracy of the answers each tier kept
report["questions"]["topic"]["accuracy_on_sample"]  # the router's accuracy on the labeled random sample
report["cost"]                                      # total, savings vs. System 2 alone, and per tier "net"
report["latency_ms"], report["cache"], report["review_queue"], report["uncalibrated"]
report["drift"]                                     # what changed since each calibration
```

- `accuracy_on_sample` has `below_target` when even the top of its interval is under the target. No threshold
  fixes that: System 2, or the question itself, is the limit.
- In `cost`, a tier with a negative `net` costs more than it saves. sureshot has no price list; pass yours. Add `"ms"`
  to price waiting.
- `drift` lists, per calibrated question and tier, `flags` for what moved: `escalation_rate`, `distribution` (the
  spread of its confidences), `accuracy`, and `change_*` for a change that started later. `unchecked` names the
  checks that lacked data, so an empty `flags` isn't read as no drift. `upstream` points at an earlier tier that
  drifted first. A flag means grade a fresh sample and calibrate again.
- `examples/drift_alert.py sureshot.db` prints what moved and exits with status 1 when anything is flagged; run it on
  a schedule.

How the drift checks keep false alarms rare however often you poll: [ARCHITECTURE.md](https://github.com/maxtmiller/sureshot/blob/main/ARCHITECTURE.md#measuring-statspy-evaluatepy-goldenpy).

### Testing and comparing models

`router.test(cases)` scores the router on answers you already know, without writing to the log. A case is
`{"state": ..., "expected": {key: answer}}`; `cases` is a list or a JSONL file.

```python
report = router.test("golden.jsonl", questions=questions)
report["overall"]    # {"n": 120, "accuracy": 0.93, "system1_accuracy": 0.81, "escalation_rate": 0.24}
report["failures"]   # [{"case": 7, "key": "urgency", "expected": 2, "got": 1, ...}, ...]
```

`evaluate(store, {"qwen": ChatDecider(...), "llama": ChatDecider(...)})` replays your graded decisions through
candidate System 1 models, to see which meets the target with the least escalation.

## HTTP API and configuration

```bash
pip install "sureshot[api]"
cp .env.example .env    # fill in the keys
sureshot serve          # or: uvicorn sureshot.api:app
curl -s localhost:8000/decide -H "Authorization: Bearer $SURESHOT_API_KEY" -H "Content-Type: application/json" \
  -d '{"state": "...", "questions": {...}, "thresholds": {"urgency": 0.6}}'
```

Or run the published image, with no clone or install (amd64 and arm64, tagged `latest` and each release's version):

```bash
docker run -p 8000:8000 -v sureshot-data:/data \
  -e SURESHOT_API_KEY=... -e SURESHOT_SYSTEM1_MODEL=gpt-4o-mini -e SURESHOT_SYSTEM1_PROVIDER=openai -e OPENAI_API_KEY=... \
  -e ANTHROPIC_API_KEY=... \
  ghcr.io/maxtmiller/sureshot
```

The image has the API and Claude as System 2, and logs to `/data/sureshot.db` (`SURESHOT_DB`), so mount a volume there
to keep the log. Run the [command line](#command-line) inside it with `docker exec <container> sureshot stats`.

| Route | Does |
| --- | --- |
| `POST /decide` | `state`, `questions` (inline, or a set's name from `SURESHOT_QUESTIONS`, or left out when it has one), optionally `images` (data URIs), `thresholds`, `escalate`. Returns an `id` when logging |
| `POST /feedback` | `{"id": ..., "labels": {"urgency": 2}}` |
| `POST /calibrate` | `{"target_accuracy": 0.95}`, optionally `min_samples`, `sample` and `confident` |
| `GET /stats` | The `stats` report; its options are query parameters, `costs` as JSON |
| `GET /review` | The review queue; `limit`, `reason`, `sample`, `confident` |
| `POST /v1/chat/completions`, `/v1/messages`, `/{version}/models/{model}:generateContent` | OpenAI, Anthropic and Gemini requests (see [below](#drop-in-for-openai-anthropic-and-gemini-apps)); the key may also come as `x-api-key`, `x-goog-api-key` or `?key=` |
| `GET /health` | No token needed |

Every other route needs `Authorization: Bearer $SURESHOT_API_KEY`. The API builds its router with `router_from_env()`,
which you can also call yourself:

| Variable | Values |
| --- | --- |
| `SURESHOT_SYSTEM1` | `chat` (default), `vllm` or `gemini` |
| `SURESHOT_SYSTEM1_MODEL`, `_URL`, `_PROVIDER`, `_API_KEY` | System 1's model (required), and a URL or a provider, and key |
| `SURESHOT_SYSTEM2` | `auto` (default), `claude`, `claude-cli`, `chat`, `gemini`, `none` |
| `SURESHOT_SYSTEM2_MODEL`, `_URL`, `_PROVIDER`, `_API_KEY` | System 2's model, URL or provider, and key |
| `SURESHOT_SYSTEM2_JSON_MODE` | `schema` (default) or `object` |
| `SURESHOT_SYSTEM1_EXTRA`, `SURESHOT_SYSTEM2_EXTRA` | JSON merged into every request, e.g. `{"reasoning_effort": "none"}` |
| `SURESHOT_LOCAL_MODEL` | A saved `LocalDecider` to put in front of System 1 |
| `SURESHOT_QUESTIONS` | A JSON or YAML [questions file](#questions), checked when the API starts |
| `SURESHOT_THRESHOLD` | Default threshold (0.8) |
| `SURESHOT_DB` | SQLite log; enables `/feedback` and `/calibrate` |
| `SURESHOT_CACHE`, `SURESHOT_CACHE_TTL` | Response cache, and seconds an answer stays valid |
| `SURESHOT_AUDIT` | Share of confident answers audited (default 0) |
| `SURESHOT_OPENAI_PASSTHROUGH_URL`, `SURESHOT_ANTHROPIC_PASSTHROUGH_URL`, `SURESHOT_GEMINI_PASSTHROUGH_URL`, each with `_API_KEY` | Where each format's requests that sureshot can't answer are forwarded |

## Drop-in for OpenAI, Anthropic and Gemini apps

An app that already asks OpenAI, Anthropic or Gemini for structured output can use sureshot by changing only its base
URL and key (`SURESHOT_API_KEY`):

```python
class Ticket(BaseModel):
    team: Literal["billing", "support"] = Field(description="Which team should handle this ticket?")
    urgent: bool
    priority: int = Field(ge=1, le=5)

OpenAI(base_url="http://localhost:8000/v1", api_key=key).beta.chat.completions.parse(model=..., messages=..., response_format=Ticket)
Anthropic(base_url="http://localhost:8000", api_key=key).messages.parse(model=..., max_tokens=..., messages=..., output_format=Ticket)
genai.Client(api_key=key, http_options={"base_url": "http://localhost:8000"}).models.generate_content(
    model=..., contents=..., config={"response_mime_type": "application/json", "response_schema": Ticket})
```

| Format | Route | Where the schema comes from |
| --- | --- | --- |
| OpenAI | `POST /v1/chat/completions` | `response_format.json_schema`, or one forced function (`tool_choice` naming it, or `"required"` with one tool) |
| Anthropic | `POST /v1/messages` | `output_config.format`, or one forced tool (`tool_choice` naming it, or `"any"` with one tool) |
| Gemini | `POST /{version}/models/{model}:generateContent` | `generationConfig.responseJsonSchema` or `responseSchema` |

sureshot turns each field of the schema into a question, routes them, and answers in the app's format with an object
matching the schema:

| Schema field | Question |
| --- | --- |
| string `enum` (a `Literal` or an `Enum`) | `choice` |
| `boolean` | `noul` |
| integer `enum`, or `minimum` to `maximum`, of 2 to 10 values | `score` |

- Each field's `description` (or `title`) is its instructions. The state is the text of every message, system prompt included.
- Question keys are the schema's name (or title, or the forced tool's name) and the field, e.g. `Ticket.team`, so apps sharing one sureshot are
  calibrated apart. The `x-sureshot` response header has the decision `id` and who answered each question; label with
  `POST /feedback {"id": ..., "labels": {"Ticket.team": "support"}}`. For a score, a label is the level's position (0 for `priority` 1).
- A request sureshot can't answer (no schema, a free-text, nested or optional field, streaming, OpenAI's `n` above 1,
  an image by URL, Gemini function calling) is forwarded unchanged to `SURESHOT_OPENAI_PASSTHROUGH_URL`,
  `SURESHOT_ANTHROPIC_PASSTHROUGH_URL` or `SURESHOT_GEMINI_PASSTHROUGH_URL` when it is set (the base URL you would give
  that SDK, with the matching `_API_KEY`), so one base URL serves the whole app. Otherwise it gets a 422. Streaming
  requests are never forwarded.
- Token usage is reported as zero: the models sureshot asked are not the one the app named.

## Command line

`pip install sureshot` adds a `sureshot` command. It reads the same environment variables as the API (and a `.env` in
the current directory when `python-dotenv` is installed), takes `--db` in place of `SURESHOT_DB`, and prints JSON.

| Command | Does |
| --- | --- |
| `sureshot serve [--host] [--port]` | Runs the HTTP API (needs `sureshot[api]`) |
| `sureshot decide [request.json] [--questions file]` | Decides one request, the body of `POST /decide`, from a file or stdin |
| `sureshot calibrate [--target] [--tier]` | [Calibrates](#calibrate-learn-each-questions-threshold) and saves thresholds |
| `sureshot retrain [--model local.joblib]` | [Retrains](#retrain-a-cheaper-first-tier) the `LocalDecider` (default `SURESHOT_LOCAL_MODEL`), saving it only when a question is promoted. A running API loads it on restart |
| `sureshot recheck [--tier]` | [Rechecks](#recheck-and-prune) saved thresholds against your labels |
| `sureshot prune --costs '{...}' [--tier]` | [Prunes](#recheck-and-prune) the questions a tier doesn't pay for |
| `sureshot stats [--costs '{...}']` | The [stats](#stats-and-drift) report |
| `sureshot review [--limit] [--reason] [--label]` | The [review queue](#review-queue-what-to-label-next). With `--label`, shows each item and asks for its true answer (Enter accepts the router's, `-` skips, Ctrl-D stops), saving each label as it is given |
| `sureshot test golden.jsonl [--questions file] [--set name] [--min-accuracy]` | [Scores](#testing-and-comparing-models) the router without logging; exits 1 below `--min-accuracy` |

`calibrate`, `retrain`, `recheck`, `stats` and `review` take `--sample` and `--confident`, the rates you give the
review queue. They run where the log is: on the machine or in the container that holds `SURESHOT_DB`. `test` needs
only the models, so it can gate a CI job:

```bash
# nightly, next to the log
0 3 * * *  sureshot retrain --sample 0.02 --confident 0.1 && sureshot calibrate --tier 1 --sample 0.02 --confident 0.1 && sureshot recheck --tier 1 --confident 0.1

# in CI, before a change to questions or models ships
sureshot test golden.jsonl --questions questions.yaml --set email_triage --min-accuracy 0.95
```

## Performance

- One process serves roughly 90 `/decide` requests a second with two model calls each; `uvicorn sureshot.api:app
  --workers 4` reached about 300, sharing the SQLite log and cache.
- Identical concurrent requests ask the model once.
- `/stats` and `/review` read the whole log; poll them rarely and use `since=`.
- SQLite is for one machine: several workers on one disk are fine, separate machines are not supported.

`examples/loadtest.py <url> questions.json --clients 32 --requests 1000` measures your own setup. What load testing
found is in [ARCHITECTURE.md](https://github.com/maxtmiller/sureshot/blob/main/ARCHITECTURE.md#concurrency).

## Tests

```bash
pip install -e ".[dev]" && pytest
```
