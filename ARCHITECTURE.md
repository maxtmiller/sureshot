# Architecture

hunch answers structured questions about a piece of content (an email, a ticket, an image) quickly and cheaply, and
only asks a slower, smarter model when the fast one is not sure. This file explains what each part does, how the parts
fit together, and why they are built the way they are.

## The idea

Two kinds of model, borrowed from the "System 1 / System 2" picture of thinking:

- **System 1** is fast and cheap, and tells you how sure it is. Its confidence comes from the probability it puts on
  each answer (token logprobs from an LLM, or class probabilities from a small classifier).
- **System 2** is slow and expensive (a stronger model) and gives an answer with a reason. It is only asked about the questions
  System 1 was unsure about.

Everything else in the library exists to make that handoff trustworthy: to know *how much to trust* System 1's
confidence, to *measure* whether it is right, and to *shrink* the share that needs System 2 over time.

```
                     questions + content
                            |
                            v
   +-------------------------------------------------------+
   |  Router.decide                                        |
   |                                                       |
   |   tier 0 (e.g. LocalDecider, ~10 ms)                  |
   |      | confident? yes -> answer                       |
   |      | no, pass the question down                     |
   |      v                                                |
   |   tier 1 (e.g. ChatDecider / LLM with logprobs)       |
   |      | confident? yes -> answer                       |
   |      | no                                             |
   |      v                                                |
   |   guard: disagrees with a confident answer? -> unsure |
   |      v                                                |
   |   System 2 (stronger model), only for unsure questions |
   +-------------------------------------------------------+
        |                      |
        v                      v
     answers              Store (SQLite log)
                               |
              labels, audits   |   calibrate / retrain / stats / review
                               v
            thresholds and a better tier 0, fed back into the Router
```

## Repository map

| Path | Role |
|---|---|
| `hunch/questions.py` | The question format (choice, score, noul), validation, JSON schema for System 2 |
| `hunch/adapters/base.py` | What adapters share: the pooled HTTP client, `Outage`, the prompts, `normalize` (the System 1 answer shape) |
| `hunch/adapters/chat.py` | Chat-completions (OpenAI-format) endpoints: `ChatDecider`, `VLLMDecider`, `ChatReasoner` |
| `hunch/adapters/http.py` | Any API you describe with two functions: `HTTPDecider`, `HTTPReasoner` |
| `hunch/adapters/gemini.py` | Gemini through its own `generateContent` API (logprobs for System 1, `responseSchema` for System 2): `GeminiDecider`, `GeminiReasoner` |
| `hunch/adapters/vendors.py` | One System 1 and System 2 class per major OpenAI-format vendor (OpenAI, Mistral, DeepSeek, xAI, Groq, Together, Fireworks, OpenRouter, Ollama) with its endpoint and key built in |
| `hunch/adapters/anthropic.py` | Claude, which has its own protocol and SDK: `ClaudeAPI`, `ClaudeCLI` (optional install) |
| `hunch/providers.py` | URLs and key variables of common OpenAI-compatible services, so `provider="groq"` replaces a URL |
| `hunch/local.py` | `LocalDecider`: an in-process classifier (embeddings + logistic regression) used as a System 1 tier or a guard |
| `hunch/router.py` | `Router`: the cascade, thresholds, retries, outage fallback, guard, audit, cache use |
| `hunch/cache.py` | `Cache`: SQLite cache of model answers |
| `hunch/store.py` | `Store`: SQLite log of decisions, labels, thresholds, calibration issues |
| `hunch/calibrate.py` | Picks per-question confidence thresholds that meet an accuracy target |
| `hunch/learn.py` | `retrain`: trains a better local tier from the log and promotes it only if it helps |
| `hunch/review.py` | The queue of questions where a person's label would help most |
| `hunch/stats.py` | Usage report from the log |
| `hunch/evaluate.py`, `hunch/golden.py` | Replay logged decisions through candidate models; score a golden dataset |
| `hunch/config.py` | `router_from_env()`: build a Router from `HUNCH_*` environment variables |
| `hunch/api.py` | FastAPI app: `/decide`, `/feedback`, `/calibrate`, `/stats`, `/review` |
| `serve/` | Dockerfile and compose file for serving a model with vLLM next to hunch |
| `examples/` | A real-model check and a load-testing script |
| `tests/` | Router behavior, calibration simulations, concurrency |

Dependencies are layered on purpose: the core needs only `httpx` and `anthropic`; the API needs `fastapi`; the local
classifier needs `scikit-learn` and `sentence-transformers`. `LocalDecider` imports them lazily, so using only the
remote models never loads them.

## One decision, end to end

`Router.decide(state, questions, images=(), thresholds=None, escalate=True)` does this:

1. **Validate** the questions.
2. **Run the tiers in order.** Tier 0 gets every question. For each question it answered, compare its confidence to
   the threshold for (this tier, this question). At or above: that tier's answer is final. Below: the question moves
   to the next tier. A tier may skip a question (a `LocalDecider` only knows the questions it was trained on), which
   also passes it down.
3. **Guard check** (if a `guard` is set). Questions a tier was confident about are shown to the guard. If the guard
   disagrees with probability at or above `guard_threshold`, the answer is pulled back and treated as unsure.
4. **Pick audits.** With `audit=0.1`, a random 10% of the still-confident answers are also sent to System 2, only to
   be graded (the answer returned still comes from System 1).
5. **Ask System 2** about the unsure questions plus the audited ones, in a single call.
6. **Assemble the result**: for each question the answer, its confidence, `source` (`hunch` or `system2`), which tier
   answered, what every tier said (`tried`), the guard's answer if it vetoed, and for escalated questions System 2's
   reason with the tier's answer kept under `hunch`. Also latency per stage, cache hits, and `degraded`.
7. **Log it** to the Store (if there is one) and return the decision id, so a label can be attached later.

Threshold precedence for a tier and question: the `thresholds` passed to `decide`, else the threshold calibration saved
in the Store for that tier and question, else the Router's default for that tier.

## The pieces, and why they are shaped this way

### Questions (`questions.py`)

A question is `{"type": "choice" | "score" | "noul", "instructions", "criteria"}`.

- `choice`: pick one of named options. `score`: pick a level from an ordered list. `noul`: yes/no.
- Every type reduces to "a probability over a small, closed set of answers", which is what makes a confidence
  meaningful. Open-ended text answers would have no calibrated confidence, so they are not supported.
- The same definition drives the System 1 prompt, the System 2 JSON schema (so System 2 cannot return an invalid
  answer), and label validation (`check_answer`), so the three can never drift apart.

### System 1 (`adapters/`, `local.py`)

All System 1 models share one contract: `decide(state, questions, images) -> {key: {answer, confidence, probabilities}}`.
`normalize` produces that shape from any model's raw output, so the Router and calibration never care which model
answered.

- **`ChatDecider`** works with any OpenAI-style chat endpoint that returns logprobs (OpenAI, vLLM, Ollama, Together,
  Groq...). Each question is asked on its own with `max_tokens=1`, and the probability mass on each option's label
  (A, B, ... / 0, 1, ... / yes, no), renormalized over the options, is the confidence. *Why:* one token means one fast
  forward pass and a clean distribution over exactly the legal answers; asking questions separately means one
  question's wording cannot bleed into another's confidence. It raises a clear error if the model puts no mass on any
  option (a "thinking" model that spends its one token starting to think), and `extra` lets you turn thinking off.
- **`VLLMDecider`** is a `ChatDecider` with local defaults, for a model you serve yourself. *Why:* the largest latency
  cost in practice was the network. A model co-located with hunch answered in about 150 ms, against about 580 ms for a
  hosted API and about 1.1 s from a laptop to a cloud GPU.
- **`HTTPDecider`** takes two functions (`request`, `parse`) for any API that is not chat-shaped.
- **`LocalDecider`** embeds the text with a small frozen model, then fits one logistic regression per question on your
  graded decisions. It runs in about 10 ms with no server. *Why:* many questions are topical, and a classifier trained
  on your own labels beats a general LLM on those while costing almost nothing. It answers only questions it was trained
  on, which is what lets it sit in front of an LLM in a chain. Its `identity` includes a hash of the trained heads, so
  retraining never serves stale cached answers.
- **One shared `httpx` client** for all model calls. *Why:* building a client per call cost about 10 ms of CPU holding
  the GIL and capped a process near 30 model calls a second no matter how many threads it had.

### System 2 (`adapters/`)

Contract: `answer(state, questions, images) -> {key: {answer, reason}}`.

- **`ClaudeAPI`** uses structured outputs with the schema from `answer_schema`, so the answer is always valid. It raises
  on refusals and truncation instead of returning a half answer.
- **`ClaudeCLI`** runs the local `claude` CLI with no tools, for use with a Claude Code login and no API key.
- **`HTTPReasoner` / `ChatReasoner`** let any chat model with structured outputs act as System 2.
- `default_system2()` picks the API if a key is set, else the CLI if installed, else none.

System 2 returns an answer and a reason but **no confidence**. That is deliberate: its answer is treated as the best
available truth for grading System 1 when no human has labeled the item.

### The Router (`router.py`)

- **Tier chain.** `system1` is one decider or a list. *Why a chain:* different models are best at different questions
  and cost very different amounts. A 10 ms classifier can settle the easy majority, an LLM the middle, a stronger model the rest.
  Each tier has its own threshold because a classifier's 0.9 and an LLM's 0.9 do not mean the same thing.
- **Failure handling.** Network errors and 408/429/5xx are retried with exponential backoff. If a model is still down
  after retries, that is an *outage* and the router falls back: a dead tier is skipped, System 2 answers everything if
  every tier is down, and if System 2 is down the last tier's answers stand (listed in `unsure`). Every fallback is
  recorded in `degraded`. A *mistake* (bad key, unknown model) is never swallowed this way: it raises. *Why:* falling
  back on an outage keeps the service up; falling back on a misconfiguration would hide a bug behind plausible-looking
  answers from the wrong model.
- **Audit.** *Why:* the log naturally contains System 2's answers only for the questions System 1 was *unsure* about,
  which is a biased sample. If calibration only saw those, it would think System 1 was worse than it is. Auditing a
  random share of confident answers gives an unbiased sample (see weights under the Store).
- **Guard.** *Why:* confidence cannot reveal errors a model is confidently wrong about. On real job emails, an LLM was
  wrong 6.6% of the time at 0.95 confidence or higher. A classifier trained on your own labels often knows exactly those
  cases; letting it veto a confident answer caught 13 of 37 such errors while escalating 3% of emails.
- **Single-flight on the cache.** When many requests ask the same question about the same text at once, the first
  asks the model and the rest wait for its answer. *Why:* without it, a burst of identical requests (a retry storm, a
  popular item) all miss the cache together and all hit the model.

### The Cache (`cache.py`)

A SQLite store of what each model said, keyed by a hash of (model identity, question name, question definition, text,
images). *Why this key:* it caches the *model's answer*, not the *decision*. Changing a threshold, recalibrating, or
adding a tier therefore needs no cache clearing, while changing the model, the prompt version or the question
definition automatically misses. A model with no `identity` is not cached.

### The Store (`store.py`)

SQLite with four tables: `decisions` (every result), `labels` (true answers), `tier_thresholds` (calibration output),
`calibration_issues` (why a question has none). It uses WAL mode and a 30 second busy timeout, which fixed `database is
locked` errors and lost decisions under concurrent load.

`Store.labeled()` is the bridge between logging and learning. It returns one row per graded question and tier with the
tier's answer and confidence, the **truth**, and a **weight**:

- Truth is a human label if one exists ("human"), else System 2's answer ("proxy").
- Escalated questions were always seen by System 2 (weight 1). Confident ones are graded only if audited, so each gets
  weight `1 / audit_rate`. Weighting by the inverse of the chance of being graded makes the rows represent *all* of a
  tier's answers, not just the unsure ones. Simulation tests check this recovers the true accuracy.

### Calibration (`calibrate.py`)

Goal: for each question and tier, the lowest confidence threshold at which that tier is at least `target_accuracy`
accurate, so everything above it can skip System 2.

- It sorts graded answers by confidence and tests slices from the most confident down, certifying each with a
  conservative one-sided bound (the lower end of a 90% Wilson interval, using the *effective* sample size so audit
  weights are not mistaken for extra evidence). The search is a **fixed-sequence test**: it stops at the first slice
  that fails, and the threshold is the last slice certified.
- *Why fixed-sequence:* taking the best of many separate tests overpromises, because with enough thresholds one will
  look good by luck. Testing in an order fixed by the confidences alone makes the bound hold for the threshold that was
  actually picked. In simulation the old method missed a 95% target 19% of the time; this one about 5%. The sequence
  starts at the first slice that could pass even with zero errors, otherwise a high target could never be certified.
- When no threshold is found, `reason` says why: `not enough data`, `not enough evidence` (the target is not ruled out,
  grade more), or `target unreachable` (even the optimistic estimate is below it). `best_target` is the highest target
  the data does support, found by search, so it is always a target that really finds a threshold.
- **Fail closed.** For the last two reasons, `calibrate` saves a threshold nothing can reach, so that tier is not
  trusted for that question and it escalates until a later calibration finds a threshold. *Why:* the alternative was to
  keep trusting a tier at a default threshold that data had just shown cannot meet the target.

### Learning from the log (`learn.py`, `review.py`)

- **`retrain`** fits a new `LocalDecider` on the log's graded decisions and promotes it **per question**, and only if,
  on held-out decisions, it meets the accuracy target and passes on fewer questions than the current classifier. The
  split is by decision id (a stable hash), so repeated runs compare like with like. Promotion also saves the new
  threshold. *Why a gate:* retraining is only worth deploying if it lowers the escalation rate at the same accuracy; a
  silent regression would be worse than no retraining. On real job emails the retrained classifier answered 55% of
  emails alone at 97.5% accuracy.
- **`review_queue`** lists ungraded questions where a person's label helps most: tiers and System 2 disagreed, an
  audit disagreed with a confident answer, or nobody was sure and System 2 was not asked. Labeling those first makes
  both calibration and retraining better, per label, than labeling at random.

### Measuring (`stats.py`, `evaluate.py`, `golden.py`)

- `stats` summarizes the log: share answered per tier, per-tier accuracy against labels (audit-weighted), latency per
  stage, cache hit rate, calibration issues, and review queue size. It answers "is this tier earning its place?"
- `evaluate` replays the logged, graded decisions through candidate System 1 models and reports each one's accuracy
  and the escalation rate it would need for the target, so a model can be chosen before it is deployed.
- `golden` (`Router.test`) scores a router on a hand-written dataset without writing to the log.

### Configuration and API (`config.py`, `api.py`)

`router_from_env()` builds a Router from `HUNCH_SYSTEM1_*`, `HUNCH_SYSTEM2_*`, `HUNCH_LOCAL_MODEL`, `HUNCH_DB`,
`HUNCH_CACHE`, `HUNCH_AUDIT` and `HUNCH_THRESHOLD`, so a deployment swaps models with no code change. The FastAPI app
wraps one shared Router, checks System 1 is reachable before serving (`ChatDecider.check`), and protects every route
except `/health` with a bearer token. Labels arrive through `/feedback`; `/calibrate` recomputes thresholds.

## How the parts form a loop

```
 decide ----> Store.log ----> (labels from people, audits from System 2)
   ^                                       |
   |                                       v
 thresholds <---- calibrate <---- Store.labeled (truth + weights)
 better tier 0 <--- retrain  <---- Store.labeled
 what to label next <--- review_queue
```

Every decision is logged. Labels and audits turn the log into graded data. Calibration turns graded data into
thresholds that keep System 1 honest. Retraining turns the same data into a cheaper, stronger tier 0. The review queue
points a person at the labels that help most. Each pass should lower the share of questions that reach System 2 at the
same accuracy.

## Concurrency

The Router is thread-safe and meant to be shared. SQLite uses WAL with a busy timeout so readers (stats, review) do
not block writers (logging). Cache misses are single-flighted. One pooled HTTP client serves every model call.
`examples/loadtest.py` drives a running API with many clients to check throughput and errors.

## Testing

- `tests/test_router.py` covers routing, chaining, thresholds, retries and outage fallback, caching, the guard, the
  local classifier, retraining, review and stats, using small fake models with scripted answers.
- `tests/test_calibration_simulation.py` simulates models whose accuracy at each confidence is known, then checks that
  calibration's thresholds, reasons and audit weights match the truth. Seeds are fixed, so it is deterministic.
- `tests/test_concurrency.py` exercises the API and Store under parallel load.
- CI runs the suite on Python 3.11 and 3.13.

## Known limits and what is not built

- Calibration quality depends on label quality. On real job emails some labels looked noisy, which caps measured accuracy.
- Isotonic or Venn-Abers calibration and conformal risk control are not implemented (issue #16).
- Per-slice calibration (for example by sender) and a verifier pass are not implemented (issue #17).
- Cost-aware thresholds (#18), drift monitoring (#19), reading hunch-ai style specs (#20) and a CLI (#8) are open.
- The guard and `LocalDecider` run in-process and are not wrapped in outage fallback.
- Only the ordering of tiers is fixed by you; the router does not learn which tier to try first.
