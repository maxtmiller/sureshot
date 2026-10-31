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
| `hunch/store.py` | `Store`: SQLite log of decisions, labels, thresholds, calibration issues, skips, drift snapshots |
| `hunch/calibrate.py` | `calibrate` picks per-question thresholds that meet an accuracy target; `recheck` tests them against labels; `prune` skips tiers that don't pay |
| `hunch/learn.py` | `retrain`: trains a better local tier from the log and promotes it only if it helps |
| `hunch/review.py` | The queue of questions where a person's label would help most |
| `hunch/stats.py` | Usage, accuracy, cost, latency and drift report from the log |
| `hunch/evaluate.py`, `hunch/golden.py` | Replay logged decisions through candidate models; score a golden dataset |
| `hunch/config.py` | `router_from_env()`: build a Router from `HUNCH_*` environment variables |
| `hunch/api.py` | FastAPI app: `/decide`, `/feedback`, `/calibrate`, `/stats`, `/review` |
| `serve/` | Dockerfile and compose file for serving a model with vLLM next to hunch |
| `examples/` | Case studies, a drift alert, a real-model check and a load-testing script |
| `tests/` | Router behavior, calibration simulations, concurrency, recorded real-model exchanges |

Dependencies are layered on purpose: the core needs only `httpx`; Claude needs `anthropic`; the API needs `fastapi`; the local
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
   disagrees with probability at or above `guard_threshold`, the answer is pulled back and treated as unsure. A
   guard that is down vetoes nothing and is listed in `degraded`.
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
  (A, B, ... / 0, 1, ... / yes, no), renormalized over the options, is the confidence. Past 26 options the labels are
  numbers, which needs a tokenizer that keeps each number one token (checked on `gpt-4o-mini` with 77 options). Only
  the top 20 tokens come back with a probability, so with many options the rest count as 0. *Why:* one token means one fast
  forward pass and a clean distribution over exactly the legal answers; asking questions separately means one
  question's wording cannot bleed into another's confidence. It raises a clear error if the model puts no mass on any
  option (a "thinking" model that spends its one token starting to think), and `extra` lets you turn thinking off.
- **`VLLMDecider`** is a `ChatDecider` with local defaults (and thinking turned off where the model supports it), for
  a model you serve yourself. *Why:* the largest latency cost in practice was the network. A model co-located with
  hunch answered in about 150 ms, against about 580 ms for a hosted API and about 1.1 s from a laptop to a cloud GPU.
  On an NVIDIA L4, Qwen2.5-7B-Instruct answered a three-question email in about 150 ms (p50 and p95), and zero-shot it
  was less accurate than `gpt-4o-mini` on a small labeled set (71% against 84%). `--max-logprobs 20` matches the 20
  top logprobs hunch requests. The compose file in `serve/` has not been run as a whole (it needs Docker with an NVIDIA
  GPU); its pieces were: the official `vllm/vllm-openai` image with the compose flags on an L4, `serve/Dockerfile`, and
  `VLLMDecider`, `check()`, `router.test`, `/decide` and `/feedback` against that server. A pip-installed vLLM may need
  `nvcc` for FlashInfer; set `VLLM_USE_FLASHINFER_SAMPLER=0` or use the official image.
- **`HTTPDecider`** takes two functions (`request`, `parse`) for any API that is not chat-shaped.
- **`LocalDecider`** embeds the text with a small frozen model, then fits one logistic regression per question on your
  graded decisions. It runs in about 10 ms with no server. *Why:* many questions are topical, and a classifier trained
  on your own labels beats a general LLM on those while costing almost nothing. It answers only questions it was trained
  on, which is what lets it sit in front of an LLM in a chain. Its `identity` includes a hash of the trained heads, so
  retraining never serves stale cached answers.
- **Vendor classes** (`adapters/vendors.py`) are the generic `Chat*` classes with a vendor's URL and key variable.
  See [Vendors](#vendors) for what was checked live.
- **One shared `httpx` client** for all model calls. *Why:* building a client per call cost about 10 ms of CPU holding
  the GIL and capped a process near 30 model calls a second no matter how many threads it had.

### System 2 (`adapters/`)

Contract: `answer(state, questions, images) -> {key: {answer, reason}}`.

- **`ClaudeAPI`** uses structured outputs with the schema from `answer_schema`, so the answer is always valid. It raises
  on refusals and truncation instead of returning a half answer.
- **`ClaudeCLI`** runs the local `claude` CLI with no tools, for use with a Claude Code login and no API key.
- **`HTTPReasoner` / `ChatReasoner`** let any chat model with structured outputs act as System 2.
- `default_system2()` picks the API if a key is set, else the CLI if installed, else none.

### Vendors

What a vendor supports is up to the vendor. A System 1 needs a model that returns logprobs, which not every model or
service does: reasoning models such as OpenAI's GPT-5 and o-series are reported not to on chat completions, and models
that think first need thinking turned off (for Gemini, `extra={"thinkingConfig": {"thinkingBudget": 0}}`; for Gemma on
Ollama, `extra={"reasoning_effort": "none"}`). A System 2 needs structured outputs, or `json_mode="object"` for a
service with only JSON mode (DeepSeek).

The Gemini adapters speak Gemini's own `generateContent` API (key from `GEMINI_API_KEY` or `GOOGLE_API_KEY`).
`GeminiReasoner` was checked live on `gemini-3.1-flash-lite` and `gemini-3-flash-preview`. `GeminiDecider` could not
be: every model available to a new key (the Gemini 3 family, Gemma 4) answered "Logprobs is not enabled for this
model", and the 2.5 family that supported them is closed to new users. OpenAI was checked live (and its exchanges are
recorded for CI in `tests/recordings/`). The other vendors' URLs and capabilities come from their public docs and
community reports, not live checks.

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
  answers from the wrong model. Decisions made while System 1 was down are logged but skipped by `calibrate`, since they
  have no System 1 confidence.
- **Audit.** *Why:* the log naturally contains System 2's answers only for the questions System 1 was *unsure* about,
  which is a biased sample. If calibration only saw those, it would think System 1 was worse than it is. Auditing a
  random share of confident answers gives an unbiased sample (see weights under the Store).
- **Guard.** *Why:* confidence cannot reveal errors a model is confidently wrong about, and a second model often knows
  them. Measurements:
  - On 506 held-out job emails with their original labels, `gpt-4o-mini` looked wrong 37 times at 0.95 confidence or
    more, and a local guard at 0.9 caught 13. Claude then reviewed every email where the label and `gpt-4o-mini`
    disagreed, and sided with `gpt-4o-mini` on 68%. Against those corrected labels `gpt-4o-mini` had 9 confident
    errors (accuracy 0.97, not 0.91), and the guard caught 0 to 2 while raising 5 to 35 false alarms: most of what it
    had "caught" was label noise. The guard learns whatever your labels say, so measure it against checked labels.
  - On 1,178 held-out items from ag_news, rotten_tomatoes and enron_spam, labels reviewed the same way, `gpt-4o-mini`
    was confident on 97% and wrong on 56 of those. At about 3% extra escalation, a second LLM disagreeing at 0.9
    caught 24 (13 false alarms) and the local guard 14 (20 false alarms); the local guard did well on topics and
    poorly on sentiment (1 of 16). Asking `gpt-4o-mini` to check its own answer caught 7 to 12 with 28 to 43 false
    alarms, and per-cluster thresholds caught 4 with 65.
  - A second LLM costs a call for every confident answer; the local guard runs in-process.
- **Probe.** A tier that failed closed on a question, or that `prune` skips, is asked only on a random `probe` share
  (default 5%). *Why:* its answers would be thrown away, so asking it every time only adds cost and latency, but with
  no answers at all a later calibration or prune could never see it improve.
- **`trust_uncalibrated=False`.** *Why:* until calibration, a tier is trusted at the default threshold, which can be far
  off. In the banking77 case study `gpt-4o-mini` kept 93% of answers on day 1 at 71% accuracy. Escalating everything
  until there is a threshold costs System 2 calls but gives calibration a complete, unbiased start.
- **Single-flight on the cache.** When many requests ask the same question about the same text at once, the first
  asks the model and the rest wait for its answer. *Why:* without it, a burst of identical requests (a retry storm, a
  popular item) all miss the cache together and all hit the model.

### The Cache (`cache.py`)

A SQLite store of what each model said, keyed by a hash of (model identity, question name, question definition, text,
images). *Why this key:* it caches the *model's answer*, not the *decision*. Changing a threshold, recalibrating, or
adding a tier therefore needs no cache clearing, while changing the model, the prompt version or the question
definition automatically misses. A model with no `identity` is not cached.

- `ClaudeAPI`, `ChatDecider`, `VLLMDecider` and `ChatReasoner` have an identity (model, URL, `extra`); an API key is
  not part of it. A trained `LocalDecider`'s identity changes when it is retrained. `HTTPDecider` and `HTTPReasoner`
  take an `identity=` string. Bump `PROMPT_VERSION` in `hunch/adapters/base.py` when a prompt changes.
- System 2 is cached per question, so a repeat call asks it only about the questions it has not answered.
- A vendor can update a model behind an alias (`gpt-4o-mini` rather than a dated snapshot). The key cannot see that,
  so cached answers keep the old behavior and drift only sees the change on texts that miss the cache.
- A cached repeat is still a logged decision, so a repeated item counts again in `calibrate` and training.
- It stores answers, not texts: the key is a hash.

### The Store (`store.py`)

SQLite with six tables: `decisions` (every result), `labels` (true answers), `tier_thresholds` (calibration output),
`calibration_issues` (why a question has none), `calibrations` (what the latest calibration saw, for drift) and
`skips` (what `prune` stopped asking). Decision ids are random uuids, or come from `Store(path, seed=)` so a
simulation that logs the same decisions in the same order gets the same ids. It uses WAL mode and a 30 second busy timeout, which fixed `database is
locked` errors and lost decisions under concurrent load.

`Store.labeled()` is the bridge between logging and learning. It returns one row per graded question and tier with the
tier's answer and confidence, the **truth**, and a **weight**:

- Truth is a human label if one exists ("human"), else System 2's answer ("proxy").
- Escalated questions were always seen by System 2 (weight 1). Confident ones are graded only if audited, so each gets
  weight `1 / audit_rate`. Weighting by the inverse of the chance of being graded makes the rows represent *all* of a
  tier's answers, not just the unsure ones. Simulation tests check this recovers the true accuracy.
- `labeled()` rows also say whether the tier was `confident` (at or above the threshold it was held to, whether or
  not its answer was kept), which the router logs with each tier's answer; `calibrate` and `recheck` need it.
  `graded()` has one row per graded question, including those no tier answered, for training.

**Random samples by id.** `in_sample(id, rate)` picks a fixed share of decisions by a hash of the id, so the same
decisions stay picked however often the queue is read. Two samples use it: `sample` (all decisions, for measuring
the router) and `confident` (the answers a tier was confident about, to confirm and recheck its threshold). Using
the same hash makes them nested. Both take precedence over the other review reasons, so a decision in a sample is
queued as such even when the models disagreed on it: if it were queued for disagreeing instead, it might be left
unlabeled, and the labels that did come in would no longer be a random pick.

The `unconfirmed` table lists thresholds `calibrate` saved that labels have not confirmed yet.

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
- **Fail closed.** For `not enough evidence`, `target unreachable` and `below target on labels`, `calibrate` saves a
  threshold nothing can reach, so that tier is not trusted for that question and it escalates until a later
  calibration finds a threshold. *Why:* the alternative was to keep trusting a tier at a default threshold that data
  had just shown cannot meet the target.
- **Graded by System 2.** Where nobody labeled, an answer is graded by System 2's, so a threshold certifies agreement
  with System 2. That overstates the tier when System 2 is often wrong: in the banking77 case study `gpt-4o-mini` was
  certified at 95% against `gpt-4.1` and was 89% to 94% right against the dataset's labels. `graded_by` reports how
  many grades above the threshold came from labels and how many from System 2. On the trec case study it was worse:
  `gpt-4o-mini` was certified at 95% from about 35 answers all graded by `gpt-4.1`, and above that threshold the two
  agreed on every question while being right 87.5% of the time. Two models that share their mistakes look perfect
  graded by each other.
- **Labels must confirm.** So with `sample=` or `confident=`, a threshold needs your labels before it is trusted.
  - *Confirming* asks whether the tier was right when it said it was sure. It counts the random labels on the answers
    the tier flagged as confident while held to this threshold or a lower one (the router logs the flag and the
    threshold with each answer), and needs them to clear the target on a one-sided 90% bound. Until then the threshold is saved but unconfirmed ("not enough labels"), and the router asks the
    tier and logs its answers but doesn't keep them.
  - *Repeat-safe.* The check runs every night, and a one-off bound checked every night passes a lucky streak more
    often than it says: with a 90% target, a classifier confirmed on 26 labels, all right, was then 88% right. So it is
    only tested when the labels reach fixed counts, each test on the first that many in the order they were given,
    with the 10% split across the tests (5%, 2.5%, 1.25%...), so all of them together are wrong at most 10% of the time
    however long they run. The first test is at the fewest labels that could pass with none wrong, and the count
    doubles after that: 25, 50, 100... for 90%, and 52, 104, 208... for 95%. A bound that holds at every check (the
    confidence sequence ruling out uses) would need 62 and 142. Labels record when they were given for this.
  - *Ruling out* also uses every random label at or above the threshold, flagged or not, weighted by the inverse of
    its chance of being picked. Either set can rule the target out, each with a confidence sequence at 2.5%; the tier
    then fails closed ("below target on labels").
  - *Why only flagged answers confirm:* every one of them was in the `confident` share at the same rate, so each label
    counts fully. Mixing in the rest weights each label by its chance of being picked, and a label from the 2% sample
    stands for up to 15 times as many decisions as one from a 30% `confident` share. In the case studies that made
    dozens of labels count as 5 to 20, and nothing could be confirmed.
  - *Why only answers held to this threshold or lower:* thresholds move. In the case studies retrain lowered the
    classifier's almost every night (0.95 to 0.82 over five nights on trec). Labels on answers held to a higher
    threshold only cover the answers above it, so counting them confirmed a lower threshold on evidence about the most
    confident answers alone, and once trusted the classifier was 84% to 93% right against a 90% target. Now a lowered
    threshold needs labels on the newly trusted band before it is confirmed.
  - *Holding a threshold still.* Since a lowered threshold has no labels on its new band, `calibrate` and `retrain`
    don't lower a saved threshold unless the labels confirm the lower one; otherwise they keep the saved one
    (`held_at`). A threshold that moved down every night would never gather labels; held still it does. Raising a
    threshold is always allowed, since labels held to the lower one cover it.
  - *Why not fail closed while unconfirmed:* a tier that is only probed is rarely confident, so labels on its
    confident answers would stop coming and it could never be confirmed. *Why the `confident` share rather than the
    answers the tier kept:* an unconfirmed tier keeps nothing.
  - Without either rate nothing needs confirming, and a threshold can rest on System 2's grades alone. `retrain`
    applies the same rule to the classifier (see below).
- **`recheck`** runs the same check on a schedule, since a certified tier is rarely calibrated again: it confirms an
  unconfirmed threshold once the labels show the target, and fails one they rule out. With a 2% sample and a tier
  keeping 15% of messages, the sample alone gives about one label per 330 messages; the `confident` share is what
  makes confirmation reachable.
- **`prune`** skips a tier on a question when `keep_rate * cost(System 2) < cost(tier)`: each answer it keeps saves a
  System 2 question, and each one it is asked costs its own. Comparing with System 2 overstates what an earlier tier
  saves when a later tier would have answered, so it errs toward asking. Waiting is priced by adding it to each
  stage's cost. In the banking77 case study `gpt-4o-mini` paid in money alone (13% of $0.0034 against $0.00008) but
  added 0.57 s to every message to save 0.95 s on one in eight.
- **How far to trust it**, from simulations with a known accuracy curve (`tests/test_calibration_simulation.py`): for
  a calibrated model with 1000 graded answers and a 90% target, about 6% of calibrations land below the target
  (nominally 10%), and the misses are under a point on average. A model that can reach 95% gets a threshold in more
  than half of calibrations on 300 answers, about 5% of them below the target. A model that cannot reach the target
  abstains in over 97% of runs. The audit weights recover a tier's accuracy to within about 3 points, where averaging
  the graded rows is off by 6 to 10 for a tier that only sees what earlier tiers passed down.

### Learning from the log (`learn.py`, `review.py`)

- **`retrain`** fits a new `LocalDecider` on the log's graded decisions (including those System 2 answered with no tier
  asked) and promotes it **per question**, and only if, on held-out decisions, it meets the accuracy target and passes
  on fewer questions than the current classifier. The
  split is by decision id (a stable hash), so repeated runs compare like with like. Promotion also saves the new
  threshold. *Why a gate:* retraining is only worth deploying if it lowers the escalation rate at the same accuracy; a
  silent regression would be worse than no retraining. On real job emails the retrained classifier answered 55% of
  emails alone at 97.5% accuracy.
- Where `retrain` keeps the current classifier it checks it again on the held-out decisions, which include everything
  graded since: `"rechecked"` updates its threshold, `"failed"` stops trusting it, and `"unproven"` (the data no longer
  shows the target but does not rule it out) leaves it as it was. *Why "unproven":* without it, a classifier near the
  target was switched off and on night after night in the banking77 case study. For the same reason the classifier is
  recalibrated by `retrain`, not `calibrate(tier=0)`, which sees only its answers since promotion: one day of those
  killed a classifier `retrain` had just certified. `retrain` also saves a drift snapshot from the classifier's
  confidences on decisions it was not trained on.
- **Confirming the classifier.** It learns mostly from System 2's answers and is graded on held-out decisions that
  are mostly System 2's answers too, so it inherits System 2's mistakes and the grading can't see them. With
  `sample=` or `confident=`, `retrain` confirms it with your labels as `calibrate` does. The classifier is free to
  run, so a new one is scored on every held-out decision a person labeled at random, but only the decisions where the
  deployed classifier answered and was confident (`graded()` rows record `confident_by`) can confirm it. So a first
  classifier is always unconfirmed until it has been asked. A new classifier is confirmed on the decisions where the
  deployed one was confident, which are close to but not exactly the ones it would be confident about. Labels that
  rule the target out stop a promotion or fail the current classifier closed.
- Measured on public datasets (200 held-out items per question, 300 training labels): a classifier then
  `gpt-4o-mini` scored 92% on a topic question (the classifier answered 71%, p50 12 ms) and 89% on sentiment (trusted
  for only 12%, p50 461 ms), against 82% and 90% for the LLM alone.
- **`review_queue`** lists ungraded questions where a person's label helps most: the random `sample` and `confident` shares
  first (the only labels that measure accuracy without bias), then where tiers and System 2 disagreed, an audit
  disagreed with a confident answer, or nobody was sure and System 2 was not asked. The hard cases make calibration
  and retraining better per label than labeling at random, but they say nothing about overall accuracy.
- An answer from System 2 with no System 1 answer to compare (System 1 was down) is not queued. System 2 gives no
  confidence, so "System 2 was also unsure" cannot be detected; disagreement is the signal. The queue is only as
  informative as System 2 is different from the tiers: with `gpt-4o-mini` as the last tier and `gpt-4.1` as System 2,
  they agreed on all 16 escalated questions in one run, so every queued item came from the audit.

### Measuring (`stats.py`, `evaluate.py`, `golden.py`)

- `stats` summarizes the log: share answered per tier, per-tier accuracy against labels (audit-weighted), accuracy on
  the labeled random sample, cost, latency per stage, cache hit rate, calibration issues, review queue size and
  drift. It answers "is this tier earning its place?" Cache hits are counted as asked, so with a cache the cost is an
  upper bound.
- **Drift** compares each calibrated question and tier with its last calibration snapshot:
  - `escalation_rate` (the share below the threshold: the inputs or the model changed), `distribution` (confidences
    over five quantile bins, which catches 0.99 becoming 0.96 with the threshold at 0.95 before it moves anything
    else) and `accuracy` (kept answers picked at random, audited or in the labeled sample, each weighted by its
    chance of being picked; a label picked any other way is left out, since it was often picked for looking wrong).
  - A flag needs evidence and size. The checks share `alpha` (0.01) and use confidence sequences (with a one-off
    interval for the calibration's own fixed sample), which hold at every sample size at once: however often you
    call `stats`, they raise a false flag at most that often, ever. The price is needing several times more answers
    than a one-off test. The escalation rate must also move by `min_escalation_change` (0.05) and the bins by
    `min_distribution_change` (0.1), so a large log does not flag 20% becoming 21%. Accuracy below target has no
    minimum.
  - Those checks pool everything since the calibration, which dilutes a change that starts months later. The
    `change_*` detectors catch that: a CUSUM per check that normal answers keep near zero, started from the edge of
    the calibration's interval, raising a false flag within n answers at most n / `false_alarm_every` (100,000) of
    the time. `change` says when each change started and was flagged; a flag stays until the next calibration.
  - Each check needs `min_samples` (30) answers since the calibration, and `unchecked` names those that could not run.
    A cheap tier keeping a few dozen answers a day needs both audits and labels: in the banking77 case study, 5%
    audits gave the classifier two or three graded answers a day.
  - `upstream` names earlier tiers of the same question that drifted or were recalibrated after this one: a later
    tier only sees what earlier ones pass down, so look at the earliest first.
  - Questions with no snapshot (never calibrated, per-call thresholds, or failed closed) learn a `baseline` from their
    first `min_samples` answers; with no threshold, accuracy is `unchecked`.
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

What load testing found, with fake models that take 20 ms (so these show hunch's own limits), on a laptop:

- One process handled `/decide` at roughly 90 requests a second with two model calls each, and in-process fakes at
  600+ a second at 128 clients with p99 under 250 ms. Four `uvicorn` workers sharing the SQLite log and cache reached
  about 300 a second with no errors and no lost decisions, including `/feedback`, `/stats` and `/review` against a
  log of 12,000 decisions. The load generator shared a process with the fake server, so treat these as a floor.
- Building an HTTP client per call held the GIL for about 10 ms and capped a process near 30 model calls a second,
  hence the shared client.
- Single-flight is per process, so with `--workers N` up to N identical requests can still reach the model at once.
- `/stats` and `/review` read the whole log (about 0.25 s of CPU each at 20,000 decisions of two questions) and slow
  that process's other requests while they run.
- Separate machines would need to share one log, which SQLite does not support.

## Testing

- `tests/test_router.py` covers routing, chaining, thresholds, retries and outage fallback, caching, the guard, the
  local classifier, retraining, review and stats, using small fake models with scripted answers.
- `tests/test_calibration_simulation.py` simulates models whose accuracy at each confidence is known, then checks that
  calibration's thresholds, reasons and audit weights match the truth. Seeds are fixed, so it is deterministic.
- `tests/test_concurrency.py` exercises the API and Store under parallel load.
- `tests/test_recorded.py` replays real OpenAI exchanges recorded in `tests/recordings/`.
- CI runs the suite on Python 3.11 and 3.13.

## Known limits and what is not built

- Calibration quality depends on label quality. On real job emails some labels looked noisy, which caps measured accuracy.
- Isotonic or Venn-Abers calibration was evaluated and not adopted: in simulation it gained nothing (#16).
- Per-slice thresholds and a self-verifier were measured and lost to a second LLM as the guard (#17).
- A CLI is not built (#8).
- `recheck` needs a few hundred labels on a tier's confident answers to rule out a target the tier misses by a few
  points, and at least 52 all-right labels to confirm a 95% threshold (25 for 90%).
- Confirmation only counts held-out labels on flagged answers, so a classifier needs roughly three times the random labels a
  deployed tier does before it is trusted.
- The guard and `LocalDecider` run in-process and are not wrapped in outage fallback.
- Only the ordering of tiers is fixed by you; the router does not learn which tier to try first.
