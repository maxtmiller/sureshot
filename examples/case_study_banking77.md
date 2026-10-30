# Case study: routing bank support messages (banking77)

An end-to-end test of hunch on a realistic job: routing a bank's support messages to one of 77 intents, over ten
simulated days. Each run turned up problems in hunch; they were fixed and the study run again. The script is
`case_study_banking77.py`; every round's numbers are in `case_study_banking77_results.json`.

## Setup

- **Data:** banking77, about 13,000 real customer messages, each labeled by a person with one of 77 intents. 3,000
  of them, 300 a day for ten days. From day 6, half the messages come from eight card-security intents (lost or
  compromised cards, unrecognized payments), to see whether drift monitoring notices.
- **Router:** a local classifier (untrained at first), then `gpt-4o-mini`, then `gpt-4.1` as System 2. Accuracy
  target 95%, 5% of confident answers audited. Nothing is trusted until it is calibrated (`trust_uncalibrated=False`),
  and a tier that failed closed is only asked about 5% of messages (`probe`).
- **Each night:** a person labels 30 items from the review queue, a random 2% of messages first (the dataset's
  labels stand in for them). `retrain` may promote a new classifier and rechecks the one it keeps. `gpt-4o-mini` is
  recalibrated whenever it has no certified threshold or drift flags it, with its threshold checked against the
  labeled sample.
- **Baselines:** the same 3,000 messages sent to `gpt-4o-mini` alone and to `gpt-4.1` alone.
- **Cost:** OpenAI list prices ($0.15 / $0.60 per million tokens for `gpt-4o-mini`, $2 / $8 for `gpt-4.1`) times the
  tokens each call really used. Asking both models about every message once cost $10.39; every run read those
  answers from a cache, so the runs themselves cost nothing more.
- **Seeds:** which answers get audited and probed is random, and it decides what gets graded, so one run can mislead:
  the final setup was run with three seeds.

## Results

| | Accuracy | Cost | Mean latency |
|---|---|---|---|
| hunch, first run | 78.8% | $8.42 | 1.31 s |
| hunch, after the first fixes (one run) | 79.4% | $8.52 | 1.15 s |
| hunch, final (three seeds) | 79.2% to 79.5% | $7.94 to $8.98 | 1.26 to 1.39 s |
| `gpt-4.1` alone | 79.3% | $10.14 | 0.95 s |
| `gpt-4o-mini` alone | 69.0% | $0.25 | 0.57 s |

hunch matches `gpt-4.1` for 11% to 22% less. No model here gets near the 95% target, and hunch says so: the labeled
random sample put the router below its target from day 2, 3 or 5, depending on the seed. It is slower than
`gpt-4.1` alone, for a reason given below.

In the final setup, by seed:

| Seed | Local classifier | Its accuracy | `gpt-4o-mini` kept | Its accuracy | Drift flags |
|---|---|---|---|---|---|
| 0 | 18% to 24% of messages from day 4 | 89% to 100% | 8% to 17% | 83% to 94% | none |
| 1 | 19% on day 10 only (promoted night 9) | 100% | 11% to 18% | 87% to 94% | none |
| 2 | 19% to 28% from day 5 | 91% to 100% | 10% to 18% | 83% to 94% | none |

## What worked

- **The local classifier.** Trained on `gpt-4.1`'s answers plus the labels, it answered about a fifth of messages
  once promoted, mostly at 95% or more against the dataset's labels, in about 50 ms, for nothing. In the first run,
  on the 326 messages it kept it was right 97.9% of the time, as often as `gpt-4.1` on the same messages (96.3%): it
  keeps the easy ones.
- **Failing closed and saying so.** `gpt-4o-mini` is badly overconfident here (88% of its answers at 0.95 confidence or
  more, 74% of those right), and the random labeling sample shows the whole router below its target.
- **Numbered labels** (new for this study, since 77 options is past the old limit of 26). 28.7% of `gpt-4o-mini`'s
  errors were one digit off the right label, against 25.8% for `gpt-4.1`, which answers with names, and 18.6% by
  chance; most of that is neighboring numbers being similar intents (the list is alphabetical).

## What each round changed

First run, and the fixes it led to (#45, #46, #47):

1. **An uncalibrated tier was trusted at the default threshold** (day 1: `gpt-4o-mini` kept 93% of answers, accuracy
   71%). `Router(trust_uncalibrated=False)` now escalates until calibration certifies a tier.
2. **A tier that failed closed was still asked about every message**, paying its cost and latency for nothing.
   `Router(probe=0.05)` now asks it about a sample only.
3. **Drift never checked the retrained classifier**, since `retrain` left no snapshot. It now saves one.
4. **Nothing said the target was out of reach.** `review_queue(sample=)` and `stats(sample=)` now measure the router on
   a random labeled sample, with `below_target`.
5. **A tier that failed closed was never checked again.** The script now recalibrates it every night.

Second run, and the fixes it led to (#48, #49, #51):

1. **Calibrating against System 2 overstates a tier** (`gpt-4o-mini` certified at 95% against `gpt-4.1`, 89% to 94%
   right against the labels). `calibrate` now reports `graded_by` and, with `sample=`, checks a threshold against
   the labeled sample.
2. **The classifier's accuracy was never checked** (two or three audited answers a day). Drift now also counts
   labeled sample answers.
3. **A drift flag on the classifier never cleared.** Running it with three seeds showed two more problems here: the
   snapshot from #46 was built partly from the decisions the classifier was trained on, so drift flagged it the day
   after promotion; and recalibrating it with `calibrate(tier=0)`, which grades only its answers since promotion,
   switched it off for good. `retrain` now leaves training decisions out of the snapshot and rechecks the classifier
   it keeps every night, stopping trusting it only when the data rules the target out (without that last rule, a
   classifier near the target was switched off and on night after night).

## What still needs to change

1. **`gpt-4o-mini` is still overstated.** It was certified on night 1 against `gpt-4.1` and kept 83% to 94% right
   against the labels all run. The sample check only runs when `calibrate` does, and nothing ran it again: drift did
   not flag `gpt-4o-mini`, and its threshold stayed certified. A certified tier should be rechecked against the
   labeled sample on a schedule, as `retrain` now does for the classifier.
2. **A tier that keeps little still costs latency on every message.** `gpt-4o-mini` kept 8% to 18% of messages but
   was asked about all of them, adding 0.57 s to most, which is why hunch is slower than `gpt-4.1` alone. `stats` can
   price this (`costs={"ms": ...}`), but nothing acts on it.
3. **Results depend on luck.** The classifier was promoted on night 3, 4 or 9 depending on the seed, and the labeled
   sample's estimate of the router's accuracy ranged from 78% to 87% against a true 79%.

## Caveats

One dataset, a simulated labeler that is always right, and plain intent names as option descriptions (better
descriptions or examples would likely raise both models' accuracy). Latency for the models is the mean from the
baseline run, since the simulation read their answers from the cache; the local classifier's is measured.
