# Case study: routing bank support messages (banking77)

A first end-to-end test of hunch on a realistic job: routing a bank's support messages to one of 77 intents, over
ten simulated days. The script is `case_study_banking77.py`; the numbers below are in `case_study_banking77_results.json`.

## Setup

- **Data:** banking77, about 13,000 real customer messages, each labeled by a person with one of 77 intents. 3,000
  of them, 300 a day for ten days. From day 6, half the messages come from eight card-security intents (lost or
  compromised cards, unrecognized payments), to see whether drift monitoring notices.
- **Router:** a local classifier (untrained at first), then `gpt-4o-mini`, then `gpt-4.1` as System 2. Accuracy
  target 95%, 5% of confident answers audited.
- **Each night:** a person labels 30 items from the review queue (the dataset's labels stand in for them),
  `retrain` may promote a new classifier, and `gpt-4o-mini` is recalibrated on the first three nights and afterwards
  whenever drift is flagged.
- **Baselines:** the same 3,000 messages sent to `gpt-4o-mini` alone and to `gpt-4.1` alone.
- **Cost:** OpenAI list prices ($0.15 / $0.60 per million tokens for `gpt-4o-mini`, $2 / $8 for `gpt-4.1`) times the
  tokens each call really used. The whole run cost $10.39 in API calls.

## Results

| | Accuracy | Cost | Mean latency |
|---|---|---|---|
| hunch | 78.8% | $8.42 | 1.31 s |
| `gpt-4.1` alone | 79.3% | $10.14 | 0.95 s |
| `gpt-4o-mini` alone | 69.0% | $0.25 | 0.57 s |

hunch matched `gpt-4.1`'s accuracy for 17% less, but was slower, and nowhere near the 95% target, which no model here
reaches.

| Day | Accuracy | Local classifier | `gpt-4o-mini` | `gpt-4.1` | Saved vs `gpt-4.1` |
|---|---|---|---|---|---|
| 1 | 71% | 0% | 93% | 7% | 86% |
| 2 to 4 | 76% to 78% | 0% | 0% | 100% | -2% |
| 5 to 10 | 78% to 85% | 16% to 19% | 0% | 81% to 84% | 13% to 17% |

(The share of messages each stage answered.)

## What worked

- **The local classifier.** Trained on `gpt-4.1`'s answers plus 290 labels, `retrain` promoted it on night 4. From
  then on it answered 16% to 19% of messages at 95% to 100% accuracy against the dataset's labels, in about 50 ms,
  for nothing. On the 326 messages it kept it was right 97.9% of the time, about as often as `gpt-4.1` on the same
  messages (96.3%): it keeps the easy ones, and on those it is as good as the big model.
- **Failing closed.** `gpt-4o-mini` is badly overconfident here: 88% of its answers are at 0.95 confidence or more,
  and only 74% of those are right. Calibration refused to trust it (the best target the data supported was 0.77 to
  0.88), so from day 2 none of its answers were kept.
- **Numbered labels** (new for this study, since 77 options is past the old limit of 26) worked. 28.7% of
  `gpt-4o-mini`'s errors were one digit off the right label, against 25.8% for `gpt-4.1`, which answers with names, and
  18.6% by chance. Most of that is neighboring numbers being similar intents (the list is alphabetical), so numbers
  add at most a little confusion.

## What needs to change

1. **An uncalibrated tier is trusted at the default threshold.** On day 1, before any calibration, `gpt-4o-mini` kept
   93% of answers at 0.8 and accuracy was 71%. A new deployment should trust nothing until calibration certifies it.
2. **A tier that failed closed is still called.** From day 2, `gpt-4o-mini` was asked about every message and its
   answer always discarded. That made hunch cost slightly more than `gpt-4.1` alone on days 2 to 4 and added 0.57 s to
   every message, which is why it is slower than `gpt-4.1` alone. It should be skipped, apart from a small sample that
   keeps the data to recalibrate it with.
3. **The accuracy drift checks never ran.** Failing closed and `retrain` both leave a snapshot with no target, so
   `accuracy` and `change_accuracy` stayed `unchecked` all ten days. After the shift, `gpt-4o-mini`'s accuracy fell 2.6
   points and nothing could have seen it. (Its confidence barely moved, 88.5% to 87.6% at 0.95 or more, so drift was
   right not to flag a confidence change.) `retrain` should save a full snapshot with its target and threshold.
4. **Nothing says the target is out of reach for System 2.** hunch grades the tiers against `gpt-4.1` where nobody
   labeled, so it cannot see that `gpt-4.1` itself is at 79%. The review queue spent all 290 labels on disagreements,
   which is right for training but says nothing about overall accuracy. A small random labeled sample, and a warning
   when System 2's accuracy on it is below the target, would.
5. **Recalibrating only on drift starves a tier.** `gpt-4o-mini` was recalibrated on nights 1 to 3 only; the best
   target its data supported rose from 0.77 to 0.88 and it was never checked again. A tier that failed closed should
   be recalibrated on a schedule.

## Caveats

One run, one dataset, a simulated labeler that is always right, and plain intent names as option descriptions (better
descriptions or examples would likely raise both models' accuracy). Latency for the models is the mean from the
baseline run, since the simulation read their answers from the cache; the local classifier's is measured.
