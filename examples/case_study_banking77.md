# Case study: routing bank support messages (banking77)

An end-to-end test of hunch on a realistic job: routing a bank's support messages to one of 77 intents, over ten
simulated days. It was run twice: once as hunch was, and again after the five changes the first run showed were
needed. The script is `case_study_banking77.py`; both runs' numbers are in `case_study_banking77_results.json`.

## Setup

- **Data:** banking77, about 13,000 real customer messages, each labeled by a person with one of 77 intents. 3,000
  of them, 300 a day for ten days. From day 6, half the messages come from eight card-security intents (lost or
  compromised cards, unrecognized payments), to see whether drift monitoring notices.
- **Router:** a local classifier (untrained at first), then `gpt-4o-mini`, then `gpt-4.1` as System 2. Accuracy
  target 95%, 5% of confident answers audited.
- **Each night:** a person labels 30 items from the review queue (the dataset's labels stand in for them),
  `retrain` may promote a new classifier, and `gpt-4o-mini` is recalibrated.
- **Baselines:** the same 3,000 messages sent to `gpt-4o-mini` alone and to `gpt-4.1` alone.
- **Cost:** OpenAI list prices ($0.15 / $0.60 per million tokens for `gpt-4o-mini`, $2 / $8 for `gpt-4.1`) times the
  tokens each call really used. Asking both models about every message once cost $10.39; both runs read those
  answers from a cache, so the runs themselves cost nothing more.

## Results

| | Accuracy | Cost | Mean latency |
|---|---|---|---|
| hunch, first run | 78.8% | $8.42 | 1.31 s |
| hunch, after the changes | 79.4% | $8.52 | 1.15 s |
| `gpt-4.1` alone | 79.3% | $10.14 | 0.95 s |
| `gpt-4o-mini` alone | 69.0% | $0.25 | 0.57 s |

hunch matches `gpt-4.1` for 16% less. No model here gets near the 95% target, and after the changes hunch says so.
On the last four days, with `gpt-4o-mini` no longer asked about every message, it is also faster than `gpt-4.1`
alone (0.85 to 0.87 s).

Share of messages each stage answered, after the changes:

| Day | Accuracy | Local classifier | `gpt-4o-mini` | `gpt-4.1` | Saved vs `gpt-4.1` |
|---|---|---|---|---|---|
| 1 | 78% | 0% | 0% | 100% | -2% |
| 2 to 4 | 76% to 78% | 0% | 15% to 18% | 82% to 85% | 12% to 14% |
| 5 to 6 | 79% to 85% | 17% to 19% | 11% to 12% | 69% to 72% | 24% to 28% |
| 7 to 10 | 79% to 82% | 17% to 21% | 0% | 79% to 83% | 17% to 20% |

## What worked from the start

- **The local classifier.** Trained on `gpt-4.1`'s answers plus the labels, `retrain` promoted it on night 4. It then
  answered 17% to 21% of messages at 88% to 98% accuracy against the dataset's labels (95% or more on five of the six
  days), in about 50 ms, for nothing.
  In the first run, on the 326 messages it kept, it was right 97.9% of the time, as often as `gpt-4.1` on the same
  messages (96.3%): it keeps the easy ones, and on those it is as good as the big model.
- **Failing closed.** `gpt-4o-mini` is badly overconfident here: 88% of its answers are at 0.95 confidence or more,
  and only 74% of those are right.
- **Numbered labels** (new for this study, since 77 options is past the old limit of 26). 28.7% of `gpt-4o-mini`'s
  errors were one digit off the right label, against 25.8% for `gpt-4.1`, which answers with names, and 18.6% by
  chance. Most of that is neighboring numbers being similar intents (the list is alphabetical).

## What the first run showed, and what the changes did

1. **An uncalibrated tier was trusted at the default threshold.** On day 1 `gpt-4o-mini` kept 93% of answers at 0.8,
   and accuracy was 71%. Now `Router(trust_uncalibrated=False)` trusts nothing until it is calibrated: day 1 sent
   everything to `gpt-4.1` (78% accurate), and that complete, unbiased day let calibration certify `gpt-4o-mini` on
   night 1 at a confidence near 1.0, which it then answered 11% to 18% of messages above.
2. **A tier that failed closed was still asked about every message.** That cost as much as it saved and added 0.57 s
   to every message. Now it is only asked about a 5% sample (`probe`): after `gpt-4o-mini` failed closed on night 6,
   days 7 to 10 averaged 0.85 to 0.87 s, where the first run's failed-closed days averaged about 1.5 s.
3. **Drift on the retrained classifier was never checked.** `retrain` left no snapshot, so drift had nothing to
   compare it to. Now it saves one, and drift flagged the day-6 shift that same night (`distribution`, then
   `escalation_rate` and the change detectors). In the first run nothing was flagged.
4. **Nothing said the target was out of reach.** Now a random 2% of messages go to the front of the review queue,
   and `stats` measures the router on them. From night 2, `below_target` was true; by day 10, 61 sampled labels gave
   78.7% (90% interval 68.9% to 86.0%) against a true 79.4%.
5. **A tier that failed closed was never checked again.** The script now recalibrates `gpt-4o-mini` every night it has
   no certified threshold, and whenever drift is flagged. (This is a choice for whoever runs hunch, not a library
   change.)

## What still needs to change

1. **Calibrating against System 2 overstates a tier.** On night 1 `gpt-4o-mini` was certified at 95% against
   `gpt-4.1`'s answers, but what it kept on days 2 to 5 was 89% to 94% right against the dataset's labels. Where
   nobody labeled, agreeing with `gpt-4.1` counts as right, and `gpt-4.1` is right 79% of the time. Calibration
   should use human labels where there are enough, or say plainly that it measured agreement with System 2.
2. **A drift flag on the classifier never clears.** `retrain` only saves a new snapshot when it promotes a classifier,
   so the day-6 flag stayed through day 10 and kept triggering recalibration. A retrain that keeps the current
   classifier should refresh its snapshot, or `calibrate(tier=0)` should be the documented way to clear it.
3. **The classifier's accuracy is still not checked.** Only 5% of its kept answers are audited, two or three a day,
   and a check needs 30. The labeled sample could count toward it, or cheap tiers could be audited more.

## Caveats

One run per setup, one dataset, a simulated labeler that is always right, and plain intent names as option
descriptions (better descriptions or examples would likely raise both models' accuracy). Latency for the models is
the mean from the baseline run, since the simulation read their answers from the cache; the local classifier's is
measured.
