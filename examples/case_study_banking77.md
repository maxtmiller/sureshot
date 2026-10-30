# Case study: routing bank support messages (banking77)

An end-to-end test of hunch on a realistic job: routing a bank's support messages to one of 77 intents, over ten
simulated days. Each round turned up problems in hunch; they were fixed and the study run again. The script is
`case_study.py` (`--dataset banking77`, the default); every round's numbers are in `case_study_banking77_results.json`.

## Setup

- **Data:** banking77, about 13,000 real customer messages, each labeled by a person with one of 77 intents. 3,000
  of them, 300 a day for ten days. From day 6, half the messages come from eight card-security intents (lost or
  compromised cards, unrecognized payments), to see whether drift monitoring notices.
- **Router:** a local classifier (untrained at first), then `gpt-4o-mini`, then `gpt-4.1` as System 2. Accuracy
  target 95%, 5% of confident answers audited. Nothing is trusted until it is calibrated (`trust_uncalibrated=False`),
  and a tier that failed closed or does not pay is only asked about 5% of messages (`probe`).
- **Each night:** a person labels 30 items from the review queue, a random 2% of messages first, then a random 10%
  of the answers a tier kept (the dataset's labels stand in for them). `retrain` may promote a new classifier and rechecks the one it keeps. `gpt-4o-mini` is
  recalibrated whenever it has no certified threshold or drift flags it, `recheck` tests its threshold against the
  labeled sample, and `prune` stops asking it when it costs more than it saves.
- **Cost:** OpenAI list prices ($0.15 / $0.60 per million tokens for `gpt-4o-mini`, $2 / $8 for `gpt-4.1`) times the
  tokens each call really used. Asking both models about every message once cost $10.39; every run read those
  answers from a cache, so the runs themselves cost nothing more.
- **Waiting:** whether `gpt-4o-mini` pays depends on what a second of waiting is worth. The final round was run with
  money only, and with waiting priced at $0.01 a second.
- **Latency:** each message is priced with typical times (`gpt-4o-mini` 0.57 s and `gpt-4.1` 0.95 s, their means in
  the baseline run; the classifier 60 ms), since the simulation reads the models' answers from a cache.
- **Luck:** which answers get audited and probed is random, and it decides what gets graded. Each setup was run three
  times. The runs in the tables below were not repeatable, since eight worker threads drew from one random
  generator in no fixed order; the simulation now decides one message at a time and seeds the log's ids, so a seed
  gives the same run every time.

## Results

| | Accuracy | Cost | Mean latency |
|---|---|---|---|
| hunch, first run | 78.8% | $8.42 | 1.31 s |
| hunch, after the first fixes (one run) | 79.4% | $8.52 | 1.15 s |
| hunch, after the second fixes (three runs) | 79.2% to 79.5% | $7.94 to $8.98 | 1.26 to 1.39 s |
| hunch, final, money only (three runs) | 79.2% to 79.3% | $8.59 to $9.14 | 1.34 to 1.40 s |
| hunch, final, waiting at $0.01 a second (three runs) | 79.2% to 79.5% | $8.57 to $9.04 | 0.93 to 0.97 s |
| `gpt-4.1` alone | 79.3% | $10.14 | 0.95 s |
| `gpt-4o-mini` alone | 69.0% | $0.25 | 0.57 s |

With waiting priced, hunch matches `gpt-4.1`'s accuracy for 11% to 15% less at the same speed, and from day 7 it is
faster (0.86 to 0.91 s). No model here gets near the 95% target, and hunch says so: the labeled random sample put the
router below its target from day 2 or 3 in five runs of six, and day 9 in the sixth.

The final round with waiting priced, by run:

| Run | Local classifier | Its accuracy | `gpt-4o-mini` | Drift flags |
|---|---|---|---|---|
| 1 | from day 5, 11% to 41% of messages | 91% to 100% | skipped from night 1 | none |
| 2 | from day 4, 16% to 24% | 86% to 100% | skipped from night 1 | none |
| 3 | from day 3, 19% to 23% | 83% to 96% | skipped from night 1 | none |

With money only, `gpt-4o-mini` paid for itself (it kept about 13% of messages for a fraction of a cent each) and was
asked about every message, and the classifier was promoted on day 8, on day 10, or not at all.

## What worked

- **The local classifier.** Trained on `gpt-4.1`'s answers plus the labels, it answered about a fifth of messages
  once promoted, mostly at 95% or more against the dataset's labels, in about 60 ms, for nothing. In the first run,
  on the 326 messages it kept it was right 97.9% of the time, as often as `gpt-4.1` on the same messages (96.3%): it
  keeps the easy ones.
- **Dropping the tier that does not pay.** Asked about every message, `gpt-4o-mini` added 0.57 s to each to save
  0.95 s on about one in eight. `prune` skipped it, and the classifier took over the easy messages.
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
3. **A drift flag on the classifier never cleared.** Three runs showed two more problems: the snapshot from #46 was
   built partly from the decisions the classifier was trained on, so drift flagged it the day after promotion; and
   recalibrating it with `calibrate(tier=0)` switched it off for good. `retrain` now leaves training decisions out of
   the snapshot and rechecks the classifier it keeps every night, stopping trusting it only when the data rules the
   target out (without that last rule, a classifier near the target was switched off and on night after night).

Third round, and the fixes it led to (#53, #54, #55):

1. **Nothing rechecked a certified tier** (`gpt-4o-mini` was certified on night 1 and never checked again).
   `recheck(store, tier, sample=)` now tests saved thresholds against the labeled sample on a schedule.
2. **A tier that keeps little still cost latency on every message**, which made hunch slower than `gpt-4.1` alone.
   `prune(store, costs, tier)` now stops asking a tier, apart from the probe sample, where it costs more than it
   saves; waiting counts if it is priced into `costs`.
3. **Skipping a tier starved the classifier.** `retrain` learned only from questions some tier had answered, so once
   `gpt-4o-mini` was skipped it had nothing new to learn from, was never promoted, and hunch cost as much as
   `gpt-4.1` alone. It now learns from every graded question (`Store.graded()`).

Fifth round, after the [trec case study](case_study_trec.md) found a tier certified on System 2's grades alone:
`calibrate(sample=, confident=)` now needs your labels on the random samples to confirm a threshold before the
router trusts it, and the review queue offers a random share of the answers a tier was confident about
(`confident=0.1`, replacing the share of kept answers). Three seeds each way:

| | Accuracy | Cost | Mean latency |
|---|---|---|---|
| money only | 79.4% to 79.5% | $8.88 to $9.73 | 1.34 to 1.46 s |
| waiting at $0.01 a second | 79.4% to 79.5% | $8.72 to $9.23 | 0.93 to 0.99 s |

`gpt-4o-mini` was certified on night 1 from 3 to 6 labels and about 50 System 2 grades, and stayed unconfirmed in
all six runs: by day 10 its labels were 91% to 93% right (27 to 49 of them, money only), short of 95%. Before, it
kept 340 to 396 answers at 90% to 91% accuracy; now it keeps none, the classifier carries the savings, and hunch
costs 4% to 14% less than `gpt-4.1` alone instead of 10% to 22%. Accuracy did not move.

Sixth round: `retrain(sample=, confident=)` holds the classifier to the same rule, checking it on every held-out
decision a person labeled at random.

| | Accuracy | Cost | Mean latency |
|---|---|---|---|
| money only | 79.3% | $10.39 | 1.56 s |
| waiting at $0.01 a second | 79.3% | $10.17 to $10.18 | 1.07 to 1.08 s |

The classifier was promoted from day 3 or 4 and stayed unconfirmed in all six runs: by day 10 its random labels were
93% to 99% right (35 to 48 of them), and the lower bound reached 0.84 to 0.91, short of 0.95. With nothing
confirmed, hunch costs 0% to 3% more than `gpt-4.1` alone and is slower. Before, the classifier's kept answers were
90% to 96% right; hunch was cheaper by claiming a 95% it did not have.

## What still needs to change

1. **`recheck` needs far more labels than a 2% sample gives.** By day 10 it had 1 to 11 labeled answers from
   `gpt-4o-mini`'s kept messages, too few to rule out 95% for a tier that was 83% to 94% right. The review queue now
   also offers a random share of the answers a tier kept (`kept=0.1`), and `recheck` counts their labels. In the
   money-only setup (three seeds, after the repeatability fix) that raised its labels by day 10 from 6 to 15 to 33
   to 56, with the router's accuracy unchanged (79.2% to 79.7%), but the upper bound on `gpt-4o-mini`'s accuracy
   still sat at 99% to 100% against its measured 91% to 93%. Ruling out 95% at that accuracy takes a few hundred
   labels on its kept answers, more than 30 labels a night can give in ten days. With waiting priced,
   `gpt-4o-mini` is pruned and keeps almost nothing, so there is nothing to recheck.
2. **Luck still decides a lot.** The classifier was promoted on day 3 or never, and the labeled sample's estimate of
   the router's accuracy on day 10 ranged from 68% to 88% (50 to 71 labels) against a true 79%, depending on which
   answers happened to be audited and labeled.
3. **The target is out of reach.** No setup gets near 95% here, because `gpt-4.1` itself is at 79%. Better option
   descriptions, examples in the prompt, or a stronger System 2 are what would move it; hunch can only say so.

## Caveats

One dataset, a simulated labeler that is always right, and plain intent names as option descriptions (better
descriptions or examples would likely raise both models' accuracy). Latency comes from typical times, not measured
ones, since the simulation reads the models' answers from a cache.
