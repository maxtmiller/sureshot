# Case study: what kind of answer does a question want? (TREC)

The second end-to-end test of hunch, on a different kind of task from [banking77](case_study_banking77.md): six
options instead of 77, short questions, and labels that follow the dataset's own conventions. The script is
`case_study.py --dataset trec`; every run's numbers are in `case_study_trec_results.json`.

## Setup

The same as banking77 apart from the data:

- **Data:** [TREC](https://huggingface.co/datasets/SetFit/TREC-QC) question classification, about 5,900 unique
  questions, each labeled by a person with the kind of answer it asks for: an abbreviation, an entity, a
  description, a person, a location or a number. The options are described with the dataset's own names
  ("Entities", "Numeric values"...). 300 questions a day for ten days; from day 6, half of them ask for a number or a
  location.
- **Router:** a local classifier (untrained at first), then `gpt-4o-mini`, then `gpt-4.1` as System 2. Target 95%,
  5% of confident answers audited, nothing trusted until calibrated, 5% probes.
- **Each night:** 30 labels from the review queue (a random 2% of messages first, then a random 10% of the answers a
  tier kept), `retrain`, `calibrate` for `gpt-4o-mini` when it has no threshold or drift flags it, `recheck`,
  `prune`.
- **Cost:** asking both models about all 3,000 questions once cost $2.26 ($0.04 for `gpt-4o-mini`, $2.22 for
  `gpt-4.1`). The six runs read those answers from the cache. Latency uses each model's mean time in that run
  (0.53 s and 0.85 s) and 60 ms for the classifier.
- **Runs:** seeds 0, 1 and 2, each with money only and with waiting priced at $0.01 a second. Runs are repeatable:
  the same seed gives the same results.

## Results

| | Accuracy | Cost | Mean latency |
|---|---|---|---|
| hunch, money only (three runs) | 85.9% to 86.0% | $1.93 to $2.07 | 1.17 to 1.32 s |
| hunch, waiting at $0.01 a second (three runs) | 85.8% to 86.0% | $1.98 to $2.17 | 0.88 to 0.93 s |
| `gpt-4.1` alone | 86.0% | $2.22 | 0.85 s |
| `gpt-4o-mini` alone | 72.8% | $0.04 | 0.53 s |

hunch matches `gpt-4.1`'s accuracy for 2% to 13% less, and only from days 7 to 10 does it beat `gpt-4.1` on speed
(0.77 to 0.87 s with waiting priced). The savings are smaller than on banking77, because here the cheap model is
rarely trustworthy. As on banking77 the 95% target is out of reach (`gpt-4.1` itself is 86% right), and the labeled
random sample says so in every run, from day 2 in two and for good from day 6 in the third.

By run, waiting priced:

| Run | Local classifier | Its accuracy | `gpt-4o-mini` | Drift flags |
|---|---|---|---|---|
| 1 | from day 7, 11% to 34% of questions | 91% to 99% | skipped from night 1 | none |
| 2 | from day 9, 8% to 12% | 92% to 100% | skipped from night 1 | none |
| 3 | from day 4, 12% to 19% | 84% to 96% | skipped from night 1 | classifier's escalation rate, day 10 |

With money only, `gpt-4o-mini` was asked about every question, since keeping 2% to 12% of them for $0.000014 each
paid against $0.00074 for `gpt-4.1`. That made hunch slower than `gpt-4.1` alone.

## What worked

- **The local classifier** again carried the savings: 96%, 99% and 91% right on the answers it kept (money-only
  runs), close to the target that `gpt-4.1` can't reach. In one run `retrain` failed it closed on day 7, after the
  shift, and no new one met the target by day 10.
- **`prune`** skipped `gpt-4o-mini` from night 1 when waiting was priced, which is what kept hunch near `gpt-4.1`'s
  speed.
- **The labeled random sample** showed the router below target in every run, and `stats` put it at 79% to 89% on day
  10 (63 to 76 labels) against a true 86%.

## What this found

1. **Calibration certified `gpt-4o-mini` on System 2's word alone.** On night 1 (two runs) or 2 (one run),
   `calibrate` set a threshold of 0.999999999998 from 32 to 35 graded answers, of which 0 to 2 had a person's label:
   the rest were graded by `gpt-4.1`. They all agreed, and 35 of 35 clears 95% on the one-sided 90% bound. Across
   all 3,000 questions, the 184 at or above that threshold agree with `gpt-4.1` every time, yet only 87.5% are right:
   both models make the same mistakes there, which grading by System 2 can never see. Its kept answers in the runs
   were 81% to 87% right. hunch reported it (`graded_by` says `{"human": 0, "system2": 35}`), but nothing acted on
   it. On banking77 the same thing happened more mildly (certified at 95%, 89% to 94% right). Fixed: see below.
2. **`recheck` still had too few labels.** With a 10% share of the kept answers queued for labels, it had 14 to 23 labels on `gpt-4o-mini`'s kept answers by
   day 10, 84% to 88% right, which leaves the upper bound at 99%. It never failed the tier.
3. **The drift was benign, and drift monitoring mostly stayed quiet.** Shifting half the traffic to numbers and
   locations did not change either model's accuracy, so no flag was the right answer, and the one flag (the
   classifier's escalation rate in one run) reflected a real change in what it saw. But the accuracy checks were
   `unchecked` in every run on day 10: neither tier had 30 randomly graded answers since its calibration, so a real
   drop in accuracy would have gone unseen too.

## After the fix: labels must confirm a threshold

`calibrate(sample=, confident=)` now saves a threshold that rests on System 2's grades as unconfirmed until your
labels on the random samples at or above it clear the target on the same bound; until then the router asks the tier
but doesn't keep its answers. The labels come from a random share of the answers a tier was confident about, kept or
not (`confident=0.1`, replacing the share of kept answers). The same six runs again:

| | Accuracy | Cost | Mean latency | `gpt-4o-mini` |
|---|---|---|---|---|
| money only, before | 85.9% to 86.0% | $1.93 to $2.07 | 1.17 to 1.32 s | trusted from night 1 or 2, 81% to 87% right |
| money only, after | 85.8% to 86.0% | $1.94 to $2.18 | 0.86 to 1.27 s | never trusted |
| waiting priced, before | 85.8% to 86.0% | $1.98 to $2.17 | 0.88 to 0.93 s | skipped by `prune` |
| waiting priced, after | 85.8% to 86.0% | $1.94 to $2.18 | 0.86 to 0.93 s | never trusted |

- In two runs, the night-1 labels on `gpt-4o-mini`'s confident answers were enough to stop `calibrate` from
  certifying it at all: with real labels in the grades it failed closed every night ("not enough evidence"), and was
  only probed.
- In the third it was certified on night 2 from 3 labels and 29 System 2 grades, and stayed unconfirmed to the end:
  19 labels by day 10, 89% right, short of what 95% needs. Without waiting priced it was still asked every question
  while unconfirmed, which cost 9% more and 0.1 s for nothing. That is the price of collecting the labels that could
  confirm it.
- Accuracy did not change, because `gpt-4o-mini`'s kept answers were few; what changed is that hunch no longer
  claims 95% for answers that were 81% to 87% right.

## Then the classifier too

`retrain(sample=, confident=)` now holds the classifier to the same rule: it is run on every held-out decision a
person labeled at random, and its threshold is unconfirmed until those labels show the target. The same six runs:

| | Accuracy | Cost | Mean latency |
|---|---|---|---|
| money only | 86.0% | $2.23 to $2.26 | 0.95 to 1.36 s |
| waiting priced | 86.0% | $2.23 | 0.95 to 0.97 s |
| `gpt-4.1` alone | 86.0% | $2.22 | 0.85 s |

The classifier was promoted in all six runs (from day 3, 6 or 8) and stayed unconfirmed. By its last check it had 5
to 27 random labels where it was confident, all right, and the lower bound reached 0.66 to 0.91; 95% needs at least 32
straight right answers on the equal-weight bound, more when labels from the 2% sample (each standing for five times
as many decisions) are mixed in. Before this change it was trusted and its kept answers were 92% to 98% right.

So with nothing confirmed, hunch here is `gpt-4.1` plus the cost and latency of asking tiers whose answers it can't
yet keep. That is the honest result at 30 labels a night over ten days: no tier has shown 95% on labels. Getting
there takes more labels on what the tiers are confident about, a longer run, or a lower target.

### Confirming on flagged answers only

Two more rounds, waiting priced, with 30% of confident answers queued for labels (`--confident 0.3`):

- Shifting labels to the confident share alone didn't help: those labels were mixed with the 2% sample's, each of
  which stood for up to 15 times as many decisions, so 19 to 36 labels counted as 4 to 10.
- Confirmation now counts only the labels on answers the tier flagged as confident, which were all picked at the same
  rate, so each counts fully. By day 10 the classifier had 18, 2 and 27 such labels, 100%, 100% and 96% right, with
  lower bounds of 0.92, 0.55 and 0.88. It needs about 32 straight right ones for 95%, so on trec it is now a
  matter of days, not of weighting. Accuracy, cost and speed were unchanged ($2.23, 0.94 to 0.97 s).

### A 90% target

To see confirmation happen within ten days, both studies were run with a 90% target (`--target 0.9`, waiting
priced, `--confident 0.3`). At first the classifier was confirmed in four runs of six, on night 8 or 10, and then
answered 60% to 89% of questions with daily savings of 55% to 83%, but its kept answers were 84% to 93% right
against 90%. The labels that confirmed it came from answers held to a higher threshold than the one it was trusted
at, since `retrain` lowered the threshold almost every night (0.95 to 0.82 over five nights here). Now only labels on
answers held to the confirmed threshold or a lower one count. With that, no run confirmed the classifier in ten
days: each night's lower threshold left the new band without labels (0 to 41 counted by day 10).

## Caveats

One dataset with its own label conventions ("Who is Stephen Hawking?" asks for a description, not a person), which
both models often miss; better option descriptions or examples would likely raise both. The labeler is the
dataset, always right by definition. Latency comes from typical times, since the runs read answers from a cache.
