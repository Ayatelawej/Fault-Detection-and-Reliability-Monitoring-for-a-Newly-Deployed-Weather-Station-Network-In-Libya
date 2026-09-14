# Quick July test: admin inspection as an output class

One isolated EF-HGB inspection classifier was trained on June development data.
It predicts whether a fault's current-hour reason reference is unresolved. After
the frozen binary detector alerts, a positive inspection decision replaces the
specific mechanism/component codes with `admin_inspection_required`. Otherwise,
the user's selected minimum-one reason policy applies. Inspection is a scored
alternative answer, not merely a warning or a way to exclude difficult rows.

## Results

### New task: specific reason OR inspection, with no unresolved-hour exclusion

Both policies are evaluated on the identical 13,565 July hours. The 843
unresolved reference fault hours have inspection as their correct target;
normal hours have no reason or inspection target. False detector alerts sent to
inspection are still false positives, and missed unknown faults are still
false negatives.

| Policy | Mechanism-or-inspection micro F1 | Component-or-inspection micro F1 |
|---|---:|---:|
| Minimum-one baseline, no inspection output | 42.17% | 41.37% |
| Learned inspection alternative | 50.66% | 49.82% |

This is an improvement of 8.49 and 8.45 percentage points on the expanded task.
It is **not** directly comparable to the previous 71.03% / 69.33% scores, which
excluded unknown-reason fault hours and had no inspection target class.

### Original task: specific reason codes on the previous evaluation population

To check whether this improves the previously discussed reason-code F1, the
specific predictions were also evaluated on the exact earlier 12,722-hour
population. Inspection does not get credit as the correct mechanism/component;
replacing a correct code with inspection loses that code's recall.

| Policy | Mechanism micro F1 | Component micro F1 |
|---|---:|---:|
| Minimum-one baseline | 71.03% | 69.33% |
| With inspection routing | 66.87% | 65.19% |

So it does not improve the original reason-code scores. Whether it is useful
depends on valuing review routing as a separate outcome; it is not a score-only
upgrade to the original reason classification task.

## Inspection outcomes

- 359 of 1,833 detector alerts were routed to inspection.
- 237 were unresolved reference faults: correct inspection outcomes.
- 85 had known current-hour reasons: unnecessary inspection under this target.
- 37 were reference non-fault hours: false detector alerts routed to inspection.
- Inspection precision: 66.02%; recall across all 843 unresolved fault hours:
  28.11%; F1: 39.43%. Unknown faults missed by the detector remain in that recall
  denominator.

## Protocol and limits

- June contained 3,854 unresolved fault hours. Training used all 9,260 fault
  hours and a fixed sample of 16,000 reference-normal hours.
- The classifier uses current-hour causal features, never true reason labels,
  source episode IDs, or future inspection decisions as predictors.
- Three June event-disjoint OOF folds selected a fusion of 0.5 full-view and
  0.5 rule-view HGB scores with threshold 0.70, then a June-only refit. The OOF
  event-weighted selection F1 was 28.74%; no July threshold sweep was run.
- Existing binary detector, specific reason heads, thresholds and dashboard
  artifacts were not changed. Hashes were verified before and after the test.
- The minimum-one baseline is the user's chosen comparison policy; the actual
  saved deployment is still the earlier abstaining reason policy. This test
  changes neither of them.
- Inspection truth here means **unresolved reference labels**, not an independent
  human judgment that a physical site inspection is necessary.
- July has now been inspected repeatedly during development discussions, so
  this is exploratory evidence, not a new untouched confirmatory holdout.

Outputs: `data/eval/july_admin_inspection_experiment/summary.csv`, `per_label.csv`,
`predictions.parquet`, `audit.json`, and an isolated inspection model. Reproduce
with `python scripts/experiment_admin_inspection.py --output <new-directory>`.
Three focused tests verify gate enforcement, exclusive routing, preservation of
inputs and baseline codes, and penalties for false alerts and missed faults.
