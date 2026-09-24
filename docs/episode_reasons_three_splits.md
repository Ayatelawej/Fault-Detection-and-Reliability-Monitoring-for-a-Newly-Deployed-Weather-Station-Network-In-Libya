# Original episode reasons: three-split EF-HGB pilot

Completed 24 September 2026. All models train on original episode mechanism and
component labels and are evaluated against those same targets. Both outputs
use minimum-one assignment. Every reference fault hour in each test partition
is included; no current-hour-support exclusion and no binary detector gate.

| Split | Test fault hours | Mechanism micro-F1 | Component micro-F1 |
|---|---:|---:|---:|
| Random | 1,389 | 89.36% | 82.08% |
| Grouped | 1,382 | 76.74% | 61.80% |
| March–April temporal holdout | 1,686 | 73.69% | 80.75% |

The temporal results reproduce the earlier episode-label pilot to numerical
tolerance. The grouped split is the saved reason-specific balanced grouped split,
not the binary spaced split. Random can share event groups across partitions;
grouped and blocked membership were checked for zero group overlap.

Same 322-feature representation, fixed HGB settings, event/class weighting,
sampled normal training negatives, and per-head validation-selected convex
fusion/threshold grid across all three runs. This compact protocol uses the
validation partition rather than the original three training OOF folds.
The policy was specified in advance, not chosen from these test scores.
March–April testing uses February validation and training on both sides of the
test period, including May–June. This is not forward-only validation.

These results measure prediction of event-associated reference reasons, which
may be established retrospectively, not contemporaneous confirmation of each
mechanism. They are exploratory results on already inspected development data.
They do not replace deployed mixed-policy or current-hour-target results.

Macro-F1 is retained in the detailed summary (mechanism/component): random
84.41%/74.07%, grouped 70.15%/59.03%, temporal 58.57%/61.30%. Temporal macro
averages three positive-support mechanisms and four components; random/grouped
have all four mechanisms and six components. Micro-F1 pools all label decisions.

Runner: `python -m scripts.experiment_episode_reasons_three_splits`.
Outputs: `data/eval/episode_reasons_three_splits_20260924_v2/`, including
summary, per-label metrics, population, validation selections, held-out predictions,
split overlap audit, and input/protected-file hashes. Deployed reason model and
July predictions were unchanged. No dashboard or report edits were made.
