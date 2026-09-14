# Reason-code redesign experiment — 12 September 2026

Completed research outputs: `data/eval/reason_code_rebuild_20260912_v2/REPORT.md`.
This experiment did not edit the authoritative illustrated Word report or
deployed models. Its design has subsequently been integrated into the
[final system](final_reason_code_system.md), with a fresh June-only refit and
frozen July scoring; the experiment's held-out metrics remain unchanged.

Additional [March–April blocked reason-code results](blocked_reason_code_test.md)
are now available using the same membership as the blocked binary EF-HGB test.
Conditional minimum-one macro-F1 is 98.65% for mechanisms and 90.22% for
components. These are separate exploratory results; no deployed models changed.

Follow-up: [July reason-only refinement tests](july_reason_only_refinement_tests.md)
records selective fallback and weather-aware reason refinement with the binary
detector kept unchanged. Those tested refiners remain undeployed; a later mixed
output release promotes only minimum-one mechanism assignment while retaining
the original component thresholds. See the final-system page.

## What was implemented

- 322 features calculated from the hourly observation clock and same-hour archived
  reference residuals. Trailing statistics use current/past observations only.
- Component-specific feature views: wind-vane heads cannot read pressure features.
  Mechanism heads omit raw channel values and use changes, trailing variation,
  standardized departures and physical/stuck evidence.
- Current-hour reason targets within accepted fault episodes. Physical breaches
  are attached to their actual hours. Stuck evidence begins when a trailing
  24-hour window confirms it, without backfilling preceding hours.
- Historical statistical adjudication remains the statistical reference at its
  recorded timestamp. Calibration retains the four historical confirmed intervals.
- Event-balanced sample weights, non-fault training negatives, and three grouped
  folds inside training for selecting convex HGB fusion weights and thresholds.
- Original random split, a new label/event-balanced grouped split, and a fixed
  chronological split: train before April; validation 8–30 April; test 8 May–30
  June. Seven-day boundary gaps and removal of crossing groups prevent overlap.
- Threshold predictions and a minimum-one policy. The latter supplies the highest
  scoring supported code if no code crosses threshold, while preserving multiple
  above-threshold codes.

## Conditional results

These scores apply to reference fault hours with a supported current-hour code,
using the minimum-one policy. They are not end-to-end alert accuracy.

| Split | Mechanism macro F1 | Component macro F1 | Exact mechanism set | Exact component set |
|---|---:|---:|---:|---:|
| Random | 95.93% | 97.43% | 97.33% | 96.44% |
| Grouped | 98.60% | 94.04% | 97.61% | 94.99% |
| Chronological | 88.38% | 88.18% | 93.69% | 90.57% |

Macro F1 averages codes with positive test support. There are no positive
calibration-offset or wind-vane targets in the chronological test. The grouped
calibration test has one positive event and the wind-vane test two; perfect scores
on these small populations do not establish broad reliability.

Chronological per-code F1: spike/impossible 72.99%, stuck/flatline 100%,
statistical anomaly 92.13%; anemometer 98.34%, barometer 81.83%, light/UV 80.93%,
rain gauge 100%, thermo-hygrometer 79.79%.

## Target coverage is part of the result

| Split | Original fault test hours | Hours with a timed code | Hours without a timed code |
|---|---:|---:|---:|
| Random | 1,389 | 786 | 603 |
| Grouped | 1,382 | 838 | 544 |
| Chronological | 2,005 | 1,442 | 563 |

The new reference has 5,406 resolved hours across the full 9,260-hour fault
population. Missing timed reason labels are unknown, not healthy labels. They are
excluded from the new conditional metrics and remain in the stored ledger.

The improved scores cannot be presented as a controlled improvement over the
old episode-broadcast target. As a diagnostic, scoring these new current-hour
predictions against the original episode labels on every chronological fault
hour gives 57.49% mechanism macro F1 and 58.46% component macro F1. Explaining an
entire episode and describing a mechanism active at this hour are different tasks.

The earlier failed 29-hour IBIRAL3 wind-vane episode has only five confirmed
current-hour wind-vane labels under the causal timing rule. It is in training in
the new grouped split and excluded by the chronological May boundary gap. The
new grouped wind-vane score therefore does not demonstrate that this particular
held-out failure was fixed. The other previously identified problem—pressure
features entering the wind-vane model—is removed by construction.

## Connection to the actual binary detector

The actual saved final random EF-HGB was evaluated with the new heads on its
original random test membership. Among its 735 correctly detected faults with
timed reason labels, minimum-one micro F1 is 98.40% for mechanisms and 98.41% for
components.

Across the eligible cascade population, including non-fault hours but excluding
fault hours with unknown timed codes, the threshold policy gives mechanism
micro F1 90.96% and component micro F1 90.56%. Forcing a code on every alert
reduces these to 88.33% and 88.24%, respectively. An alert is not a guarantee
that the fault reference is positive; false alarms acquire false explanations.

The generic `summary.csv` cascade rows use a separate experimental HGB gate
trained on the matching split, not the saved operational EF-HGB. That gate is
weaker and is not a replacement candidate. Its scores must not be attributed to
the production detector. The time split has ample binary fault support, but this
work does not constitute a full chronological retraining/evaluation of the final
binary EF-HGB system.

Follow-up (14 September): a [bounded chronological binary EF-HGB test](chronological_fault_detection_test.md)
now refits the final configuration on pre-April data using the existing tensor
and labels. May–June test F1 is 71.96%; this remains a model-transfer diagnostic,
not a fully causal reconstruction of the complete pipeline. Deployment is unchanged.

## Verification and limitations

Six targeted unit tests pass: target timing, no stuck backfill, feature truncation,
component feature isolation, group integrity, chronology and minimum-one behavior.
All 26 real stations pass the future-truncation feature check. All 30 serialized
reason heads replay the stored probabilities on 116 sampled rows. Source hashes
confirm the original datasets, labels, tensor and split manifest were unchanged.
Numerical fitting was limited to two threads.

Historical statistical reference labels still use retrospective adjudication;
archived reference arrival times have not been verified. Physical and stuck
labels share their defining evidence with the rule feature branch, so high
performance partly measures recovery of deterministic rules. The dataset has
already been inspected during development. A fresh prospective evaluation,
support across more independent events/stations, and treatment of fault hours
without timed labels are required before treating the full code set as deployable.

## Reproduction

```powershell
python scripts/rebuild_reason_code_experiment.py --output data/eval/reason_code_rebuild_new_run
python scripts/summarize_reason_code_rebuild.py --output data/eval/reason_code_rebuild_new_run
python -m pytest tests/test_reason_code_rebuild.py -q
```

The runner requires a new output directory so a completed experiment is not
overwritten. See the output directory for per-code support, split membership,
selected thresholds, model bundles, current/original target comparisons and
sample-level prediction ledgers. The first partial run is marked aborted and is
not used in the completed results.
