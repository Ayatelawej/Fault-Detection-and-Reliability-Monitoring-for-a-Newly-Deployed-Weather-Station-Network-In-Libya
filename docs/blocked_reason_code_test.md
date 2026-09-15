# March–April blocked reason-code test — 14 September 2026

Completed isolated EF-HGB reason-head experiment using exactly the same split
membership as the [blocked binary fault test](blocked_fault_detection_test.md).
The deployed models, July ledger, dashboard and Word report were not changed.

## Method and population

Training includes June 2025–January 2026 and May–June 2026. February is validation;
March–April is the held-out test block. Seven-day post-boundary gaps at February,
March and May starts mean validation begins February 8, testing March 8 and later
training May 8. Whole connected fault groups crossing partitions are excluded.

| Partition | Total hours | Reference faults | Faults with reason labels | Unknown-reason faults |
|---|---:|---:|---:|---:|
| Training | 84,377 | 5,866 | 3,165 | 2,701 |
| Validation | 10,411 | 577 | 307 | 270 |
| Test | 25,260 | 1,686 | 1,157 | 529 |

All ten heads were freshly fitted with the existing redesigned method: 322
features, sensor-local component views, mechanism views excluding raw channel
levels, event/class weighting, up to 16,000 normal training negatives per axis,
and three grouped training-only OOF folds to choose fusion weights and thresholds.
February reason scores are sanity checks, not used to select the reason policy.
No test rows enter fitting or OOF selection. Models were saved before test scoring.

The binary gate is the saved matching-split EF-HGB, not the deployed model fitted
on different memberships. Both its original 0.30 threshold and its February-only
selected 0.35 threshold are reported. No test-based reason refinement was made.

## Conditional reason classification

These results assume a reference fault with a supported reason target (1,157
hours), independent of whether the binary detector catches it. Macro-F1 averages
only codes with positive test support: three mechanisms and four components.

| Reason policy | Mechanism macro-F1 | Component macro-F1 | Mechanism micro-F1 | Component micro-F1 |
|---|---:|---:|---:|---:|
| Thresholds | 94.15% | 85.78% | 96.99% | 97.79% |
| Minimum-one on both axes | 98.65% | 90.22% | 99.23% | 98.64% |
| Mixed: minimum-one mechanism, threshold component | 98.65% | 85.78% | 99.23% | 97.79% |

For comparison with the existing conditional minimum-one experiments:

| Reason experiment split | Mechanism macro-F1 | Component macro-F1 |
|---|---:|---:|
| Original random | 95.93% | 97.43% |
| Label/event-balanced grouped | 98.60% | 94.04% |
| March–April blocked | 98.65% | 90.22% |

The reason experiment's balanced grouped split is not identical to the binary
detector's original spaced membership; do not silently rename one as the other.
Test populations and supported labels differ, so these are not controlled
estimates of which split or method is intrinsically better.

## Detection plus explanation

End-to-end micro-F1 includes normal hours, detector false alarms and missed
resolved faults. It excludes 529 fault hours with unknown reason labels, leaving
24,731 metric hours. Those 529 hours remain in the full prediction ledger and
in binary fault-detection evaluation.

| Binary threshold | Reason policy | Mechanism micro-F1 | Component micro-F1 |
|---|---|---:|---:|
| Fixed 0.30 | Thresholds | 88.65% | 89.51% |
| Fixed 0.30 | Minimum-one both | 86.05% | 85.54% |
| Fixed 0.30 | Mixed | 86.05% | 89.51% |
| February-selected 0.35 | Thresholds | 89.12% | 89.90% |
| February-selected 0.35 | Minimum-one both | 86.88% | 86.47% |
| February-selected 0.35 | Mixed | 86.88% | 89.90% |

Minimum-one helps when a fault is known, but can turn false binary alarms into
false reason assignments. The mixed policy here is reported because it is the
current deployed policy, not because it wins every metric on this test. Binary
F1 is 87.01% at 0.30 and 87.32% at February-selected 0.35.

## Per-code evidence and limitations

Conditional minimum-one results:

| Code | Positive hours | Positive events | F1 |
|---|---:|---:|---:|
| Spike/impossible | 32 | 31 | 100.00% |
| Stuck/flatline | 919 | 18 | 100.00% |
| Statistical anomaly | 216 | 159 | 95.95% |
| Anemometer | 979 | 55 | 99.39% |
| Barometer | 115 | 106 | 96.46% |
| Light/UV | 61 | 44 | 98.36% |
| Thermo-hygrometer | 14 | 5 | 66.67% |

There are no positive calibration-offset, rain-gauge or wind-vane targets in this
test. Their recall is not established. Stuck and anemometer observations dominate
the positive hours, explaining why micro-F1 is stronger than component macro-F1.
Thermo-hygrometer support is small and performance weaker; the aggregate must not
be presented as uniformly high performance across all components.

This is an exploratory blocked-period classification evaluation with training
on both sides of the test block. The period was selected after prior development
experiments, and feature/target design and binary hyperparameters were already
informed by the broader dataset. It is not independent prospective validation.
Existing preprocessing and historical statistical-label timing limitations
remain; rolling features use observation history, including preceding held-out
observations, without using their test labels in estimator fitting. Physical and
stuck targets also share rule evidence with the features. Labels are evidence-
derived references, not independently confirmed sensor diagnoses.

## Verification and reproduction

- Exact membership agreement with the blocked binary test; no shared rows/groups
  between training, validation and test, or between OOF fitting/scoring folds.
- Unknown-reason fault rows excluded from supervised reason fitting, not
  converted into normal negative examples.
- All ten fitted heads scored from serialized files; unsupported heads would
  remain unavailable to minimum-one fallback.
- SHA-256 checks confirmed all protected inputs/deployed artifacts were unchanged.
- 24 relevant tests passed, including the new training-population and fallback
  checks. Two-thread runtime was about 86 seconds; training has finished.

Artifacts: `data/eval/blocked_reason_codes_20260914/` contains the frozen plan,
population, per-code support, selections, fitted models, per-code and aggregate
metrics, sample prediction ledger and audit. Earlier experiment files are retained.

```powershell
python scripts/experiment_blocked_reason_codes.py --output data/eval/blocked_reason_codes_new_run
python -m pytest tests/test_blocked_reason_codes.py tests/test_blocked_fault_detection.py tests/test_reason_code_rebuild.py tests/test_final_reason_codes.py -q
```
