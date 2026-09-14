# July 2026: frozen reason-code evaluation

The frozen final heads were evaluated without refitting, changing thresholds,
or changing the dashboard's saved predictions. July deployment predictions were
reproduced exactly. Results use the same current-hour target construction as
the redesign: accepted episode membership AND evidence at the actual hour.

## Head performance on known faults

These results isolate the reason heads on **858 reference fault hours with
supported current-hour codes**, including hours missed by the binary detector.
The minimum-one diagnostic retains above-threshold codes, then supplies the
highest-scoring supported code if none crosses threshold. It is the policy used
in the redesign's headline conditional comparison, **not the deployed policy**.

| Policy | Axis | Micro precision | Micro recall | Micro F1 | Macro F1 | Exact code-set match |
|---|---|---:|---:|---:|---:|---:|
| Minimum-one diagnostic | Mechanism | 98.41% | 97.63% | 98.01% | 98.85% | 95.92% |
| Minimum-one diagnostic | Component | 93.78% | 94.84% | 94.31% | 90.53% | 91.72% |
| Deployed threshold policy | Mechanism | 97.35% | 58.08% | 72.75% | 80.76% | 55.13% |
| Deployed threshold policy | Component | 95.48% | 71.16% | 81.54% | 84.58% | 67.13% |

The distinction matters: the deployed threshold policy was not the same policy
as the experiment's headline minimum-one results. It retains high precision on
known faults but loses recall through abstention. The reason heads' ranking
ability is substantially better than those deployed threshold results suggest.

## Deployed detector-to-reason pipeline

This includes detector false positives and missed faults, then applies the
unchanged deployed reason thresholds. The evaluation population contains all
11,864 reference non-fault hours and the 858 resolved fault hours (12,722 total).

| Axis | Micro precision | Micro recall | Micro F1 | Macro F1 |
|---|---:|---:|---:|---:|
| Mechanism | 81.41% | 54.92% | 65.59% | 76.90% |
| Component | 77.26% | 67.12% | 71.83% | 66.31% |

On the 749 resolved fault hours actually detected by EF-HGB, deployed conditional
micro F1 is 76.18% for mechanisms and 85.06% for components.

Forcing minimum-one on every detector alert is not equivalent to supplying
known faults: its end-to-end micro F1 is 71.03% mechanism and 69.33% component,
with precision falling to 60.84% and 59.10%. It therefore cannot simply inherit
the high known-fault scores. These are fixed-policy diagnostics, not July-based
threshold selection; no deployment change was made after seeing them.

### Deployed per-code end-to-end F1

| Mechanism | Positive hours | F1 |
|---|---:|---:|
| Spike / impossible | 246 | 100.00% |
| Stuck / flatline | 127 | 100.00% |
| Statistical anomaly | 512 | 30.71% |

| Component | Positive hours | F1 |
|---|---:|---:|
| Anemometer | 160 | 89.27% |
| Barometer | 101 | 81.15% |
| Light / UV | 10 | 12.50% |
| Rain gauge | 190 | 100.00% |
| Thermo-hygrometer | 430 | 48.64% |

## Coverage and limitations

- Frozen July binary population: 13,565 hours; 1,701 reference fault hours.
- Current-hour reasons resolve 858 fault hours; **843 remain unknown** and are
  excluded from timed reason metrics, not relabelled healthy. This is about half
  the original fault-hour population, so these metrics are not a complete
  assessment of every July fault hour.
- Binary detector: 1,361 true-positive hours, 472 false alerts and 340 missed
  fault hours. Of its true positives, 749 have timed reasons and 612 do not.
- There are no positive calibration-offset or wind-vane targets in this frozen
  July reference. Their recall/F1 is not established here. Macro F1 averages
  positively supported codes; micro precision still penalises unsupported-code
  predictions. Separately recorded continuing calibration conditions were not
  inserted as new labels into the frozen July target contract.
- Statistical references were reconstructed using the existing same-month/hour
  leave-one-out evidence gate, frozen June detector thresholds, and archived
  external residuals. July 2025 context is included where available, consistent
  with the reference function's month grouping. This is retrospective weak-label
  construction, not a claim of live reference availability or physical diagnosis.
- The old episode-broadcast target is reported separately in the CSV under
  `original_episode_fault_hours_diagnostic`. It is not substituted for the new
  current-hour task, nor directly comparable to its headline F1.

## Reproduction and checks

Run `python scripts/evaluate_july_reason_codes.py --output <new-directory>`.
The existing output is `data/eval/july_2026_reason_code_evaluation/`:
`summary.csv`, `per_label.csv`, `aligned_reference.parquet`, reconstructed
statistical evidence, and `audit.json` with source checksums.

Checks verified exact agreement of episode fault membership with the frozen
binary truth, no timed reasons outside accepted fault hours, identical saved
deployment predictions, and unchanged model/input hashes. Deployed micro
precision/recall/F1 were independently recomputed from the saved reason strings
and aligned reference, without invoking model inference again.
