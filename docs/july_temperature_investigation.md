# July temperature investigation and bounded fix options

## Finding: real shared warming is a plausible driver of excess alerts

The project data supports a temperature-distribution change, not simply a
failure of the reason classifiers:

- Of 472 binary false-alert hours, 367 (77.75%) have temperature statistical
  flags, but none have temperature physical-limit or stuck flags in the saved
  detector scores. These are reference-label false positives, not independently
  inspected sensor-health judgments.
- Temperature/humidity evidence on these false alerts does not pass the
  reconstructed retrospective statistical label gate. Most failed records say
  `context_not_outlier`; multiple channel records can belong to one hour.
- Matching station and UTC-hour groups with at least five observed days in both
  June and July, and matching available reference temperatures, produces 540
  common groups. Equal-weight average July-minus-June warming is **3.15°C in
  station observations and 3.06°C in reference weather**. This supports shared
  environmental warming rather than treating every raised temperature as a
  sensor-specific problem. It does **not** establish that July 2026 was hotter
  than historical Julys; no long-term climate-normal comparison was performed.

The frozen robust-z baseline is station/pooled median-and-MAD based rather than
season/hour conditional. Previously measured temperature-score medians were
0.25 in training-normal hours, 2.42 in July-normal hours and 4.48 in July false
alerts. The model relies heavily on sensor-group flags combining different
detectors. This combination can mistake seasonally shifted statistical evidence
for sensor-fault evidence. It is an evidence-backed hypothesis, not a proof of
the cause of every individual alert.

## What was tested without retraining

Two fixed, conservative shadow safeguards were evaluated, with no threshold
sweep or selection using July labels. Each could suppress an existing alert
only when:

1. Temperature statistical flags are present, with no other-channel statistical
   flags and no current physical or confirmed-stuck evidence.
2. All three temperature channels (low/average/high) lie within 3.5 MAD of their
   previous 30 same-hour days, with at least 15 observations.
3. The current station-minus-reference temperature residual is within three
   standard deviations of its previous 720-hour residual history, with at least
   240 observations and a 0.3°C scale floor.

The stricter variant additionally requires analogous agreement with spatial
neighbours, with a 0.5°C residual scale floor. Missing required evidence preserves
the alert. These are new diagnostic rules, not calibrated probabilities.

### July fault detection

| Policy | Precision | Recall | F1 | False alerts removed | Labelled fault hours lost |
|---|---:|---:|---:|---:|---:|
| Existing detector | 74.25% | 80.01% | **77.02%** | 0 | 0 |
| Temperature context + reference | 77.23% | 73.37% | 75.25% | 104 | 113 |
| Temperature context + reference + neighbours | 77.35% | 74.07% | 75.68% | 103 | 101 |

Both safeguards lose fault-detection F1. They also slightly reduce F1 on the
existing development random validation and test populations: validation goes
from 92.44% to 91.92% / 91.96%, and test from 92.02% to 91.84% / 91.96%.
These controls are not a new chronological backtest.

### July minimum-one reason cascade

| Policy | Mechanism micro F1 | Component micro F1 |
|---|---:|---:|
| Existing detector + minimum-one | 71.03% | 69.33% |
| Temperature context + reference | 73.80% | 72.07% |
| Temperature context + reference + neighbours | 73.89% | 72.16% |

The reason scores rise because fewer false alerts get false explanations.
However, **89 of the stricter safeguard's 101 lost fault hours have unknown
current-hour reason labels**, so they are excluded from reason F1 but correctly
counted as misses in fault-detection F1. Only 12 lost resolved fault hours enter
the reason metric. The apparent reason-score gain must not hide deterioration
in the primary fault detector. Neither safeguard was deployed.

This also exposes a target-timing issue: an hour can look consistent with current
weather yet still belong to an accepted fault-labelled episode. A blanket
same-hour veto is not guaranteed to preserve the original episode-derived binary
target. Unknown reasons do not prove the binary reference is wrong.

## Potential fixes within a small scope

1. **Keep weather agreement as supporting information, not an automatic veto.**
   A dashboard annotation could identify temperature-driven alerts consistent
   with reference weather while retaining the alert. This improves interpretation,
   not F1, and does not require a new output class or model retraining.
2. **If a numerical improvement is required, test one temperature-specific
   refinement only.** Add the small causal temperature/reference feature set to
   a narrowly scoped statistical-alert classifier or detector update, keeping
   physical/stuck processing and the reason heads fixed. Train/select only with
   pre-test data and measure false alarms removed *and* genuine fault hours lost.
   Retain recent fault evidence/continuity as an input so current weather
   agreement does not erase an ongoing fault alert. No such trained refinement
   was implemented here, and improved F1 is not promised.
3. **Maintain artifact consistency as a separate engineering check.** Preserve
   preprocessing hashes and historical-prefix parity. The earlier controlled
   replay found nearly unchanged development F1 despite feature differences, so
   preprocessing cleanup alone is not an established cure for this temperature
   behaviour. No full-project reconstruction is justified by this test alone.

Do not broadly raise temperature limits, remove temperature flags, or change the
global fault threshold solely to make July look better. That would also affect
the substantial population of labelled temperature-related faults. July has now
been inspected repeatedly, so any follow-up result is exploratory; pre-July
validation and a later untouched period are needed for independent confirmation.

## Verification and limits

- No detector/reason refit, deployment change, health change or label edit.
  Protected model/data hashes remained unchanged.
- All-station delete-the-future checks passed for the new guard feature builder,
  plus two focused unit tests for missing/conflicting evidence and prefix safety.
- Legacy stored stuck flags backfill earlier hours. The guard's protection uses
  the existing current-hour causal primitives instead; passing its prefix check
  does **not** establish causality of the inherited binary predictions.
- Same-hour archived weather and peer residuals were used. Their actual arrival
  times in a live feed remain unverified. No retrospective July label-gate output
  was fed into the guard.
- The minimum-one reason policy remains an evaluated alternative to the saved
  abstaining deployment; this investigation switches neither policy.

Artifacts: `data/eval/july_temperature_investigation/` contains binary/reason
comparisons, July shadow predictions, matched temperature changes and a hash/
protocol audit. Reproduce with
`python scripts/investigate_july_temperature.py --output <new-directory>`.
