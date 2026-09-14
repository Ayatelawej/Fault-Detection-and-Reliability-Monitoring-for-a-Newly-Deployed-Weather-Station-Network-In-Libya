# July: selective fallback and weather-aware reason-only tests

Neither exploratory test achieved over 70% on both reason categories. No
production model, alert, reason ledger or dashboard was changed. Binary detection
remains exactly the frozen July output: **77.02% F1**.

## July results

All reason results are end-to-end micro-F1 on the same 12,722 hours: 11,864
reference-normal hours and 858 faults with resolved current-hour reasons. The 843
unknown-reason fault hours remain excluded from reason metrics, not from fault
detection. Assignment counts below include all 1,833 detector alerts, including
unknown-reason faults.

| Policy | Mechanism F1 | Component F1 | Alerts assigned a mechanism / component |
|---|---:|---:|---:|
| Current deployed thresholds | 65.59% | 71.83% | 638 / 862 |
| Existing always-assign comparison | 71.03% | 69.33% | 1,833 / 1,833 |
| Pre-July-selected selective fallback | 65.59% | 71.83% | 638 / 862 |
| Pre-July-selected weather refinement | 65.93% | 66.85% | 586 / 605 |
| Weather refinement plus selected fallback | 65.93% | 66.85% | 586 / 605 |

## Test 1: selective fallback

Existing above-threshold codes are preserved. For an otherwise empty alerted
category, a fallback can assign the highest finite score only if it meets both
a score floor and a top-versus-second score margin. Normal detector hours never
receive codes. Scores are not calibrated diagnostic probabilities.

The fixed grid used score floors 0, 0.3, 0.5 and 0.7, and margins 0, 0.1, 0.2 and
0.3, plus the unchanged/no-fallback option. Each category was selected separately
using cascade micro-F1 on the original pre-July random validation membership,
with the actual saved binary detector and the corresponding train-only reason
heads. Neither base model was refitted.

**No fallback won for either category.** Selected validation F1 was 91.01%
mechanism and 91.20% component. These are policy-selection scores, not new
independent test results. Transferring this selected no-change policy to the final
June-refitted heads reproduces July's current deployment exactly.

## Test 2: weather-aware reason refinement, not alert suppression

Two small HGB classifiers refine only the statistical-anomaly mechanism and the
thermo-hygrometer component. They use same-category base reason scores plus 20
weather/history features: recent same-hour temperature departures, station-minus-
reference and station-minus-neighbour residual departures, recent temperature
changes, prior evidence continuity, coverage and calendar phase. They do not
consume station IDs, episode IDs, true labels at inference, or July target gates.
Other reason-head scores are unchanged. Unlike the previous weather safeguards,
these classifiers never cancel or add fault alerts.

The classifiers were trained within the pre-July validation alerts with resolved
reason targets or normal references. Three event-disjoint folds generated
out-of-fold refinement scores within that subset. Fixed blending weights
0, 0.25, 0.5 and 1 were compared; zero keeps the original head unchanged. For
nonzero weights the threshold candidates were the original threshold and
0.3, 0.5, 0.7 and 0.9. Selection used validation cascade micro-F1, including
missed resolved faults. The classifiers were then refitted on the eligible
validation alerts, and all settings frozen before loading July evaluation labels.

Selected mechanism refinement: 25% new weather score / 75% original score,
threshold 0.7; selection F1 91.42%. Selected component refinement: 100% new score,
threshold 0.9; selection F1 91.40%.

On July, mechanism precision rose from 81.41% to 85.43%, but recall fell from
54.92% to 53.67%. Component precision rose from 77.26% to 84.95%, but recall fell
from 67.12% to 55.11%. Thus the weather heads became more selective about giving
reasons, especially components. The component recall loss outweighed the precision
gain. These are missing explanations, **not removed fault alerts**.

The combined variant is a fixed composition of the two separately selected
methods, not an additional July-tuned search. Because fallback selection chose
no fallback, its predictions equal weather refinement alone.

## Interpretation and limits

- No method was tuned to July and no second parameter search followed these
  results. July has nonetheless already been examined repeatedly, so this is
  exploratory evidence, not an untouched confirmatory test.
- The original random base-model split shares fault events across train and
  validation. Grouping the weather classifier's internal folds does not remove
  the inherited overlap in its base scores.
- Refinements selected using split-specific development heads are transferred to
  the final all-development reason heads. Their score distributions and original
  thresholds can differ. This small test does not establish that weather context
  cannot help; it tests this particular bounded refinement and transfer procedure.
- The unchanged option won the fallback grid; this is not proof that every
  possible fallback rule is inferior.
- Reference labels remain evidence-derived rather than independently confirmed
  hardware diagnoses. Archived weather arrival times remain unverified.
- The weather-feature future-truncation check passed for every station at
  15 July 12:00 UTC. It does not certify the inherited binary pipeline as causal.

## Artifacts and verification

Runner: `scripts/experiment_reason_only_refinements.py`.
Output: `data/eval/july_reason_only_refinements/` contains `summary.csv`,
`development_selection.csv`, `predictions.parquet`, isolated
`weather_reason_refiners.joblib`, and `audit.json`.

Protected input/model/dashboard hashes remained unchanged. The saved deployment
reason strings were reproduced exactly; binary probabilities and decisions were
matched to the original ledger. Four new unit tests and nine existing relevant
tests passed (13 total), covering fallback margins/missing scores, multiple-code
preservation, gate enforcement, target-only refinement, input preservation and
existing weather/reason integration.

Reproduce with a fresh output directory:

```powershell
python scripts/experiment_reason_only_refinements.py --output data/eval/july_reason_only_refinements_new
python -m pytest tests/test_reason_only_refinements.py tests/test_temperature_shadow_guard.py tests/test_weather_notes_and_continuity.py tests/test_final_reason_codes.py -q
```

Recommendation: do not promote either tested change as a full-system improvement.
Keep the current release unchanged pending an explicit policy decision.
