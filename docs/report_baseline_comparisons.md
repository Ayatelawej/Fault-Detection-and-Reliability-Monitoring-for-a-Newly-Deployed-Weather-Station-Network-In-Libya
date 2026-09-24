# Additional report baselines — 24 September 2026

## Split names and scope

Use **stratified random holdout** and **blocked temporal holdout** in the main
comparison. The former preserves binary class proportions; it can share event
groups between partitions. The latter uses February validation, March–April
testing and eligible training observations on both sides, not forward-only
validation. Grouped/spaced tests are not redundant with random assignment: they
address event separation. Preserve their results as supplementary experiments;
do not rename the reason-specific grouped split as the binary spaced split.

## Binary logistic baseline

Logistic regression uses the same historical 67 columns and saved partitions as
the original binary comparison. Median imputation and standardisation fit only
training rows. Validation maximises the minimum of precision, recall and F1
over C = 0.1, 1, 10, the original seven fault weights, and thresholds 0.30–0.80.
Test data does not select settings. Convergence warnings cause the runner to fail.

| Split | Validation F1 | Test accuracy | Test precision | Test recall | Test F1 |
|---|---:|---:|---:|---:|---:|
| Stratified random | 87.51% | 98.13% | 86.62% | 87.19% | 86.90% |
| Blocked temporal | 90.48% | 98.09% | 86.21% | 84.93% | 85.57% |

The blocked logistic model has validation minimum(P,R,F1) = 90.33%, compared
with 89.60% for the main blocked EF-HGB comparison. Logistic therefore wins
that validation rule in the expanded binary comparison, although its test F1
is lower (85.57% versus EF-HGB 86.86%). Do not claim EF-HGB remains the validation
winner among all newly tested methods. Deployment was not changed.

This is a historical matched-feature comparison, not a repaired causal detector:
the original retrospective stuck-indicator limitation affects these binary inputs.

## Episode-reason baselines

Plain HGB uses the full allowed feature view for each head, with the same fixed
HGB settings, training rows and event/class weighting as the EF-HGB full arm.
Logistic uses the same head-specific view, training-only imputation/scaling,
the same sample weighting, and validation-selected C. Per-code thresholds use
validation event-weighted F1; both axes use minimum-one output. Train and test
targets are the original episode labels. All test reference faults are included.
These classifiers are assessed separately from binary detection.

| Split | Model | Mechanism micro-F1 | Component micro-F1 |
|---|---|---:|---:|
| Stratified random | Logistic regression | 88.58% | 79.88% |
| Stratified random | HGB | 88.92% | 81.83% |
| Stratified random | EF-HGB | 89.36% | 82.08% |
| Blocked temporal | Logistic regression | 77.45% | 74.11% |
| Blocked temporal | HGB | 73.86% | 78.76% |
| Blocked temporal | EF-HGB | 73.69% | 80.75% |

EF-HGB leads both random-test outputs and the temporal component test. Logistic
leads the temporal mechanism test. These test comparisons do not constitute
validation-based selection of a new combined deployment. EF-HGB figures reuse
the verified matching episode-target experiment predictions, not a new fit.
Detailed macro/per-class scores and test rows are retained in the output files.

## What does “detected fault hours only” mean?

All three conditions—episode-level targets, minimum-one on both outputs, and
detector-positive hours—are implemented in the active July release. Its output
assignments were already restricted to alerts; the earlier 73.69%/80.75% figures
instead evaluated all reference faults in the March–April test.

For the matching historical split gates, EF-HGB reason scores on **correctly
detected faults only** are:

| Split | Correctly detected fault hours | Mechanism micro-F1 | Component micro-F1 |
|---|---:|---:|---:|
| Stratified random | 1,286 | 89.20% | 81.72% |
| Blocked temporal | 1,450 | 71.58% | 81.72% |

The detector gates are the saved random EF-HGB at 0.30 and the original blocked
EF-HGB at its February-selected 0.35 threshold. The blocked gate is not the later
main-model-comparison EF-HGB at 0.50; keep this provenance explicit.

The **active July refit**, evaluated against original episode reasons, gives:

| Evaluation population | Hours | Mechanism micro-F1 | Component micro-F1 |
|---|---:|---:|---:|
| Correctly detected reference faults | 1,361 | 79.44% | 68.49% |
| All detector alerts, including false alarms | 1,833 | 69.46% | 59.01% |
| Full detector-to-reason ledger, including missed faults | 13,565 | 63.39% | 54.34% |

No unknown-current-hour exclusions are used here. All 1,701 July reference fault
hours have original episode reasons. Correctly detected-fault evaluation excludes
340 missed fault hours and 472 false alerts by definition, not because their
reason predictions were incorrect. July is a previously explored extension.

## Ridge and ordinary linear forecast baselines

Both baselines use the same horizon-specific numeric features, station identity,
recency weighting and purged timestamp partitions as the deployed nonlinear
forecasts. Numeric missing-value imputation and standardisation fit training
data only, with one-hot station encoding. Validation MAE selects direct prediction
or a correction to the deployed horizon's baseline. For corrections, validation
also selects the correction weight; Ridge additionally selects regularisation
strength from 0.1, 1, 10, 100 and 1000. Final baseline models refit training plus
validation, then score the original test partition. Outputs are clipped to 0–100.

These are regressors, not band classifiers. At 1h both linear candidates select
a zero correction weight, so their reported output is the standalone baseline.
Features and recency settings were originally selected for nonlinear models;
this is not an exhaustive linear-specific feature search.

### Validation MAE (selection criterion)

| Horizon | Deployed nonlinear | Ordinary linear | Ridge | Lowest MAE in this comparison |
|---|---:|---:|---:|---|
| 1h | 1.188 | 1.223 | 1.223 | Deployed |
| 3h | 2.441 | 2.773 | 2.778 | Deployed |
| 6h | 3.518 | 4.171 | 4.134 | Deployed |
| 12h | 5.070 | 5.539 | 5.531 | Deployed |
| 24h | 6.432 | 6.573 | 6.547 | Deployed |
| 48h | 7.734 | 7.813 | 7.731 | Ridge |
| 72h | 8.262 | 8.400 | 8.252 | Ridge |
| 96h | 8.773 | 8.770 | 8.632 | Ridge |
| 120h | 9.116 | 9.336 | 9.261 | Deployed |
| 144h | 9.701 | 10.016 | 9.939 | Deployed |
| 168h | 10.289 | 10.188 | 9.974 | Ridge |

### Test results

The deployed nonlinear family is CatBoost except at 144h, where it is HGB.

| Horizon | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| 1h | Deployed | 1.27 | 2.31 | 0.971 |
| 1h | Ordinary linear | 1.31 | 2.38 | 0.969 |
| 1h | Ridge | 1.31 | 2.38 | 0.969 |
| 3h | Deployed | 2.55 | 3.93 | 0.917 |
| 3h | Ordinary linear | 3.71 | 7.97 | 0.657 |
| 3h | Ridge | 3.24 | 5.57 | 0.832 |
| 6h | Deployed | 3.84 | 6.02 | 0.807 |
| 6h | Ordinary linear | 4.91 | 8.67 | 0.599 |
| 6h | Ridge | 5.27 | 10.66 | 0.395 |
| 12h | Deployed | 4.88 | 7.05 | 0.746 |
| 12h | Ordinary linear | 5.89 | 9.51 | 0.538 |
| 12h | Ridge | 5.94 | 9.28 | 0.560 |
| 24h | Deployed | 6.15 | 8.80 | 0.631 |
| 24h | Ordinary linear | 7.03 | 10.68 | 0.456 |
| 24h | Ridge | 7.02 | 10.35 | 0.489 |
| 48h | Deployed | 7.69 | 10.69 | 0.491 |
| 48h | Ordinary linear | 10.10 | 17.99 | -0.442 |
| 48h | Ridge | 8.56 | 13.12 | 0.233 |
| 72h | Deployed | 8.46 | 11.62 | 0.424 |
| 72h | Ordinary linear | 9.31 | 13.18 | 0.259 |
| 72h | Ridge | 10.34 | 14.88 | 0.056 |
| 96h | Deployed | 8.47 | 11.92 | 0.414 |
| 96h | Ordinary linear | 10.00 | 13.83 | 0.211 |
| 96h | Ridge | 11.18 | 15.38 | 0.024 |
| 120h | Deployed | 8.60 | 12.21 | 0.394 |
| 120h | Ordinary linear | 12.08 | 19.27 | -0.509 |
| 120h | Ridge | 10.62 | 15.10 | 0.073 |
| 144h | Deployed | 8.84 | 12.65 | 0.364 |
| 144h | Ordinary linear | 9.50 | 13.79 | 0.244 |
| 144h | Ridge | 10.70 | 15.40 | 0.057 |
| 168h | Deployed | 9.09 | 12.89 | 0.353 |
| 168h | Ordinary linear | 11.36 | 19.22 | -0.440 |
| 168h | Ridge | 10.81 | 15.59 | 0.053 |

The deployed nonlinear model has lower test MAE at every horizon in this
comparison. Nevertheless, Ridge has lower validation MAE at 48, 72, 96 and 168h.
The first two differences are very small (0.003 and 0.011 points). Do not claim
the deployed models win the expanded validation comparison at every horizon,
or use test results to retroactively change the stated selection rule. No model
was promoted. The earlier direct-CatBoost pilot is separate; at 168h its validation
MAE of 9.913 is lower than this Ridge result, so this table is not a global winner
ranking across every experiment ever run.

The original splits and deployed test metrics were reproduced before comparison.
All evaluations are transmitting-origin station-hours on previously inspected
development periods, not new independent July validation.

## Artifacts

- `data/eval/report_classification_baselines_20260924/`: binary metrics/grid/models,
  reason metrics/per-label counts/predictions, selections and protected-file audit.
- `data/eval/episode_reason_alert_scopes_20260924/`: July populations and metrics.
- `data/eval/report_linear_forecasts_20260924/`: forecast baselines, separate
  validation grid, predictions and serialized experimental models.

No production model, July prediction ledger, dashboard or Word report was changed
by these comparison runs. The tables here are report-ready additions, not an
automatic edit of the illustrated report.
