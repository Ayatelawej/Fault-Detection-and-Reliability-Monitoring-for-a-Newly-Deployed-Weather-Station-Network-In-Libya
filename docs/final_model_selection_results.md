# Final model comparison and selected release

Completed comparisons for the main development dataset. Validation tables come first; test tables are reporting only. The classification comparison covers HGB, EF-HGB and RGFN.

## Selection and interpretation

- Binary: maximise the minimum of temporal validation precision, recall and F1.
- Mechanisms and components: maximise temporal validation micro-F1 separately; minimum-one output on both axes, original episode reasons, all reference fault hours.
- Forecasts: minimise temporal validation MAE per horizon. Band accuracy is a separate secondary metric, never the selection criterion.
- Stratified random holdout preserves binary class proportions. Blocked temporal classification uses February validation and March–April testing, with training on both sides. Forecasting uses its original horizon-purged chronological partitions, not the classification months.
- Normalised regression score (%) = 100 − MAE on the 0–100 health scale. This is not the percentage of correct forecasts. Band accuracy is the percentage in the correct band (<40, 40–<60, 60–<80, >=80).

## Validation results

### Binary fault detection

#### Stratified random holdout

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| HGB | 98.90% | 92.36% | 92.22% | 92.29% |
| EF-HGB | 98.92% | 92.44% | 92.44% | 92.44% |
| RGFN | 98.53% | 89.66% | 89.81% | 89.70% |

#### Blocked temporal holdout

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| HGB | 98.84% | 89.58% | 89.43% | 89.51% |
| EF-HGB | 98.89% | 90.23% | 89.60% | 89.91% |
| RGFN | 98.70% | 88.53% | 88.25% | 88.31% |

### Mechanism and component classification

#### Stratified random holdout: mechanism

| Model | Fault hours | Micro precision | Micro recall | Micro-F1 | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| HGB | 1389 | 92.01% | 86.06% | 88.93% | 82.69% |
| EF-HGB | 1389 | 92.09% | 85.72% | 88.79% | 83.38% |
| RGFN | 1389 | 88.36% | 92.58% | 90.42% | 76.66% |

#### Stratified random holdout: component

| Model | Fault hours | Micro precision | Micro recall | Micro-F1 | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| HGB | 1389 | 96.90% | 71.21% | 82.09% | 73.11% |
| EF-HGB | 1389 | 97.12% | 71.52% | 82.37% | 73.70% |
| RGFN | 1389 | 94.46% | 70.64% | 80.83% | 71.29% |

#### Blocked temporal holdout: mechanism

| Model | Fault hours | Micro precision | Micro recall | Micro-F1 | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| HGB | 577 | 98.62% | 85.76% | 91.74% | 72.09% |
| EF-HGB | 577 | 99.31% | 86.21% | 92.30% | 70.67% |
| RGFN | 577 | 91.84% | 84.41% | 87.97% | 69.63% |

#### Blocked temporal holdout: component

| Model | Fault hours | Micro precision | Micro recall | Micro-F1 | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| HGB | 577 | 91.94% | 86.04% | 88.89% | 67.69% |
| EF-HGB | 577 | 96.20% | 89.41% | 92.68% | 72.37% |
| RGFN | 577 | 93.31% | 89.57% | 91.40% | 75.17% |

### Health forecasting — transmitting-origin hours

| Horizon (h) | Model | MAE | RMSE | R² | Normalised score (%) | Band accuracy (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Linear regression | 1.22 | 2.46 | 0.951 | 98.78 | 93.61 |
| 1 | Ridge | 1.22 | 2.46 | 0.951 | 98.78 | 93.61 |
| 1 | HGB | 1.19 | 2.41 | 0.953 | 98.81 | 93.74 |
| 1 | CatBoost | 1.19 | 2.39 | 0.954 | 98.81 | 93.86 |
| 3 | Linear regression | 2.77 | 4.17 | 0.861 | 97.23 | 85.67 |
| 3 | Ridge | 2.78 | 4.17 | 0.861 | 97.22 | 85.68 |
| 3 | HGB | 2.46 | 4.15 | 0.862 | 97.54 | 86.94 |
| 3 | CatBoost | 2.44 | 4.21 | 0.858 | 97.56 | 87.10 |
| 6 | Linear regression | 4.17 | 5.63 | 0.751 | 95.83 | 79.27 |
| 6 | Ridge | 4.13 | 5.62 | 0.752 | 95.87 | 79.41 |
| 6 | HGB | 3.81 | 5.77 | 0.738 | 96.19 | 80.06 |
| 6 | CatBoost | 3.52 | 5.24 | 0.784 | 96.48 | 81.60 |
| 12 | Linear regression | 5.54 | 7.38 | 0.597 | 94.46 | 73.58 |
| 12 | Ridge | 5.53 | 7.37 | 0.598 | 94.47 | 73.63 |
| 12 | HGB | 5.14 | 6.86 | 0.651 | 94.86 | 75.16 |
| 12 | CatBoost | 5.07 | 6.83 | 0.655 | 94.93 | 75.48 |
| 24 | Linear regression | 6.57 | 8.66 | 0.505 | 93.43 | 70.22 |
| 24 | Ridge | 6.55 | 8.63 | 0.508 | 93.45 | 70.27 |
| 24 | HGB | 6.49 | 8.88 | 0.479 | 93.51 | 70.90 |
| 24 | CatBoost | 6.43 | 8.73 | 0.497 | 93.57 | 70.98 |
| 48 | Linear regression | 7.81 | 10.81 | 0.370 | 92.19 | 67.38 |
| 48 | Ridge | 7.73 | 10.72 | 0.380 | 92.27 | 67.53 |
| 48 | HGB | 7.75 | 11.00 | 0.347 | 92.25 | 67.44 |
| 48 | CatBoost | 7.71 | 10.93 | 0.356 | 92.29 | 67.49 |
| 72 | Linear regression | 8.40 | 12.01 | 0.327 | 91.60 | 66.21 |
| 72 | Ridge | 8.25 | 11.74 | 0.357 | 91.75 | 66.79 |
| 72 | HGB | 8.36 | 12.23 | 0.302 | 91.64 | 65.84 |
| 72 | CatBoost | 8.24 | 12.18 | 0.308 | 91.76 | 66.80 |
| 96 | Linear regression | 8.77 | 12.71 | 0.318 | 91.23 | 65.11 |
| 96 | Ridge | 8.63 | 12.41 | 0.349 | 91.37 | 65.27 |
| 96 | HGB | 8.92 | 13.20 | 0.264 | 91.08 | 64.27 |
| 96 | CatBoost | 8.77 | 13.13 | 0.272 | 91.23 | 65.46 |
| 120 | Linear regression | 9.34 | 13.87 | 0.241 | 90.66 | 64.00 |
| 120 | Ridge | 9.26 | 13.49 | 0.282 | 90.74 | 64.28 |
| 120 | HGB | 9.21 | 13.78 | 0.251 | 90.79 | 65.40 |
| 120 | CatBoost | 9.12 | 13.82 | 0.246 | 90.88 | 66.04 |
| 144 | Linear regression | 10.02 | 14.39 | 0.199 | 89.98 | 61.71 |
| 144 | Ridge | 9.94 | 14.04 | 0.237 | 90.06 | 62.18 |
| 144 | HGB | 9.70 | 14.10 | 0.231 | 90.30 | 62.68 |
| 144 | CatBoost | 9.88 | 14.22 | 0.217 | 90.12 | 61.91 |
| 168 | Linear regression | 10.19 | 14.54 | 0.170 | 89.81 | 61.43 |
| 168 | Ridge | 9.97 | 14.17 | 0.212 | 90.03 | 62.00 |
| 168 | HGB | 10.19 | 14.94 | 0.124 | 89.81 | 60.16 |
| 168 | CatBoost | 9.91 | 14.49 | 0.176 | 90.09 | 62.05 |

## Test results — not used for selection

### Binary fault detection

#### Stratified random holdout

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| HGB | 98.85% | 91.39% | 92.51% | 91.95% |
| EF-HGB | 98.86% | 91.47% | 92.58% | 92.02% |
| RGFN | 98.51% | 89.74% | 89.39% | 89.54% |

#### Blocked temporal holdout

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| HGB | 98.27% | 87.42% | 86.54% | 86.97% |
| EF-HGB | 98.26% | 87.67% | 86.06% | 86.86% |
| RGFN | 97.99% | 84.56% | 85.98% | 85.15% |

### Mechanism and component classification

#### Stratified random holdout: mechanism

| Model | Fault hours | Micro precision | Micro recall | Micro-F1 | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| HGB | 1389 | 92.09% | 85.96% | 88.92% | 83.95% |
| EF-HGB | 1389 | 92.26% | 86.63% | 89.36% | 84.41% |
| RGFN | 1389 | 86.95% | 91.28% | 89.06% | 79.60% |

#### Stratified random holdout: component

| Model | Fault hours | Micro precision | Micro recall | Micro-F1 | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| HGB | 1389 | 96.23% | 71.18% | 81.83% | 73.59% |
| EF-HGB | 1389 | 96.18% | 71.59% | 82.08% | 74.07% |
| RGFN | 1389 | 93.34% | 70.20% | 80.13% | 70.93% |

#### Blocked temporal holdout: mechanism

| Model | Fault hours | Micro precision | Micro recall | Micro-F1 | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| HGB | 1686 | 79.33% | 69.09% | 73.86% | 58.75% |
| EF-HGB | 1686 | 79.37% | 68.77% | 73.69% | 58.57% |
| RGFN | 1686 | 94.49% | 68.10% | 79.15% | 63.40% |

#### Blocked temporal holdout: component

| Model | Fault hours | Micro precision | Micro recall | Micro-F1 | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| HGB | 1686 | 88.50% | 70.95% | 78.76% | 58.19% |
| EF-HGB | 1686 | 91.70% | 72.14% | 80.75% | 61.30% |
| RGFN | 1686 | 84.63% | 69.96% | 76.60% | 57.72% |

### Health forecasting — transmitting-origin hours

| Horizon (h) | Model | MAE | RMSE | R² | Normalised score (%) | Band accuracy (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Linear regression | 1.31 | 2.38 | 0.969 | 98.69 | 93.35 |
| 1 | Ridge | 1.31 | 2.38 | 0.969 | 98.69 | 93.35 |
| 1 | HGB | 1.29 | 2.35 | 0.970 | 98.71 | 93.44 |
| 1 | CatBoost | 1.27 | 2.31 | 0.971 | 98.73 | 93.62 |
| 3 | Linear regression | 3.71 | 7.97 | 0.657 | 96.29 | 85.51 |
| 3 | Ridge | 3.24 | 5.57 | 0.832 | 96.76 | 85.95 |
| 3 | HGB | 2.62 | 4.04 | 0.912 | 97.38 | 88.01 |
| 3 | CatBoost | 2.55 | 3.93 | 0.917 | 97.45 | 88.29 |
| 6 | Linear regression | 4.91 | 8.67 | 0.599 | 95.09 | 79.22 |
| 6 | Ridge | 5.27 | 10.66 | 0.395 | 94.73 | 76.99 |
| 6 | HGB | 3.59 | 5.53 | 0.837 | 96.41 | 82.18 |
| 6 | CatBoost | 3.84 | 6.02 | 0.807 | 96.16 | 83.42 |
| 12 | Linear regression | 5.89 | 9.51 | 0.538 | 94.11 | 74.37 |
| 12 | Ridge | 5.94 | 9.28 | 0.560 | 94.06 | 72.48 |
| 12 | HGB | 5.39 | 8.79 | 0.606 | 94.61 | 75.40 |
| 12 | CatBoost | 4.88 | 7.05 | 0.746 | 95.12 | 77.74 |
| 24 | Linear regression | 7.03 | 10.68 | 0.456 | 92.97 | 68.41 |
| 24 | Ridge | 7.02 | 10.35 | 0.489 | 92.98 | 67.22 |
| 24 | HGB | 6.96 | 10.75 | 0.449 | 93.04 | 67.04 |
| 24 | CatBoost | 6.15 | 8.80 | 0.631 | 93.85 | 71.53 |
| 48 | Linear regression | 10.10 | 17.99 | -0.442 | 89.90 | 60.98 |
| 48 | Ridge | 8.56 | 13.12 | 0.233 | 91.44 | 62.62 |
| 48 | HGB | 7.63 | 10.65 | 0.494 | 92.37 | 64.80 |
| 48 | CatBoost | 7.57 | 10.59 | 0.501 | 92.43 | 64.87 |
| 72 | Linear regression | 9.31 | 13.18 | 0.259 | 90.69 | 59.79 |
| 72 | Ridge | 10.34 | 14.88 | 0.056 | 89.66 | 56.69 |
| 72 | HGB | 8.03 | 11.48 | 0.438 | 91.97 | 63.83 |
| 72 | CatBoost | 8.11 | 11.55 | 0.432 | 91.89 | 62.89 |
| 96 | Linear regression | 10.00 | 13.83 | 0.211 | 90.00 | 57.44 |
| 96 | Ridge | 11.18 | 15.38 | 0.024 | 88.82 | 53.85 |
| 96 | HGB | 8.35 | 11.93 | 0.413 | 91.65 | 62.25 |
| 96 | CatBoost | 8.47 | 11.92 | 0.414 | 91.53 | 61.17 |
| 120 | Linear regression | 12.08 | 19.27 | -0.509 | 87.92 | 54.01 |
| 120 | Ridge | 10.62 | 15.10 | 0.073 | 89.38 | 54.84 |
| 120 | HGB | 8.56 | 12.30 | 0.385 | 91.44 | 62.68 |
| 120 | CatBoost | 8.60 | 12.21 | 0.394 | 91.40 | 62.23 |
| 144 | Linear regression | 9.50 | 13.79 | 0.244 | 90.50 | 59.59 |
| 144 | Ridge | 10.70 | 15.40 | 0.057 | 89.30 | 55.17 |
| 144 | HGB | 8.84 | 12.65 | 0.364 | 91.16 | 61.39 |
| 144 | CatBoost | 8.76 | 12.44 | 0.384 | 91.24 | 61.43 |
| 168 | Linear regression | 11.36 | 19.22 | -0.440 | 88.64 | 56.35 |
| 168 | Ridge | 10.81 | 15.59 | 0.053 | 89.19 | 55.12 |
| 168 | HGB | 9.13 | 13.08 | 0.333 | 90.87 | 61.12 |
| 168 | CatBoost | 12.03 | 16.12 | -0.012 | 87.97 | 50.16 |

## Validation-selected system

Binary: **EF-HGB**, threshold **0.50**. Mechanism: **EF-HGB**. Component: **EF-HGB**. Both reason outputs require at least one code.

| Forecast horizon (h) | Selected model | Validation MAE |
| --- | --- | --- |
| 1 | CatBoost | 1.19 |
| 3 | CatBoost | 2.44 |
| 6 | CatBoost | 3.52 |
| 12 | CatBoost | 5.07 |
| 24 | CatBoost | 6.43 |
| 48 | CatBoost | 7.71 |
| 72 | CatBoost | 8.24 |
| 96 | Ridge | 8.63 |
| 120 | CatBoost | 9.12 |
| 144 | HGB | 9.70 |
| 168 | CatBoost | 9.91 |

The release refits selected configurations using eligible development data through June. The tables above belong to the held-out experiment models, not the final refits.

## July evaluation of the selected refits

### Binary detector

| Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 94.27% | 77.86% | 75.90% | 76.87% |

### Reasons: distinguish the evaluation populations

| Output | Population | Hours | Micro precision | Micro recall | Micro-F1 | Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| mechanism | all_reference_fault_hours | 1701 | 78.37% | 82.58% | 80.42% | 77.57% |
| mechanism | correctly_detected_fault_hours | 1291 | 77.44% | 83.11% | 80.18% | 79.59% |
| mechanism | all_alerts | 1658 | 63.13% | 83.11% | 71.75% | 74.08% |
| mechanism | end_to_end | 13565 | 63.13% | 64.55% | 63.83% | 69.18% |
| component | all_reference_fault_hours | 1701 | 61.71% | 73.71% | 67.18% | 64.63% |
| component | correctly_detected_fault_hours | 1291 | 63.25% | 77.17% | 69.52% | 83.18% |
| component | all_alerts | 1658 | 50.62% | 77.17% | 61.14% | 78.21% |
| component | end_to_end | 13565 | 50.62% | 60.37% | 55.07% | 57.22% |

All-reference-fault evaluation bypasses the binary gate; correctly-detected-fault evaluation excludes missed faults and false alarms. All-alert evaluation includes false alarms but not missed faults. End-to-end includes both. Original episode reasons cover all 1,701 July reference fault hours: no current-hour-reason exclusions are applied.

### Health forecasts

| Horizon (h) | Model | Hours | MAE | RMSE | R² | Normalised score (%) | Band accuracy (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CatBoost | 13799 | 1.37 | 2.38 | 0.973 | 98.63 | 93.94 |
| 3 | CatBoost | 13781 | 2.60 | 4.07 | 0.922 | 97.40 | 88.42 |
| 6 | CatBoost | 13719 | 3.84 | 5.88 | 0.843 | 96.16 | 82.27 |
| 12 | CatBoost | 13603 | 4.90 | 7.51 | 0.756 | 95.10 | 78.01 |
| 24 | CatBoost | 13378 | 6.41 | 9.70 | 0.623 | 93.59 | 73.22 |
| 48 | CatBoost | 12942 | 8.22 | 12.90 | 0.428 | 91.78 | 68.18 |
| 72 | CatBoost | 12575 | 9.44 | 15.13 | 0.293 | 90.56 | 65.97 |
| 96 | Ridge | 12209 | 9.74 | 14.66 | 0.372 | 90.26 | 60.70 |
| 120 | CatBoost | 11766 | 10.80 | 17.11 | 0.170 | 89.20 | 61.11 |
| 144 | HGB | 11360 | 11.55 | 17.88 | 0.109 | 88.45 | 58.93 |
| 168 | CatBoost | 10977 | 11.59 | 17.86 | 0.154 | 88.41 | 57.17 |

Only July origins with an observed future health target are scored. Longer horizons have fewer eligible July origins; unknown August targets are not invented. The dashboard shows 1/3/6/12/24h; selected models and July evaluation extend to 168h.

## Protocol and limitations

Binary comparisons reuse the historical 67-feature pipeline, including the previously identified retrospective stuck indicators. These results do not repair or establish fully causal binary detection. Binary RGFN results are the saved multi-seed summaries.

Reason RGFN is a new single-seed adaptation of the existing one-hour MLP gated architecture to the same head-specific feature views as HGB/EF-HGB (322-feature library). It is not the retired current-hour-target result. All heads use original episode targets; per-head fusion/thresholds and neural checkpoints use validation, never test. Macro-F1 averages only positive-support classes, whose membership differs between periods.

Forecasts share feature scope, recency settings and baseline at each horizon. HGB/CatBoost search 100/200/300 iterations and direct versus residual prediction; linear/Ridge reproduce the earlier validation-selected grid. This is not an exhaustive feature search. At 1h the linear candidates select zero correction, so their output is the standalone baseline.

These development splits and July have been explored previously. July is a retrospective extension, not a newly untouched external validation. Archived weather-reference arrival-time availability remains unverified. Grouped/spaced comparisons are retained as supplementary results. Obsolete exploratory scripts and logistic classification experiments were removed from the submission checkout; a recoverable external snapshot preserves their history.

Release paths: `data/model/final_system_20260924/` and `data/eval/final_system_release_20260924/`. Required earlier forecast templates are retained as reproduction inputs; other historical artifacts are recoverable from the external cleanup snapshot. This document supplies replacement report tables; the user’s Word report has not been rewritten.
