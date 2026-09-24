# Health-score regression: alternative correction baselines

All three methods predict continuous 0–100 health scores using a baseline plus a learned regression correction: CatBoost except at 144h, where the saved model uses HGB. No classification metric is used here.

At each horizon, the feature set, iterations and recency settings match the existing model. Each baseline has a separately trained residual target; validation MAE selects correction strength. Training plus validation is then used for refitting. All original split digests match, and roll-forward control predictions reproduce the saved models. Test metrics were independently recalculated from saved predictions.

This is an exploratory extension on previously examined test periods. Settings were originally selected for roll-forward; this is not a separate full hyperparameter search for each baseline. Metrics cover transmitting-origin station-hours. The temporal split is shared by methods at each horizon, but its exact dates and population vary across horizons.

Each table cell is **MAE / RMSE / R²**. MAE and RMSE are health-score points (lower is better); R² is dimensionless (higher is better). Bold marks the method selected by validation MAE, including in the test table.

## Validation

| Horizon | Roll-forward + model | Persistence + model | Trend + model |
|---|---:|---:|---:|
| 1h | **1.19 / 2.39 / 0.954** | 1.20 / 2.39 / 0.954 | 1.25 / 2.42 / 0.953 |
| 3h | 2.45 / 4.29 / 0.853 | **2.44 / 4.21 / 0.858** | 2.58 / 4.28 / 0.853 |
| 6h | **3.52 / 5.24 / 0.784** | 3.53 / 5.22 / 0.786 | 3.81 / 5.46 / 0.765 |
| 12h | 5.09 / 6.93 / 0.644 | **5.07 / 6.83 / 0.655** | 5.18 / 6.94 / 0.643 |
| 24h | 6.51 / 8.98 / 0.468 | **6.43 / 8.73 / 0.497** | 7.97 / 10.83 / 0.226 |
| 48h | 7.92 / 11.08 / 0.337 | **7.73 / 10.91 / 0.358** | 10.77 / 14.69 / -0.165 |
| 72h | 8.41 / 12.31 / 0.293 | **8.26 / 12.14 / 0.312** | 13.05 / 17.86 / -0.488 |
| 96h | 8.96 / 13.39 / 0.243 | **8.77 / 13.13 / 0.272** | 15.18 / 20.71 / -0.811 |
| 120h | 9.36 / 13.74 / 0.255 | **9.12 / 13.82 / 0.246** | 9.79 / 14.80 / 0.136 |
| 144h | 10.02 / 14.68 / 0.165 | **9.70 / 14.10 / 0.231** | 10.41 / 15.46 / 0.075 |
| 168h | 10.39 / 15.18 / 0.096 | **10.29 / 14.83 / 0.137** | 10.44 / 15.42 / 0.067 |

## Test

| Horizon | Roll-forward + model | Persistence + model | Trend + model |
|---|---:|---:|---:|
| 1h | **1.27 / 2.31 / 0.971** | 1.26 / 2.28 / 0.972 | 1.31 / 2.31 / 0.971 |
| 3h | 2.53 / 3.95 / 0.916 | **2.55 / 3.93 / 0.917** | 2.73 / 4.14 / 0.908 |
| 6h | **3.84 / 6.02 / 0.807** | 3.51 / 5.30 / 0.850 | 3.78 / 5.50 / 0.839 |
| 12h | 4.61 / 6.75 / 0.767 | **4.88 / 7.05 / 0.746** | 4.72 / 6.87 / 0.759 |
| 24h | 6.07 / 8.69 / 0.640 | **6.15 / 8.80 / 0.631** | 7.48 / 10.44 / 0.481 |
| 48h | 7.80 / 10.50 / 0.509 | **7.69 / 10.69 / 0.491** | 10.29 / 13.97 / 0.131 |
| 72h | 8.65 / 11.50 / 0.437 | **8.46 / 11.62 / 0.424** | 12.25 / 16.66 / -0.183 |
| 96h | 8.63 / 11.86 / 0.419 | **8.47 / 11.92 / 0.414** | 13.95 / 19.23 / -0.527 |
| 120h | 8.70 / 12.20 / 0.395 | **8.60 / 12.21 / 0.394** | 9.11 / 12.87 / 0.327 |
| 144h | 9.47 / 13.56 / 0.269 | **8.84 / 12.65 / 0.364** | 9.39 / 13.42 / 0.284 |
| 168h | 9.16 / 12.79 / 0.363 | **9.09 / 12.89 / 0.353** | 9.75 / 13.98 / 0.238 |

## Validation choices

- Roll-forward: 1h, 6h.
- Persistence: 3h, 12h, 24h, 48h, 72h, 96h, 120h, 144h, 168h.

## Explanation for the report

> The forecasting model predicts a continuous health score. Earlier summaries emphasised accuracy after converting those predictions into health bands, which did not directly measure numerical forecast error. I have revised the evaluation to report MAE, RMSE and R², and compared three baseline-plus-regression formulations using validation MAE for selection. The underlying task remains health-score regression.

No production model or dashboard was changed by this comparison.
