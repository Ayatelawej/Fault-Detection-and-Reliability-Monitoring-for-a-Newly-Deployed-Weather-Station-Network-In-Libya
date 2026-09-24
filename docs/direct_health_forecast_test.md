# Direct score versus baseline-plus-correction forecast — 24 September 2026

Isolated pilot at 24, 48, 72 and 168 hours. Same CatBoost family, numeric features
(53 except 181 at 168h), iterations, recency weights, station identifier, and
purged temporal partitions as the deployed forecasts. Models fit training data
for validation and training+validation for test scoring. Direct regression learns
future health level and clips its prediction to 0–100, without adding a baseline
or multiplying by a correction weight. Current health remains an input; no
explicit baseline columns are present in the selected feature sets.

The deployed residual forecasts at these horizons use persistence plus a learned
correction. The control exactly reproduced validation MAE and saved test metrics.

| Horizon | Current validation MAE | Direct validation MAE | Current test MAE | Direct test MAE |
|---|---:|---:|---:|---:|
| 24h | 6.43 | 6.49 | 6.15 | 6.88 |
| 48h | 7.73 | 7.86 | 7.69 | 9.30 |
| 72h | 8.26 | 8.48 | 8.46 | 9.34 |
| 168h | 10.29 | 9.91 | 9.09 | 12.03 |

| Horizon | Current test RMSE | Direct test RMSE | Current test R² | Direct test R² |
|---|---:|---:|---:|---:|
| 24h | 8.80 | 10.43 | 0.631 | 0.482 |
| 48h | 10.69 | 12.27 | 0.491 | 0.329 |
| 72h | 11.62 | 12.51 | 0.424 | 0.333 |
| 168h | 12.89 | 16.12 | 0.353 | -0.012 |

Validation favours the existing residual formulation at 24/48/72h, but direct
prediction at 168h (MAE improvement 0.376 points, about 3.65%). The week-ahead
validation winner generalises worse on the inspected test period. Report this
as a validation/test disagreement; do not retrospectively describe the existing
168h model as the winner of the expanded validation comparison. No model was
promoted. The pilot does not support simply removing baselines as a general fix.

Limitations: model settings originally selected for residual learning, not a
full direct-model search; previously inspected test periods; transmitting-origin
station-hours only, not continued-outage projections or July-only performance.
Long-horizon uncertainty may be driven by future incidents and distribution
changes, but this comparison does not establish their causes.

Runner: `scripts/experiment_direct_health_forecast.py`.
Results/predictions/audit: `data/eval/direct_health_forecast_20260924/`.
Protected deployed forecast models and July prediction ledger hashes unchanged.
Saved experimental direct-model wrappers must use `model.predict`, not the
residual `predict_health` method; they are not deployment bundles.
