# Deployed health forecasts: 20 September 2026

This is the preserved previous release. The 24 September comparison and full-June
refits supersede it; see [final model-selection results](final_model_selection_results.md)
and `data/eval/final_system_release_20260924/release_manifest.json`.

The transmitting-origin release selects the baseline-plus-regression formulation
with the lowest validation MAE at each horizon. Roll-forward is selected at 1h
and 6h; persistence is selected at all other horizons. These are learned
corrections to those baselines, not the standalone baselines. CatBoost is used
except at 144h, which uses histogram gradient boosting.

The canonical saved models were replaced after every serialized model reproduced
its experimental test predictions. July's five-horizon dashboard prediction
cache, metrics and model hashes were refreshed without July training or tuning.
The dashboard displays 1, 3, 6, 12 and 24 hours; saved long-horizon models extend
through 168 hours. Full-outage continued-outage projections are unchanged.

Release manifest, exact metrics and recoverable previous artifacts:
`data/eval/health_forecast_release_20260920/`.
Original experiment evaluations remain historical, not active-release scores.
Rerunning the original training workflow does not automatically repeat this
three-baseline selection; use the explicit release workflow for promotion.

## Test results for validation-selected policies

These results cover transmitting-origin station-hours in the original purged
test partitions, not July. Exact dates and row populations vary by horizon.
This is an exploratory comparison on previously inspected periods, not a new
independent validation. Selection uses validation MAE, not the best test result.

| Horizon | MAE | RMSE | R² | Band accuracy | Normalized regression score |
|---|---:|---:|---:|---:|---:|
| 1h | 1.27 | 2.31 | 0.971 | 93.62% | 98.73% |
| 3h | 2.55 | 3.93 | 0.917 | 88.29% | 97.45% |
| 6h | 3.84 | 6.02 | 0.807 | 83.42% | 96.16% |
| 12h | 4.88 | 7.05 | 0.746 | 77.74% | 95.12% |
| 24h | 6.15 | 8.80 | 0.631 | 71.53% | 93.85% |
| 48h | 7.69 | 10.69 | 0.491 | 64.58% | 92.31% |
| 72h | 8.46 | 11.62 | 0.424 | 60.79% | 91.54% |
| 96h | 8.47 | 11.92 | 0.414 | 61.17% | 91.53% |
| 120h | 8.60 | 12.21 | 0.394 | 62.23% | 91.40% |
| 144h | 8.84 | 12.65 | 0.364 | 61.39% | 91.16% |
| 168h | 9.09 | 12.89 | 0.353 | 60.93% | 90.91% |

Band accuracy is the fraction whose predicted and actual scores fall in the same
band: Critical below 40, Degraded 40–<60, Watch 60–<80, Healthy at least 80.
It is a secondary evaluation of regression outputs, not a classification model.

There is no universal regression accuracy percentage. The explicitly defined
normalized regression score here is `100 × (1 − MAE / 100)`, or `100 − MAE`,
because the health scale spans 0–100. It measures average error relative to the
full scale, not the fraction of correct forecasts. It must accompany MAE, RMSE
and R², and must not be described simply as accuracy. A high normalized score
does not imply high explained variance or superiority over a baseline.

## Coordinated outages

Removed the midnight-versus-other timing subtypes. All 430 coordinated events
and 47 coordinated windows retain their original membership, timestamps and
durations. Legacy subtype strings are accepted only for backwards compatibility.
Migration backups are in `data/eval/coordinated_outage_migration_20260920/`.

Suggested replacement for the report:

> Overlapping outages across multiple stations are marked as coordinated
> outages. This indicates shared timing, not a confirmed common cause.

The user's Word report was not edited by this release.

## Why spatial evidence uses three channels

The implemented neighbour layer compares pressure, temperature and dew point
against a median of available neighbours within 200 km (at least two required).
These channels are a practical choice for broad-scale consistency checks, though
elevation and local weather still matter. This was not demonstrated to be an
optimal channel subset by an ablation experiment.

Wind is sensitive to local exposure and terrain; solar irradiance varies with
clouds and shading. Applying the same broad-radius median test indiscriminately
could mistake genuine differences for faults. External gridded comparisons are
a different reference, aligned to each station's location and time; their wider
channel coverage does not establish that nearby station values should match.
Spatial wind/solar checks are possible, but require channel-specific design and
validation, which this release does not add.

Physical context: [WMO measurement guide](https://www.weather.gov/media/epz/mesonet/CWOP-WMO8.pdf)
and [observed spatial variability in solar irradiance](https://arxiv.org/abs/2307.06980).
