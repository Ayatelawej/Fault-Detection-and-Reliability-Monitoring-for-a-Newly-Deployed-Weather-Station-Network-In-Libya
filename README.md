# Weather Station Network Fault Detection and Reliability Monitoring

This repository contains the hour-level fault-detection and
reliability-monitoring system for a 26-station personal weather-station
network in Libya. The canonical station-hour dataset is frozen through
**2026-06-30** and contains 166,017 rows.

## Input and ownership boundary

The analysis begins from the frozen hourly dataset at
`data/merged/station_hourly_merged.csv`. The underlying public five-minute
observations are available as the *Mozn Weather Dataset: Libya Station-Time
Weather Observations* ([Zenodo DOI 10.5281/zenodo.21811533](https://doi.org/10.5281/zenodo.21811533)).
Source retrieval, raw-grid construction, staging, and canonical merge tooling
are intentionally outside this repository. Those steps belong to the data
provider's operational workflow and are not required to inspect, reproduce, or
run the reliability analysis from the frozen project dataset.

The live detection unit is a **station-hour**. The pipeline
derives multi-label episode labels, expands them into hourly
fault/not-fault targets, and trains a gradient-boosted baseline and a
Reliability-Aware Gated Fusion Network (RGFN). It also contains a separate
causal fault- and outage-risk experiment, plus a transparent causal
station-health score for the operational scorecard.

## Current scope

- `data/labels/episode_labels.csv` is the live label source. It contains
  4,799 episodes: 1,121 fault, 3,533 benign, and 145 borderline-review.
  Borderline-review episodes are excluded from model training and evaluation.
- The hourly detection experiments report every predefined configuration.
  Held-out test metrics are comparisons, not a mechanism for naming a winner.
  A controlled 1/3/5/7/12/24-hour HGB comparison selected the one-hour
  representation under both partitions. The final EF-HGB detector combines a
  full-feature HGB with context and rule-evidence specialists using
  validation-selected weights. It was selected over plain HGB and the
  one-hour Reliability-Aware Gated Fusion Network under both partitions.
  Compact development, window-ablation, and July results are tracked in
  `data/report/`.
- Availability events, network windows, and the rule-derived full/partial
  availability classification are current through June 2026. The legacy full
  outage series remains frozen at 2,398 events; partial outages are descriptive
  sensor-group availability findings, not a supervised-model result.
  The 6/12/24-hour outage- and fault-risk branches construct labels on a
  continuous clock-hour grid. A horizon asks whether a full outage or a
  confirmed labelled fault occurs strictly after the scored hour. Chronological
  partitions are assigned by timestamp and independently horizon-purged.
  The completed direct causal forecasting route uses 1,014 audited, as-of-time
  features, train-derived class weights, and validation-only operating-point
  selection. A separate discrete-time recurrent-event hazard comparison fits
  one 1-hour logistic or shallow-boosted hazard per target, then derives 6/12/24h
  risk under a fixed-score-time-covariate assumption using 45 numeric fields plus
  regularised station indicators.
- `scripts/build_station_health.py` creates an explainable 0–100 current-state
  score from seven-day transmission reliability, sensor completeness, causal
  rule-evidence burden, causal external-reference consistency, and recent
  operational stability. During an active outage it applies a fixed, causal
  exponential duration multiplier rather than forcing the score to zero, so a
  newly dropped station remains distinguishable from a prolonged outage. A
  causal score cap also prevents absent telemetry from looking like improving
  evidence during the same outage. It does not use reviewed labels or
  completed calibration-corroboration findings as live score inputs; its source-truncation audit
  verifies that a later observation cannot alter an earlier score.
- `scripts/build_station_health.py --scorecard` assembles the current
  26-station operational view at a parameterised UTC hour. Its default skips
  terminal-padding hours with no observed transmissions. It infers all five
  transmitting-origin health forecasts, suppresses them during full outages,
  and reports current/trailing causal rule evidence explicitly as a proxy
  rather than mislabelling the retrospective detector or event reason-code
  experiments as live deployment outputs.
- `scripts/run_dashboard.py` presents the delivered system as a chronological
  July 2026 replay because no live station feed is available. It preserves all
  26 stations and uses only the validation-selected EF-HGB binary detector. Its
  three compact operational views move from the network table, to one station,
  to the saved detector evidence behind a predicted fault event.
  Events are segmented from consecutive predicted-positive hours and never
  use reviewed labels or ground-truth episode identifiers.

## Pipeline

| Stage                                               | Front door                              | Purpose                                                                                                                                                                                                          |
| --------------------------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset audit and reliability foundations           | `scripts/build_reliability_foundations.py` | Validate the frozen hourly dataset and rebuild row states plus full, partial, and network-outage artifacts.                                                                                                    |
| Public reference data                               | `scripts/fetch_reference_data.py`       | Fetch/cache active-station ERA5/Open-Meteo reference data.                                                                                                                                                       |
| Detection features                                  | `scripts/rebuild_detection_features.py` | Rebuild statistical, external, spatial, and feature-matrix evidence.                                                                                                                                             |
| Episode labels                                      | `scripts/build_labels.py`               | Produce the live reproducible multi-label episode labels.                                                                                                                                                        |
| Hour-level dataset                                  | `scripts/build_hourly_dataset.py`       | Build short and long past-window tensors.                                                                                                                                                                        |
| Model training                                      | `scripts/train_hourly_detection.py`     | Run baseline, split-comparison, calibration, or RGFN workflows.                                                                                                                                                  |
| RGFN tuning                                         | `scripts/tune_hourly_detection.py`      | Run or resume RGFN architecture/feature experiments.                                                                                                                                                             |
| Station health, forecast, and operational scorecard | `scripts/build_station_health.py`       | Build the causal current-state score; use `--forecast` for the post-hoc five-horizon evaluation, or `--scorecard` for a causal per-station snapshot from the frozen scores and saved transmitting-origin models. |
| Forecast-risk construction and evaluation           | `scripts/evaluate_outage_risk.py`       | Build label/split diagnostics, run the direct causal HGB comparison, or run the compact discrete-hazard comparison after continuous-clock label construction and timestamp purging.                              |
| Report assets                                       | `scripts/generate_report_assets.py`     | Generate methodology or results figures.                                                                                                                                                                         |
| July replay dashboard                               | `scripts/run_dashboard.py`              | Replay saved July operational outputs hour by hour in three simple tables/charts: network state, station health, and predicted-event detector evidence. It performs no inference or fitting.                     |

See [REPO_MAP.md](REPO_MAP.md) for the detailed data flow, module index,
artifact map, concepts, and test coverage.

## Key current artifacts

- `docs/report/Ayat_Elawej_EC499_Report_FINAL.docx` and its PDF export - latest
  report draft, aligned with the frozen EF-HGB, health-forecast, July,
  scorecard, and dashboard results.
- `data/merged/station_hourly_merged.csv` — canonical June-inclusive station-hour data.
- `data/processed/hourly_row_states.parquet` — current availability states.
- `data/processed/availability_events.parquet` and
  `data/processed/network_outage_windows.csv` — current June-inclusive
  availability evidence.
- `data/processed/hourly_availability_classification.parquet` and
  `data/processed/partial_outage_events.parquet` — current rule-derived
  full/partial availability outputs. `availability_report.txt` records their
  definitions, duration statistics, and sensor-group availability.
- `data/processed/station_health_scores.parquet` - causal per-station-hour
  score components, total, band, and input diagnostics. Its companion summary,
  report, invariant hashes, delete-the-future audit, fixed outage-duration
  curve, and hard-zero-versus-progressive comparison tables are tracked
  alongside it.
- `data/processed/station_operational_scorecard.csv` and its report,
  invariant JSON, and delete-the-future audit - the latest 26-station
  operational snapshot. The table contains weighted health-component points,
  five forecast horizons, current/trailing causal fault evidence, and trailing
  reliability fields with explicit scope/null statuses.
- `data/eval/health_forecast/` - readable CSV, JSON, and text outputs from the
  primary health-forecast evaluation: frozen legacy diagnostics;
  validation-only iteration, feature-set, recency, model-family, and alpha
  traces; pooled/regime test metrics; direct-classification metrics and
  confusion matrices; calibration, feature importance, causality audit,
  split/invariant records, and a model manifest. Large prediction and model
  binaries remain excluded.
  `health_forecast_model_manifest.json` is authoritative for the active saved
  models. Persistence, trailing-24-hour trend, and no-new-incident roll-forward
  remain mandatory baselines.
- `data/report/health_forecast_2_to_7_day_accuracy.csv` - compact tracked
  accuracy curve for the supervisor-requested Day-2 through Day-7 extension.
  The reproducible method and the existing-versus-long-feature comparison are
  documented in `docs/health_forecast_2_to_7_day_experiment.md`. The readable
  evaluation traces for both compared feature sets are retained below
  `data/eval/health_forecast_long_horizon/`.
- `data/labels/episode_labels.csv` — live label source for the hourly path.
- `data/report/hgb_history_window_comparison.csv` — controlled HGB history
  ablation; validation selects one hour under both partitions.
- `data/report/one_hour_detection_model_comparison.csv` — EF-HGB, HGB, and
  RGFN on the same one-hour input and exact partitions; validation selects
  EF-HGB.
- `data/report/one_hour_detection_july_comparison.csv` — frozen out-of-time
  comparison of one-hour HGB and the validation-selected EF-HGB.
- `data/hourly_detection/one_hour_final/` - readable final-development model
  comparison, EF-HGB validation grid, metrics, and reports. Saved model and
  tensor binaries remain excluded.
- `data/eval/one_hour_candidate/july_ef_hgb_binary_metrics.csv`, its manifest,
  and its report - the final frozen July EF-HGB evaluation package.
- `data/eval/july_2026_health/`, `july_2026_health_forecast/`, and
  `july_2026_scoring/` - frozen local July replay inputs for health, the selected
  five-horizon forecast policies, and the selected EF-HGB detector. The
  readable July health metrics, manifest, and report are tracked; large replay
  tables remain excluded.
- `outputs/figures/selected_ef_hgb_class_distribution.png`,
  `selected_ef_hgb_confusion_matrices.png`, and `selected_ef_hgb_roc_pr_curves.png` -
  verified development-versus-July evaluation figures.

Incident-risk experiments are future-work investigations rather than deployed
system outputs. Their generated evaluation files are not retained in this
minimal repository; `scripts/evaluate_outage_risk.py` can recreate them when a
deliberate experiment is required.

`data/processed/station_reliability_summary.csv` is rebuilt from the current
June-inclusive availability layer. The uptime/timeline figures and unsuffixed
historical review outputs remain older checkpoint artifacts and are not current
June-wide report sources.

## Running the system

Install dependencies and run the regression suite first:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

The repository starts from the frozen hourly dataset. Rebuild its audit and
availability foundations before any dependent analysis when those artifacts
need to be regenerated:

```powershell
python scripts/build_reliability_foundations.py
```

Rebuild reference-dependent features and the live hourly route only after all
required inputs are available:

```powershell
python scripts/fetch_reference_data.py
python scripts/rebuild_detection_features.py --five-min-dir data/external/mozn_weather_dataset/per_station_weather_data
python scripts/build_labels.py
python scripts/build_hourly_dataset.py --mask-mode per_hour
```

The optional full feature rebuild uses the public Zenodo five-minute files for
exact-hour reference alignment. Download them separately into the ignored path
shown above, or pass another local directory with `--five-min-dir`. It does not
fetch, stage, merge, or rewrite the frozen hourly dataset.

The canonical `per_hour` mask is the reported configuration. The `per_feature`
mode remains available only as an optional mask-sensitivity experiment.

Build, calibrate, compare, and evaluate the final one-hour detector:

```powershell
python scripts/build_hourly_dataset.py --window-hours 1 --window-output data/hourly_detection/one_hour_final/hourly_detection_01h.npz
python scripts/train_hourly_detection.py calibration --tensor data/hourly_detection/one_hour_final/hourly_detection_01h.npz --output-dir data/hourly_detection/one_hour_final --save-selected-models
python scripts/train_hourly_detection.py one-hour-comparison --tensor data/hourly_detection/one_hour_final/hourly_detection_01h.npz --manifest data/hourly_detection/hourly_baseline_split_manifest.csv --baseline-metrics data/hourly_detection/one_hour_final/hourly_short_calibration_metrics.json --output-dir data/hourly_detection/one_hour_final
python scripts/train_hourly_detection.py evidence-fusion --tensor data/hourly_detection/one_hour_final/hourly_detection_01h.npz --manifest data/hourly_detection/hourly_baseline_split_manifest.csv --baseline-metrics data/hourly_detection/one_hour_final/hourly_short_calibration_metrics.json --output-dir data/hourly_detection/one_hour_final
```

Build the current operational health score independently of model training:

```powershell
python scripts/build_station_health.py
```

This writes the 0-100 station-hour score table, summary, source-truncation
audit, and four scorecard figures. It requires only the canonical hourly data
and exact-hour public reference cache; it does not retrain a model.

Evaluate the health-score forecasting comparison without rebuilding the frozen
current-health score:

```powershell
python scripts/build_station_health.py --forecast
```

This runs the frozen post-hoc rebuild on the residual to no-new-incident
roll-forward at 1, 3, 6, 12, and 24 hours. It selects iteration count, core
versus expanded features, recency weighting, HGB/CatBoost/core-only Ridge
family, and alpha (including zero) within transmitting and full-outage regimes
using chronological validation only. It refits accepted learned policies on
train plus validation and evaluates each frozen policy once on test. The
delivered regression, deterioration, trajectory, and future-band results use
stations transmitting at forecast origin; full-outage regression is retained
as a low-confidence supplement because recovery depends on unobserved
maintenance. It writes ignored evaluation/model artifacts and three tracked
figures, including the 1-to-24-hour degradation curve. The strict roll-forward reads no future
observations, outages, faults, labels, or reference values. The saved policy
wrappers require a prepared horizon-origin feature frame containing the causal
baseline columns; they are not a raw-telemetry inference API.

Assemble the combined operational snapshot without retraining:

```powershell
python scripts/build_station_health.py --scorecard
python scripts/build_station_health.py --scorecard --reference-hour 2026-06-30T21:00:00Z
```

The default resolves the latest canonical hour with at least one observed
transmission, because the final two canonical hours are terminal padding. An
explicit timezone-aware whole hour is honoured exactly. Partial outages remain
inside the transmitting forecast scope; full-outage forecasts are marked not
applicable. The output does not join held-out detector/reason-code ledgers.

To reconstruct the corrected label/split diagnostics without fitting a model:

```powershell
python scripts/evaluate_outage_risk.py --target outage
python scripts/evaluate_outage_risk.py --target fault
```

The forecast experiment is a single predeclared six-configuration evaluation.
It reads the historical detector-matrix schema only to document excluded fields;
it does not use detector-matrix values as model inputs. Do not rerun it to tune
against held-out test results.

```powershell
python scripts/evaluate_outage_risk.py --target all --train-risk-models
```

The compact recurrent-event comparison uses the same saved onset partitions. It
fits one 1-hour hazard per target/method and derives each forecast horizon by
holding score-time covariates fixed:

```powershell
python scripts/evaluate_outage_risk.py --target all --train-discrete-hazard
```

It is a comparison experiment, not a replacement for the frozen direct route.
Its fault target remains retrospective-only until a causal all-hour detector
history is available.

The report-asset command defaults to the methodology figures. The results
route performs a prerequisite check because it relies on historical evidence
inputs that are not regenerated by every feature rebuild.

Generate the selected EF-HGB class-distribution, confusion-matrix, ROC, and
precision-recall figures without training:

```powershell
python scripts/generate_report_assets.py --set july-evaluation
```

Run the operational demonstration as a frozen July replay:

```powershell
python -m streamlit run scripts/run_dashboard.py
```

The replay stops at the last July hour with at least one real transmission,
excluding the two all-station terminal-padding hours. It contains no model
performance panel or ground-truth input. Transmitting stations use the selected
learned health forecasts; full-outage stations show a separately labelled
causal continued-outage projection that assumes the outage persists and does
not predict recovery. Predicted fault events expose a compact per-channel table
of detector scores, frozen thresholds, and margins. Short status lines say
whether external-reference and spatial-neighbour evidence were available.
