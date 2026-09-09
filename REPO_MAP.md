# Repository map

This is a map of the active analysis repository after the input-boundary cleanup (2026-09-05). It is organised around the pipeline, so it can be read as both a table of contents and an architecture reference.

The live deployed modelling unit is a **station-hour**, not an episode. The repository has twelve meaningful executable entry-point scripts in `scripts/`: the delivered analysis pipeline and explicitly retained experiment/report runners.

> **Input boundary:** this repository starts from the frozen 26-station hourly dataset. Source retrieval, raw-grid construction, staging, and canonical merge code are intentionally excluded.

## 1. Overview

The project monitors a 26-station personal weather-station network. It starts from a frozen hourly dataset, audits those observations, derives rule and reference-data evidence, labels fault episodes, expands those labels into station-hour targets, and trains hour-level fault detectors. A separate reliability branch provides full/partial availability monitoring, a causal current station-health score, a 1/3/6/12/24-hour transmitting-station health-forecast comparison, and future-risk experiments.

The main path is:

```text
frozen hourly dataset
  -> audit + full/partial/network availability
  -> ERA5/Open-Meteo reference cache
  -> detection features and rule evidence
  -> episode labels
  -> hour-level tensors
  -> baseline / RGFN training and tuning
  -> causal station-health scorecard and forecast comparison
  -> causal forecast-risk construction/evaluation and report figures
```

The public five-minute source dataset is available at DOI `10.5281/zenodo.21811533`. Its acquisition and assembly workflow is outside this university-facing analysis repository. `scripts/build_reliability_foundations.py` begins with the frozen hourly CSV and rebuilds the audit and availability artifacts without modifying that input.

## 2. Pipeline

### Stage 1 — Frozen dataset audit and availability

**Purpose.** Validate the frozen station-hour input, classify row states, and rebuild full-outage events, coordinated network windows, partial-outage events, structural-gap evidence, and the per-station reliability summary.

**Front door.** `scripts/build_reliability_foundations.py`

**Main modules.** `src/features/run_data_audit.py` and `row_state.py` build quality and row-state outputs; `src/availability/build_availability_events.py` constructs full/partial availability; `build_network_outage_windows.py` constructs coordinated windows; and `build_station_reliability_summary.py` writes the station rollup.

**Inputs and outputs.** It reads `data/merged/station_hourly_merged.csv` and `station_registry.csv`. It never rewrites the frozen hourly dataset. It writes audit/missingness tables, hourly row states, full-outage events, network windows, partial-outage outputs, a structural-gap audit, the availability report, and the current per-station summary.

**Tests.** `tests/test_data_audit.py`, `tests/test_availability.py`, and `src/config/test_paths.py`

### Stage 2 — Public reference data

**Purpose.** Fetch and cache public ERA5/Open-Meteo reference observations for active registered stations, so station measurements can later be compared with an external reference.

**Front door.** `scripts/fetch_reference_data.py`

**Main modules.** The front door performs the fetch; `src/rules/config.py` supplies reference variables, cache location, date range, and comparison settings.

**Inputs and outputs.** It reads the tracked station registry and checks the merged-data header. It writes `data/external/reference_hourly/<station>.parquet` and `data/external/reference_manifest.csv`, both ignored local cache artifacts.

**Tests.** `tests/test_external_fetch.py`, `tests/test_external_conventions.py`

**Current scope.** The active reference-cache contract covers 2025-06-15 through 2026-07-31 (9,888 expected UTC hours). Active caches must exactly match that complete hourly index. Development statistics and model choices remain frozen through 2026-06-30; the July reference rows support the separate out-of-time evaluation and do not refit the development system. `IJANZO4` is a retired station retained as a separate historical cache. Extending beyond July requires an explicit update to the reference date range and its dynamically derived expected index.

### Stage 3 — Feature construction

**Purpose.** Rebuild the evidence used for detection and modelling from canonical data: statistical scores, events, episodes, clusters/review queue, external residuals, spatial residuals, and the feature matrix.

**Front door.** `scripts/rebuild_detection_features.py`

**Main modules.** `src/features/rebuild.py` is the orchestrator. It calls `src/rules/score.py`, `events.py`, `episodes.py`, `clustering.py`, `review_queue.py`, `external_residuals.py`, `external_features.py`, `spatial_residuals.py`, `spatial_offsets.py`, and `feature_matrix.py`.

**Inputs and outputs.** It reads the frozen hourly data, station registry, the public-reference cache, and separately downloaded public five-minute observations. It writes statistical score/event/episode/cluster artifacts, a review queue, external/spatial feature tables, and `data/features/feature_matrix.parquet`. It contains no source-fetch, staging, or canonical-merge capability. Current feature construction is retrospective, so it must not be represented as prospective live feature generation.

**Tests.** `tests/test_statistical_anomaly.py`, `tests/test_external_residuals.py`, `tests/test_external_features.py`, `tests/test_spatial_residuals.py`, `tests/test_spatial_offsets.py`, `tests/test_feature_matrix.py`, and `tests/test_stuck_confirmation.py`

### Stage 4 — Statistical detection evidence

**Purpose.** Turn physical-limit, robust-z-score, Isolation Forest, and rolling-variance signals into channel events and station episodes, then add contextual, external, and spatial evidence. This is a substage of feature construction rather than an extra script.

**Front door.** `scripts/rebuild_detection_features.py` for score/event/episode construction; `scripts/build_labels.py` applies the final statistical gate during labelling.

**Main modules.** `src/rules/physical_limits.py`, `detectors/robust_zscore.py`, `detectors/isolation_forest.py`, `detectors/rolling_variance.py`, `score.py`, `events.py`, `episodes.py`, `statistical_gate.py`, `external_residuals.py`, and `spatial_residuals.py`.

**Inputs and outputs.** The generated evidence is carried into feature tables and, during the next stage, into statistical-anomaly decisions and ranked review files.

**Tests.** `tests/test_statistical_anomaly.py`, `tests/test_statistical_gate.py`, `tests/test_external_residuals.py`, and `tests/test_spatial_residuals.py`

### Stage 5 — Episode labelling

**Purpose.** Build the reproducible episode labels that the live hourly system uses. It recomputes scores, events, and episodes; applies mechanism/component rules; uses the contextual statistical gate; and resolves eligible calibration cases with calibration-offset corroboration.

**Front door.** `scripts/build_labels.py`

**Main modules.** `src/rules/labelling.py`, `statistical_gate.py`, `calibration_corroboration.py`, `score.py`, `events.py`, `episodes.py`, `external_offsets.py`, and `channel_handlers.py`.

**Inputs and outputs.** It reads canonical merged data, external and spatial residual artifacts, and an ignored frozen legacy label inventory solely to create a historical crosswalk. Its live output is the tracked `data/labels/episode_labels.csv`. It also writes `data/labels/calibration_offset_corroboration.csv` plus ignored crosswalk and review files.

**Tests.** `tests/test_labelling.py`, `tests/test_statistical_gate.py`, `tests/test_calibration_corroboration.py`, and `tests/test_stuck_confirmation.py`

**Important distinction.** The live label file contains 4,799 episodes: 1,121 `fault`, 3,533 `benign`, and 145 `borderline_review`. It is the only label source used by the hourly dataset/model path. Retired reconciliation evidence is not part of this minimal repository.

### Stage 6 — Hour-level detection dataset

**Purpose.** Expand episode labels across the full station-hour record and create past-only feature windows for binary fault detection. The window assembly itself is past-only, while some underlying detection features are retrospective snapshots.

**Front door.** `scripts/build_hourly_dataset.py`

**Main modules.** `src/model/hourly_detection.py` builds labels and tensors; `src/model/feature_spec.py` defines the feature and target vocabulary.

**Inputs and outputs.** It reads canonical merged data, `data/features/feature_matrix.parquet`, and `data/labels/episode_labels.csv`. Its retained reason-code route writes the historical seven-hour and 49-hour tensors. The isolated `--window-hours 1 --window-output ...` route builds the final binary detector tensor without replacing those inputs. Generated tensors are ignored by Git because they are large.

**Tests.** `tests/test_hourly_detection_dataset.py`

### Stage 7 — Hour-level model training

**Purpose.** Train the binary gradient-boosted baseline and RGFN, compare split strategies, calibrate weights/operating thresholds, and run a conditional multi-label reason-code experiment for fault hours. The delivered reason-code result is mechanism-only and retrospective at the connected-event level; component outputs remain exploratory and are not claimed as a deployed capability. All predefined configurations are reported; held-out test metrics do not select or name a winner.

**Front door.** `scripts/train_hourly_detection.py` with one of `baseline`, `split-comparison`, `calibration`, `one-hour-comparison`, `evidence-fusion`, `one-hour-july`, `rgfn`, or `reason-codes`.

**Main modules.** `src/workflows/train_hourly_baseline.py`, `train_hourly_split_comparison.py`, `train_hourly_calibration.py`, and `train_hourly_rgfn.py`; `src/model/hourly_baseline.py`, `hourly_calibration.py`, `hourly_rgfn.py`, and `hourly_rgfn_training.py`.

**Inputs and outputs.** Binary routes read ignored hourly tensors and write split manifests, metrics, reports, and model bundles under `data/hourly_detection/`. The controlled HGB ablation covers 1/3/5/7/12/24-hour inputs and selects one hour from validation under both partitions. The one-hour comparison gives HGB and RGFN the same engineered input and exact memberships. RGFN uses an MLP sensor-context branch, independent rule-evidence branch, and learned reliability gate. The final EF-HGB instead fits full-feature, context, and rule-evidence HGB branches on training only, then selects their convex fusion weights and threshold from validation only. EF-HGB is validation-selected over HGB under both partitions. `one-hour-july` applies the frozen random EF-HGB to the exact pre-existing July evaluation population without refitting or threshold changes. Compact results are retained in `data/report/hgb_history_window_comparison.csv`, `one_hour_detection_model_comparison.csv`, and `one_hour_detection_july_comparison.csv`. The separate retrospective reason-code experiment retains its original seven-hour development input and does not feed the July dashboard.

**Tests.** `tests/test_hourly_baseline.py`, `tests/test_hourly_calibration.py`, `tests/test_rgfn_hourly.py`, and `tests/test_front_door_scripts.py`

### Stage 8 — RGFN tuning

**Purpose.** Run or resume the RGFN architecture/feature hyperparameter experiments while keeping validation selection separate from the final one-time test evaluation. The report retains the complete predefined-configuration table rather than using a test metric to name an arm.

**Front door.** `scripts/tune_hourly_detection.py` with `run` or `resume`.

**Main modules.** `src/workflows/tune_hourly_detection.py`, `resume_hourly_tuning.py`, `src/model/hourly_rgfn_tuning.py`, `hourly_rgfn_tuning_features.py`, and `hourly_rgfn_tuning_logistic.py`.

**Inputs and outputs.** It reads the short tensor plus the baseline split manifest, calibration metrics, and earlier RGFN results. It writes tuning tables, reports, validation/screen files, and checkpoints below `data/hourly_detection/models/rgfn_hourly_tuning/`.

**Tests.** `tests/test_rgfn_tuning.py`, `tests/test_rgfn_tuning_logistic.py`, and `tests/test_front_door_scripts.py`

### Stage 9 — Causal station-health scorecard

**Purpose.** Build an auditable 0-100 current-state score for each station-hour. It is a fixed weighted definition rather than a trained model: availability (30), sensor completeness (20), causal rule-evidence burden (25), causal external-reference consistency (15), and recent operational stability (10).

**Front door.** `scripts/build_station_health.py`

**Main modules.** `src/availability/health_score.py` builds the continuous raw station clock, derives causal detector and exact-hour reference evidence through shared forecast-source helpers, calculates the five normalized components, applies a fixed causal duration multiplier and a no-improvement cap during active full or partial outages, validates source truncation, writes score artifacts, and renders scorecard figures.

**Inputs and outputs.** It reads the tracked canonical station-hour dataset and the ignored exact-hour public-reference cache. It writes tracked `data/processed/station_health_scores.parquet`, per-station summary/report/invariant/audit artifacts, a fixed duration-curve table, old-versus-new comparison tables, and five figures in `outputs/figures/`. It deliberately does not consume reviewed episode labels, completed availability-event tables, or completed calibration-corroboration findings as score inputs. Calibration corroboration is used only to select a retrospective confirmed-offset sanity-check trajectory for the figure.

**Tests.** `tests/test_availability.py` and `tests/test_front_door_scripts.py`

### Stage 10 — Health-score forecasting

**Purpose.** Forecast health at 1, 3, 6, 12, and 24 hours as a residual to the strict no-new-incident roll-forward. The delivered scope is stations transmitting at forecast origin, including partial outages. Full-outage origins are selected separately and retained only as a low-confidence supplement because recovery depends on unobserved maintenance. Validation may reject a learned correction through alpha zero or a deterministic policy. A separate supervisor-requested experiment extends evaluation to exact 48/72/96/120/144/168-hour horizons without replacing the deployed models. This is an explicitly post-hoc, purged chronological evaluation, not a replacement for the current health score.

**Front door.** `scripts/build_station_health.py --forecast`; the isolated extension uses `--long-horizon-forecast --long-horizon-features {current,extended}`.

**Main modules.** `src/availability/health_forecast.py` builds a 53-numeric-feature core, the original expanded historical union, and a separately bounded long-horizon representation containing causal 7/14/30-day summaries and target-calendar context. It constructs exact continuous-clock targets and H-purged chronological splits; compares absolute-error HGB, MAE CatBoost, and core-only Ridge residual models; selects iterations, feature union, recency weighting, model family, and alpha on validation only; refits on train plus validation; and fits transmitting-origin deterioration, trajectory, and health-band classifiers. It also prepares target-free causal inference frames for saved policies and writes selected-candidate permutation importance, transmitting-only calibration, cross-horizon degradation tables, bounds and delete-the-future audits, figures, model hashes, and a deployment manifest. `src/availability/risk_eval.py` supplies the generic timestamp-partition/purge and regression-metric primitives.

**Inputs and outputs.** It reads the frozen causal `data/processed/station_health_scores.parquet` plus static station elevation from the canonical dataset. It does not rewrite health, availability, labels, detection models, or reason-code models. Ignored artifacts below `data/eval/health_forecast/` contain legacy diagnostics, the master comparison, validation traces, pooled/regime metrics, classification/confusion outputs, predictions, transmitting-population checks, cross-horizon degradation tables, calibration, feature audit, split digests, invariants, and `health_forecast_model_manifest.json`. The long-horizon extension writes equivalent isolated artifacts below ignored `data/eval/health_forecast_long_horizon/` and models below `data/model/health_forecast_long_horizon/`; its compact accuracy result is tracked at `data/report/health_forecast_2_to_7_day_accuracy.csv` and interpreted in `docs/health_forecast_2_to_7_day_experiment.md`. Each saved forecast policy consumes a prepared causal horizon-origin frame containing the baseline columns; it is not a raw-telemetry entry point. The deployed stage writes `outputs/figures/health_forecast_baseline_comparison.png`, `outputs/figures/health_forecast_level_calibration.png`, and `outputs/figures/health_forecast_horizon_degradation.png`.

**Tests.** `tests/test_availability.py` and `tests/test_outage_risk.py`

### Stage 11 — Combined operational scorecard

**Purpose.** Assemble one causal, current row per station for operational display. It joins present transmission/full/partial-outage state, weighted health components, saved transmitting-origin forecasts, causal detector evidence, and trailing reliability history without retraining any model or modifying upstream artifacts.

**Front door.** `scripts/build_station_health.py --scorecard`, with optional `--reference-hour` for an exact timezone-aware hour.

**Main modules.** `src/availability/operational_scorecard.py` resolves the reference hour, preserves the full 26-station registry roster, computes current/trailing evidence and reliability fields, prepares target-free forecast inputs, enforces forecast scope, audits expected nulls and join inconsistencies, and validates every derived field by deleting future source rows. `src/availability/health_forecast.py` supplies the saved-policy inference frame.

**Inputs and outputs.** It reads the canonical station-hour data, registry, frozen causal health table, current availability classification, live labels only for provenance checks, causal feature sources, and the five saved health-forecast policies. It writes `data/processed/station_operational_scorecard.csv`, a plain-text report, invariant hashes, and a delete-the-future comparison. The default reference is the latest hour with at least one actual transmission; explicit reference hours are honoured exactly. Full-outage forecasts are marked not applicable. Retrospective held-out detector/reason-code ledgers and completed calibration-corroboration intervals are not presented as live predictions.

**Tests.** `tests/test_availability.py` and `tests/test_front_door_scripts.py`

### Stage 12 — Causal forecast-risk construction and evaluation

**Purpose.** Construct and evaluate two separate future-risk problems: whether a station will experience a full outage, or a confirmed labelled sensor fault, in 6, 12, or 24 hours. Both use a station's continuous clock-hour grid and look strictly forward from the scored hour. Outage risk uses materialised structural gaps as outage targets. Fault risk uses the live hourly label construction; excluded hours remain on the clock but are neutral within a future target window, while warmup hours are history-only and never scored.

**Front door.** `scripts/evaluate_outage_risk.py --target outage`, `scripts/evaluate_outage_risk.py --target fault`, the fixed direct experiment `scripts/evaluate_outage_risk.py --target all --train-risk-models`, or the compact recurrent-event comparison `scripts/evaluate_outage_risk.py --target all --train-discrete-hazard`.

**Main modules.** `src/availability/risk_dataset.py`, `risk_model.py`, and `risk_eval.py`, with shared threshold metrics in `src/model/binary_metrics.py`.

**Inputs and outputs.** The outage path reads `data/processed/hourly_availability_classification.parquet` and frozen availability evidence. The fault path rebuilds live hourly labels in memory from `data/labels/episode_labels.csv`, the canonical hourly source, and feature evidence. For each horizon, both paths build continuous-clock-hour targets, assign whole timestamps to chronological partitions, and purge the horizon immediately before each boundary. The completed fixed direct route uses 1,014 predeclared causal inputs: current and lagged raw station telemetry, strictly prior rolling statistics, exact-hour ERA5/Open-Meteo residuals, same-hour peer disagreement with installation-date-aware topology, causal detector history, and shifted operational/context features. The discrete-hazard comparison instead uses 45 predeclared numeric fields plus regularised station indicators, fits one 1-hour onset hazard per target/method on the H24-safe risk set, Platt-calibrates on H24 validation only, and derives 6/12/24h risk by holding score-time covariates fixed. Its direct split membership is asserted against the saved onset manifest, all compact inputs pass delete-the-future checks, and its upstream artifacts are hashed before and after. Fault recurrence history is chronologically prior but derived from reviewed labels, so the fault-hazard result is explicitly retrospective-only. Both routes retain test-blind selection and none clears the 0.80 precision/recall/F1 criterion; forecast risk remains a measured limitation rather than a deployable claim. Existing `outage_risk_*` prototype artifacts remain stale preliminary evidence.

**Tests.** `tests/test_outage_risk.py`, `tests/test_availability.py`, and `tests/test_binary_metrics.py`

### Stage 13 — Report assets

**Purpose.** Produce methodology figures and the result/evidence figures used in the report.

**Front door.** `scripts/generate_report_assets.py` with `--set methodology`, `--set results`, `--set july-evaluation`, or `--set all`.

**Main modules.** `src/workflows/build_methodology_figures.py`, `build_result_figures.py`, `build_july_evaluation_figures.py`, and `src/rules/result_figures.py`.

**Inputs and outputs.** Methodology figures use the registry and hourly row states and write to `outputs/figures/`. Results figures use canonical data, detection features, and historical reviewed evidence queues and write to `figures/results/`. The July-evaluation route reloads the selected EF-HGB development test and frozen July probability ledger without fitting, then writes the paired class-distribution, confusion-matrix, ROC, and precision-recall figures to `outputs/figures/`.

**Tests.** `tests/test_result_figures.py`, `tests/test_front_door_scripts.py`

**Inherited dependency to know about.** The result-figure route expects historical evidence files such as `data/features/systemic_external_evidence.csv`, `data/labels/external_offset_queue_reviewed.csv`, and `data/labels/spatial_anomaly_queue.csv`. They are preserved local/report inputs rather than outputs clearly regenerated by the active feature-rebuild front door.

### Stage 14 — July replay dashboard

**Purpose.** Demonstrate the completed operational system when a live station feed is unavailable by replaying the independent July period hour by hour.

**Front door.** `python -m streamlit run scripts/run_dashboard.py`

**Main modules.** `src/dashboard/replay.py` loads five saved operational tables plus the station registry, assembles exact-hour 26-station snapshots, segments consecutive selected-EF-HGB positive hours into causal predicted events, and prepares a compact detector-margin table plus external/spatial availability text. `scripts/run_dashboard.py` uses Streamlit's built-in table and line-chart components for three simple views: Network, Station, and Evidence. It contains no map, performance panel, or ground-truth view.

**Inputs and outputs.** The replay reads saved July health scores, selected forecast predictions, only `station_id`, `hour_utc`, `random_probability`, and `random_prediction` from the frozen one-hour EF-HGB ledger, long-form statistical scores, the small neighbour graph, and the station registry. External-reference availability is read from the health rows already loaded. It writes nothing, loads no model, trains nothing, and excludes the final two all-station terminal-padding hours from its replay clock. Components shown in the event view are detector-evidence groups rather than component-model predictions.

**Tests.** `tests/test_availability.py` covers station preservation, mutually exclusive network categories, selected-HGB joins, full-outage continuation projections, past-only history, predicted-event breaks, and future-duration isolation. `tests/test_front_door_scripts.py` asserts that evaluation metrics and ground-truth fields do not enter the dashboard source.

## 3. Module index

Line counts are physical lines in the current working tree. `__init__.py` files are package markers unless noted.

### `src/availability/`

| Module | Lines | What it does |
|---|---:|---|
| `__init__.py` | 0 | Package marker. |
| `build_availability_events.py` | 803 | Preserves the frozen full-outage series, derives group-level full/partial/online station-hour availability, materializes structural grid gaps, builds partial events, and writes the descriptive availability report. |
| `build_network_outage_windows.py` | 357 | Groups concurrent station events into network windows and classifies them as network-midnight, other network, or local. |
| `build_station_reliability_summary.py` | 299 | Builds the current per-station availability rollup, including full/partial outage counts and hours plus per-group availability over transmitting hours. |
| `health_forecast.py` | 3,391 | Builds causal five-horizon health-forecast features and targets, strict no-new-incident roll-forward baselines, regime-specific residual HGB/CatBoost/Ridge selection, transmitting-origin direct classifiers, target-free saved-policy inference frames, comprehensive diagnostics, deployment artifacts, output tables/figures, and feature source-truncation validation. |
| `health_score.py` | 1,599 | Builds the causal, transparent 0-100 station-health score from raw transmission, group presence, causal detector evidence, exact-hour reference residuals, and strictly prior/current operational history; also writes scorecard tables, figures, progressive outage handling, and source-truncation validation. |
| `operational_scorecard.py` | 929 | Joins the complete station roster to current availability, weighted health components, saved transmitting-origin forecasts, causal fault evidence, and trailing reliability history; enforces forecast/deployment scope and audits causality, joins, nulls, and upstream hashes. |
| `risk_dataset.py` | 3420 | Builds continuous-clock-hour outage and fault-risk labels plus the direct causal and compact discrete-hazard feature routes. It reconstructs raw, exact-hour reference, spatial, detector, and strictly prior recurrence evidence as of the scored hour; validates future isolation; materialises structural gaps; and supplies live hourly-fault reconstruction and causality-audit evidence. |
| `risk_eval.py` | 670 | Creates generic timestamp-based partitions with boundary purges, reports label/split characteristics, and supplies binary-risk and numeric-regression metric helpers. |
| `risk_model.py` | 1260 | Retains historical risk baselines and defines direct HGB plus regularised logistic and shallow-boosted discrete-hazard routes, train-only class weighting, validation-only calibration/threshold selection, stationary hazard cumulation, and reference baselines. |

### `src/dashboard/`

| Module | Lines | What it does |
|---|---:|---|
| `replay.py` | 303 | Loads the minimum saved July operational artifacts, segments causal predicted events, assembles exact-hour station state, and prepares compact detector evidence and layer-availability text. |

### `src/config/`

| Module | Lines | What it does |
|---|---:|---|
| `__init__.py` | 0 | Package marker. |
| `paths.py` | 182 | Central project paths, frozen canonical schema/count expectations, measurement columns, availability, station-health, health-forecast, operational-scorecard, and risk artifacts, and directory helpers. |
| `test_paths.py` | 74 | Pytest dataset-contract checks for the canonical file. It lives unusually under `src/config/`, but is actively collected; it is not runtime configuration. |

### `src/features/`

| Module | Lines | What it does |
|---|---:|---|
| `__init__.py` | 0 | Package marker. |
| `build_station_registry.py` | 111 | Rebuilds station present-rate/status-class fields in the registry. It is a support utility, not currently called by a front door. |
| `rebuild.py` | 241 | Orchestrates statistical, external, spatial, and feature-matrix regeneration from frozen/public inputs, with isolated outputs for non-canonical experiments. |
| `row_state.py` | 137 | Defines and assigns station-hour availability states such as warm-up, true outage, online, and padded absence. |
| `run_data_audit.py` | 446 | Produces row states, audit/missingness tables, coverage plots, and a heatmap from the frozen hourly dataset. |

### `src/model/`

| Module | Lines | What it does |
|---|---:|---|
| `__init__.py` | 1 | Package marker. |
| `binary_metrics.py` | 49 | Shared binary precision, recall, F1, confusion metrics, and maximum-F1 threshold selection for outage risk. |
| `feature_spec.py` | 170 | Defines live hourly continuous/static/rule feature vocabulary and mechanism/component axes. It also retains unused V1-era family/window constants. |
| `hourly_baseline.py` | 3,189 | Validates/loads tensors, creates random and spaced splits, trains the class-weighted HistGradientBoosting baseline, defines the selected EF-HGB probability fusion, and implements conditional multi-label reason-code fitting, grouped training-only out-of-fold threshold selection, retrospective event aggregation, and neutral comparison tables. |
| `hourly_calibration.py` | 224 | Performs validation-only weight/threshold grid selection and operating-point reporting for the baseline. |
| `hourly_detection.py` | 667 | Core hour-level assembly: loads merged/features/live labels, derives display states, and writes past-only short/long tensors for both mask modes. |
| `hourly_rgfn.py` | 309 | Defines the Reliability-Aware Gated Fusion Network, including the final one-hour MLP encoder, legacy GRU/Conv encoders, rule branch, learned gate, mask validation, and optional ten-output exploratory reason-code heads. |
| `hourly_rgfn_training.py` | 1,789 | Prepares/scales splits, trains RGFN candidates, aggregates seeds, saves checkpoints, writes comparisons/reports, and trains/evaluates the time-boxed exploratory joint multi-label reason-code RGFN with grouped training-only OOF threshold selection. |
| `hourly_rgfn_tuning.py` | 1,268 | Main RGFN architecture/hyperparameter tuning engine: screening, selection, final evaluation, checkpoints, and tables. |
| `hourly_rgfn_tuning_features.py` | 286 | Builds causal past-only feature augmentations for tuning arms from tensors and raw hourly data. |
| `hourly_rgfn_tuning_logistic.py` | 257 | Runs the logistic-regression tuning comparison using flattened hourly and augmented features, reusing the shared train-only logistic scaler. |

### `src/references/`

| Module | Lines | What it does |
|---|---:|---|
| `__init__.py` | 0 | Empty namespace marker. The current reference fetcher lives in `scripts/fetch_reference_data.py`; this package has no active contained module or caller. |

### `src/rules/`

| Module | Lines | What it does |
|---|---:|---|
| `__init__.py` | 0 | Package marker. |
| `baselines.py` | 48 | Chooses station-specific or pooled median/MAD baselines based on coverage. |
| `channel_handlers.py` | 39 | Applies wind-direction and precipitation transforms and maps channels to components. |
| `clustering.py` | 146 | Builds episode feature vectors and HDBSCAN clusters. The current hourly labels do not consume clusters. |
| `config.py` | 287 | Central detector thresholds, physical/contextual/reference settings, mappings, and feature/label paths; it mixes live settings with historical replay settings. |
| `detectors/__init__.py` | 0 | Detector package marker. |
| `detectors/isolation_forest.py` | 31 | One-dimensional Isolation Forest detector. |
| `detectors/robust_zscore.py` | 37 | Robust median/MAD z-score detector. |
| `detectors/rolling_variance.py` | 47 | Rolling-variance flatline detector with zero-value handling. |
| `episodes.py` | 146 | Merges aligned channel events into station episodes with channel/component/detector evidence. |
| `events.py` | 102 | Groups contiguous flagged station/channel hours into detector events. |
| `external_features.py` | 277 | Creates leakage-guarded external residual/offset/solar/fleet features; some systemic-evidence helpers support historical report assets. |
| `external_offsets.py` | 558 | Detects persistent external residual offsets and solar-ratio runs. Calibration corroboration uses its channel detector; its broad queue route is historical. |
| `external_residuals.py` | 636 | Aligns station, five-minute, and ERA5/Open-Meteo readings; derives residual baselines, z-scores, and source-agreement diagnostics. |
| `feature_matrix.py` | 235 | Joins statistical, external, spatial, and context features into the leakage-guarded matrix consumed by hour-level detection. |
| `labelling.py` | 513 | Core multi-label episode classifier for spike, stuck, statistical, and calibration mechanisms, components, periods, and frozen-label crosswalks. |
| `calibration_corroboration.py` | 465 | Detects sustained calibration-offset runs, records borderline review, resolves eligible cases, and writes resolution evidence. |
| `physical_limits.py` | 41 | Provides hard/suspect physical-limit flags and physical-kind lookup. |
| `result_figures.py` | 507 | Computes and plots historical external/spatial/offset evidence figures for the report-assets workflow. |
| `review_queue.py` | 318 | Converts clustered episodes into review-priority queue artifacts; the live hourly model does not train from this queue. |
| `score.py` | 220 | Applies physical, stuck, robust-z, and Isolation Forest evidence per station/channel and writes reasoned anomaly scores. |
| `spatial_offsets.py` | 402 | Builds live spatial feature columns/neighbor-presence fields and retains older spatial-offset candidate/queue functions. |
| `spatial_residuals.py` | 265 | Builds station neighbour graphs, neighbour medians, spatial residual baselines, and z-scores. |
| `statistical_gate.py` | 810 | Implements contextual + detector + ERA5 statistical evidence, review tables, and ranked benign review. |
| `stuck_confirmation.py` | 552 | Supplies five-minute stuck confirmation and dominant-channel helpers. Its main confirmation route is historical; `external_residuals.py` currently imports its five-minute path constant. |

### `src/workflows/`

| Module | Lines | What it does |
|---|---:|---|
| `__init__.py` | 1 | Package marker. |
| `build_methodology_figures.py` | 421 | Generates station map, architecture, row-state, outage-flow, and stuck-wind methodology figures. |
| `build_result_figures.py` | 122 | Loads historical evidence artifacts and delegates result-figure production. |
| `build_july_evaluation_figures.py` | 238 | Reconstructs the selected one-hour EF-HGB development test, reads the frozen July probability ledger, and generates paired class-distribution, confusion-matrix, ROC, and precision-recall figures without fitting. |
| `resume_hourly_tuning.py` | 1,084 | Restores tuning state/checkpoints and resumes one named tuning phase or report generation. |
| `train_hourly_baseline.py` | 1,464 | Runs baseline matrix creation, split manifest, artifacts, metrics, importance reporting, and the conditional `reason-codes` mode with logistic/gradient-boosted/RGFN comparison, retrospective event aggregation, and validation-only configuration selection. |
| `train_hourly_calibration.py` | 146 | Runs the HGB validation-only weight/threshold sweep, writes its report, and optionally saves the selected model for each partition. |
| `train_hourly_rgfn.py` | 853 | Retains the earlier GRU/Conv workflow, runs the one-hour HGB-versus-gated-MLP comparison, fits the validation-selected EF-HGB branches, and performs frozen July evaluation. |
| `train_hourly_split_comparison.py` | 316 | Compares 70/15/15 and 80/20 baseline split configurations. |
| `tune_hourly_detection.py` | 604 | Orchestrates tuning arms, augmentation checks, artifacts, and reports. |

### Mixed live/historical support code

`src/rules/config.py`, `feature_spec.py`, `external_features.py`, `external_offsets.py`, `spatial_offsets.py`, and `stuck_confirmation.py` still provide live functionality but retain selected V1/Stage 5 logic or historical report support. They remain active because removing their historical portions safely requires separating the live interfaces first.

## 4. Data map

### Core tracked data and evidence

| Artifact | Produced by | Consumed by | Git status / role |
|---|---|---|---|
| `data/merged/station_hourly_merged.csv` | Frozen project input | Audit, reference validation, feature rebuild, labelling, hourly dataset | **Tracked canonical source**; 166,017 station-hours. Its acquisition/assembly code is outside this repository. |
| `data/merged/station_registry.csv` | Frozen project input | Audit, reference fetch, feature rebuild, methodology figures | **Tracked** station metadata. |
| `data/processed/data_audit_summary.csv`, `hourly_row_states.parquet`, missingness tables | `build_reliability_foundations.py` | Availability, methodology figures | **Tracked audit/reliability evidence**. |
| `data/processed/availability_events.parquet`, `network_outage_windows.csv` | `build_reliability_foundations.py` | Outage-risk evaluation/reporting | **Tracked frozen full-outage evidence**: 2,398 events and 47 windows. |
| `data/processed/hourly_availability_classification.parquet`, `partial_outage_events.parquet`, `structural_availability_gaps.csv`, `availability_report.txt` | `build_reliability_foundations.py` | Per-station summary, operations, reporting, corrected outage-risk labels | **Tracked June-current operational availability evidence**. Partial outages are rule-derived, not supervised-model results; full-outage hours, including materialised structural gaps, are the outage-risk target source. |
| `data/processed/station_reliability_summary.csv` | `build_reliability_foundations.py` | Reliability reporting/figures | **Tracked June-current summary**, dynamically anchored to the current data end. |
| `data/processed/station_health_scores.parquet`, `station_health_summary.csv`, `station_health_report.txt`, invariant/source-truncation audit files, duration curve, and comparison tables | `build_station_health.py` | Operational scorecard and future health-forecast design | **Tracked causal current-state health artifacts.** The score uses raw station-hour observations and exact-hour external references, not reviewed labels or completed calibration-corroboration intervals. Active full outages use `exp(-d/24)` and partial outages use `exp(-d/72)` before the existing fixed weighted sum, where `d` is active duration in completed hours. |
| `data/eval/health_forecast/`, `data/model/health_forecast/`, and `outputs/figures/health_forecast_*.png` | `build_station_health.py --forecast` | Residual health-policy evaluation, classification, and report figures | The model/evaluation files are **ignored local artifacts**. They retain frozen Part A diagnostics, validation-only selection traces, the regression master table, pooled/regime and classification/confusion results, predictions, calibration, importance, feature causality validation, split/invariant records, and the authoritative active-model manifest. The two figures are tracked report assets. |
| `data/processed/station_operational_scorecard.csv`, `station_operational_scorecard_report.txt`, invariant JSON, and delete-the-future audit | `build_station_health.py --scorecard` | Dashboard data layer and direct operational review | **Tracked causal 26-station snapshot.** The default uses the latest hour with a real transmission, stores explicit history/forecast scope statuses, marks full-outage forecasts not applicable, and excludes retrospective held-out detector/reason-code ledgers from live claims. |
| `data/eval/one_hour_candidate/`, `july_2026_health/`, `july_2026_health_forecast/`, `july_2026_scoring/`, and `july_2026_features/` | Frozen July workflows | `run_dashboard.py`, July evaluation figures, final report | **Ignored local replay evidence.** `one_hour_candidate/` contains the selected one-hour EF-HGB July ledger. The dashboard reads only its operational columns, selected health policies, detector scores, and the neighbour graph. It never reads July truth, reviewed episode IDs, RGFN output, reason-code output, detailed residual tables, or evaluation fields. |
| `outputs/figures/selected_ef_hgb_class_distribution.png`, `selected_ef_hgb_confusion_matrices.png`, and `selected_ef_hgb_roc_pr_curves.png` | `generate_report_assets.py --set july-evaluation` | Final report and supervisor review | **Tracked report assets** rebuilt from the exact selected development model/test partition and frozen July probability ledger without fitting. |
| `data/labels/episode_labels.csv` | `build_labels.py` | `build_hourly_dataset.py` | **Tracked live label source**. It contains the 1,121-fault population used by the models. |
| `data/labels/calibration_offset_corroboration.csv` | `build_labels.py` | Report/evidence review | **Tracked calibration-offset corroboration evidence**. |
| `outputs/figures/station_uptime_bar.png`, `outputs/figures/network_offline_fraction_timeline.png` | Historical availability plotting utilities | Historical reporting only | **Tracked May-era figures**, not June-current evidence. |
| Other `outputs/figures/`, `figures/results/` assets | Figure workflows | Report | **Tracked report assets**; verify each asset's source period before using it as a current result. |

### Generated local artifacts

| Artifact / directory | Produced by | Used by | Git status / note |
|---|---|---|---|
| `data/external/reference_hourly/` and manifest | Reference fetch | Feature rebuild | **Ignored.** Active public-reference cache. |
| `data/external/mozn_weather_dataset/per_station_weather_data/` | User-supplied public Zenodo download | Optional feature rebuild | **Ignored.** Public five-minute observation files; the repository contains no downloader or assembly implementation. |
| `data/features/` | Detection-feature rebuild | Labels, hourly dataset, result figures | **Ignored.** Includes statistical, external, spatial, and `feature_matrix.parquet` artifacts. |
| Statistical score/event/episode/cluster parquet files | Detection-feature rebuild | Review/provenance | **Ignored.** Recomputed evidence. |
| Label crosswalk and review CSVs other than the allowlisted evidence above | Label build | Review/report provenance | **Ignored.** The frozen legacy inventory is optional: when present it creates a historical crosswalk, and when absent the live labels are still written normally. |
| `data/hourly_detection/` | Hourly dataset, training, tuning | Training/tuning/reporting | **Ignored.** Labels/tensors, split manifests, metrics, models, and checkpoints are generated local artifacts. |

### Data ownership shorthand

- **Live model labels:** `data/labels/episode_labels.csv`
- **Canonical observational record:** `data/merged/station_hourly_merged.csv`
- **Rebuildable local model artifacts:** `data/features/` and `data/hourly_detection/`
- **Public observation/reference caches:** `data/external/`

## 5. Key concepts

### Episode and station-hour

- A **station-hour** is one `(station_id, hour)` observation row. It is the unit used by the deployed detection dataset and model.
- An **event** is a contiguous run of flagged hours for one station/channel.
- An **episode** is a station-level interval formed by merging temporally aligned events. It is an intermediate labelling unit and can carry multiple mechanisms and components.
- The hour-level model expands an episode across its inclusive hours, then builds a past-only window ending at each hour: seven hours for the short tensor and 49 hours for the long tensor.

### Fault mechanisms and components

The live target vocabulary has four multi-label fault mechanisms:

1. `spike_impossible`
2. `stuck_flatline`
3. `statistical_anomaly`
4. `calibration_offset`

The six supported components are:

1. `anemometer`
2. `barometer`
3. `light_uv`
4. `rain_gauge`
5. `thermo_hygrometer`
6. `wind_vane`

Mechanisms and components are multi-label: one episode/hour can involve more than one of either. There is no live `multiple faults` class. A raw `other` mapping can exist during processing, but it is not an accepted hourly target axis.

The **reason-code experiment** is conditional: it receives only fault-hours and reuses the binary detector's frozen split manifest. Logistic regression and gradient boosting fit one classifier per label; the RGFN has shared temporal/evidence streams with mechanism and exploratory component outputs. Thresholds are selected by grouped cross-validation entirely inside each outer training partition, using mean fold-level out-of-fold F1. The delivered analysis aggregates mechanism predictions over retrospective ground-truth-connected event groups and selects its method/rule from complete spaced validation groups only. Calibration offset is excluded from that selection because its validation support is below five positive complete events. Held-out scores recover the rule-derived label taxonomy rather than independently field-verified root causes. The comparison reports all methods without using test scores to choose a winner. It is not a source-ID-free live dashboard feature; predicted-event segmentation and closure are future work.

### Fault, benign, clean, and excluded

`fault_hour` is the only training target: 1 for fault and 0 for not-fault. `display_state` gives the useful finer distinction:

| Display state | Meaning | Training use |
|---|---|---|
| `fault` | The hour lies inside at least one `label_state=fault` episode. Mechanisms/components are unioned if needed. | `fault_hour = 1` |
| `benign` | The hour is not labelled fault, but at least one detector flag fired. | `fault_hour = 0` |
| `clean` | The hour is not labelled fault and no detector flag fired. | `fault_hour = 0` |
| `excluded` | The hour is within a `borderline_review` episode, or lacks eligible source/evidence. | No training/evaluation label |

`benign` versus `clean` is dashboard metadata rather than a second model target. It shows whether a non-fault hour was a detector candidate or entirely normal.

### Detection evidence and calibration-offset corroboration

The code does not define a universal numbered “three-layer” architecture. The accurate terminology is:

1. **Rule/statistical screening:** physical limits, robust z-score, Isolation Forest, and rolling variance create evidence at the station/channel/hour level.
2. **Context and external evidence:** the statistical gate uses same-channel detector evidence, physical/stuck checks, station hour-of-day/month context, and ERA5 information where comparison is available. Path A uses ERA5 directional agreement. The externally comparable single-detector Path B instead requires a strong absolute ERA5 residual; because either sign qualifies, it represents strong external discrepancy rather than same-direction corroboration.
3. **Calibration-offset corroboration:** this stage specifically resolves sustained calibration-offset candidates from external residuals and spatial corroboration, retaining uncertain cases as `borderline_review` rather than forcing a fault label.

### Mask mode

The canonical mask mode is `per_hour`, with shape `(N, window, 1)`: it states whether an hourly row is present. The reproducible experiment `per_feature` uses `(N, window, n_continuous)` and records NaN availability for each continuous feature. Both retain `time_since_last`; default code and reported results use `per_hour`.

### Random and spaced splits

- The **random** split is deterministic and stratified; the baseline defaults to 70/15/15, while the comparison runner also evaluates a true 80/20 setup with no validation partition.
- The **spaced** split keeps connected positive source episodes together and assigns benign/clean rows within month/display-state strata.
- The spaced split is useful for avoiding a positive episode being split across partitions, but it is **not** a purged-time or held-station evaluation. It does not explicitly eliminate overlap between nearby sliding windows, so it should not be described as independent prospective validation.

### Causal station health

The station-health score is a current operational state, not a trained model. Each sufficient-history station-hour stores five 0-1 components and their fixed weighted 0-100 total: trailing seven-day transmission availability (30 points), transmitting-hour sensor completeness across the six groups (20), exponentially recency-weighted causal rule-evidence burden (25), exact-hour external-reference consistency with prior-only persistence evidence (15), and recent operational stability (10). During a continuous full or partial outage, a fixed causal duration multiplier and a no-improvement base-score cap make health decline progressively rather than forcing an immediate hard zero.

The health builder is intentionally stricter than a retrospective dashboard reconstruction: it does not use reviewed `fault_hour` labels, episode boundaries, completed availability events, historical feature matrices, or completed calibration-corroboration intervals. Its fault component is therefore named causal rule-evidence burden, and confirmed calibration-corroboration records are annotations for a sanity-check figure only. Sixteen raw/reference source-truncation rebuilds verify that changing future inputs cannot change a historical score row.

### Health-score forecasting

The forecasting extension predicts the residual `health(t+H) - rollforward(t+H)` and reconstructs bounded health as `clip(rollforward + alpha * residual, 0, 100)` for `H` in 1, 3, 6, 12, and 24 hours; change is derived from that bounded level. Every horizon uses timestamp-based chronological train/validation/test partitions with an H-hour purge immediately before both boundaries. Iterations, 53-feature core versus 181-feature historical union, recency weighting, HGB/CatBoost/core-only Ridge family, and alpha (including zero) are selected on validation only. Final learned policies are refit on train plus validation and evaluated once on test. The evaluation is post-hoc because earlier 6/12/24-hour test results influenced this frozen extension and scope definition.

Three predeclared baselines remain mandatory: current-value persistence, least-squares extrapolation of the immediately trailing 24-hour health slope, and the strict no-new-incident roll-forward. The roll-forward retains a state already active at the forecast origin, advances its 7-day and 30-day health histories mechanically, preserves its causal reference component, and never reads actual future telemetry, events, labels, or reference values. Regression policies are separated into transmitting origins (including partial outages) and full-outage origins, with the latter marked low-confidence and supplementary. The primary regression, direct -5/-10 deterioration, three-class trajectory, and future-band evaluations all use the same transmitting-origin station-hours. Direction accuracy, balanced metrics, supports, confusion matrices, selected-candidate importance, transmitting-only calibration, bounds, serialization, cross-horizon degradation, and full feature source-truncation checks accompany the result.

### Combined operational scorecard

The operational scorecard is a causal join and presentation layer, not another model. At one parameterised reference hour it preserves all 26 registry stations, reports current full/partial/transmitting state, weighted health components, current and trailing causal detector evidence, trailing uptime/outage history, and the five saved health forecasts. Partial-outage stations remain forecast-eligible because they are transmitting; active full outages receive an explicit not-applicable status rather than a number. Historical held-out detector predictions, ground-truth episode-segmented reason-code outputs, and completed calibration-corroboration intervals are excluded from live prediction fields because they are not deployment-safe at the reference hour.

### Causal forecast risk

Forecast risk is a distinct task from retrospective hour-level detection. At a scored station-hour `t`, the 6/12/24-hour target asks whether a fault or full outage occurs in `(t, t+H]`. The prediction route uses a continuous hourly grid and purges the H hours immediately before each timestamp partition boundary, so a retained row never derives its target from a later partition.

The frozen direct route uses 1,014 audited as-of-time features, including reconstructed raw, exact-hour reference, spatial, detector, availability, and station-context evidence. The separate discrete-hazard comparison deliberately narrows this to 45 numeric fields plus regularised station indicators: calendar, static context, prior availability, detector counts, five pressure/temperature/wind summaries, and strictly prior target recurrence. It fits a 1-hour hazard then cumulates it to each reporting horizon under a fixed-covariate assumption. Historical detector-matrix values, labels, episodes, forward five-minute snapshots, whole-day solar ratios, and full-series anomaly baselines remain excluded from the direct route. Fault recurrence is a reviewed-label history and makes only the fault-hazard comparison retrospective-only; outage recurrence is observable after an event closes.

For each fixed target/horizon configuration, the model derives the base positive weight from the training partition, sweeps a predefined weight/threshold grid on validation, and selects the highest validation minimum of precision, recall, and F1. Test is evaluated once after that choice is frozen. Base-rate and persistence predictors are reference baselines; fault persistence is retrospective label-history evidence and is not a live dashboard feature. The first causal run did not meet the 0.80 precision/recall/F1 criterion in any configuration, so it is a documented limitation rather than a deployment result.

### RGFN and its Evidence Gate

**RGFN** means **Reliability-Aware Gated Fusion Network**. It combines:

- a temporal encoder over continuous features, mask, and normalised `time_since_last` (GRU or convolutional variant),
- a rule-evidence branch through an MLP,
- static context, and
- an Evidence Gate that learns `alpha = sigmoid(...)` and combines the temporal and rule logits as `alpha * temporal_logit + (1 - alpha) * rule_logit` before the final sigmoid probability.

## 6. How to run it

Run commands from the repository root. The examples below are the complete operational sequence; do not run data-writing steps casually against the canonical dataset.

### Check the code first

```powershell
python -m pytest -q
```

### 1. Validate the frozen dataset and rebuild reliability foundations

```powershell
python scripts/build_reliability_foundations.py
```

This reads but never rewrites `data/merged/station_hourly_merged.csv`. It regenerates audit, full-outage, partial-outage, coordinated-window, and station-reliability outputs. The public source observations are documented in the README; acquisition and hourly assembly are outside this repository.

### 2. Extend/fetch public reference data

```powershell
python scripts/fetch_reference_data.py
```

For a period beyond June 2026, update the reference date range and expected row count in `src/rules/config.py` first. Use `--force` only when intentionally refreshing the local reference cache.

### 3. Rebuild detection features

```powershell
python scripts/rebuild_detection_features.py --five-min-dir data/external/mozn_weather_dataset/per_station_weather_data
```

This creates the statistical/external/spatial evidence and `feature_matrix.parquet` needed by labels and hourly tensors. It expects the separately downloaded public Zenodo per-station files but contains no acquisition, staging, or merge route.

### 4. Rebuild live labels

```powershell
python scripts/build_labels.py
```

This front door currently has no argument parser, so `--help` is not a harmless command: it would start the real label build. The frozen legacy label inventory is optional and only controls whether a historical crosswalk is emitted.

### 5. Build the hour-level dataset

```powershell
python scripts/build_hourly_dataset.py --mask-mode per_hour
```

`per_hour` is the canonical/reported configuration. Use `--mask-mode per_feature` only when intentionally reproducing the mask experiment.

### 6. Train the detection models

```powershell
python scripts/train_hourly_detection.py baseline
python scripts/train_hourly_detection.py split-comparison
python scripts/train_hourly_detection.py calibration
python scripts/train_hourly_detection.py rgfn
python scripts/train_hourly_detection.py reason-codes
python scripts/train_hourly_detection.py reason-codes --postprocess-existing
```

The required order is baseline, calibration, then RGFN: the RGFN route relies on the baseline split manifest and calibration outputs. `reason-codes` also relies on the frozen baseline split manifest, but leaves all binary artifacts unchanged. Its `--postprocess-existing` route performs retrospective event aggregation and validation-only method/rule selection; it is not a live dashboard feed. The split-comparison route is useful experiment/report evidence rather than a prerequisite for RGFN training.

### 7. Run or resume tuning (optional experiment path)

```powershell
python scripts/tune_hourly_detection.py run
python scripts/tune_hourly_detection.py resume --phase random
```

Valid resume phases are `random`, `spaced_arm1`, `spaced_screen`, `spaced_finalize`, and `report`.

### 8. Build the causal station-health scorecard

```powershell
python scripts/build_station_health.py
```

This needs the canonical merged station-hour dataset and a complete exact-hour public-reference cache. It writes the per-station-hour components and total, ranked station summary, plain-text measurement report, input hashes, source-truncation audit, and four scorecard figures. It does not rebuild labels or train a model.

### 9. Evaluate the health-score forecast comparison

```powershell
python scripts/build_station_health.py --forecast
```

This leaves the current health score unchanged. It writes its five-horizon models and detailed evaluation artifacts below ignored `data/model/health_forecast/` and `data/eval/health_forecast/`, plus three tracked forecast figures. The primary results are transmitting-origin forecasts; full-outage regression remains a low-confidence supplement.

The isolated Day-2 through Day-7 comparison is reproduced with:

```powershell
python scripts/build_station_health.py --long-horizon-forecast --long-horizon-features current
python scripts/build_station_health.py --long-horizon-forecast --long-horizon-features extended
```

These commands do not replace the deployed 1/3/6/12/24-hour model directory. The first uses only the original feature definitions; the second lets validation additionally consider the explicitly named long-horizon representation.

### 10. Assemble the combined operational scorecard

```powershell
python scripts/build_station_health.py --scorecard
python scripts/build_station_health.py --scorecard --reference-hour 2026-06-30T21:00:00Z
```

The default chooses the latest canonical hour containing at least one observed transmission; this avoids presenting terminal padding as a network-wide live outage. The explicit reference must be a timezone-aware whole hour. The command reads frozen health and forecast artifacts, writes the 26-station machine-readable table and plain report, and verifies upstream hashes plus delete-the-future comparisons before accepting the snapshot.

### 11. Construct forecast-risk labels or run the fixed causal experiment

```powershell
python scripts/evaluate_outage_risk.py --target outage
python scripts/evaluate_outage_risk.py --target fault
```

These commands construct and report corrected 6/12/24-hour availability-outage or sensor-fault labels and splits. Each uses continuous clock-hour horizon arithmetic, puts every station at a timestamp in the same chronological partition, and purges the relevant horizon before partition boundaries.

The causal forecast route is separate and evaluates all six target/horizon configurations in one fixed pass:

```powershell
python scripts/evaluate_outage_risk.py --target all --train-risk-models
```

The completed fixed pass used 1,014 audited as-of-time features: raw readings at the scored hour, prior-only raw summaries, exact-hour ERA5/Open-Meteo residuals, same-hour peer disagreement, causal detector history, and shifted operational/context features. It did not use stored detector-matrix values, labels, episode fields, forward five-minute snapshots, whole-day solar ratios, or full-series anomaly baselines. Eight deterministic source-truncation checks validated every feature after future raw, reference, peer, availability, and future-topology information was removed (8,112 comparisons; zero failures). For each configuration, the implementation derives the positive class weight from train, sweeps predefined weights and thresholds on validation, freezes the validation-maximin operating point, then evaluates test once. The experiment did not produce a deployable forecast claim and its generated artifacts are not retained in the minimal repository. The command remains available only to reproduce this future-work investigation deliberately; do not rerun it in response to held-out test performance.

The discrete-time hazard comparison is separate. It verifies rebuilt direct partitions against the saved onset manifest, fits one validation-calibrated 1-hour hazard per target/method, and derives 6/12/24-hour risk by holding the score-time covariates fixed:

```powershell
python scripts/evaluate_outage_risk.py --target all --train-discrete-hazard
```

Do not use its fault result as a live dashboard input: its strictly-prior fault-recurrence source is reviewed ground truth rather than an online detector ledger.

### 12. Generate report figures

```powershell
python scripts/generate_report_assets.py --set methodology
python scripts/generate_report_assets.py --set july-evaluation
```

`methodology` is the default public-safe figure set. `july-evaluation` recreates the verified selected-EF-HGB class distribution, confusion matrices, ROC curves, and precision-recall curves without retraining. `--set results` and `--set all` preflight the historical evidence inputs described in Stage 13 before writing figures.

### 13. Run the July operational replay

```powershell
python -m streamlit run scripts/run_dashboard.py
```

The replay uses saved July health, forecast, selected-EF-HGB, rule-score, neighbour, and registry artifacts. It contains no live acquisition route, model loading, fitting, performance panel, ground-truth field, map, or elaborate evidence plot. Transmitting stations use the selected learned health policy; a full-outage station receives a clearly labelled causal continuation projection that assumes the outage persists and does not predict recovery. Event evidence uses one compact table and two layer-availability lines.

### Recheck after changes

```powershell
python -m pytest -q
git status --short
```
