from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MERGED_DIR = DATA_DIR / "merged"
EXTERNAL_DIR = DATA_DIR / "external"
LABELS_DIR = DATA_DIR / "labels"
PROCESSED_DIR = DATA_DIR / "processed"
EVALUATION_DIR = DATA_DIR / "eval"
MODEL_DIR = DATA_DIR / "model"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

MERGED_DATASET_PATH = MERGED_DIR / "station_hourly_merged.csv"
STATION_REGISTRY_PATH = MERGED_DIR / "station_registry.csv"
DATA_AUDIT_SUMMARY_PATH = PROCESSED_DIR / "data_audit_summary.csv"

HOURLY_ROW_STATES_PATH = PROCESSED_DIR / "hourly_row_states.parquet"
AVAILABILITY_EVENTS_PATH = PROCESSED_DIR / "availability_events.parquet"
NETWORK_OUTAGE_WINDOWS_PATH = PROCESSED_DIR / "network_outage_windows.csv"
AVAILABILITY_CLASSIFICATION_PATH = (
    PROCESSED_DIR / "hourly_availability_classification.parquet"
)
PARTIAL_OUTAGE_EVENTS_PATH = PROCESSED_DIR / "partial_outage_events.parquet"
STRUCTURAL_AVAILABILITY_GAPS_PATH = (
    PROCESSED_DIR / "structural_availability_gaps.csv"
)
STATION_RELIABILITY_SUMMARY_PATH = (
    PROCESSED_DIR / "station_reliability_summary.csv"
)
AVAILABILITY_REPORT_PATH = PROCESSED_DIR / "availability_report.txt"
STATION_HEALTH_SCORES_PATH = PROCESSED_DIR / "station_health_scores.parquet"
STATION_HEALTH_SUMMARY_PATH = PROCESSED_DIR / "station_health_summary.csv"
STATION_HEALTH_REPORT_PATH = PROCESSED_DIR / "station_health_report.txt"
STATION_HEALTH_INVARIANTS_PATH = PROCESSED_DIR / "station_health_invariants.json"
STATION_HEALTH_CAUSALITY_AUDIT_PATH = (
    PROCESSED_DIR / "station_health_delete_future_validation.csv"
)
STATION_HEALTH_CAUSALITY_SUMMARY_PATH = (
    PROCESSED_DIR / "station_health_delete_future_summary.csv"
)
STATION_HEALTH_PROGRESSIVE_COMPARISON_PATH = (
    PROCESSED_DIR / "station_health_progressive_comparison.csv"
)
STATION_HEALTH_PROGRESSIVE_RANKING_PATH = (
    PROCESSED_DIR / "station_health_progressive_station_ranking.csv"
)
STATION_HEALTH_OUTAGE_DURATION_CURVE_PATH = (
    PROCESSED_DIR / "station_health_outage_duration_curve.csv"
)
STATION_HEALTH_OUTAGE_TRAJECTORY_PATH = (
    PROCESSED_DIR / "station_health_outage_trajectory.csv"
)
HEALTH_FORECAST_DIR = EVALUATION_DIR / "health_forecast"
HEALTH_FORECAST_MODEL_DIR = MODEL_DIR / "health_forecast"
HEALTH_FORECAST_LONG_HORIZON_DIR = EVALUATION_DIR / "health_forecast_long_horizon"
HEALTH_FORECAST_LONG_HORIZON_MODEL_DIR = MODEL_DIR / "health_forecast_long_horizon"
FROZEN_RULE_STATISTICS_DIR = MODEL_DIR / "frozen_rule_statistics"
STATION_OPERATIONAL_SCORECARD_PATH = (
    PROCESSED_DIR / "station_operational_scorecard.csv"
)
STATION_OPERATIONAL_SCORECARD_REPORT_PATH = (
    PROCESSED_DIR / "station_operational_scorecard_report.txt"
)
STATION_OPERATIONAL_SCORECARD_INVARIANTS_PATH = (
    PROCESSED_DIR / "station_operational_scorecard_invariants.json"
)
STATION_OPERATIONAL_SCORECARD_CAUSALITY_PATH = (
    PROCESSED_DIR / "station_operational_scorecard_delete_future_validation.csv"
)

STATION_COVERAGE_FIGURE_PATH = FIGURES_DIR / "station_coverage_timeline.png"
MISSINGNESS_HEATMAP_PATH = FIGURES_DIR / "missingness_heatmap.png"
HEALTH_DISTRIBUTION_FIGURE_PATH = FIGURES_DIR / "health_distribution.png"
HEALTH_COMPONENT_DISTRIBUTIONS_FIGURE_PATH = (
    FIGURES_DIR / "health_component_distributions.png"
)
HEALTH_STATION_TIMESERIES_FIGURE_PATH = FIGURES_DIR / "health_station_timeseries.png"
HEALTH_COMPONENT_CORRELATION_FIGURE_PATH = (
    FIGURES_DIR / "health_component_correlation.png"
)
HEALTH_OUTAGE_DURATION_TRAJECTORY_FIGURE_PATH = (
    FIGURES_DIR / "health_outage_duration_trajectory.png"
)
HEALTH_FORECAST_BASELINE_COMPARISON_FIGURE_PATH = (
    FIGURES_DIR / "health_forecast_baseline_comparison.png"
)
HEALTH_FORECAST_CALIBRATION_FIGURE_PATH = (
    FIGURES_DIR / "health_forecast_level_calibration.png"
)
HEALTH_FORECAST_HORIZON_DEGRADATION_FIGURE_PATH = (
    FIGURES_DIR / "health_forecast_horizon_degradation.png"
)
HEALTH_FORECAST_LONG_HORIZON_BASELINE_COMPARISON_FIGURE_PATH = (
    FIGURES_DIR / "health_forecast_long_horizon_baseline_comparison.png"
)
HEALTH_FORECAST_LONG_HORIZON_CALIBRATION_FIGURE_PATH = (
    FIGURES_DIR / "health_forecast_long_horizon_level_calibration.png"
)
HEALTH_FORECAST_LONG_HORIZON_DEGRADATION_FIGURE_PATH = (
    FIGURES_DIR / "health_forecast_long_horizon_degradation.png"
)

CANONICAL_TIMEZONE = "UTC"
EXPECTED_FROZEN_N_ROWS = 166_017
EXPECTED_FROZEN_N_COLS = 41
EXPECTED_STATION_COUNT = 26

CANONICAL_COLUMN_ORDER = [
    "station_id",
    "hour_utc",
    "n_raw_records",
    "latitude",
    "longitude",
    "qc_status",
    "epoch",
    "solar_radiation_high_wm2",
    "uv_high",
    "winddir_avg_deg",
    "humidity_avg_pct",
    "humidity_high_pct",
    "humidity_low_pct",
    "temp_avg_c",
    "temp_high_c",
    "temp_low_c",
    "windspeed_avg_kmh",
    "windspeed_high_kmh",
    "windspeed_low_kmh",
    "windgust_avg_kmh",
    "windgust_high_kmh",
    "windgust_low_kmh",
    "dewpoint_avg_c",
    "dewpoint_high_c",
    "dewpoint_low_c",
    "windchill_avg_c",
    "windchill_high_c",
    "windchill_low_c",
    "heatindex_avg_c",
    "heatindex_high_c",
    "heatindex_low_c",
    "pressure_max_hpa",
    "pressure_min_hpa",
    "pressure_trend_hpa",
    "precip_rate_mmh",
    "precip_total_mm",
    "timestamp_utc_dt",
    "timestamp_utc",
    "timestamp_local",
    "data_present",
    "elevation",
]

NON_MEASUREMENT_COLUMNS = [
    "station_id",
    "hour_utc",
    "n_raw_records",
    "latitude",
    "longitude",
    "qc_status",
    "epoch",
    "timestamp_utc_dt",
    "timestamp_utc",
    "timestamp_local",
    "data_present",
    "elevation",
]

MEASUREMENT_COLUMNS = [
    column for column in CANONICAL_COLUMN_ORDER
    if column not in NON_MEASUREMENT_COLUMNS
]

REQUIRED_ID_COLUMNS = ["station_id", "hour_utc", "data_present"]
STATION_METADATA_COLUMNS = ["station_id", "latitude", "longitude", "elevation"]
TIMESTAMP_COLUMNS = ["hour_utc", "timestamp_utc_dt", "timestamp_utc", "timestamp_local"]

DIRECTORIES_TO_CREATE = [
    MERGED_DIR,
    EXTERNAL_DIR,
    LABELS_DIR,
    PROCESSED_DIR,
    EVALUATION_DIR,
    MODEL_DIR,
    FIGURES_DIR,
]


def ensure_directories() -> None:
    for path in DIRECTORIES_TO_CREATE:
        path.mkdir(parents=True, exist_ok=True)
