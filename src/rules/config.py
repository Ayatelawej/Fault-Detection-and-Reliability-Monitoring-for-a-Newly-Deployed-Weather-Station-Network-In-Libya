from datetime import date


ROBUST_ZSCORE_FLAG_PERCENTILE = 99.7
ROLLING_VARIANCE_WINDOW_HOURS = 24
ROLLING_VARIANCE_FLAG_THRESHOLD = 1e-6
CONTEXTUAL_BASELINE_MIN_SAMPLES = 15
CONTEXTUAL_OUTLIER_Z_THRESHOLD = 3.5
ERA5_AGREEMENT_Z_THRESHOLD = 3.0
CONTEXTUAL_MAD_FLOORS = {
    "temp_avg_c": 0.1,
    "temp_high_c": 0.1,
    "temp_low_c": 0.1,
    "humidity_avg_pct": 0.5,
    "humidity_high_pct": 0.5,
    "humidity_low_pct": 0.5,
    "windspeed_avg_kmh": 0.1,
    "windspeed_high_kmh": 0.1,
    "windspeed_low_kmh": 0.1,
    "windgust_avg_kmh": 0.1,
    "windgust_high_kmh": 0.1,
    "windgust_low_kmh": 0.1,
    "winddir_sin": 0.01,
    "winddir_cos": 0.01,
    "pressure_max_hpa": 0.1,
    "pressure_min_hpa": 0.1,
    "pressure_trend_hpa": 0.05,
    "precip_rate_mmh": 0.05,
    "precip_total_mm": 0.05,
    "solar_radiation_high_wm2": 10.0,
    "uv_high": 0.1,
}
ISOLATION_FOREST_CONTAMINATION = 0.003
COVERAGE_FLOOR_HOURS = 1500
HDBSCAN_MIN_CLUSTER_SIZE = 30
HDBSCAN_MIN_SAMPLES = 10
REVIEW_SUSTAINED_NOISE_HOURS = 24
EXTERNAL_MODELS = "era5_seamless"
EXTERNAL_CELL_SELECTION = "land"
EXTERNAL_CACHE_DIR = "data/external/reference_hourly"
EXTERNAL_START_DATE = "2025-06-15"
EXTERNAL_END_DATE = "2026-07-31"
EXTERNAL_EXCLUDED_STATION_IDS = frozenset({"IJANZO4"})
EXTERNAL_API_URL = "https://archive-api.open-meteo.com/v1/archive"


def _identity(value):
    return value


def _kmh_to_ms(value):
    return value / 3.6


EXTERNAL_CHANNEL_MAP = {
    "pressure_max_hpa": {
        "openmeteo_variable": "pressure_msl",
        "conversion": _identity,
        "unit": "hPa",
    },
    "temp_avg_c": {
        "openmeteo_variable": "temperature_2m",
        "conversion": _identity,
        "unit": "degC",
    },
    "dewpoint_avg_c": {
        "openmeteo_variable": "dew_point_2m",
        "conversion": _identity,
        "unit": "degC",
    },
    "windspeed_avg_kmh": {
        "openmeteo_variable": "wind_speed_10m",
        "conversion": _kmh_to_ms,
        "unit": "m/s",
    },
    "solar_radiation_high_wm2": {
        "openmeteo_variable": "shortwave_radiation",
        "conversion": _identity,
        "unit": "W/m2",
    },
}
EXTERNAL_HOURLY_VARS = list(
    dict.fromkeys(
        spec["openmeteo_variable"]
        for spec in EXTERNAL_CHANNEL_MAP.values()
    )
)
EXTERNAL_EXPECTED_ROWS = (
    (date.fromisoformat(EXTERNAL_END_DATE) - date.fromisoformat(EXTERNAL_START_DATE)).days + 1
) * 24
EXTERNAL_TIME_LAG_HOURS = 0
EXTERNAL_CHANNEL_SOURCE = {
    "pressure": "5min_snapshot",
    "temp": "5min_snapshot",
    "dewpoint": "5min_snapshot",
    "wind": "hourly_mean",
    "solar": "5min_trailing_mean",
}
EXTERNAL_SNAPSHOT_TOLERANCE_MIN = 15
EXTERNAL_SOLAR_MEAN_MIN_SLOTS = 8
EXTERNAL_BASELINE_WINDOW_HOURS = 720
EXTERNAL_BASELINE_MIN_HOURS = 240
EXTERNAL_HOURBIN_WINDOW_DAYS = 30
EXTERNAL_HOURBIN_MIN_DAYS = 15
EXTERNAL_MAD_FLOORS = {
    "pressure": 0.8,
    "temp": 0.3,
    "dewpoint": 0.4,
    "wind": 0.3,
    "solar": 10.0,
}
EXTERNAL_SOLAR_DAYLIGHT_MIN = 10.0
EXTERNAL_CLEARSKY_REF_MIN = 300.0
EXTERNAL_RESIDUALS_PATH = "data/features/external_residuals.parquet"
EXTERNAL_OFFSET_CHANNELS = ["pressure", "temp", "dewpoint"]
EXTERNAL_OFFSET_MIN_FLEET_STATIONS = 8
EXTERNAL_OFFSET_SPREAD_FLOORS = {
    "pressure": 1.0,
    "temp": 0.8,
    "dewpoint": 0.8,
}
EXTERNAL_OFFSET_PHYSICAL_FLOORS = {
    "pressure": 3.0,
    "temp": 2.0,
    "dewpoint": 2.5,
}
EXTERNAL_OFFSET_MIN_DENSITY = 0.6
EXTERNAL_OFFSET_SCORE_HIGH = 3.0
EXTERNAL_OFFSET_SCORE_MED = 2.0
EXTERNAL_OFFSET_MED_STABILITY_MAX = {
    "pressure": 0.8,
    "temp": 0.5,
    "dewpoint": 0.6,
}
EXTERNAL_OFFSET_STABILITY_MAX = {
    "pressure": 3.0,
    "temp": 1.5,
    "dewpoint": 2.0,
}
EXTERNAL_OFFSET_GAP_HOURS = {"pressure": 168, "temp": 72, "dewpoint": 72}
EXTERNAL_SOLAR_QUEUE_ENABLED = True
EXTERNAL_OFFSET_MIN_DAYS = 7
EXTERNAL_INSUFFICIENT_MIN_HOURS = 1000
EXTERNAL_RATIO_WINDOW_DAYS = 14
EXTERNAL_RATIO_MIN_DAYS = 7
EXTERNAL_RATIO_MIN_HOURS_PER_DAY = 3
EXTERNAL_RATIO_MAX = 1.15
EXTERNAL_RATIO_MIN = 0.75
EXTERNAL_REL_RATIO_MAX = 1.25
EXTERNAL_REL_RATIO_MIN = 0.80
EXTERNAL_OFFSET_QUEUE_PATH = "data/labels/external_offset_queue.csv"
EXTERNAL_FEATURES_PATH = "data/features/external_features.parquet"
EXTERNAL_SYSTEMIC_EVIDENCE_PATH = "data/features/systemic_external_evidence.csv"
EXTERNAL_ARRAY_CHANNELS = ["temp", "dewpoint", "wind", "solar"]
EXTERNAL_SYSTEMIC_DISAGREE_Z = 3.0
EXTERNAL_TEMP_OFFSET_WINDOW_HOURS = 720
EXTERNAL_TEMP_OFFSET_MIN_HOURS = 240
SPATIAL_NEIGHBOR_RADIUS_KM = 200
SPATIAL_MIN_NEIGHBORS = 2
SPATIAL_MIN_NEIGHBORS_PRESENT = 2
SPATIAL_CHANNELS = ["pressure", "temp", "dewpoint"]
SPATIAL_CHANNEL_COLUMNS = {
    "pressure": "pressure_max_hpa",
    "temp": "temp_avg_c",
    "dewpoint": "dewpoint_avg_c",
}
SPATIAL_BASELINE_WINDOW_HOURS = 720
SPATIAL_BASELINE_MIN_HOURS = 240
SPATIAL_MAD_FLOORS = {"pressure": 0.5, "temp": 0.5, "dewpoint": 0.5}
SPATIAL_NEIGHBORS_PATH = "data/features/spatial_neighbors.csv"
SPATIAL_RESIDUALS_PATH = "data/features/spatial_residuals.parquet"
SPATIAL_OFFSET_CHANNEL = "pressure"
SPATIAL_OFFSET_PHYSICAL_FLOOR_HPA = 4.0
SPATIAL_OFFSET_SCORE_HIGH_HPA = 6.0
SPATIAL_OFFSET_STABILITY_MAX_HPA = 4.0
SPATIAL_OFFSET_MIN_DENSITY = 0.6
SPATIAL_OFFSET_MIN_DAYS = 7
SPATIAL_OFFSET_GAP_HOURS = 168
SPATIAL_INSUFFICIENT_MIN_HOURS = 1000
SPATIAL_FEATURES_PATH = "data/features/spatial_features.parquet"
SPATIAL_ANOMALY_QUEUE_PATH = "data/labels/spatial_anomaly_queue.csv"
STATISTICAL_FEATURES_PATH = "data/features/statistical_features.parquet"
FEATURE_MATRIX_PATH = "data/features/feature_matrix.parquet"
RESULT_FIGURES_DIR = "figures/results"
BENIGN_AUDIT_EXT_PRESSURE_LOW = -10.0
BENIGN_AUDIT_EXT_PRESSURE_HIGH = 3.0
BENIGN_AUDIT_SPATIAL_ABS = 5.0
BENIGN_AUDIT_ARRAY_Z = 3.0
BENIGN_AUDIT_MIN_VALID_HOURS = 6
BENIGN_AUDIT_CONFIG_SCALE = 30.0
BENIGN_AUDIT_CANDIDATES_PATH = "data/labels/benign_audit_candidates.csv"
BENIGN_AUDIT_BOUNDARY_DAYS = 21
BENIGN_AUDIT_TRIAGE_PATH = "data/labels/benign_audit_triage.csv"
IJABAL13_OFFSET_START = "2025-10-04 23:00"
IJABAL15_OFFSET_START = "2025-10-05 16:00"
IJABAL15_OFFSET_END = "2025-10-09 18:00"
BENIGN_RESOLUTION_LOG_PATH = "data/labels/benign_resolution_log.csv"
LABEL_PRIORITY = {
    "systemic_array": 1,
    "stuck_flatline": 1,
    "spike_impossible": 2,
    "calibration_offset": 3,
    "spatial_anomaly": 4,
    "benign": 5,
}
SPATIAL_EXCLUDED_NEIGHBORS = [
    "IBIRAL3",
    "IJABAL13",
    "IJABAL14",
    "IMISRA13",
    "INUQAT9",
    "INUQAT10",
    "IZAWIY7",
    "IZAWIY5",
]
SPATIAL_OFFSET_REQUIRE_EXTERNAL_AGREEMENT = True
SPATIAL_OFFSET_EXTERNAL_MARGIN_HPA = 2.0
PHYSICAL_LIMIT_RULES = {
    "temp_avg_c": {"min": -60.0, "max": 60.0, "kind": "temperature"},
    "temp_high_c": {"min": -60.0, "max": 60.0, "kind": "temperature"},
    "temp_low_c": {"min": -60.0, "max": 60.0, "kind": "temperature"},
    "humidity_avg_pct": {"min": 0.0, "max": 100.0, "kind": "humidity"},
    "humidity_high_pct": {"min": 0.0, "max": 100.0, "kind": "humidity"},
    "humidity_low_pct": {"min": 0.0, "max": 100.0, "kind": "humidity"},
    "windspeed_avg_kmh": {"min": 0.0, "max": 250.0, "kind": "wind"},
    "windspeed_high_kmh": {"min": 0.0, "max": 250.0, "kind": "wind"},
    "windspeed_low_kmh": {"min": 0.0, "max": 250.0, "kind": "wind"},
    "windgust_avg_kmh": {"min": 0.0, "max": 300.0, "kind": "wind"},
    "windgust_high_kmh": {"min": 0.0, "max": 300.0, "kind": "wind"},
    "windgust_low_kmh": {"min": 0.0, "max": 300.0, "kind": "wind"},
    "winddir_avg_deg": {"min": 0.0, "max": 360.0, "kind": "wind_direction"},
    "pressure_max_hpa": {"min": 870.0, "max": 1085.0, "kind": "pressure"},
    "pressure_min_hpa": {"min": 870.0, "max": 1085.0, "kind": "pressure"},
    "pressure_trend_hpa": {"max_abs": 20.0, "kind": "pressure_trend"},
    "precip_rate_mmh": {"min": 0.0, "max": 1000.0, "kind": "rain_rate"},
    "precip_total_mm": {"min": 0.0, "max": 1000.0, "kind": "rain_total"},
    "solar_radiation_high_wm2": {"min": 0.0, "max": 1600.0, "kind": "solar"},
    "uv_high": {"min": 0.0, "max": 25.0, "kind": "uv"},
}
PHYSICAL_SUSPECT_RULES = {
    "solar_radiation_high_wm2": {"max": 1100.0, "kind": "solar"},
    "uv_high": {"max": 16.0, "kind": "uv"},
    "windspeed_avg_kmh": {"max": 150.0, "kind": "wind"},
    "windspeed_high_kmh": {"max": 150.0, "kind": "wind"},
    "windspeed_low_kmh": {"max": 150.0, "kind": "wind"},
    "windgust_avg_kmh": {"max": 180.0, "kind": "wind"},
    "windgust_high_kmh": {"max": 180.0, "kind": "wind"},
    "windgust_low_kmh": {"max": 180.0, "kind": "wind"},
    "precip_rate_mmh": {"max": 300.0, "kind": "rain_rate"},
    "precip_total_mm": {"max": 300.0, "kind": "rain_total"},
    "pressure_trend_hpa": {"max_abs": 20.0, "kind": "pressure_trend"},
}

CHANNEL_BASELINE_WINDOWS = {
    "pressure_max_hpa": 30 * 24,
    "pressure_min_hpa": 30 * 24,
    "precip_total_mm": 14 * 24,
    "precip_rate_mmh": 14 * 24,
}

CHANNELS_REQUIRING_CIRCULAR_TRANSFORM = ["winddir_avg_deg"]
CHANNELS_REQUIRING_LOG_TRANSFORM = ["precip_total_mm", "precip_rate_mmh"]
CHANNELS_EXCLUDED_FROM_STATISTICAL_LAYER = [
    "dewpoint_avg_c",
    "dewpoint_high_c",
    "dewpoint_low_c",
    "windchill_avg_c",
    "windchill_high_c",
    "windchill_low_c",
    "heatindex_avg_c",
    "heatindex_high_c",
    "heatindex_low_c",
]
STUCK_IGNORE_ZERO_CHANNELS = [
    "precip_rate_mmh",
    "solar_radiation_high_wm2",
    "uv_high",
    "windspeed_low_kmh",
    "windgust_low_kmh",
]
STUCK_SKIP_CHANNELS = ["precip_total_mm"]
SENSOR_GROUP_PREFIXES = {
    "temp": "thermo_hygrometer",
    "humidity": "thermo_hygrometer",
    "windspeed": "anemometer",
    "windgust": "anemometer",
    "winddir": "wind_vane",
    "precip": "rain_gauge",
    "solar": "light_uv",
    "uv": "light_uv",
    "pressure": "barometer",
}
