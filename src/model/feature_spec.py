from __future__ import annotations

CONTINUOUS_FEATURES = [
    "r_pressure",
    "offset_level_pressure",
    "z_pressure",
    "spatial_offset_level_pressure",
    "z_spatial_pressure",
    "z_spatial_temp",
    "z_spatial_dewpoint",
    "ext_abs_z_array_mean",
    "ext_n_valid_array",
    "z_temp",
    "z_dewpoint",
    "z_wind",
    "rel_ratio_solar",
    "offset_level_temp",
    "offset_level_dewpoint",
    "stat_rolling_variance_pressure_max_hpa",
    "stat_rolling_variance_temp_avg_c",
    "stat_rolling_variance_windspeed_avg_kmh",
    "stat_rolling_variance_winddir_cos",
    "stat_zscore_pressure_trend_hpa",
    "stat_zscore_temp_avg_c",
    "stat_zscore_windspeed_avg_kmh",
    "stat_zscore_solar_radiation_high_wm2",
    "n_neighbors_present_pressure",
]

STATIC_FEATURES = [
    "ctx_elevation",
    "ctx_n_neighbors",
    "spatial_isolated",
]

EPISODE_STATIC_FEATURES = [
    "episode_log_duration_hours",
    "episode_duration_ge_7h",
    "episode_duration_ge_24h",
    "episode_duration_ge_72h",
    "episode_shape_max_any_detector_flag_run",
    "episode_co_fraction_hours_ge2_groups",
    "episode_co_total_distinct_groups",
    "episode_agree_pressure_external_spatial_strength_mean",
    "episode_agree_pressure_external_spatial_sign_fraction",
    "episode_drift_pressure_abs_mean",
]

MODEL_STATIC_FEATURES = STATIC_FEATURES + EPISODE_STATIC_FEATURES

SCALED_STATIC_FEATURES = [
    "ctx_elevation",
    "ctx_n_neighbors",
    "episode_log_duration_hours",
    "episode_shape_max_any_detector_flag_run",
    "episode_co_total_distinct_groups",
    "episode_agree_pressure_external_spatial_strength_mean",
    "episode_drift_pressure_abs_mean",
]

UNSCALED_STATIC_FEATURES = [
    "spatial_isolated",
    "episode_duration_ge_7h",
    "episode_duration_ge_24h",
    "episode_duration_ge_72h",
    "episode_co_fraction_hours_ge2_groups",
    "episode_agree_pressure_external_spatial_sign_fraction",
]

RULE_EVIDENCE_FLAGS = [
    "stat_flag_stuck_windspeed_avg_kmh",
    "stat_flag_stuck_windgust_avg_kmh",
    "stat_flag_stuck_windspeed_high_kmh",
    "stat_flag_stuck_windgust_high_kmh",
    "stat_flag_stuck_winddir_cos",
    "stat_flag_stuck_winddir_sin",
    "stat_flag_physical_solar_radiation_high_wm2",
    "stat_flag_physical_suspect_solar_radiation_high_wm2",
    "stat_flag_physical_precip_total_mm",
    "stat_flag_physical_precip_rate_mmh",
    "stat_flag_physical_pressure_trend_hpa",
    "stat_flag_physical_suspect_uv_high",
    "stat_sensor_group_flag_anemometer",
    "stat_sensor_group_flag_barometer",
    "stat_sensor_group_flag_light_uv",
    "stat_sensor_group_flag_other",
    "stat_sensor_group_flag_rain_gauge",
    "stat_sensor_group_flag_thermo_hygrometer",
    "stat_sensor_group_flag_wind_vane",
]

TARGET_COLUMNS = [
    "binary_fault",
    "family",
]





MECHANISM_LABEL_NAMES = (
    "spike_impossible",
    "stuck_flatline",
    "statistical_anomaly",
    "calibration_offset",
)

COMPONENT_LABEL_NAMES = (
    "anemometer",
    "barometer",
    "light_uv",
    "rain_gauge",
    "thermo_hygrometer",
    "wind_vane",
)

MECHANISM_TARGET_COLUMNS = tuple(
    f"mechanism_{name}" for name in MECHANISM_LABEL_NAMES
)
COMPONENT_TARGET_COLUMNS = tuple(
    f"component_{name}" for name in COMPONENT_LABEL_NAMES
)

ID_COLUMNS = [
    "episode_id",
    "station_id",
    "start_hour",
    "end_hour",
]

ALLOWED_TARGET_COLUMNS = ID_COLUMNS + TARGET_COLUMNS

FORBIDDEN_SUBSTRINGS = [
    "label",
    "family",
    "fault",
    "target",
    "reason",
    "priority",
    "support_tier",
    "episode_id",
    "binary_fault",
]

WINDOW_BEFORE_ONSET = 60
WINDOW_AFTER_ONSET = 12
WINDOW_LEN = 72

FAMILY_TO_IDX = {
    "benign": 0,
    "stuck_flatline": 1,
    "systemic_array": 2,
    "spike_impossible": 3,
    "calibration_offset": 4,
    "spatial_anomaly": 5,
}

IDX_TO_FAMILY = {index: family for family, index in FAMILY_TO_IDX.items()}


def rule_evidence_feature_names() -> list[str]:
    names = []
    for column in RULE_EVIDENCE_FLAGS:
        names.append(f"{column}_any")
        names.append(f"{column}_rate")
    return names


def feature_columns_in_X() -> list[str]:
    return CONTINUOUS_FEATURES + MODEL_STATIC_FEATURES + rule_evidence_feature_names()
