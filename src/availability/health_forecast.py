from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.availability.build_availability_events import (
    AVAILABILITY_CLASS_FULL_OUTAGE,
    AVAILABILITY_CLASS_ONLINE,
    AVAILABILITY_CLASS_PARTIAL_OUTAGE,
    SENSOR_GROUP_ORDER,
)
from src.availability.health_score import (
    FAULT_EVIDENCE_HALF_LIFE_HOURS,
    FAULT_EVIDENCE_RATE_CAP,
    HEALTH_BANDS,
    HEALTH_COMPONENT_COLUMNS,
    HEALTH_HISTORY_HOURS,
    HEALTH_WEIGHTS,
    STABILITY_EVENT_CAP,
    STABILITY_HISTORY_HOURS,
    outage_duration_multiplier,
)
from src.availability.risk_dataset import build_causal_detector_evidence
from src.availability.risk_eval import (
    regression_error_improvement_percent,
    regression_metrics,
    split_timestamp_partitions,
)
from src.config.paths import MEASUREMENT_COLUMNS


HEALTH_FORECAST_HORIZONS = (1, 3, 6, 12, 24)
HEALTH_FORECAST_LONG_HORIZONS = (48, 72, 96, 120, 144, 168)
HEALTH_TREND_WINDOW_HOURS = 24
HEALTH_FORECAST_METHODS = (
    "persistence",
    "recent_trend_24h",
    "no_new_incident_roll_forward",
    "selected_residual_forecast",
)
HEALTH_FORECAST_ALPHA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
HEALTH_FORECAST_RECENCY_HALF_LIVES_DAYS = (None, 90, 60, 30)
HEALTH_FORECAST_TREE_ITERATION_GRID = (100, 200, 300)
HEALTH_FORECAST_MODEL_FAMILIES = ("hist_gradient_boosting", "catboost", "ridge")
HEALTH_FORECAST_FEATURE_SETS = ("core", "full_engineered")
HEALTH_FORECAST_LONG_FEATURE_SETS = (*HEALTH_FORECAST_FEATURE_SETS, "long_horizon")
HEALTH_FORECAST_CLASSIFIER_ITERATIONS = 200
HEALTH_FORECAST_CLASSIFIER_THRESHOLD_GRID = tuple(
    float(value) for value in np.linspace(0.10, 0.90, 17)
)
HEALTH_FORECAST_HGB_PARAMETERS = {
    "loss": "absolute_error",
    "learning_rate": 0.05,
    "max_leaf_nodes": 7,
    "max_depth": 3,
    "min_samples_leaf": 80,
    "l2_regularization": 10.0,
    "early_stopping": False,
    "random_state": 2026,
}
HEALTH_FORECAST_CATBOOST_PARAMETERS = {
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "learning_rate": 0.05,
    "depth": 4,
    "l2_leaf_reg": 20.0,
    "random_seed": 2026,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
}
HEALTH_FORECAST_RIDGE_ALPHA = 10.0
HEALTH_FORECAST_NETWORK_FEATURES = (
    "feature_fraction_other_stations_transmitting",
    "feature_network_median_health_slope_24h",
)
HEALTH_FORECAST_CORE_FEATURES = (
    "feature_current_health_total",
    "feature_current_health_availability",
    "feature_current_health_sensor_completeness",
    "feature_current_health_fault_burden",
    "feature_current_health_reference_consistency",
    "feature_current_health_stability",
    "feature_health_slope_6h",
    "feature_health_slope_12h",
    "feature_health_slope_24h",
    "feature_health_slope_72h",
    "feature_full_outage_run_hours",
    "feature_partial_outage_run_hours",
    "feature_causal_fault_evidence_rate_7d",
    "feature_hours_since_last_outage",
    "feature_hours_since_last_fault_evidence",
    "feature_missing_fraction_now",
    "feature_missing_fraction_24h",
    *tuple(
        name
        for group in SENSOR_GROUP_ORDER
        for name in (
            f"feature_sensor_group_present_now_{group}",
            f"feature_sensor_group_present_fraction_24h_{group}",
        )
    ),
    *tuple(
        name
        for detector in ("physical", "stuck", "deviation")
        for name in (
            *(f"feature_detector_{detector}_count_24h_{group}" for group in SENSOR_GROUP_ORDER),
            f"feature_detector_{detector}_count_24h_any",
        )
    ),
    "feature_station_age_hours",
    *HEALTH_FORECAST_NETWORK_FEATURES,
)
HEALTH_FORECAST_LONG_FEATURES = (
    *HEALTH_FORECAST_CORE_FEATURES,
    *tuple(
        f"feature_health_{statistic}_{window}h"
        for window in (168, 336, 720)
        for statistic in ("mean", "std", "min", "max")
    ),
    *tuple(f"feature_health_slope_{window}h" for window in (168, 336, 720)),
    *tuple(f"feature_missing_fraction_{window}h" for window in (168, 336, 720)),
    *tuple(f"feature_transmitting_fraction_{window}h" for window in (168, 336, 720)),
    *tuple(
        f"feature_sensor_group_present_fraction_{window}h_{group}"
        for window in (168, 336, 720)
        for group in SENSOR_GROUP_ORDER
    ),
    *tuple(f"feature_health_band_fraction_{window}h_{band.lower()}" for window in (168, 336, 720) for band in HEALTH_BANDS),
    "feature_hours_in_current_health_band",
    "feature_target_hour_sin",
    "feature_target_hour_cos",
    "feature_target_day_of_year_sin",
    "feature_target_day_of_year_cos",
)
HEALTH_FORECAST_BAND_INTERVALS = (
    ("healthy", 80.0, np.inf),
    ("watch", 60.0, 80.0),
    ("degraded", 40.0, 60.0),
    ("critical", -np.inf, 40.0),
)
HEALTH_FORECAST_TARGET_CALENDAR_FEATURES = (
    "feature_target_hour_sin",
    "feature_target_hour_cos",
    "feature_target_day_of_year_sin",
    "feature_target_day_of_year_cos",
)


@dataclass(frozen=True)
class HealthForecastFeatureBundle:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    station_ids: tuple[str, ...]
    elevation_by_station: dict[str, float]


@dataclass(frozen=True)
class HealthForecastRun:
    metrics: pd.DataFrame
    improvements: pd.DataFrame
    conditional_metrics: pd.DataFrame
    direction_metrics: pd.DataFrame
    deterioration_metrics: pd.DataFrame
    deterioration_confusion: pd.DataFrame
    trajectory_metrics: pd.DataFrame
    trajectory_confusion: pd.DataFrame
    band_metrics: pd.DataFrame
    band_confusion: pd.DataFrame
    feature_importance: pd.DataFrame
    calibration: pd.DataFrame
    predictions: pd.DataFrame
    selection_trace: pd.DataFrame
    iteration_comparison: pd.DataFrame
    feature_ablation: pd.DataFrame
    recency_comparison: pd.DataFrame
    model_family_comparison: pd.DataFrame
    alpha_selection: pd.DataFrame
    feature_audit: pd.DataFrame
    split_digests: dict[str, object]
    models: dict[str, object]
    model_artifact_hashes: dict[str, str]
    retired_model_artifacts: tuple[str, ...]
    forecast_model_round_trip_verified: bool


def _require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(f"{name} is missing required columns: {', '.join(missing)}")


def _normalise_scores(scores: pd.DataFrame) -> pd.DataFrame:
    required = [
        "station_id",
        "hour_utc",
        "health_total",
        "availability_class",
        "is_transmitting",
        "causal_fault_evidence",
        "stability_adverse_now",
        "stability_hard_fault_evidence",
        "outage_base_score_cap",
        "base_health_reference_consistency",
        "full_outage_run_hours",
        "partial_outage_run_hours",
        *HEALTH_COMPONENT_COLUMNS,
        *[f"base_{component}" for component in HEALTH_COMPONENT_COLUMNS],
        *[f"sensor_group_present_{group}" for group in SENSOR_GROUP_ORDER],
        *MEASUREMENT_COLUMNS,
    ]
    _require_columns(scores, required, "station health scores")
    result = scores.copy(deep=True)
    result["station_id"] = result["station_id"].astype(str)
    result["hour_utc"] = pd.to_datetime(result["hour_utc"], utc=True, errors="coerce")
    if result[["station_id", "hour_utc"]].isna().any().any():
        raise ValueError("station health scores contain invalid station-hour keys")
    if result.duplicated(["station_id", "hour_utc"]).any():
        raise ValueError("station health scores contain duplicate station-hour keys")
    result = result.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(drop=True)
    for _, station in result.groupby("station_id", sort=False):
        hours = station["hour_utc"].diff().dropna()
        if not hours.eq(pd.Timedelta(hours=1)).all():
            raise ValueError("station health scores must retain a continuous hourly clock")
    return result


def _elevation_mapping(
    station_metadata: pd.DataFrame | None,
    station_ids: tuple[str, ...],
) -> dict[str, float]:
    if station_metadata is None or station_metadata.empty:
        return {station_id: np.nan for station_id in station_ids}
    _require_columns(station_metadata, ["station_id", "elevation"], "station metadata")
    source = station_metadata.loc[:, ["station_id", "elevation"]].copy()
    source["station_id"] = source["station_id"].astype(str)
    source["elevation"] = pd.to_numeric(source["elevation"], errors="coerce")
    values = source.groupby("station_id", sort=False)["elevation"].first()
    return {station_id: float(values.get(station_id, np.nan)) for station_id in station_ids}


def _hours_since_event(hours: pd.Series, event: pd.Series, cap: int) -> pd.Series:
    timestamps = pd.to_datetime(hours, utc=True, errors="coerce")
    event_time = timestamps.where(event.fillna(False).astype(bool)).ffill()
    elapsed = timestamps.sub(event_time).dt.total_seconds().div(3600.0)
    return elapsed.fillna(float(cap)).clip(lower=0.0, upper=float(cap))


def _linear_slope(values: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    length = int(window)
    result = np.full(len(numeric), np.nan, dtype=float)
    if length <= 1 or len(numeric) < length:
        return pd.Series(result, index=values.index, dtype=float)
    time_index = np.arange(length, dtype=float)
    denominator = float(((time_index - time_index.mean()) ** 2).sum())
    kernel = time_index[::-1]
    sums = np.convolve(
        np.nan_to_num(numeric, nan=0.0), np.ones(length), mode="full"
    )[: len(numeric)]
    weighted = np.convolve(
        np.nan_to_num(numeric, nan=0.0), kernel, mode="full"
    )[: len(numeric)]
    counts = np.convolve(
        np.isfinite(numeric).astype(float), np.ones(length), mode="full"
    )[: len(numeric)]
    valid = np.arange(len(numeric)) >= length - 1
    valid &= np.isclose(counts, float(length))
    result[valid] = (
        weighted[valid] - time_index.mean() * sums[valid]
    ) / denominator
    return pd.Series(result, index=values.index, dtype=float)


def _add_station_features(station: pd.DataFrame) -> pd.DataFrame:
    result = station.copy(deep=True).reset_index(drop=True)
    feature_values: dict[str, pd.Series | np.ndarray] = {}
    hours = result["hour_utc"]
    health = pd.to_numeric(result["health_total"], errors="coerce")
    for lag in (6, 12, 24, 72, 168, 336, 720):
        lagged_hour = hours.shift(lag)
        exact = hours.sub(lagged_hour).dt.total_seconds().div(3600.0).eq(lag)
        feature_values[f"feature_health_slope_{lag}h"] = (
            (health - health.shift(lag)) / float(lag)
        ).where(exact)
    feature_values["feature_health_trend_slope_24h"] = _linear_slope(
        health, HEALTH_TREND_WINDOW_HOURS
    )
    measurement = result.loc[:, list(MEASUREMENT_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    missing_now = measurement.isna().mean(axis=1)
    feature_values["feature_missing_fraction_now"] = missing_now
    for window in (6, 24, 72, 168, 336, 720):
        feature_values[f"feature_missing_fraction_{window}h"] = missing_now.rolling(
            window, min_periods=1
        ).mean()
        feature_values[f"feature_transmitting_fraction_{window}h"] = (
            result["is_transmitting"].astype(float).rolling(window, min_periods=1).mean()
        )
    outage = result["availability_class"].isin(
        [AVAILABILITY_CLASS_FULL_OUTAGE, AVAILABILITY_CLASS_PARTIAL_OUTAGE]
    )
    feature_values["feature_hours_since_last_outage"] = _hours_since_event(
        hours, outage, STABILITY_HISTORY_HOURS
    )
    feature_values["feature_hours_since_last_fault_evidence"] = _hours_since_event(
        hours,
        result["causal_fault_evidence"].fillna(False).astype(bool),
        STABILITY_HISTORY_HOURS,
    )
    for group in SENSOR_GROUP_ORDER:
        present = result[f"sensor_group_present_{group}"].astype(float)
        feature_values[f"feature_sensor_group_present_now_{group}"] = present
        feature_values[f"feature_sensor_group_present_fraction_24h_{group}"] = present.rolling(
            24, min_periods=1
        ).mean()
        for window in (168, 336, 720):
            feature_values[f"feature_sensor_group_present_fraction_{window}h_{group}"] = present.rolling(
                window, min_periods=1
            ).mean()
    for window in (168, 336, 720):
        history = health.rolling(window, min_periods=min(24, window))
        feature_values[f"feature_health_mean_{window}h"] = history.mean()
        feature_values[f"feature_health_std_{window}h"] = history.std(ddof=0)
        feature_values[f"feature_health_min_{window}h"] = history.min()
        feature_values[f"feature_health_max_{window}h"] = history.max()
        for band, lower, upper in HEALTH_FORECAST_BAND_INTERVALS:
            in_band = health.ge(float(lower)) & health.lt(float(upper))
            feature_values[f"feature_health_band_fraction_{window}h_{band.lower()}"] = in_band.astype(float).rolling(
                window, min_periods=min(24, window)
            ).mean()
    current_band = result["health_band"].astype(str)
    band_start = current_band.ne(current_band.shift(1)).cumsum()
    feature_values["feature_hours_in_current_health_band"] = current_band.groupby(band_start).cumcount().add(1).astype(float)
    detector = build_causal_detector_evidence(result).reset_index(drop=True)
    for kind in ("physical", "stuck", "deviation"):
        for group in SENSOR_GROUP_ORDER:
            source = f"causal_detector_{kind}_count_24h_{group}"
            feature_values[f"feature_detector_{kind}_count_24h_{group}"] = pd.to_numeric(
                detector[source], errors="coerce"
            )
        source = f"causal_detector_{kind}_any_count_24h"
        feature_values[f"feature_detector_{kind}_count_24h_any"] = pd.to_numeric(
            detector[source], errors="coerce"
        )
    for column in MEASUREMENT_COLUMNS:
        values = pd.to_numeric(result[column], errors="coerce")
        feature_values[f"feature_sensor_now_{column}"] = values
        feature_values[f"feature_sensor_prior_mean_24h_{column}"] = values.shift(1).rolling(
            24, min_periods=6
        ).mean()
        feature_values[f"feature_sensor_delta_6h_{column}"] = values - values.shift(6)
    return pd.concat([result, pd.DataFrame(feature_values, index=result.index)], axis=1)


def build_health_forecast_features(
    scores: pd.DataFrame,
    *,
    station_metadata: pd.DataFrame | None = None,
    station_ids: tuple[str, ...] | None = None,
) -> HealthForecastFeatureBundle:
    source = _normalise_scores(scores)
    known_station_ids = (
        tuple(sorted(source["station_id"].unique())) if station_ids is None else tuple(station_ids)
    )
    if set(source["station_id"].unique()).difference(known_station_ids):
        raise ValueError("forecast station identifiers do not cover the score source")
    elevation = _elevation_mapping(station_metadata, known_station_ids)
    stations = [
        _add_station_features(station)
        for _, station in source.groupby("station_id", sort=False)
    ]
    result = pd.concat(stations, ignore_index=True)
    scoreable = pd.to_numeric(result["health_total"], errors="coerce").notna()
    transmitting = result["is_transmitting"].fillna(False).astype(bool) & scoreable
    cohort_size = scoreable.groupby(result["hour_utc"], sort=False).transform("sum").astype(float)
    cohort_transmitting = transmitting.groupby(result["hour_utc"], sort=False).transform("sum").astype(float)
    peer_count = cohort_size - scoreable.astype(float)
    peer_transmitting = cohort_transmitting - transmitting.astype(float)
    result["feature_fraction_other_stations_transmitting"] = np.divide(
        peer_transmitting,
        peer_count,
        out=np.full(len(result), np.nan, dtype=float),
        where=peer_count.to_numpy(dtype=float) > 0.0,
    )
    network_median_health = (
        pd.to_numeric(result["health_total"], errors="coerce")
        .where(scoreable)
        .groupby(result["hour_utc"], sort=True)
        .median()
        .sort_index()
    )
    if not network_median_health.empty:
        continuous_network_hours = pd.date_range(
            network_median_health.index.min(),
            network_median_health.index.max(),
            freq="h",
        )
        network_median_health = network_median_health.reindex(continuous_network_hours)
    network_median_slope = _linear_slope(
        network_median_health, HEALTH_TREND_WINDOW_HOURS
    )
    result["feature_network_median_health_slope_24h"] = result["hour_utc"].map(
        network_median_slope
    )
    extra: dict[str, pd.Series | np.ndarray] = {}
    extra["feature_current_health_total"] = pd.to_numeric(
        result["health_total"], errors="coerce"
    )
    for component in HEALTH_COMPONENT_COLUMNS:
        extra[f"feature_current_{component}"] = pd.to_numeric(
            result[component], errors="coerce"
        )
    for column in (
        "full_outage_run_hours",
        "partial_outage_run_hours",
        "causal_fault_evidence_rate_7d",
        "reference_severity",
        "stability_event_starts_30d",
        "stability_hours_since_last_adverse",
        "health_history_hours",
    ):
        extra[f"feature_{column}"] = pd.to_numeric(result[column], errors="coerce")
    extra["feature_station_elevation"] = result["station_id"].map(elevation)
    extra["feature_station_age_hours"] = result.groupby("station_id", sort=False).cumcount()
    hour = result["hour_utc"].dt.hour.astype(float)
    day = result["hour_utc"].dt.dayofyear.astype(float)
    extra["feature_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    extra["feature_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    extra["feature_day_of_year_sin"] = np.sin(2.0 * np.pi * day / 366.0)
    extra["feature_day_of_year_cos"] = np.cos(2.0 * np.pi * day / 366.0)
    for station_id in known_station_ids:
        extra[f"feature_station_{station_id}"] = result["station_id"].eq(station_id).astype(float)
    result = pd.concat([result, pd.DataFrame(extra, index=result.index)], axis=1)
    excluded = {
        "station_id",
        "hour_utc",
        "availability_class",
        "absent_sensor_groups",
        "health_status",
        "health_band",
    }
    feature_columns = tuple(
        column
        for column in result.columns
        if column.startswith("feature_") and column not in excluded
    )
    if not feature_columns:
        raise RuntimeError("health forecast construction produced no features")
    return HealthForecastFeatureBundle(
        frame=result.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(drop=True),
        feature_columns=feature_columns,
        station_ids=known_station_ids,
        elevation_by_station=elevation,
    )


def _rolling_sum(values: np.ndarray, window: int, *, min_periods: int) -> np.ndarray:
    if int(window) == 0:
        return np.zeros(len(values), dtype=float)
    series = pd.Series(values, dtype=float)
    return series.rolling(int(window), min_periods=int(min_periods)).sum().to_numpy(dtype=float)


def _project_fault_burden(
    evidence: np.ndarray,
    future_evidence: np.ndarray,
    step: int,
) -> np.ndarray:
    window = int(HEALTH_HISTORY_HOURS)
    keep = window - int(step)
    if keep < 0:
        raise ValueError("roll-forward step must be shorter than the health window")
    if keep == 0:
        rate = np.nan_to_num(future_evidence, nan=0.0)
        return 1.0 - np.clip(rate / float(FAULT_EVIDENCE_RATE_CAP), 0.0, 1.0)
    weights = np.exp(
        -np.log(2.0) * np.arange(window, dtype=float) / float(FAULT_EVIDENCE_HALF_LIFE_HOURS)
    )
    retained = np.convolve(
        np.nan_to_num(evidence, nan=0.0), weights[int(step) :], mode="full"
    )[: len(evidence)]
    retained[: keep - 1] = np.nan
    appended_weight = float(weights[: int(step)].sum())
    rate = (retained + future_evidence * appended_weight) / float(weights.sum())
    return 1.0 - np.clip(rate / float(FAULT_EVIDENCE_RATE_CAP), 0.0, 1.0)


def _project_station_no_new_incident(station: pd.DataFrame, horizon: int) -> pd.Series:
    source = station.reset_index(drop=True).copy()
    h = int(horizon)
    if h <= 0 or h > HEALTH_HISTORY_HOURS:
        raise ValueError("health forecast horizon must be between one and 168 hours")
    n_rows = len(source)
    full = source["availability_class"].eq(AVAILABILITY_CLASS_FULL_OUTAGE).to_numpy()
    partial = source["availability_class"].eq(AVAILABILITY_CLASS_PARTIAL_OUTAGE).to_numpy()
    online = source["availability_class"].eq(AVAILABILITY_CLASS_ONLINE).to_numpy()
    active = full | partial
    future_transmitting = (~full).astype(float)
    evidence_future = source["causal_fault_evidence"].fillna(False).astype(float).to_numpy()
    hard_future = source["stability_hard_fault_evidence"].fillna(False).astype(bool).to_numpy()
    future_adverse = active | hard_future
    starts = source["stability_adverse_now"].fillna(False).astype(bool)
    starts = (starts & ~starts.shift(1, fill_value=False)).astype(float).to_numpy()
    prior_cap = pd.to_numeric(source["outage_base_score_cap"], errors="coerce").to_numpy(dtype=float)
    previous_minimum = prior_cap.copy()
    final_components: dict[str, np.ndarray] = {}
    final_base_total = np.full(n_rows, np.nan, dtype=float)
    for step in range(1, h + 1):
        retained_window = int(HEALTH_HISTORY_HOURS) - step
        transmitted = source["is_transmitting"].fillna(False).astype(float).to_numpy()
        availability = (
            _rolling_sum(transmitted, retained_window, min_periods=retained_window)
            + float(step) * future_transmitting
        ) / float(HEALTH_HISTORY_HOURS)
        component_values: dict[str, np.ndarray] = {
            "health_availability": availability,
            "health_fault_burden": _project_fault_burden(
                source["causal_fault_evidence"].fillna(False).astype(float).to_numpy(),
                evidence_future,
                step,
            ),
            "health_reference_consistency": pd.to_numeric(
                source["base_health_reference_consistency"], errors="coerce"
            ).fillna(1.0).to_numpy(dtype=float),
        }
        group_components: list[np.ndarray] = []
        denominator = (
            _rolling_sum(transmitted, retained_window, min_periods=retained_window)
            + float(step) * future_transmitting
        )
        for group in SENSOR_GROUP_ORDER:
            existing = source[f"sensor_group_present_{group}"].fillna(False).astype(float).to_numpy()
            future_present = np.where(full, 0.0, np.where(partial, existing, 1.0))
            numerator = (
                _rolling_sum(existing, retained_window, min_periods=retained_window)
                + float(step) * future_present
            )
            group_components.append(
                np.divide(
                    numerator,
                    denominator,
                    out=np.zeros(n_rows, dtype=float),
                    where=denominator > 0.0,
                )
            )
        component_values["health_sensor_completeness"] = np.mean(
            np.vstack(group_components), axis=0
        )
        retained_stability = int(STABILITY_HISTORY_HOURS) - step
        event_starts = _rolling_sum(starts, retained_stability, min_periods=1)
        recurrence = 1.0 - np.clip(
            event_starts / float(STABILITY_EVENT_CAP), 0.0, 1.0
        )
        current_since = pd.to_numeric(
            source["stability_hours_since_last_adverse"], errors="coerce"
        ).fillna(float(STABILITY_HISTORY_HOURS)).to_numpy(dtype=float)
        future_since = np.where(
            future_adverse,
            0.0,
            np.clip(current_since + float(step), 0.0, float(STABILITY_HISTORY_HOURS)),
        )
        recovery = np.clip(
            future_since / float(STABILITY_HISTORY_HOURS), 0.0, 1.0
        )
        component_values["health_stability"] = 0.5 * recurrence + 0.5 * recovery
        base_total = np.zeros(n_rows, dtype=float)
        for component, weight in HEALTH_WEIGHTS.items():
            base_total += np.nan_to_num(component_values[component], nan=0.0) * float(weight)
        current_active_minimum = np.minimum(previous_minimum, base_total)
        previous_minimum = np.where(active, current_active_minimum, previous_minimum)
        if step == h:
            final_components = component_values
            final_base_total = base_total
    full_duration = pd.to_numeric(source["full_outage_run_hours"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    partial_duration = pd.to_numeric(source["partial_outage_run_hours"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    final_full_duration = np.where(full, full_duration + float(h), 0.0)
    final_partial_duration = np.where(partial, partial_duration + float(h), 0.0)
    duration = outage_duration_multiplier(
        pd.Series(final_full_duration), pd.Series(final_partial_duration)
    ).to_numpy(dtype=float)
    cap_multiplier = np.where(
        active & (final_base_total > 0.0),
        np.clip(previous_minimum / final_base_total, 0.0, 1.0),
        1.0,
    )
    projected = np.zeros(n_rows, dtype=float)
    for component, weight in HEALTH_WEIGHTS.items():
        projected += (
            np.nan_to_num(final_components[component], nan=0.0)
            * cap_multiplier
            * duration
            * float(weight)
        )
    valid = pd.to_numeric(source["health_total"], errors="coerce").notna().to_numpy()
    projected[~valid] = np.nan
    projected[online & ~np.isfinite(projected)] = np.nan
    return pd.Series(np.clip(projected, 0.0, 100.0), index=station.index, dtype=float)


def roll_forward_health_no_new_incident(
    scores: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    source = _normalise_scores(scores)
    if int(horizon) == 0:
        return pd.to_numeric(source["health_total"], errors="coerce").astype(float)
    values = [
        _project_station_no_new_incident(station, int(horizon))
        for _, station in source.groupby("station_id", sort=False)
    ]
    result = pd.concat(values).reindex(source.index)
    return result.astype(float)


def _recent_trend_projection(source: pd.DataFrame, horizon: int) -> pd.Series:
    values: list[pd.Series] = []
    for _, station in source.groupby("station_id", sort=False):
        slope = _linear_slope(
            pd.to_numeric(station["health_total"], errors="coerce"),
            HEALTH_TREND_WINDOW_HOURS,
        )
        projected = pd.to_numeric(station["health_total"], errors="coerce") + float(horizon) * slope
        fallback = pd.to_numeric(station["health_total"], errors="coerce")
        values.append(projected.where(projected.notna(), fallback).clip(0.0, 100.0))
    return pd.concat(values).reindex(source.index).astype(float)


def attach_health_forecast_inference_baselines(
    frame: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    result = frame.copy(deep=True)
    h = int(horizon)
    if h <= 0 or h > HEALTH_HISTORY_HOURS:
        raise ValueError("health forecast horizon must be between one and 168 hours")
    current = pd.to_numeric(result["health_total"], errors="coerce")
    result["baseline_persistence_level"] = current
    result["baseline_trend_level"] = _recent_trend_projection(result, h)
    result["baseline_no_new_incident_level"] = roll_forward_health_no_new_incident(
        result, h
    )
    for name in ("persistence", "trend", "no_new_incident"):
        result[f"baseline_{name}_delta"] = (
            result[f"baseline_{name}_level"] - current
        )
    result["origin_condition"] = np.select(
        [
            result["availability_class"].eq(AVAILABILITY_CLASS_ONLINE),
            result["availability_class"].isin(
                [AVAILABILITY_CLASS_FULL_OUTAGE, AVAILABILITY_CLASS_PARTIAL_OUTAGE]
            ),
        ],
        ["transmitting_online", "active_outage"],
        default="other",
    )
    return result


def _attach_target_calendar_features(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    result = frame.copy(deep=True)
    target_hour = result["hour_utc"] + pd.Timedelta(hours=int(horizon))
    hour_value = target_hour.dt.hour.astype(float)
    day_value = target_hour.dt.dayofyear.astype(float)
    result["feature_target_hour_sin"] = np.sin(2.0 * np.pi * hour_value / 24.0)
    result["feature_target_hour_cos"] = np.cos(2.0 * np.pi * hour_value / 24.0)
    result["feature_target_day_of_year_sin"] = np.sin(2.0 * np.pi * day_value / 366.0)
    result["feature_target_day_of_year_cos"] = np.cos(2.0 * np.pi * day_value / 366.0)
    return result


def health_forecast_inference_frame(
    bundle: HealthForecastFeatureBundle,
    horizon: int,
) -> pd.DataFrame:
    result = attach_health_forecast_inference_baselines(bundle.frame, int(horizon))
    result = _attach_target_calendar_features(result, int(horizon))
    scoreable = pd.to_numeric(result["health_total"], errors="coerce").notna()
    return result.loc[scoreable].reset_index(drop=True)


def _attach_targets_and_baselines(
    frame: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    result = attach_health_forecast_inference_baselines(frame, horizon)
    h = int(horizon)
    target_level = pd.Series(np.nan, index=result.index, dtype=float)
    target_hour = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    for _, station in result.groupby("station_id", sort=False):
        target_level.loc[station.index] = pd.to_numeric(
            station["health_total"], errors="coerce"
        ).shift(-h).to_numpy(dtype=float)
        target_hour.loc[station.index] = station["hour_utc"].shift(-h).to_numpy()
    exact = target_hour.sub(result["hour_utc"]).dt.total_seconds().div(3600.0).eq(h)
    result["label_end_utc"] = result["hour_utc"] + pd.Timedelta(hours=h)
    result = _attach_target_calendar_features(result, h)
    result["target_health_total"] = target_level.where(exact)
    current = pd.to_numeric(result["health_total"], errors="coerce")
    result["target_delta_health"] = (result["target_health_total"] - current).where(
        current.notna()
    )
    scoreable = (
        current.notna()
        & result["target_health_total"].notna()
        & result["target_delta_health"].notna()
    )
    return result.loc[scoreable].reset_index(drop=True)


def build_health_forecast_dataset(
    scores: pd.DataFrame,
    *,
    station_metadata: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = HEALTH_FORECAST_HORIZONS,
) -> HealthForecastFeatureBundle:
    bundle = build_health_forecast_features(scores, station_metadata=station_metadata)
    source = bundle.frame
    output = source.copy(deep=True)
    for horizon in horizons:
        horizon_frame = _attach_targets_and_baselines(source, int(horizon))
        horizon_frame = horizon_frame.set_index(["station_id", "hour_utc"])
        for column in (
            "target_health_total",
            "target_delta_health",
            "label_end_utc",
            "baseline_persistence_level",
            "baseline_trend_level",
            "baseline_no_new_incident_level",
            "baseline_persistence_delta",
            "baseline_trend_delta",
            "baseline_no_new_incident_delta",
            "origin_condition",
        ):
            output[f"{column}_{int(horizon)}h"] = pd.MultiIndex.from_frame(
                output[["station_id", "hour_utc"]]
            ).map(horizon_frame[column])
    return HealthForecastFeatureBundle(
        frame=output,
        feature_columns=bundle.feature_columns,
        station_ids=bundle.station_ids,
        elevation_by_station=bundle.elevation_by_station,
    )


def health_forecast_horizon_frame(
    bundle: HealthForecastFeatureBundle,
    horizon: int,
) -> pd.DataFrame:
    h = int(horizon)
    source = bundle.frame.copy(deep=True)
    rename = {
        f"target_health_total_{h}h": "target_health_total",
        f"target_delta_health_{h}h": "target_delta_health",
        f"label_end_utc_{h}h": "label_end_utc",
        f"baseline_persistence_level_{h}h": "baseline_persistence_level",
        f"baseline_trend_level_{h}h": "baseline_trend_level",
        f"baseline_no_new_incident_level_{h}h": "baseline_no_new_incident_level",
        f"baseline_persistence_delta_{h}h": "baseline_persistence_delta",
        f"baseline_trend_delta_{h}h": "baseline_trend_delta",
        f"baseline_no_new_incident_delta_{h}h": "baseline_no_new_incident_delta",
        f"origin_condition_{h}h": "origin_condition",
    }
    _require_columns(source, list(rename), "health forecast horizon frame")
    result = source.rename(columns=rename)
    result = result.loc[
        result["target_health_total"].notna() & result["target_delta_health"].notna()
    ].copy()
    result["target_health_total"] = pd.to_numeric(
        result["target_health_total"], errors="coerce"
    )
    result["target_delta_health"] = pd.to_numeric(
        result["target_delta_health"], errors="coerce"
    )
    result = _attach_target_calendar_features(result, h)
    return result.reset_index(drop=True)


def validate_delete_future_health_forecast_features(
    scores: pd.DataFrame,
    *,
    station_metadata: pd.DataFrame | None = None,
    full_bundle: HealthForecastFeatureBundle | None = None,
    sample_size: int = 8,
) -> pd.DataFrame:
    full = (
        build_health_forecast_features(scores, station_metadata=station_metadata)
        if full_bundle is None
        else full_bundle
    )
    candidates = full.frame.loc[
        full.frame["health_total"].notna(), ["station_id", "hour_utc"]
    ].sort_values(["hour_utc", "station_id"], kind="mergesort")
    if candidates.empty:
        raise RuntimeError("no scored station-hours are available for health forecast causality validation")
    positions = np.linspace(0, len(candidates) - 1, num=min(int(sample_size), len(candidates)), dtype=int)
    keys = candidates.iloc[np.unique(positions)].drop_duplicates().reset_index(drop=True)
    source = _normalise_scores(scores)
    rows: list[dict[str, object]] = []
    for key in keys.itertuples(index=False):
        cutoff = pd.Timestamp(key.hour_utc)
        truncated = build_health_forecast_features(
            source.loc[source["hour_utc"].le(cutoff)].copy(),
            station_metadata=station_metadata,
            station_ids=full.station_ids,
        )
        full_row = full.frame.loc[
            full.frame["station_id"].eq(str(key.station_id))
            & full.frame["hour_utc"].eq(cutoff),
            list(full.feature_columns),
        ]
        truncated_row = truncated.frame.loc[
            truncated.frame["station_id"].eq(str(key.station_id))
            & truncated.frame["hour_utc"].eq(cutoff),
            list(full.feature_columns),
        ]
        if len(full_row) != 1 or len(truncated_row) != 1:
            raise RuntimeError("health forecast delete-the-future audit could not recover its score row")
        left = full_row.iloc[0].to_numpy(dtype=float)
        right = truncated_row.iloc[0].to_numpy(dtype=float)
        passed = np.isclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True)
        for feature, original, as_of, is_equal in zip(
            full.feature_columns, left, right, passed, strict=True
        ):
            rows.append(
                {
                    "station_id": str(key.station_id),
                    "hour_utc": cutoff,
                    "feature": feature,
                    "full_value": float(original) if np.isfinite(original) else np.nan,
                    "as_of_value": float(as_of) if np.isfinite(as_of) else np.nan,
                    "passed": bool(is_equal),
                }
            )
        full_station = str(
            full.frame.loc[
                full.frame["station_id"].eq(str(key.station_id))
                & full.frame["hour_utc"].eq(cutoff),
                "station_id",
            ].iloc[0]
        )
        truncated_station = str(
            truncated.frame.loc[
                truncated.frame["station_id"].eq(str(key.station_id))
                & truncated.frame["hour_utc"].eq(cutoff),
                "station_id",
            ].iloc[0]
        )
        rows.append(
            {
                "station_id": str(key.station_id),
                "hour_utc": cutoff,
                "feature": "station_id_categorical",
                "full_value": full_station,
                "as_of_value": truncated_station,
                "passed": full_station == truncated_station,
            }
        )
    return pd.DataFrame(rows)


def summarize_health_forecast_feature_audit(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame(
            [
                {
                    "sample_rows_validated": 0,
                    "features_validated": 0,
                    "comparisons": 0,
                    "failed_comparisons": 0,
                    "all_passed": False,
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "sample_rows_validated": int(
                    audit[["station_id", "hour_utc"]].drop_duplicates().shape[0]
                ),
                "features_validated": int(audit["feature"].nunique()),
                "comparisons": int(len(audit)),
                "failed_comparisons": int((~audit["passed"].astype(bool)).sum()),
                "all_passed": bool(audit["passed"].astype(bool).all()),
            }
        ]
    )


def _feature_matrix(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> np.ndarray:
    return (
        frame.loc[:, list(feature_columns)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float32)
    )


@dataclass
class FittedHealthForecastModel:
    family: str
    feature_columns: tuple[str, ...]
    estimator: object
    station_encoder: object | None = None
    alpha: float = 1.0
    final_policy: str = "learned_residual"
    horizon_h: int | None = None
    regime: str | None = None
    feature_set: str | None = None
    recency_half_life_days: int | None = None
    iterations: int | None = None
    residual_baseline: str = "no_new_incident_roll_forward"

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.family == "hist_gradient_boosting":
            if self.station_encoder is None:
                raise RuntimeError("HGB station encoder is missing")
            numeric = _feature_matrix(frame, self.feature_columns)
            station = self.station_encoder.transform(frame.loc[:, ["station_id"]]).astype(np.float32)
            return np.asarray(self.estimator.predict(np.column_stack([numeric, station])), dtype=float)
        source = _categorical_model_frame(frame, self.feature_columns)
        return np.asarray(self.estimator.predict(source), dtype=float).reshape(-1)

    def predict_health(self, frame: pd.DataFrame) -> np.ndarray:
        if self.final_policy == "direct_regression":
            return np.clip(self.predict(frame), 0.0, 100.0)
        if self.final_policy in {
            "persistence",
            "recent_trend_24h",
            "no_new_incident_roll_forward",
        }:
            return _baseline_level_predictions(frame)[self.final_policy]
        baseline_name = getattr(self, "residual_baseline", "no_new_incident_roll_forward")
        baseline = _baseline_level_predictions(frame)[baseline_name]
        return np.clip(baseline + float(self.alpha) * self.predict(frame), 0.0, 100.0)

    def predict_delta(self, frame: pd.DataFrame) -> np.ndarray:
        current = pd.to_numeric(frame["health_total"], errors="coerce").to_numpy(dtype=float)
        return self.predict_health(frame) - current


@dataclass
class ResidualCandidate:
    family: str
    feature_set: str
    feature_columns: tuple[str, ...]
    recency_half_life_days: int | None
    iterations: int | None
    alpha: float
    validation_metrics: dict[str, float]
    model: FittedHealthForecastModel
    validation_residual_prediction: np.ndarray


def _feature_columns_for_set(
    bundle: HealthForecastFeatureBundle,
    feature_set: str,
) -> tuple[str, ...]:
    if feature_set == "core":
        missing = sorted(set(HEALTH_FORECAST_CORE_FEATURES).difference(bundle.frame.columns))
        if missing:
            raise KeyError(f"core health-forecast features are missing: {', '.join(missing)}")
        return tuple(HEALTH_FORECAST_CORE_FEATURES)
    if feature_set == "full_engineered":
        long_only = set(HEALTH_FORECAST_LONG_FEATURES).difference(
            HEALTH_FORECAST_CORE_FEATURES
        )
        return tuple(
            column for column in bundle.feature_columns if column not in long_only
        )
    if feature_set == "long_horizon":
        available = set(bundle.frame.columns).union(HEALTH_FORECAST_TARGET_CALENDAR_FEATURES)
        missing = sorted(set(HEALTH_FORECAST_LONG_FEATURES).difference(available))
        if missing:
            raise KeyError(f"long-horizon health-forecast features are missing: {', '.join(missing)}")
        return tuple(HEALTH_FORECAST_LONG_FEATURES)
    raise ValueError(feature_set)


def _categorical_model_frame(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    numeric = (
        frame.loc[:, list(feature_columns)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )
    numeric["station_id"] = frame["station_id"].astype(str).to_numpy()
    return numeric


def _recency_weights(frame: pd.DataFrame, half_life_days: int | None) -> np.ndarray | None:
    if half_life_days is None:
        return None
    hours = pd.to_datetime(frame["hour_utc"], utc=True, errors="coerce")
    newest = hours.max()
    age_hours = (newest - hours).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    weights = np.exp(-np.log(2.0) * age_hours / (24.0 * float(half_life_days)))
    return np.clip(weights, 1.0e-4, 1.0)


def _fit_health_forecast_model(
    family: str,
    train: pd.DataFrame,
    target: np.ndarray,
    *,
    feature_columns: tuple[str, ...],
    iterations: int | None,
    sample_weight: np.ndarray | None,
) -> FittedHealthForecastModel:
    if family == "hist_gradient_boosting":
        if iterations is None:
            raise ValueError("HGB iterations must be specified")
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )
        station = encoder.fit_transform(train.loc[:, ["station_id"]]).astype(np.float32)
        numeric = _feature_matrix(train, feature_columns)
        matrix = np.column_stack([numeric, station])
        categorical = np.zeros(matrix.shape[1], dtype=bool)
        categorical[-1] = True
        estimator = HistGradientBoostingRegressor(
            **HEALTH_FORECAST_HGB_PARAMETERS,
            max_iter=int(iterations),
            categorical_features=categorical,
        )
        estimator.fit(matrix, target, sample_weight=sample_weight)
        return FittedHealthForecastModel(family, feature_columns, estimator, encoder)
    if family == "catboost":
        if iterations is None:
            raise ValueError("CatBoost iterations must be specified")
        estimator = CatBoostRegressor(
            **HEALTH_FORECAST_CATBOOST_PARAMETERS,
            iterations=int(iterations),
        )
        estimator.fit(
            _categorical_model_frame(train, feature_columns),
            target,
            cat_features=["station_id"],
            sample_weight=sample_weight,
        )
        return FittedHealthForecastModel(family, feature_columns, estimator)
    if family == "ridge":
        numeric_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]
        )
        preprocessor = ColumnTransformer(
            [
                ("numeric", numeric_pipeline, list(feature_columns)),
                (
                    "station",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ["station_id"],
                ),
            ],
            remainder="drop",
        )
        estimator = Pipeline(
            [
                ("prepare", preprocessor),
                ("model", Ridge(alpha=HEALTH_FORECAST_RIDGE_ALPHA, solver="lsqr")),
            ]
        )
        fit_parameters = {} if sample_weight is None else {"model__sample_weight": sample_weight}
        estimator.fit(_categorical_model_frame(train, feature_columns), target, **fit_parameters)
        return FittedHealthForecastModel(family, feature_columns, estimator)
    raise ValueError(family)


def _residual_target(frame: pd.DataFrame) -> np.ndarray:
    actual = pd.to_numeric(frame["target_health_total"], errors="coerce").to_numpy(dtype=float)
    baseline = pd.to_numeric(
        frame["baseline_no_new_incident_level"], errors="coerce"
    ).to_numpy(dtype=float)
    return actual - baseline


def _health_from_residual(
    frame: pd.DataFrame,
    residual_prediction: np.ndarray,
    alpha: float,
) -> np.ndarray:
    baseline = pd.to_numeric(
        frame["baseline_no_new_incident_level"], errors="coerce"
    ).to_numpy(dtype=float)
    return np.clip(baseline + float(alpha) * np.asarray(residual_prediction, dtype=float), 0.0, 100.0)


def _select_residual_alpha(
    validation: pd.DataFrame,
    residual_prediction: np.ndarray,
) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    actual = pd.to_numeric(
        validation["target_health_total"], errors="coerce"
    ).to_numpy(dtype=float)
    rows: list[dict[str, float]] = []
    for alpha in HEALTH_FORECAST_ALPHA_GRID:
        predicted = _health_from_residual(validation, residual_prediction, alpha)
        rows.append({"alpha": float(alpha), **regression_metrics(actual, predicted)})
    selected = min(rows, key=lambda row: (float(row["mae"]), float(row["alpha"])))
    return float(selected["alpha"]), {
        "mae": float(selected["mae"]),
        "rmse": float(selected["rmse"]),
        "r2": float(selected["r2"]),
    }, rows


def _fit_residual_candidate(
    family: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_set: str,
    feature_columns: tuple[str, ...],
    recency_half_life_days: int | None,
    iterations: int | None,
) -> tuple[ResidualCandidate, list[dict[str, float]]]:
    model = _fit_health_forecast_model(
        family,
        train,
        _residual_target(train),
        feature_columns=feature_columns,
        iterations=iterations,
        sample_weight=_recency_weights(train, recency_half_life_days),
    )
    residual_prediction = model.predict(validation)
    alpha, metrics, alpha_rows = _select_residual_alpha(validation, residual_prediction)
    return (
        ResidualCandidate(
            family=family,
            feature_set=feature_set,
            feature_columns=feature_columns,
            recency_half_life_days=recency_half_life_days,
            iterations=iterations,
            alpha=alpha,
            validation_metrics=metrics,
            model=model,
            validation_residual_prediction=residual_prediction,
        ),
        alpha_rows,
    )


def _candidate_record(
    candidate: ResidualCandidate,
    *,
    horizon: int,
    regime: str,
    stage: str,
) -> dict[str, object]:
    return {
        "horizon_h": int(horizon),
        "regime": regime,
        "selection_stage": stage,
        "model_family": candidate.family,
        "feature_set": candidate.feature_set,
        "n_numeric_features": int(len(candidate.feature_columns)),
        "station_encoding": (
            "one_hot_categorical" if candidate.family == "ridge" else "native_categorical"
        ),
        "recency_half_life_days": candidate.recency_half_life_days,
        "iterations": candidate.iterations,
        "alpha": float(candidate.alpha),
        "validation_mae": float(candidate.validation_metrics["mae"]),
        "validation_rmse": float(candidate.validation_metrics["rmse"]),
        "validation_r2": float(candidate.validation_metrics["r2"]),
        "selection_partition": "validation",
        "test_metrics_accessed": False,
    }


@dataclass
class FittedHealthClassifier:
    feature_columns: tuple[str, ...]
    station_encoder: OrdinalEncoder
    estimator: HistGradientBoostingClassifier
    decision_threshold: float | None = None
    task: str | None = None
    horizon_h: int | None = None
    scope: str | None = None

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = _feature_matrix(frame, self.feature_columns)
        station = self.station_encoder.transform(frame.loc[:, ["station_id"]]).astype(np.float32)
        return np.column_stack([numeric, station])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.decision_threshold is not None:
            classes = list(self.estimator.classes_)
            if True not in classes:
                raise RuntimeError("binary classifier has no positive class")
            probability = self.predict_proba(frame)[:, classes.index(True)]
            return probability >= float(self.decision_threshold)
        return np.asarray(self.estimator.predict(self._matrix(frame)))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(self._matrix(frame)), dtype=float)


def _fit_health_classifier(
    train: pd.DataFrame,
    target: np.ndarray,
    *,
    feature_columns: tuple[str, ...],
) -> FittedHealthClassifier:
    if len(np.unique(np.asarray(target))) < 2:
        raise ValueError("health classifier training requires at least two target classes")
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
    )
    station = encoder.fit_transform(train.loc[:, ["station_id"]]).astype(np.float32)
    numeric = _feature_matrix(train, feature_columns)
    matrix = np.column_stack([numeric, station])
    categorical = np.zeros(matrix.shape[1], dtype=bool)
    categorical[-1] = True
    estimator = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=HEALTH_FORECAST_CLASSIFIER_ITERATIONS,
        max_leaf_nodes=7,
        max_depth=3,
        min_samples_leaf=80,
        l2_regularization=10.0,
        early_stopping=False,
        categorical_features=categorical,
        random_state=2026,
    )
    estimator.fit(matrix, target, sample_weight=compute_sample_weight("balanced", target))
    return FittedHealthClassifier(feature_columns, encoder, estimator)


def _binary_classification_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    score: np.ndarray,
) -> dict[str, float | int]:
    truth = np.asarray(actual, dtype=bool)
    verdict = np.asarray(predicted, dtype=bool)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth.astype(int), verdict.astype(int), average="binary", zero_division=0
    )
    pr_auc = (
        float(average_precision_score(truth.astype(int), np.asarray(score, dtype=float)))
        if truth.any() and (~truth).any()
        else np.nan
    )
    return {
        "n": int(len(truth)),
        "support_positive": int(truth.sum()),
        "support_predicted_positive": int(verdict.sum()),
        "accuracy": float(accuracy_score(truth, verdict)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, verdict)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": pr_auc,
    }


def _select_binary_threshold(actual: np.ndarray, probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float | int]] = []
    for threshold in HEALTH_FORECAST_CLASSIFIER_THRESHOLD_GRID:
        metrics = _binary_classification_metrics(
            actual,
            np.asarray(probability, dtype=float) >= float(threshold),
            probability,
        )
        rows.append({"threshold": float(threshold), **metrics})
    trace = pd.DataFrame(rows)
    selected = trace.sort_values(
        ["f1", "balanced_accuracy", "threshold"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]
    return float(selected["threshold"]), trace


def _trajectory_labels(delta: np.ndarray) -> np.ndarray:
    values = np.asarray(delta, dtype=float)
    return np.select(
        [values <= -5.0, values >= 5.0],
        ["Deteriorating", "Improving"],
        default="Stable",
    ).astype(str)


def _multiclass_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    labels: tuple[str, ...],
) -> dict[str, float | int]:
    truth = np.asarray(actual, dtype=str)
    verdict = np.asarray(predicted, dtype=str)
    recalls = recall_score(truth, verdict, labels=list(labels), average=None, zero_division=0)
    output: dict[str, float | int] = {
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, verdict)),
        "balanced_accuracy": float(
            balanced_accuracy_score(truth, verdict)
        ),
        "macro_f1": float(
            f1_score(truth, verdict, labels=list(labels), average="macro", zero_division=0)
        ),
    }
    for label, value in zip(labels, recalls, strict=True):
        output[f"recall_{label.lower()}"] = float(value)
        output[f"support_{label.lower()}"] = int((truth == label).sum())
    return output


def _confusion_rows(
    actual: np.ndarray,
    predicted: np.ndarray,
    labels: tuple[str, ...],
    **metadata: object,
) -> list[dict[str, object]]:
    matrix = confusion_matrix(actual, predicted, labels=list(labels))
    rows: list[dict[str, object]] = []
    for actual_label, counts in zip(labels, matrix, strict=True):
        for predicted_label, count in zip(labels, counts, strict=True):
            rows.append(
                {
                    **metadata,
                    "actual_class": actual_label,
                    "predicted_class": predicted_label,
                    "count": int(count),
                }
            )
    return rows


def _regime_subset(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
    if regime == "transmitting_origin":
        return frame.loc[frame["is_transmitting"].fillna(False).astype(bool)].copy()
    if regime == "full_outage_origin":
        return frame.loc[
            frame["availability_class"].eq(AVAILABILITY_CLASS_FULL_OUTAGE)
        ].copy()
    raise ValueError(regime)


def _baseline_level_predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "persistence": pd.to_numeric(
            frame["baseline_persistence_level"], errors="coerce"
        ).to_numpy(dtype=float),
        "recent_trend_24h": pd.to_numeric(
            frame["baseline_trend_level"], errors="coerce"
        ).to_numpy(dtype=float),
        "no_new_incident_roll_forward": pd.to_numeric(
            frame["baseline_no_new_incident_level"], errors="coerce"
        ).to_numpy(dtype=float),
    }


def _select_regime_configuration(
    bundle: HealthForecastFeatureBundle,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    horizon: int,
    regime: str,
    selection_rows: list[dict[str, object]],
    iteration_rows: list[dict[str, object]],
    ablation_rows: list[dict[str, object]],
    recency_rows: list[dict[str, object]],
    family_rows: list[dict[str, object]],
    alpha_rows: list[dict[str, object]],
    feature_sets: tuple[str, ...],
) -> tuple[ResidualCandidate, str]:
    cache: dict[tuple[str, str, int | None, int | None], ResidualCandidate] = {}

    def evaluate(
        family: str,
        feature_set: str,
        half_life: int | None,
        iterations: int | None,
        stage: str,
    ) -> ResidualCandidate:
        key = (family, feature_set, half_life, iterations)
        if key not in cache:
            candidate, candidate_alpha_rows = _fit_residual_candidate(
                family,
                train,
                validation,
                feature_set=feature_set,
                feature_columns=_feature_columns_for_set(bundle, feature_set),
                recency_half_life_days=half_life,
                iterations=iterations,
            )
            cache[key] = candidate
            for row in candidate_alpha_rows:
                alpha_rows.append(
                    {
                        "horizon_h": int(horizon),
                        "regime": regime,
                        "selection_stage": stage,
                        "model_family": family,
                        "feature_set": feature_set,
                        "recency_half_life_days": half_life,
                        "iterations": iterations,
                        **row,
                        "selection_partition": "validation",
                        "test_metrics_accessed": False,
                    }
                )
        candidate = cache[key]
        selection_rows.append(
            _candidate_record(candidate, horizon=horizon, regime=regime, stage=stage)
        )
        return candidate

    hgb_iterations = [
        evaluate(
            "hist_gradient_boosting",
            "core",
            None,
            int(iterations),
            "hgb_iteration",
        )
        for iterations in HEALTH_FORECAST_TREE_ITERATION_GRID
    ]
    for candidate in hgb_iterations:
        iteration_rows.append(
            _candidate_record(
                candidate,
                horizon=horizon,
                regime=regime,
                stage="hgb_iteration",
            )
        )
    selected_iterations = min(
        hgb_iterations,
        key=lambda candidate: (
            float(candidate.validation_metrics["mae"]),
            int(candidate.iterations or 0),
        ),
    ).iterations

    ablation_candidates = [
        evaluate(
            "hist_gradient_boosting",
            feature_set,
            None,
            selected_iterations,
            "feature_ablation",
        )
        for feature_set in feature_sets
    ]
    for candidate in ablation_candidates:
        ablation_rows.append(
            _candidate_record(
                candidate,
                horizon=horizon,
                regime=regime,
                stage="feature_ablation",
            )
        )
    selected_feature_set = min(
        ablation_candidates,
        key=lambda candidate: (
            float(candidate.validation_metrics["mae"]),
            feature_sets.index(candidate.feature_set),
        ),
    ).feature_set

    recency_candidates = [
        evaluate(
            "hist_gradient_boosting",
            selected_feature_set,
            half_life,
            selected_iterations,
            "recency_weighting",
        )
        for half_life in HEALTH_FORECAST_RECENCY_HALF_LIVES_DAYS
    ]
    for candidate in recency_candidates:
        recency_rows.append(
            _candidate_record(
                candidate,
                horizon=horizon,
                regime=regime,
                stage="recency_weighting",
            )
        )
    selected_recency = min(
        recency_candidates,
        key=lambda candidate: (
            float(candidate.validation_metrics["mae"]),
            HEALTH_FORECAST_RECENCY_HALF_LIVES_DAYS.index(
                candidate.recency_half_life_days
            ),
        ),
    ).recency_half_life_days

    hgb_family_candidate = evaluate(
        "hist_gradient_boosting",
        selected_feature_set,
        selected_recency,
        selected_iterations,
        "model_family",
    )
    catboost_iterations = [
        evaluate(
            "catboost",
            selected_feature_set,
            selected_recency,
            int(iterations),
            "catboost_iteration",
        )
        for iterations in HEALTH_FORECAST_TREE_ITERATION_GRID
    ]
    for candidate in catboost_iterations:
        iteration_rows.append(
            _candidate_record(
                candidate,
                horizon=horizon,
                regime=regime,
                stage="catboost_iteration",
            )
        )
    catboost_family_candidate = min(
        catboost_iterations,
        key=lambda candidate: (
            float(candidate.validation_metrics["mae"]),
            int(candidate.iterations or 0),
        ),
    )
    ridge_family_candidate = evaluate(
        "ridge",
        "core",
        selected_recency,
        None,
        "model_family",
    )
    family_candidates = [
        hgb_family_candidate,
        catboost_family_candidate,
        ridge_family_candidate,
    ]
    for candidate in family_candidates:
        family_rows.append(
            _candidate_record(
                candidate,
                horizon=horizon,
                regime=regime,
                stage="model_family",
            )
        )
    selected = min(
        family_candidates,
        key=lambda candidate: (
            float(candidate.validation_metrics["mae"]),
            HEALTH_FORECAST_MODEL_FAMILIES.index(candidate.family),
        ),
    )

    final_policy = (
        "no_new_incident_roll_forward"
        if np.isclose(float(selected.alpha), 0.0)
        else "learned_residual"
    )
    if regime == "full_outage_origin":
        actual = pd.to_numeric(
            validation["target_health_total"], errors="coerce"
        ).to_numpy(dtype=float)
        baselines = _baseline_level_predictions(validation)
        deterministic = {
            method: regression_metrics(actual, predicted)
            for method, predicted in baselines.items()
            if method in {"persistence", "no_new_incident_roll_forward"}
        }
        deterministic_method, deterministic_metrics = min(
            deterministic.items(), key=lambda item: (float(item[1]["mae"]), item[0])
        )
        if float(selected.validation_metrics["mae"]) >= float(deterministic_metrics["mae"]):
            final_policy = deterministic_method
    selection_rows.append(
        {
            **_candidate_record(
                selected,
                horizon=horizon,
                regime=regime,
                stage="frozen_configuration",
            ),
            "selected": True,
            "final_policy": final_policy,
            "post_hoc_evaluation": True,
        }
    )
    return selected, final_policy


def _refit_residual_candidate(
    candidate: ResidualCandidate,
    combined: pd.DataFrame,
) -> FittedHealthForecastModel:
    return _fit_health_forecast_model(
        candidate.family,
        combined,
        _residual_target(combined),
        feature_columns=candidate.feature_columns,
        iterations=candidate.iterations,
        sample_weight=_recency_weights(combined, candidate.recency_half_life_days),
    )


def _selected_feature_importance(
    candidate: ResidualCandidate,
    validation: pd.DataFrame,
    *,
    horizon: int,
    regime: str,
) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    sample = validation.sort_values(["hour_utc", "station_id"], kind="mergesort").iloc[
        : min(1_000, len(validation))
    ].copy()
    actual = pd.to_numeric(sample["target_health_total"], errors="coerce").to_numpy(dtype=float)
    baseline_prediction = _health_from_residual(
        sample,
        candidate.model.predict(sample),
        candidate.alpha,
    )
    baseline_mae = float(regression_metrics(actual, baseline_prediction)["mae"])
    generator = np.random.default_rng(2026 + int(horizon))
    rows: list[dict[str, object]] = []
    for feature in (*candidate.feature_columns, "station_id"):
        shuffled = sample.copy(deep=True)
        shuffled[feature] = generator.permutation(shuffled[feature].to_numpy())
        prediction = _health_from_residual(
            shuffled,
            candidate.model.predict(shuffled),
            candidate.alpha,
        )
        rows.append(
            {
                "horizon_h": int(horizon),
                "regime": regime,
                "model_family": candidate.family,
                "feature_set": candidate.feature_set,
                "feature": feature,
                "importance_mean": float(regression_metrics(actual, prediction)["mae"])
                - baseline_mae,
                "selection_partition": "validation",
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["importance_mean", "feature"], ascending=[False, True], kind="mergesort")
        .head(20)
        .reset_index(drop=True)
    )


def _delta_predictions_from_levels(
    frame: pd.DataFrame,
    level_predictions: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    current = pd.to_numeric(frame["health_total"], errors="coerce").to_numpy(dtype=float)
    return {
        method: np.asarray(predicted, dtype=float) - current
        for method, predicted in level_predictions.items()
    }


def _direction_metric_rows(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    horizon: int,
    scope: str,
) -> list[dict[str, object]]:
    train_delta = pd.to_numeric(train["target_delta_health"], errors="coerce").to_numpy(dtype=float)
    train_valid = np.abs(train_delta) >= 1.0
    train_direction = np.sign(train_delta[train_valid]).astype(int)
    if len(train_direction):
        values, counts = np.unique(train_direction, return_counts=True)
        majority_direction = int(values[np.argmax(counts)])
    else:
        majority_direction = -1
    actual_delta = pd.to_numeric(test["target_delta_health"], errors="coerce").to_numpy(dtype=float)
    valid = np.abs(actual_delta) >= 1.0
    actual = np.sign(actual_delta[valid]).astype(int)
    methods = {
        **predictions,
        "training_majority_direction": np.full(len(test), float(majority_direction)),
    }
    rows: list[dict[str, object]] = []
    for method, predicted_delta in methods.items():
        predicted = np.sign(np.asarray(predicted_delta, dtype=float)[valid]).astype(int)
        rows.append(
            {
                "horizon_h": int(horizon),
                "scope": scope,
                "method": method,
                "n": int(valid.sum()),
                "majority_direction": int(majority_direction),
                "direction_accuracy": float(accuracy_score(actual, predicted))
                if len(actual)
                else np.nan,
                "direction_balanced_accuracy": float(
                    balanced_accuracy_score(actual, predicted)
                )
                if len(actual)
                else np.nan,
                "zero_delta_prediction_fraction": float((predicted == 0).mean())
                if len(predicted)
                else np.nan,
            }
        )
    return rows


def _direct_classification_outputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    horizon: int,
    feature_columns: tuple[str, ...],
    selected_level_prediction: np.ndarray,
    models: dict[str, object],
    selection_rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    train_all = train.copy()
    validation_all = validation.copy()
    test_all = test.copy()
    if len(selected_level_prediction) != len(test_all):
        raise ValueError("selected level predictions do not match the full test partition")
    selected_level_all = np.asarray(selected_level_prediction, dtype=float)

    train = _regime_subset(train_all, "transmitting_origin")
    validation = _regime_subset(validation_all, "transmitting_origin")
    test = _regime_subset(test_all, "transmitting_origin")
    combined_transmitting = pd.concat([train, validation], ignore_index=True).sort_values(
        ["hour_utc", "station_id"], kind="mergesort"
    )
    supplied = pd.Series(
        selected_level_all,
        index=pd.MultiIndex.from_frame(test_all[["station_id", "hour_utc"]]),
    )
    selected_level = supplied.reindex(
        pd.MultiIndex.from_frame(test[["station_id", "hour_utc"]])
    ).to_numpy(dtype=float)
    level_predictions = {
        **_baseline_level_predictions(test),
        "selected_residual_forecast": selected_level,
    }
    delta_predictions = _delta_predictions_from_levels(test, level_predictions)
    deterioration_rows: list[dict[str, object]] = []
    deterioration_confusion: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    trajectory_confusion: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []
    band_confusion: list[dict[str, object]] = []

    for threshold_points in (5, 10):
        y_train = pd.to_numeric(train["target_delta_health"], errors="coerce").to_numpy(dtype=float) <= -float(threshold_points)
        y_validation = pd.to_numeric(validation["target_delta_health"], errors="coerce").to_numpy(dtype=float) <= -float(threshold_points)
        y_test = pd.to_numeric(test["target_delta_health"], errors="coerce").to_numpy(dtype=float) <= -float(threshold_points)
        classifier = _fit_health_classifier(train, y_train, feature_columns=feature_columns)
        validation_probability = classifier.predict_proba(validation)[:, list(classifier.estimator.classes_).index(True)]
        selected_threshold, threshold_trace = _select_binary_threshold(
            y_validation, validation_probability
        )
        selected_validation = threshold_trace.loc[
            threshold_trace["threshold"].eq(selected_threshold)
        ].iloc[0]
        selection_rows.append(
            {
                "horizon_h": int(horizon),
                "regime": "transmitting_origin",
                "selection_stage": f"deterioration_{threshold_points}_threshold",
                "model_family": "hist_gradient_boosting_classifier",
                "feature_set": "core",
                "n_numeric_features": int(len(feature_columns)),
                "station_encoding": "native_categorical",
                "iterations": HEALTH_FORECAST_CLASSIFIER_ITERATIONS,
                "threshold": float(selected_threshold),
                "validation_f1": float(selected_validation["f1"]),
                "validation_balanced_accuracy": float(
                    selected_validation["balanced_accuracy"]
                ),
                "selection_partition": "validation",
                "test_metrics_accessed": False,
                "selected": True,
            }
        )
        final_classifier = _fit_health_classifier(
            combined_transmitting,
            pd.to_numeric(
                combined_transmitting["target_delta_health"], errors="coerce"
            ).to_numpy(dtype=float)
            <= -float(threshold_points),
            feature_columns=feature_columns,
        )
        final_classifier.decision_threshold = float(selected_threshold)
        final_classifier.task = f"deterioration_{threshold_points}pt"
        final_classifier.horizon_h = int(horizon)
        final_classifier.scope = "transmitting_origin"
        test_probability = final_classifier.predict_proba(test)[
            :, list(final_classifier.estimator.classes_).index(True)
        ]
        direct_prediction = test_probability >= selected_threshold
        model_key = f"deterioration_{threshold_points}pt_{horizon}h"
        models[model_key] = final_classifier
        baseline_methods = {
            method: np.asarray(delta, dtype=float) <= -float(threshold_points)
            for method, delta in delta_predictions.items()
        }
        for method, predicted in {
            **baseline_methods,
            "direct_classifier": direct_prediction,
        }.items():
            score = (
                test_probability
                if method == "direct_classifier"
                else -np.asarray(delta_predictions[method], dtype=float)
            )
            deterioration_rows.append(
                {
                    "horizon_h": int(horizon),
                    "scope": "transmitting_origin",
                    "drop_threshold_points": int(threshold_points),
                    "method": method,
                    "selected_threshold": float(selected_threshold)
                    if method == "direct_classifier"
                    else np.nan,
                    **_binary_classification_metrics(y_test, predicted, score),
                }
            )
            deterioration_confusion.extend(
                _confusion_rows(
                    np.where(y_test, "Drop", "No drop"),
                    np.where(predicted, "Drop", "No drop"),
                    ("No drop", "Drop"),
                    horizon_h=int(horizon),
                    scope="transmitting_origin",
                    drop_threshold_points=int(threshold_points),
                    method=method,
                )
            )

    trajectory_labels = ("Deteriorating", "Stable", "Improving")
    y_test_trajectory = _trajectory_labels(
        pd.to_numeric(test["target_delta_health"], errors="coerce").to_numpy(dtype=float)
    )
    trajectory_classifier = _fit_health_classifier(
        combined_transmitting,
        _trajectory_labels(
            pd.to_numeric(
                combined_transmitting["target_delta_health"], errors="coerce"
            ).to_numpy(dtype=float)
        ),
        feature_columns=feature_columns,
    )
    trajectory_classifier.task = "trajectory"
    trajectory_classifier.horizon_h = int(horizon)
    trajectory_classifier.scope = "transmitting_origin"
    direct_trajectory = trajectory_classifier.predict(test).astype(str)
    models[f"trajectory_{horizon}h"] = trajectory_classifier
    selection_rows.append(
        {
            "horizon_h": int(horizon),
            "regime": "transmitting_origin",
            "selection_stage": "trajectory_classifier_fixed_configuration",
            "model_family": "hist_gradient_boosting_classifier",
            "feature_set": "core",
            "n_numeric_features": int(len(feature_columns)),
            "station_encoding": "native_categorical",
            "iterations": HEALTH_FORECAST_CLASSIFIER_ITERATIONS,
            "selection_partition": "fixed_a_priori_then_train_plus_validation",
            "test_metrics_accessed": False,
            "selected": True,
        }
    )
    for method, predicted in {
        **{
            method: _trajectory_labels(delta)
            for method, delta in delta_predictions.items()
        },
        "direct_classifier": direct_trajectory,
    }.items():
        trajectory_rows.append(
            {
                "horizon_h": int(horizon),
                "scope": "transmitting_origin",
                "method": method,
                **_multiclass_metrics(y_test_trajectory, predicted, trajectory_labels),
            }
        )
        trajectory_confusion.extend(
            _confusion_rows(
                y_test_trajectory,
                predicted,
                trajectory_labels,
                horizon_h=int(horizon),
                scope="transmitting_origin",
                method=method,
            )
        )

    band_labels = tuple(HEALTH_BANDS)
    y_test_band = _band_labels(
        pd.to_numeric(test["target_health_total"], errors="coerce").to_numpy(dtype=float)
    )
    band_classifier = _fit_health_classifier(
        combined_transmitting,
        _band_labels(
            pd.to_numeric(
                combined_transmitting["target_health_total"], errors="coerce"
            ).to_numpy(dtype=float)
        ),
        feature_columns=feature_columns,
    )
    band_classifier.task = "health_band"
    band_classifier.horizon_h = int(horizon)
    band_classifier.scope = "transmitting_origin"
    direct_band = band_classifier.predict(test).astype(str)
    models[f"health_band_{horizon}h"] = band_classifier
    current_band = _band_labels(
        pd.to_numeric(test["health_total"], errors="coerce").to_numpy(dtype=float)
    )
    transition = y_test_band != current_band
    for method, predicted in {
        **{method: _band_labels(level) for method, level in level_predictions.items()},
        "direct_classifier": direct_band,
    }.items():
        metrics = _multiclass_metrics(y_test_band, predicted, band_labels)
        metrics["transition_support"] = int(transition.sum())
        metrics["transition_accuracy"] = (
            float(accuracy_score(y_test_band[transition], predicted[transition]))
            if transition.any()
            else np.nan
        )
        band_rows.append(
            {
                "horizon_h": int(horizon),
                "scope": "transmitting_origin",
                "method": method,
                **metrics,
            }
        )
        band_confusion.extend(
            _confusion_rows(
                y_test_band,
                predicted,
                band_labels,
                horizon_h=int(horizon),
                scope="transmitting_origin",
                method=method,
            )
        )
    selection_rows.append(
        {
            "horizon_h": int(horizon),
            "regime": "transmitting_origin",
            "selection_stage": "health_band_classifier_fixed_configuration",
            "model_family": "hist_gradient_boosting_classifier",
            "feature_set": "core",
            "n_numeric_features": int(len(feature_columns)),
            "station_encoding": "native_categorical",
            "iterations": HEALTH_FORECAST_CLASSIFIER_ITERATIONS,
            "selection_partition": "fixed_a_priori_then_train_plus_validation",
            "test_metrics_accessed": False,
            "selected": True,
        }
    )
    return {
        "deterioration_metrics": deterioration_rows,
        "deterioration_confusion": deterioration_confusion,
        "trajectory_metrics": trajectory_rows,
        "trajectory_confusion": trajectory_confusion,
        "band_metrics": band_rows,
        "band_confusion": band_confusion,
    }


def _metric_rows(
    frame: pd.DataFrame,
    *,
    horizon: int,
    target: str,
    predictions: dict[str, np.ndarray],
    scope: str,
) -> list[dict[str, object]]:
    actual_column = "target_health_total" if target == "level" else "target_delta_health"
    actual = pd.to_numeric(frame[actual_column], errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for method, predicted in predictions.items():
        valid = np.isfinite(actual) & np.isfinite(predicted)
        if not valid.any():
            metrics = {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
        else:
            metrics = regression_metrics(actual[valid], predicted[valid])
        rows.append(
            {
                "horizon_h": int(horizon),
                "target": target,
                "scope": scope,
                "method": method,
                "n": int(valid.sum()),
                **metrics,
            }
        )
    return rows


def _classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual.astype(int), predicted.astype(int), average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy_score(actual.astype(int), predicted.astype(int))),
    }


def _band_labels(values: np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    return np.select(
        [numeric >= 80.0, numeric >= 60.0, numeric >= 40.0],
        ["Healthy", "Watch", "Degraded"],
        default="Critical",
    ).astype(str)


def _split_digest(frame: pd.DataFrame) -> str:
    digest = sha256()
    for row in frame.loc[:, ["station_id", "hour_utc"]].itertuples(index=False):
        digest.update(f"{row.station_id}|{pd.Timestamp(row.hour_utc).isoformat()}\n".encode("utf-8"))
    return digest.hexdigest()


def _calibration_table(
    actual: np.ndarray,
    predicted: np.ndarray,
    horizon: int,
    method: str = "selected_residual_forecast",
    scope: str = "transmitting_origin",
) -> pd.DataFrame:
    source = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if source.empty:
        return pd.DataFrame()
    source["prediction_decile"] = pd.qcut(
        source["predicted"].rank(method="first"), q=min(10, len(source)), labels=False
    ).astype(int) + 1
    result = source.groupby("prediction_decile", as_index=False).agg(
        n=("actual", "size"),
        predicted_mean=("predicted", "mean"),
        actual_mean=("actual", "mean"),
        actual_min=("actual", "min"),
        actual_max=("actual", "max"),
    )
    result.insert(0, "horizon_h", int(horizon))
    result.insert(1, "target", "level")
    result.insert(2, "method", method)
    result.insert(3, "scope", scope)
    return result


def _comparison_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (horizon, target, scope), group in metrics.groupby(
        ["horizon_h", "target", "scope"], sort=True
    ):
        model = group.loc[group["method"].eq("selected_residual_forecast")]
        if len(model) != 1:
            continue
        model_row = model.iloc[0]
        for baseline in HEALTH_FORECAST_METHODS[:-1]:
            reference = group.loc[group["method"].eq(baseline)]
            if len(reference) != 1:
                continue
            base = reference.iloc[0]
            denominator = 1.0 - float(base["r2"])
            r2_error_reduction = (
                100.0 * (float(model_row["r2"]) - float(base["r2"])) / denominator
                if np.isfinite(denominator) and not np.isclose(denominator, 0.0)
                else np.nan
            )
            rows.append(
                {
                    "horizon_h": int(horizon),
                    "target": target,
                    "scope": scope,
                    "baseline": baseline,
                    "model": "selected_residual_forecast",
                    "mae_improvement_pct": regression_error_improvement_percent(
                        float(base["mae"]), float(model_row["mae"])
                    ),
                    "rmse_improvement_pct": regression_error_improvement_percent(
                        float(base["rmse"]), float(model_row["rmse"])
                    ),
                    "r2_difference": float(model_row["r2"]) - float(base["r2"]),
                    "r2_error_reduction_pct": r2_error_reduction,
                }
            )
    return pd.DataFrame(rows)


def _master_comparison_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (horizon, target, scope), group in metrics.groupby(
        ["horizon_h", "target", "scope"], sort=True
    ):
        indexed = group.set_index("method")
        if not {
            "persistence",
            "no_new_incident_roll_forward",
        }.issubset(indexed.index):
            continue
        persistence = indexed.loc["persistence"]
        roll_forward = indexed.loc["no_new_incident_roll_forward"]
        for method, result in indexed.iterrows():
            rows.append(
                {
                    "horizon_h": int(horizon),
                    "target": target,
                    "scope": scope,
                    "method": method,
                    "n": int(result["n"]),
                    "mae": float(result["mae"]),
                    "rmse": float(result["rmse"]),
                    "r2": float(result["r2"]),
                    "mae_improvement_vs_persistence_pct": regression_error_improvement_percent(
                        float(persistence["mae"]), float(result["mae"])
                    ),
                    "rmse_improvement_vs_persistence_pct": regression_error_improvement_percent(
                        float(persistence["rmse"]), float(result["rmse"])
                    ),
                    "mae_improvement_vs_roll_forward_pct": regression_error_improvement_percent(
                        float(roll_forward["mae"]), float(result["mae"])
                    ),
                    "rmse_improvement_vs_roll_forward_pct": regression_error_improvement_percent(
                        float(roll_forward["rmse"]), float(result["rmse"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def _health_forecast_degradation_tables(
    run: HealthForecastRun,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    regression = _master_comparison_table(run.metrics)
    regression = regression.loc[
        regression["target"].eq("level")
        & regression["scope"].eq("transmitting_origin")
    ].sort_values(["horizon_h", "method"], kind="mergesort").reset_index(drop=True)
    band = run.band_metrics.loc[
        run.band_metrics["scope"].eq("transmitting_origin")
    ].sort_values(["horizon_h", "method"], kind="mergesort").reset_index(drop=True)
    return regression, band


def _validate_transmitting_population_counts(run: HealthForecastRun) -> pd.DataFrame:
    regression = run.metrics.loc[
        run.metrics["target"].eq("level")
        & run.metrics["scope"].eq("transmitting_origin")
        & run.metrics["method"].eq("selected_residual_forecast"),
        ["horizon_h", "n"],
    ].drop_duplicates()
    if regression["horizon_h"].duplicated().any():
        raise RuntimeError("transmitting regression has duplicate horizon populations")
    expected = {
        int(row.horizon_h): int(row.n) for row in regression.itertuples(index=False)
    }
    if not expected:
        raise RuntimeError("transmitting regression population is missing")

    records: list[dict[str, int]] = []
    for horizon, expected_n in sorted(expected.items()):
        row: dict[str, int] = {
            "horizon_h": horizon,
            "regression_n": expected_n,
        }
        for name, frame in (
            ("trajectory", run.trajectory_metrics),
            ("band", run.band_metrics),
            ("deterioration", run.deterioration_metrics),
        ):
            subset = frame.loc[
                frame["horizon_h"].eq(horizon)
                & frame["scope"].eq("transmitting_origin")
            ]
            counts = sorted(set(pd.to_numeric(subset["n"], errors="raise").astype(int)))
            if counts != [expected_n]:
                raise RuntimeError(
                    f"{name} population at {horizon}h does not match transmitting regression: "
                    f"expected {expected_n}, found {counts}"
                )
            row[f"{name}_n"] = counts[0]
        calibration_n = int(
            pd.to_numeric(
                run.calibration.loc[
                    run.calibration["horizon_h"].eq(horizon)
                    & run.calibration["scope"].eq("transmitting_origin"),
                    "n",
                ],
                errors="raise",
            ).sum()
        )
        if calibration_n != expected_n:
            raise RuntimeError(
                f"calibration population at {horizon}h does not match transmitting "
                f"regression: expected {expected_n}, found {calibration_n}"
            )
        row["calibration_n"] = calibration_n
        records.append(row)
    return pd.DataFrame(records)


def _legacy_part_a_diagnostics() -> dict[str, pd.DataFrame]:
    signed_summary = pd.DataFrame(
        [
            ("overall", 34892, -14.401663, -17.534893, 17.619613),
            ("transmitting_origin", 28871, -15.936331, -18.154911, 17.795009),
            ("full_outage_origin", 6021, -7.042852, 0.276510, 16.778583),
            ("online", 25681, -16.748705, -18.186751, 17.479345),
            ("partial_outage", 3190, -9.396339, -17.575401, 20.336249),
            ("test_month_2026_05", 16796, -14.697652, -17.328871, 17.195784),
            ("test_month_2026_06", 18096, -14.126937, -17.774326, 18.012996),
            ("future_drop_at_least_5", 8150, -4.226781, -7.125642, 10.635261),
            ("no_future_drop_at_least_5", 26742, -17.502601, -19.923768, 19.748193),
        ],
        columns=["group", "n", "mean_signed_error", "median_signed_error", "mae"],
    )
    station_values = {
        "I90583612": (-9.906706, -15.112930),
        "IALWAH18": (6.409048, 5.950783),
        "IBARAS3": (-18.375058, -19.874206),
        "IBIRAL3": (-19.931008, -20.921194),
        "IDERNA7": (-18.588742, -19.611050),
        "IJABAL13": (-18.728869, -20.154206),
        "IJABAL14": (-16.745557, -17.163984),
        "IJABAL15": (-17.507222, -18.301960),
        "IJABAL16": (-17.271880, -19.361461),
        "IJANZO2": (-17.658720, -20.718724),
        "IJANZO3": (-17.097862, -18.073115),
        "IJANZO4": (9.271195, 9.141113),
        "IMISRA12": (-16.259572, -17.237891),
        "IMISRA13": (-18.844953, -19.636305),
        "IMURQU5": (-17.360436, -22.680083),
        "IMURQU6": (-15.368578, -16.221892),
        "IMURQU7": (-16.922721, -18.932672),
        "INALUT3": (-20.952514, -22.045477),
        "INUQAT10": (-14.573931, -14.964597),
        "INUQAT8": (-15.479054, -17.557533),
        "INUQAT9": (-20.029818, -20.655593),
        "ITAHLI1": (-14.803980, -15.167429),
        "ITRIPO32": (-0.396205, 2.230121),
        "ITRIPO33": (-23.053547, -23.517557),
        "IZAWIY5": (-11.218777, -11.752520),
        "IZAWIY7": (-13.047765, -14.513171),
    }
    stations = pd.DataFrame(
        [
            {
                "station_id": station,
                "n": 1342,
                "mean_signed_error": values[0],
                "median_signed_error": values[1],
            }
            for station, values in station_values.items()
        ]
    )
    validation_test = pd.DataFrame(
        [
            ("validation", "persistence", 34216, 7.719746, 12.156494, 0.772171),
            ("validation", "no_new_incident_roll_forward", 34216, 10.692857, 16.684007, 0.570866),
            ("validation", "legacy_hgb", 34216, 10.414566, 14.233138, 0.687685),
            ("test", "persistence", 34892, 7.321222, 11.821250, 0.748457),
            ("test", "no_new_incident_roll_forward", 34892, 10.684404, 17.504223, 0.448469),
            ("test", "legacy_hgb", 34892, 17.619613, 19.639723, 0.305687),
        ],
        columns=["partition", "method", "n", "mae", "rmse", "r2"],
    )
    return {
        "signed_error_summary": signed_summary,
        "signed_error_by_station": stations,
        "validation_test": validation_test,
    }


def _format_frame(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False) if not frame.empty else "No rows."


def build_health_forecast_report(run: HealthForecastRun) -> str:
    run_horizons = tuple(
        sorted(pd.to_numeric(run.metrics["horizon_h"], errors="raise").astype(int).unique())
    )
    horizon_text = "/".join(str(horizon) for horizon in run_horizons)
    long_horizon_run = bool(run_horizons) and min(run_horizons) >= 48
    title = (
        "HEALTH SCORE FORECASTING: TWO-TO-SEVEN-DAY FEATURE COMPARISON"
        if long_horizon_run
        else "HEALTH SCORE FORECASTING: FIVE-HORIZON TRANSMITTING-STATION REPORT"
    )
    methodological_disclosure = (
        "Methodological disclosure: this supervisor-requested 2-to-7-day extension evaluates the predeclared "
        f"{horizon_text}-hour grid. Model family, feature set, recency weighting, alpha, and all other "
        "data-driven choices were selected on chronological validation only; final models were refitted on "
        "train plus validation and each frozen configuration was evaluated once on test. The original "
        "1/3/6/12/24-hour deployed forecast and its artifacts are not modified by this comparison."
        if long_horizon_run
        else "Methodological disclosure: the first test evaluation at commit 8c7b748 and the 6/12/24-hour residual-forecaster results at commit a26881d influenced this extension and operational rescope. The five-horizon grid and transmitting-origin primary scope were locked before this run. Every data-driven model, feature, recency, alpha, and threshold choice below used chronological validation only; trajectory and band classifier configurations were fixed a priori; final models were refitted on train plus validation; and each frozen configuration was evaluated once on test. These are post-hoc model-improvement results, not a clean single-shot test."
    )
    audit_summary = summarize_health_forecast_feature_audit(run.feature_audit)
    diagnostics = _legacy_part_a_diagnostics()
    master_comparison = _master_comparison_table(run.metrics)
    frozen = run.selection_trace.loc[
        run.selection_trace["selection_stage"].eq("frozen_configuration")
    ].copy()
    selected_columns = [
        column
        for column in (
            "horizon_h",
            "regime",
            "model_family",
            "feature_set",
            "recency_half_life_days",
            "iterations",
            "alpha",
            "validation_mae",
            "validation_rmse",
            "validation_r2",
            "final_policy",
        )
        if column in frozen.columns
    ]
    overall_improvement = run.improvements.loc[
        run.improvements["scope"].eq("overall")
        & run.improvements["target"].eq("level")
        & run.improvements["baseline"].isin(
            ["persistence", "no_new_incident_roll_forward"]
        )
    ]
    horizon_conclusions: list[str] = []
    for horizon in run_horizons:
        group = run.metrics.loc[
            run.metrics["horizon_h"].eq(horizon)
            & run.metrics["target"].eq("level")
            & run.metrics["scope"].eq("overall")
        ].set_index("method")
        if not {
            "selected_residual_forecast",
            "persistence",
            "no_new_incident_roll_forward",
        }.issubset(group.index):
            continue
        selected_mae = float(group.loc["selected_residual_forecast", "mae"])
        persistence_mae = float(group.loc["persistence", "mae"])
        roll_mae = float(group.loc["no_new_incident_roll_forward", "mae"])
        roll_improvement = regression_error_improvement_percent(roll_mae, selected_mae)
        persistence_improvement = regression_error_improvement_percent(
            persistence_mae, selected_mae
        )
        horizon_conclusions.append(
            f"{horizon}h: selected regime-policy MAE={selected_mae:.3f}; "
            f"{roll_improvement:+.2f}% versus roll-forward and "
            f"{persistence_improvement:+.2f}% versus persistence."
        )
    regime_conclusions: list[str] = []
    final_policies = run.selection_trace.loc[
        run.selection_trace["selection_stage"].eq(
            "final_refit_and_single_test_evaluation"
        )
    ]
    for policy in final_policies.itertuples(index=False):
        group = run.metrics.loc[
            run.metrics["horizon_h"].eq(int(policy.horizon_h))
            & run.metrics["target"].eq("level")
            & run.metrics["scope"].eq(str(policy.regime))
        ].set_index("method")
        if not {
            "selected_residual_forecast",
            "persistence",
            "no_new_incident_roll_forward",
        }.issubset(group.index):
            continue
        selected_mae = float(group.loc["selected_residual_forecast", "mae"])
        persistence_mae = float(group.loc["persistence", "mae"])
        roll_mae = float(group.loc["no_new_incident_roll_forward", "mae"])
        decision = (
            f"accepted {policy.model_family} residual correction at alpha={float(policy.alpha):g}"
            if str(policy.final_policy) == "learned_residual"
            else f"rejected the learned correction and selected {policy.final_policy}"
        )
        regime_conclusions.append(
            f"{int(policy.horizon_h)}h {policy.regime}: {decision}; "
            f"test MAE={selected_mae:.3f}, "
            f"{regression_error_improvement_percent(roll_mae, selected_mae):+.2f}% versus roll-forward, "
            f"{regression_error_improvement_percent(persistence_mae, selected_mae):+.2f}% versus persistence."
        )
    primary_regression = master_comparison.loc[
        master_comparison["target"].eq("level")
        & master_comparison["scope"].eq("transmitting_origin")
    ].reset_index(drop=True)
    primary_band = run.band_metrics.loc[
        run.band_metrics["scope"].eq("transmitting_origin")
    ].reset_index(drop=True)
    primary_band_confusion = run.band_confusion.loc[
        run.band_confusion["scope"].eq("transmitting_origin")
        & run.band_confusion["method"].isin(
            ["persistence", "selected_residual_forecast", "direct_classifier"]
        )
    ].reset_index(drop=True)
    primary_trajectory = run.trajectory_metrics.loc[
        run.trajectory_metrics["scope"].eq("transmitting_origin")
    ].reset_index(drop=True)
    primary_trajectory_confusion = run.trajectory_confusion.loc[
        run.trajectory_confusion["scope"].eq("transmitting_origin")
        & run.trajectory_confusion["method"].isin(
            ["persistence", "selected_residual_forecast", "direct_classifier"]
        )
    ].reset_index(drop=True)
    primary_deterioration = run.deterioration_metrics.loc[
        run.deterioration_metrics["scope"].eq("transmitting_origin")
    ].reset_index(drop=True)
    primary_deterioration_confusion = run.deterioration_confusion.loc[
        run.deterioration_confusion["scope"].eq("transmitting_origin")
        & run.deterioration_confusion["method"].isin(
            ["persistence", "selected_residual_forecast", "direct_classifier"]
        )
    ].reset_index(drop=True)
    primary_direction = run.direction_metrics.loc[
        run.direction_metrics["scope"].eq("transmitting_origin")
    ].reset_index(drop=True)
    primary_frozen = frozen.loc[frozen["regime"].eq("transmitting_origin")].copy()
    supplementary_frozen = frozen.loc[frozen["regime"].eq("full_outage_origin")].copy()
    supplementary_regression = master_comparison.loc[
        master_comparison["target"].eq("level")
        & master_comparison["scope"].eq("full_outage_origin")
    ].reset_index(drop=True)
    transmitting_importance = run.feature_importance.loc[
        run.feature_importance["regime"].eq("transmitting_origin")
    ].reset_index(drop=True)
    regression_degradation, band_degradation = _health_forecast_degradation_tables(run)
    alpha_zero_horizons = sorted(
        pd.to_numeric(
            primary_frozen.loc[
                pd.to_numeric(primary_frozen["alpha"], errors="coerce").eq(0.0),
                "horizon_h",
            ],
            errors="raise",
        ).astype(int).tolist()
    )
    alpha_zero_note = (
        "Validation selected alpha=0 at "
        + ", ".join(f"{horizon}h" for horizon in alpha_zero_horizons)
        + "; at those horizons the deterministic no-new-incident roll-forward is the frozen policy."
        if alpha_zero_horizons
        else "Validation selected a non-zero residual correction at every transmitting-origin horizon."
    )
    supplementary_alpha_zero_horizons = sorted(
        pd.to_numeric(
            supplementary_frozen.loc[
                pd.to_numeric(supplementary_frozen["alpha"], errors="coerce").eq(0.0),
                "horizon_h",
            ],
            errors="raise",
        ).astype(int).tolist()
    )
    supplementary_alpha_zero_note = (
        "In the supplementary full-outage regime, validation selected alpha=0 at "
        + ", ".join(f"{horizon}h" for horizon in supplementary_alpha_zero_horizons)
        + "; the learned correction was rejected at those horizons."
        if supplementary_alpha_zero_horizons
        else "The supplementary full-outage regime retained a non-zero correction at every horizon."
    )
    short_horizon_notes: list[str] = []
    for horizon in (1, 3):
        group = primary_regression.loc[
            primary_regression["horizon_h"].eq(horizon)
        ].set_index("method")
        if {"persistence", "selected_residual_forecast"}.issubset(group.index):
            short_horizon_notes.append(
                f"{horizon}h: persistence MAE={float(group.loc['persistence', 'mae']):.3f}; "
                f"selected-policy MAE={float(group.loc['selected_residual_forecast', 'mae']):.3f}. "
                "Strong persistence is expected because the health score normally changes little over this interval."
            )
    supplementary_notes: list[str] = []
    for horizon in (12, 24):
        group = supplementary_regression.loc[
            supplementary_regression["horizon_h"].eq(horizon)
        ].set_index("method")
        if {
            "persistence",
            "no_new_incident_roll_forward",
            "selected_residual_forecast",
        }.issubset(group.index):
            selected_mae = float(group.loc["selected_residual_forecast", "mae"])
            persistence_mae = float(group.loc["persistence", "mae"])
            roll_mae = float(group.loc["no_new_incident_roll_forward", "mae"])
            supplementary_notes.append(
                f"{horizon}h full-outage origins: selected MAE={selected_mae:.3f}; "
                f"{regression_error_improvement_percent(roll_mae, selected_mae):+.2f}% versus "
                f"roll-forward and {regression_error_improvement_percent(persistence_mae, selected_mae):+.2f}% "
                "versus persistence. Recovery timing depends on unobserved intervention, so this remains supplementary and low-confidence."
            )
    lines = [
        title,
        "",
        methodological_disclosure,
        "Historical implementation disclosure: one earlier pre-report rebuild was discarded after code review found specification deviations in Ridge feature scope, trajectory/band population, and the network trend definition. The a26881d implementation corrected those issues. This later run deliberately supersedes the earlier overall-population classification scope with the predeclared transmitting-origin deployment scope; no candidate grid, threshold rule, or model choice was changed in response to the current test metrics.",
        "Primary target: residual = health(t+H) - no-new-incident-roll-forward(t+H). Production health is clip(roll-forward + alpha * predicted residual, 0, 100), with validation-only alpha in {0, 0.1, 0.25, 0.5, 0.75, 1}. Delta is derived from the bounded level, so its physical bounds are automatic.",
        "",
        "1. SCOPE",
        "The delivered forecast is defined for stations transmitting at the forecast origin, including stations in a partial outage. A station already in a full outage requires intervention rather than a health forecast, and its recovery time depends on maintenance response that is not observed in this dataset. This is a predeclared operational scope definition, not the deletion of inconvenient results: full-outage-origin regression remains in Section 6 as a low-confidence supplement.",
        "Classification metrics, calibration, and the primary regression table all use the same transmitting-origin station-hours. Persistence, fixed trailing-24-hour trend, and no-new-incident roll-forward remain mandatory comparators.",
        "",
        "2. PRIMARY REGRESSION: TRANSMITTING ORIGINS",
        f"MAE, RMSE, R2, sample count, and percentage improvement versus persistence and roll-forward for the {horizon_text}-hour horizons and all four methods:",
        _format_frame(primary_regression),
        "Short-horizon interpretation:",
        *short_horizon_notes,
        "",
        "3. PRIMARY CLASSIFICATION: TRANSMITTING ORIGINS",
        "3a. Four-band future-health classification. The persistence row is the current-band comparator; transition_accuracy is computed only where the future band differs from the current band. Per-band recall and support are included:",
        _format_frame(primary_band),
        "3a confusion matrices for persistence, selected residual forecast, and the direct classifier:",
        _format_frame(primary_band_confusion),
        "3b. Three-class trajectory with a fixed +/-5-point deadband:",
        _format_frame(primary_trajectory),
        "3b confusion matrices for persistence, selected residual forecast, and the direct classifier:",
        _format_frame(primary_trajectory_confusion),
        "3c. Direction-only supporting metric for non-trivial absolute changes of at least one point:",
        _format_frame(primary_direction),
        "",
        "4. DETERIORATION CLASSIFIERS: TRANSMITTING ORIGINS",
        "The -5-point result is primary and -10-point result secondary. Direct-classifier thresholds were selected on validation only. Accuracy, balanced accuracy, precision, recall, F1, PR-AUC, and support are all shown:",
        _format_frame(primary_deterioration),
        "Confusion matrices for persistence, selected residual forecast, and the direct classifier:",
        _format_frame(primary_deterioration_confusion),
        "",
        "5. VALIDATION-SELECTED TRANSMITTING CONFIGURATION PER HORIZON",
        _format_frame(primary_frozen.loc[:, selected_columns]),
        alpha_zero_note,
        "Alpha=0 is a valid validation result: it means the learned correction was rejected and the deterministic roll-forward was sufficient at that horizon.",
        "",
        "6. SUPPLEMENTARY FULL-OUTAGE-ORIGIN REGRESSION",
        "These rows are retained for transparency but are not the delivered forecast. They are low-confidence because recovery depends on unobserved maintenance intervention:",
        _format_frame(supplementary_regression),
        "Validation-selected supplementary configurations:",
        _format_frame(supplementary_frozen.loc[:, selected_columns]),
        supplementary_alpha_zero_note,
        *supplementary_notes,
        "",
        "7. VALIDATION-ONLY TOP-20 FEATURE IMPORTANCE: TRANSMITTING MODELS",
        _format_frame(transmitting_importance),
        "",
        f"8. DEGRADATION ACROSS {horizon_text}-HOUR HORIZONS",
        "8a. Transmitting-origin regression by horizon:",
        _format_frame(regression_degradation),
        "8b. Transmitting-origin four-band metrics by horizon:",
        _format_frame(band_degradation),
        f"The companion horizon-degradation figure plots selected-policy regression error and band metrics at {horizon_text} hours. Deterioration across horizon is expected because uncertainty accumulates; persistence remains a mandatory comparator.",
        "",
        "APPENDIX A. DIAGNOSTICS OF THE FIRST FORECASTER",
        "A0. Provenance: these frozen diagnostics were replayed from commit 8c7b748 before rebuilding. The legacy prediction ledger SHA-256 is DB0D05BD739561C58E1014488F6FD08FEC6E0B07398B2F07742DF92F305B4FB5 and its metrics-table SHA-256 is C99A97970401907527EE56ADA9B0997E5F30C973C9A5B1B8F92EFC70BD644BB1. Signed error is predicted minus actual; meaningful future drop means health(t+24)-health(t) <= -5.",
        "A1. Loss: squared_error was inherited from sklearn's default; it was not aligned with MAE evaluation.",
        "A2. Early stopping: automatic early stopping was already disabled. All six models ran 250 fixed iterations, so no internal random validation split was used.",
        "A3. Final fit: the test-evaluated models were trained on train only, not refitted on train plus validation.",
        "A4. Bounds: levels were clipped to 0-100. Deltas were only clipped to -100..100; 113 12-hour and 242 24-hour test predictions violated the origin-specific lower bound.",
        "A5. Station identity: 26 one-hot numeric columns were used. There was no ordered integer, but also no native categorical treatment.",
        "A6. The 24-hour HGB signed-error decomposition (predicted minus actual):",
        _format_frame(diagnostics["signed_error_summary"]),
        "A6b. Per-station signed errors; n=1,342 per station:",
        _format_frame(diagnostics["signed_error_by_station"]),
        "A7. Replayed validation versus test, using the unchanged saved model:",
        _format_frame(diagnostics["validation_test"]),
        "Diagnosis: the old HGB already lost to persistence on validation, establishing a formulation failure. Temporal shift compounded it: mean signed bias moved from +1.12 on validation to -14.40 on test.",
        "",
        "APPENDIX B. COMPLETE VALIDATION-ONLY CONFIGURATION SELECTION",
        "B1. Tree iteration comparison:",
        _format_frame(run.iteration_comparison),
        "B2. Core versus full-engineered feature ablation. Core has 53 numeric state/evidence fields plus categorical station identity. Expanded has 181 numeric fields: the legacy 179-feature union plus the two required network features; it retains 26 legacy station-dummy fields in addition to native categorical identity, so this is a historical full-union ablation rather than a purely de-duplicated raw-feature set:",
        _format_frame(run.feature_ablation),
        "B3. Recency-weighting comparison; half-life is in days and blank means unweighted:",
        _format_frame(run.recency_comparison),
        "B4. Model-family comparison:",
        _format_frame(run.model_family_comparison),
        "B5. Frozen regime configurations and selected alpha:",
        _format_frame(frozen.loc[:, selected_columns]),
        "The alpha grid includes zero. Alpha zero rejects the learned correction and makes that candidate equal roll-forward; the separately validated full-outage safety policy may then choose persistence instead.",
        "",
        "B6. COMPLETE SINGLE HELD-OUT REGRESSION LEDGER",
        "B6a. Audit table: MAE, RMSE, R2, and percentage improvement against persistence and roll-forward for the pooled diagnostic view and both origin regimes. Section 2, not the pooled rows here, is the primary delivered result:",
        _format_frame(master_comparison),
        "B6b. Percentage improvement of the selected regime policy over every baseline, including R2 error reduction:",
        _format_frame(run.improvements),
        "B6c. Pooled diagnostic future-level improvement against the two principal baselines:",
        _format_frame(overall_improvement),
        "B6d. Pooled diagnostic per-horizon conclusion:",
        *horizon_conclusions,
        "B6e. Per-horizon, per-regime correction decision:",
        *regime_conclusions,
        "A positive MAE/RMSE improvement percentage means lower error than the named baseline. R2 error reduction is the change in unexplained sum of squares, avoiding misleading percentages of negative R2 values.",
        "",
        "APPENDIX C. COMPLETE TRANSMITTING-ORIGIN CLASSIFICATION LEDGER",
        "C1. Direction accuracy with the training-majority baseline and balanced accuracy:",
        _format_frame(run.direction_metrics),
        "C2. Direct meaningful-deterioration classification; -5 is primary and -10 secondary. Direct-classifier PR-AUC uses probabilities; forecast policies use continuous negative predicted delta as their deterioration ranking score:",
        _format_frame(run.deterioration_metrics),
        "C2 confusion matrices:",
        _format_frame(run.deterioration_confusion),
        "C3. Three-class trajectory classification:",
        _format_frame(run.trajectory_metrics),
        "C3 confusion matrices:",
        _format_frame(run.trajectory_confusion),
        "C4. Future health-band classification. Current-band persistence, balanced accuracy, macro F1, per-band recall, and transition-only accuracy are mandatory context:",
        _format_frame(run.band_metrics),
        "C4 confusion matrices:",
        _format_frame(run.band_confusion),
        "",
        "APPENDIX D. BASELINES AND ASSUMPTION CHECK",
        "Roll-forward is structurally matched but can underperform persistence because it deliberately continues current fault evidence, partial group missingness, and active outage duration while assuming no recovery. The test comparison therefore exposes recovery-timing and persistence-assumption error; persistence remains a mandatory comparator rather than being replaced by roll-forward.",
        "",
        "APPENDIX E. VALIDATION-ONLY TOP-20 FEATURE IMPORTANCE FOR BOTH REGIMES",
        _format_frame(run.feature_importance),
        "",
        "APPENDIX F. TRANSMITTING-ORIGIN HELD-OUT LEVEL CALIBRATION BY PREDICTION DECILE",
        _format_frame(run.calibration),
        "",
        "APPENDIX G. DELETE-THE-FUTURE FEATURE AUDIT",
        _format_frame(audit_summary),
        f"The same audited causal origin features feed all reported horizons. Exact-clock target construction and the horizon-specific {horizon_text}-hour boundary purges are verified separately, so the delete-the-future result applies to every reported horizon.",
        "Station identity is an immutable categorical key from the predeclared station registry. It is audited explicitly, every validation/test station occurs in training, and its encoder is fit only on train or train-plus-validation as appropriate.",
    ]
    return "\n".join(lines) + "\n"


def generate_health_forecast_figures(
    run: HealthForecastRun,
    *,
    comparison_path: Path,
    calibration_path: Path,
    degradation_path: Path,
) -> dict[str, Path]:
    horizons = tuple(
        sorted(pd.to_numeric(run.metrics["horizon_h"], errors="raise").astype(int).unique())
    )
    comparison = run.metrics.loc[
        run.metrics["scope"].eq("transmitting_origin")
        & run.metrics["target"].eq("level")
    ].copy()
    display_names = {
        "persistence": "Persistence",
        "recent_trend_24h": "Trend\n(24 h)",
        "no_new_incident_roll_forward": "Roll-forward",
        "selected_residual_forecast": "Selected\nresidual",
    }
    figure, axes = plt.subplots(1, len(horizons), figsize=(18, 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, horizon in zip(axes, horizons, strict=True):
        subset = comparison.loc[comparison["horizon_h"].eq(horizon)].set_index("method")
        ordered = [method for method in HEALTH_FORECAST_METHODS if method in subset.index]
        axis.bar(
            np.arange(len(ordered)),
            [float(subset.loc[method, "mae"]) for method in ordered],
            color=["#7f8c8d", "#f39c12", "#8e44ad", "#2471a3"],
        )
        axis.set_xticks(
            np.arange(len(ordered)),
            [display_names[method] for method in ordered],
            rotation=0,
            fontsize=9,
        )
        axis.set_title(f"{horizon}-hour level")
        axis.set_ylabel("Held-out test MAE")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Transmitting-origin post-hoc residual health forecast versus fixed baselines"
    )
    figure.tight_layout()
    Path(comparison_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(comparison_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    calibration = run.calibration.loc[
        run.calibration["scope"].eq("transmitting_origin")
    ].copy()
    figure, axes = plt.subplots(1, len(horizons), figsize=(18, 4.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for axis, horizon in zip(axes, horizons, strict=True):
        subset = calibration.loc[calibration["horizon_h"].eq(horizon)]
        axis.plot([0, 100], [0, 100], linestyle="--", color="0.4", label="ideal")
        if not subset.empty:
            axis.plot(subset["predicted_mean"], subset["actual_mean"], marker="o", color="#2471a3", label="model")
        axis.set_title(f"{horizon}-hour level")
        axis.set_xlabel("Mean predicted health")
        axis.set_ylabel("Mean actual health")
        axis.set_xlim(0, 100)
        axis.set_ylim(0, 100)
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best")
    figure.suptitle(
        "Transmitting-origin held-out health-level calibration by prediction decile"
    )
    figure.tight_layout()
    Path(calibration_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(calibration_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    regression_degradation, band_degradation = _health_forecast_degradation_tables(run)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {
        "persistence": "#7f8c8d",
        "recent_trend_24h": "#f39c12",
        "no_new_incident_roll_forward": "#8e44ad",
        "selected_residual_forecast": "#2471a3",
        "direct_classifier": "#1e8449",
    }
    for method in HEALTH_FORECAST_METHODS:
        subset = regression_degradation.loc[
            regression_degradation["method"].eq(method)
        ].sort_values("horizon_h")
        if not subset.empty:
            axes[0].plot(
                subset["horizon_h"],
                subset["mae"],
                marker="o",
                color=colors[method],
                label=display_names[method].replace("\n", " "),
            )
    axes[0].set_title("Regression error")
    axes[0].set_xlabel("Forecast horizon (hours)")
    axes[0].set_ylabel("Held-out test MAE")
    axes[0].set_xticks(horizons)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    for method, metric, linestyle, label in (
        ("persistence", "accuracy", "--", "Current-band persistence accuracy"),
        ("selected_residual_forecast", "accuracy", "-", "Selected-policy accuracy"),
        ("selected_residual_forecast", "balanced_accuracy", "-.", "Selected-policy balanced accuracy"),
        ("selected_residual_forecast", "macro_f1", ":", "Selected-policy macro F1"),
        ("direct_classifier", "accuracy", "-", "Direct-classifier accuracy"),
    ):
        subset = band_degradation.loc[
            band_degradation["method"].eq(method)
        ].sort_values("horizon_h")
        if not subset.empty:
            axes[1].plot(
                subset["horizon_h"],
                subset[metric],
                marker="o",
                linestyle=linestyle,
                color=colors[method],
                label=label,
            )
    axes[1].set_title("Four-band classification")
    axes[1].set_xlabel("Forecast horizon (hours)")
    axes[1].set_ylabel("Held-out test score")
    axes[1].set_xticks(horizons)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    figure.suptitle("Transmitting-origin health forecast degradation from 1 to 24 hours")
    figure.tight_layout()
    Path(degradation_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(degradation_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "comparison_figure": Path(comparison_path),
        "calibration_figure": Path(calibration_path),
        "degradation_figure": Path(degradation_path),
    }


def _write_frame(frame: pd.DataFrame, path: Path, *, parquet: bool = False) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if parquet:
        frame.to_parquet(destination, index=False)
    else:
        frame.to_csv(destination, index=False)
    return destination


def run_health_forecast(
    scores: pd.DataFrame,
    *,
    station_metadata: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = HEALTH_FORECAST_HORIZONS,
    feature_audit_samples: int = 8,
    model_directory: Path | None = None,
    feature_sets: tuple[str, ...] | None = None,
) -> HealthForecastRun:
    horizons = tuple(int(value) for value in horizons)
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("health forecast horizons must be non-empty and unique")
    selected_feature_sets = (
        tuple(feature_sets)
        if feature_sets is not None
        else (
            HEALTH_FORECAST_LONG_FEATURE_SETS
            if max(horizons) >= 48
            else HEALTH_FORECAST_FEATURE_SETS
        )
    )
    if not selected_feature_sets or not set(selected_feature_sets).issubset(
        HEALTH_FORECAST_LONG_FEATURE_SETS
    ):
        raise ValueError("health forecast feature sets are invalid")
    bundle = build_health_forecast_dataset(
        scores,
        station_metadata=station_metadata,
        horizons=horizons,
    )
    audit_bundle = HealthForecastFeatureBundle(
        frame=bundle.frame.drop(
            columns=[
                column
                for column in bundle.frame.columns
                if column.startswith("target_")
                or column.startswith("label_end_")
                or column.startswith("baseline_")
                or column.startswith("origin_condition_")
            ],
            errors="ignore",
        ),
        feature_columns=bundle.feature_columns,
        station_ids=bundle.station_ids,
        elevation_by_station=bundle.elevation_by_station,
    )
    feature_audit = validate_delete_future_health_forecast_features(
        scores,
        station_metadata=station_metadata,
        full_bundle=audit_bundle,
        sample_size=feature_audit_samples,
    )
    audit_summary = summarize_health_forecast_feature_audit(feature_audit)
    if not bool(audit_summary.loc[0, "all_passed"]):
        raise RuntimeError("health forecast feature delete-the-future validation failed")

    metric_rows: list[dict[str, object]] = []
    conditional_rows: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    deterioration_rows: list[dict[str, object]] = []
    deterioration_confusion_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    trajectory_confusion_rows: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []
    band_confusion_rows: list[dict[str, object]] = []
    importance_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    iteration_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    recency_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    split_digests: dict[str, object] = {}
    models: dict[str, object] = {}
    forecast_round_trip_cases: dict[
        str, tuple[pd.DataFrame, np.ndarray, np.ndarray]
    ] = {}

    for horizon in horizons:
        h = int(horizon)
        frame = health_forecast_horizon_frame(bundle, h)
        split = split_timestamp_partitions(
            frame,
            target_columns=("target_health_total", "target_delta_health"),
            horizon_h=h,
        )
        train = split["train"]
        validation = split["validation"]
        test = split["test"]
        if not all(isinstance(value, pd.DataFrame) for value in (train, validation, test)):
            raise TypeError("health forecast split did not return data frames")
        train_stations = set(train["station_id"].astype(str))
        unseen_stations = sorted(
            set(validation["station_id"].astype(str))
            .union(set(test["station_id"].astype(str)))
            .difference(train_stations)
        )
        if unseen_stations:
            raise RuntimeError(
                "health forecast validation/test contain stations absent from training: "
                + ", ".join(unseen_stations)
            )
        split_digests[str(h)] = {
            "metadata": {
                name: value.isoformat() if isinstance(value, pd.Timestamp) else value
                for name, value in split["metadata"].items()
            },
            "train": _split_digest(train),
            "validation": _split_digest(validation),
            "test": _split_digest(test),
            "purged": _split_digest(split["purged"]),
        }

        selected_by_regime: dict[str, ResidualCandidate] = {}
        policy_by_regime: dict[str, str] = {}
        for regime in ("transmitting_origin", "full_outage_origin"):
            regime_train = _regime_subset(train, regime)
            regime_validation = _regime_subset(validation, regime)
            candidate, policy = _select_regime_configuration(
                bundle,
                regime_train,
                regime_validation,
                horizon=h,
                regime=regime,
                selection_rows=selection_rows,
                iteration_rows=iteration_rows,
                ablation_rows=ablation_rows,
                recency_rows=recency_rows,
                family_rows=family_rows,
                alpha_rows=alpha_rows,
                feature_sets=selected_feature_sets,
            )
            selected_by_regime[regime] = candidate
            policy_by_regime[regime] = policy
            importance_frames.append(
                _selected_feature_importance(
                    candidate,
                    regime_validation,
                    horizon=h,
                    regime=regime,
                )
            )

        selected_level = np.full(len(test), np.nan, dtype=float)
        selected_policy = np.full(len(test), "unassigned", dtype=object)
        selected_family = np.full(len(test), "none", dtype=object)
        confidence = np.full(len(test), "standard", dtype=object)
        for regime in ("transmitting_origin", "full_outage_origin"):
            candidate = selected_by_regime[regime]
            policy = policy_by_regime[regime]
            regime_train = _regime_subset(train, regime)
            regime_validation = _regime_subset(validation, regime)
            regime_test = _regime_subset(test, regime)
            combined = pd.concat([regime_train, regime_validation], ignore_index=True).sort_values(
                ["hour_utc", "station_id"], kind="mergesort"
            )
            mask = (
                test["is_transmitting"].fillna(False).astype(bool).to_numpy()
                if regime == "transmitting_origin"
                else test["availability_class"].eq(AVAILABILITY_CLASS_FULL_OUTAGE).to_numpy()
            )
            if policy == "learned_residual":
                final_model = _refit_residual_candidate(candidate, combined)
                final_model.alpha = float(candidate.alpha)
                final_model.final_policy = policy
                final_model.horizon_h = h
                final_model.regime = regime
                final_model.feature_set = candidate.feature_set
                final_model.recency_half_life_days = candidate.recency_half_life_days
                final_model.iterations = candidate.iterations
                family_name = candidate.family
            else:
                final_model = FittedHealthForecastModel(
                    family="deterministic",
                    feature_columns=(),
                    estimator=None,
                    alpha=0.0,
                    final_policy=policy,
                    horizon_h=h,
                    regime=regime,
                    feature_set=None,
                    recency_half_life_days=None,
                    iterations=None,
                )
                family_name = "deterministic"
            regime_prediction = final_model.predict_health(regime_test)
            regime_delta = final_model.predict_delta(regime_test)
            current_regime_health = pd.to_numeric(
                regime_test["health_total"], errors="coerce"
            ).to_numpy(dtype=float)
            if not np.allclose(
                regime_delta,
                regime_prediction - current_regime_health,
                rtol=1.0e-10,
                atol=1.0e-10,
            ):
                raise RuntimeError("saved regime wrapper does not reproduce its selected delta")
            forecast_model_key = f"forecast_{regime}_{h}h"
            models[forecast_model_key] = final_model
            verification_frame = regime_test.iloc[: min(128, len(regime_test))].copy()
            forecast_round_trip_cases[forecast_model_key] = (
                verification_frame,
                final_model.predict_health(verification_frame),
                final_model.predict_delta(verification_frame),
            )
            selected_level[mask] = regime_prediction
            selected_policy[mask] = policy
            selected_family[mask] = family_name
            if regime == "full_outage_origin":
                confidence[mask] = "low"
            selection_rows.append(
                {
                    "horizon_h": h,
                    "regime": regime,
                    "selection_stage": "final_refit_and_single_test_evaluation",
                    "model_family": family_name,
                    "feature_set": candidate.feature_set,
                    "recency_half_life_days": candidate.recency_half_life_days,
                    "iterations": candidate.iterations,
                    "alpha": candidate.alpha,
                    "final_policy": policy,
                    "refit_partition": "train_plus_validation",
                    "prediction_bounds": "health_level_0_100_and_origin_specific_delta",
                    "test_metrics_accessed_during_selection": False,
                    "test_evaluations": 1,
                    "post_hoc_evaluation": True,
                }
            )
        if not np.isfinite(selected_level).all():
            raise RuntimeError(f"health forecast policy left unassigned test rows at {h} hours")

        level_predictions = {
            **_baseline_level_predictions(test),
            "selected_residual_forecast": selected_level,
        }
        delta_predictions = _delta_predictions_from_levels(test, level_predictions)
        for target, predictions in (("level", level_predictions), ("delta", delta_predictions)):
            metric_rows.extend(
                _metric_rows(
                    test,
                    horizon=h,
                    target=target,
                    predictions=predictions,
                    scope="overall",
                )
            )
            for regime in ("transmitting_origin", "full_outage_origin"):
                mask = (
                    test["is_transmitting"].fillna(False).astype(bool).to_numpy()
                    if regime == "transmitting_origin"
                    else test["availability_class"].eq(AVAILABILITY_CLASS_FULL_OUTAGE).to_numpy()
                )
                rows = _metric_rows(
                    test.loc[mask],
                    horizon=h,
                    target=target,
                    predictions={method: prediction[mask] for method, prediction in predictions.items()},
                    scope=regime,
                )
                metric_rows.extend(rows)
                conditional_rows.extend(rows)
            actual_column = "target_health_total" if target == "level" else "target_delta_health"
            actual = pd.to_numeric(test[actual_column], errors="coerce").to_numpy(dtype=float)
            for method, prediction in predictions.items():
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "horizon_h": h,
                            "target": target,
                            "method": method,
                            "station_id": test["station_id"].to_numpy(),
                            "hour_utc": test["hour_utc"].to_numpy(),
                            "availability_class": test["availability_class"].to_numpy(),
                            "is_transmitting": test["is_transmitting"].to_numpy(),
                            "forecast_policy": selected_policy
                            if method == "selected_residual_forecast"
                            else method,
                            "model_family": selected_family
                            if method == "selected_residual_forecast"
                            else "deterministic",
                            "confidence": confidence
                            if method == "selected_residual_forecast"
                            else "baseline",
                            "actual": actual,
                            "predicted": prediction,
                        }
                    )
                )
        transmitting_mask = test["is_transmitting"].fillna(False).astype(bool).to_numpy()
        transmitting_train = _regime_subset(train, "transmitting_origin")
        transmitting_test = _regime_subset(test, "transmitting_origin")
        transmitting_level = selected_level[transmitting_mask]
        calibration_frames.append(
            _calibration_table(
                pd.to_numeric(
                    transmitting_test["target_health_total"], errors="coerce"
                ).to_numpy(dtype=float),
                transmitting_level,
                h,
                scope="transmitting_origin",
            )
        )

        direction_rows.extend(
            _direction_metric_rows(
                transmitting_train,
                transmitting_test,
                {
                    method: prediction[transmitting_mask]
                    for method, prediction in delta_predictions.items()
                },
                horizon=h,
                scope="transmitting_origin",
            )
        )

        classification = _direct_classification_outputs(
            train,
            validation,
            test,
            horizon=h,
            feature_columns=tuple(HEALTH_FORECAST_CORE_FEATURES),
            selected_level_prediction=selected_level,
            models=models,
            selection_rows=selection_rows,
        )
        deterioration_rows.extend(classification["deterioration_metrics"])
        deterioration_confusion_rows.extend(classification["deterioration_confusion"])
        trajectory_rows.extend(classification["trajectory_metrics"])
        trajectory_confusion_rows.extend(classification["trajectory_confusion"])
        band_rows.extend(classification["band_metrics"])
        band_confusion_rows.extend(classification["band_confusion"])

    model_artifact_hashes: dict[str, str] = {}
    retired_model_artifacts: list[str] = []
    forecast_model_round_trip_verified = False
    if model_directory is not None:
        directory = Path(model_directory)
        directory.mkdir(parents=True, exist_ok=True)
        resolved_directory = directory.resolve()
        expected_files = {
            f"health_forecast_{name}.joblib" for name in models
        }
        for previous in directory.glob("health_forecast_*.joblib"):
            if previous.resolve().parent != resolved_directory:
                raise RuntimeError("health forecast model cleanup escaped its output directory")
            if previous.name not in expected_files:
                retired_model_artifacts.append(previous.name)
                previous.unlink()
        for name, model in models.items():
            model_path = directory / f"health_forecast_{name}.joblib"
            joblib.dump(model, model_path)
            digest = sha256()
            with model_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            model_artifact_hashes[name] = digest.hexdigest()
            if name in forecast_round_trip_cases:
                verification_frame, expected_level, expected_delta = (
                    forecast_round_trip_cases[name]
                )
                loaded = joblib.load(model_path)
                loaded_level = loaded.predict_health(verification_frame)
                loaded_delta = loaded.predict_delta(verification_frame)
                if not np.allclose(
                    loaded_level, expected_level, rtol=1.0e-10, atol=1.0e-10
                ) or not np.allclose(
                    loaded_delta, expected_delta, rtol=1.0e-10, atol=1.0e-10
                ):
                    raise RuntimeError(
                        f"saved health forecast model {name} failed round-trip verification"
                    )
        forecast_model_round_trip_verified = (
            len(forecast_round_trip_cases) == 2 * len(horizons)
        )
        if not forecast_model_round_trip_verified:
            raise RuntimeError("not every regime forecast model was round-trip verified")

    metrics = pd.DataFrame(metric_rows)
    importance = (
        pd.concat([frame for frame in importance_frames if not frame.empty], ignore_index=True)
        if any(not frame.empty for frame in importance_frames)
        else pd.DataFrame()
    )
    return HealthForecastRun(
        metrics=metrics,
        improvements=_comparison_rows(metrics),
        conditional_metrics=pd.DataFrame(conditional_rows),
        direction_metrics=pd.DataFrame(direction_rows),
        deterioration_metrics=pd.DataFrame(deterioration_rows),
        deterioration_confusion=pd.DataFrame(deterioration_confusion_rows),
        trajectory_metrics=pd.DataFrame(trajectory_rows),
        trajectory_confusion=pd.DataFrame(trajectory_confusion_rows),
        band_metrics=pd.DataFrame(band_rows),
        band_confusion=pd.DataFrame(band_confusion_rows),
        feature_importance=importance,
        calibration=pd.concat(calibration_frames, ignore_index=True),
        predictions=pd.concat(prediction_frames, ignore_index=True),
        selection_trace=pd.DataFrame(selection_rows),
        iteration_comparison=pd.DataFrame(iteration_rows),
        feature_ablation=pd.DataFrame(ablation_rows),
        recency_comparison=pd.DataFrame(recency_rows),
        model_family_comparison=pd.DataFrame(family_rows),
        alpha_selection=pd.DataFrame(alpha_rows),
        feature_audit=feature_audit,
        split_digests=split_digests,
        models=models,
        model_artifact_hashes=model_artifact_hashes,
        retired_model_artifacts=tuple(sorted(retired_model_artifacts)),
        forecast_model_round_trip_verified=forecast_model_round_trip_verified,
    )


def _manifest_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _health_forecast_model_manifest(run: HealthForecastRun) -> dict[str, object]:
    stages = run.selection_trace["selection_stage"].astype(str)
    frozen_regression = run.selection_trace.loc[
        stages.eq("frozen_configuration")
    ].copy()
    deployment_policies = run.selection_trace.loc[
        stages.eq("final_refit_and_single_test_evaluation")
    ].copy()
    classifier_selection = run.selection_trace.loc[
        stages.str.startswith("deterioration_")
        | stages.isin(
            {
                "trajectory_classifier_fixed_configuration",
                "health_band_classifier_fixed_configuration",
            }
        )
    ].copy()
    saved_models: list[dict[str, object]] = []
    for name, model in sorted(run.models.items()):
        record: dict[str, object] = {
            "model_key": name,
            "model_file": f"health_forecast_{name}.joblib",
            "sha256": run.model_artifact_hashes.get(name),
            "wrapper_type": type(model).__name__,
        }
        if isinstance(model, FittedHealthForecastModel):
            record.update(
                {
                    "task": (
                        "residual_health_regression"
                        if model.final_policy == "learned_residual"
                        else "deterministic_health_policy"
                    ),
                    "horizon_h": model.horizon_h,
                    "regime": model.regime,
                    "model_family": model.family,
                    "final_policy": model.final_policy,
                    "alpha": float(model.alpha),
                    "feature_set": model.feature_set,
                    "recency_half_life_days": model.recency_half_life_days,
                    "iterations": model.iterations,
                    "n_numeric_features": int(len(model.feature_columns)),
                    "station_encoding": (
                        "one_hot_categorical"
                        if model.family == "ridge"
                        else (
                            "native_categorical"
                            if model.family in {"hist_gradient_boosting", "catboost"}
                            else "not_applicable"
                        )
                    ),
                }
            )
        elif isinstance(model, FittedHealthClassifier):
            record.update(
                {
                    "task": model.task,
                    "horizon_h": model.horizon_h,
                    "scope": model.scope,
                    "decision_threshold": model.decision_threshold,
                    "n_numeric_features": int(len(model.feature_columns)),
                    "station_encoding": "native_categorical",
                }
            )
        saved_models.append(record)
    policy_records = _manifest_records(deployment_policies)
    for record in policy_records:
        model_key = f"forecast_{record['regime']}_{int(record['horizon_h'])}h"
        if model_key not in run.models:
            raise RuntimeError(f"deployment manifest cannot find selected model {model_key}")
        record["model_key"] = model_key
        record["model_file"] = f"health_forecast_{model_key}.joblib"
        record["confidence"] = (
            "low" if record.get("regime") == "full_outage_origin" else "standard"
        )
    return {
        "post_hoc_model_improvement_evaluation": True,
        "post_hoc_extension_influenced_by_commit": "a26881d",
        "forecast_horizons_h": sorted(
            pd.to_numeric(run.metrics["horizon_h"], errors="raise").astype(int).unique().tolist()
        ),
        "primary_forecast_scope": "transmitting_origin",
        "full_outage_scope": "supplementary_low_confidence",
        "part_a_legacy_source_commit": "8c7b748",
        "part_a_legacy_prediction_sha256": "db0d05bd739561c58e1014488f6fd08fec6e0b07398b2f07742df92f305b4fb5",
        "part_a_legacy_metrics_sha256": "c99a97970401907527ee56ada9b0997e5f30c973c9a5b1b8f92efc70bd644bb1",
        "selection_partition": "chronological_validation_only",
        "final_fit_partition": "train_plus_validation",
        "frozen_regression_configurations": _manifest_records(frozen_regression),
        "deployment_regime_policies": policy_records,
        "classifier_configurations": _manifest_records(classifier_selection),
        "saved_models": saved_models,
        "retired_model_artifacts": list(run.retired_model_artifacts),
        "forecast_model_round_trip_verified": run.forecast_model_round_trip_verified,
    }


def write_health_forecast_outputs(
    run: HealthForecastRun,
    *,
    output_directory: Path,
    comparison_figure_path: Path,
    calibration_figure_path: Path,
    degradation_figure_path: Path,
    input_hashes_before: dict[str, str],
    input_hashes_after: dict[str, str],
) -> dict[str, Path]:
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("an upstream health-forecast artifact changed during the run")
    audit_summary = summarize_health_forecast_feature_audit(run.feature_audit)
    if not bool(audit_summary.loc[0, "all_passed"]):
        raise RuntimeError("health forecast feature causality audit failed")
    selection_flag_columns = [
        column
        for column in (
            "test_metrics_accessed",
            "test_metrics_accessed_during_selection",
        )
        if column in run.selection_trace.columns
    ]
    for column in selection_flag_columns:
        accessed = run.selection_trace[column].eq(True)
        if accessed.any():
            raise RuntimeError(f"selection trace reports test access in {column}")
    selected_level = run.predictions.loc[
        run.predictions["target"].eq("level")
        & run.predictions["method"].eq("selected_residual_forecast"),
        "predicted",
    ]
    if not selected_level.between(0.0, 100.0, inclusive="both").all():
        raise RuntimeError("selected health-level predictions violate 0-100 bounds")
    selected_predictions = run.predictions.loc[
        run.predictions["method"].eq("selected_residual_forecast")
    ]
    level_rows = selected_predictions.loc[
        selected_predictions["target"].eq("level"),
        ["horizon_h", "station_id", "hour_utc", "actual", "predicted"],
    ].rename(columns={"actual": "actual_level", "predicted": "predicted_level"})
    delta_rows = selected_predictions.loc[
        selected_predictions["target"].eq("delta"),
        ["horizon_h", "station_id", "hour_utc", "actual", "predicted"],
    ].rename(columns={"actual": "actual_delta", "predicted": "predicted_delta"})
    if len(level_rows) != len(delta_rows):
        raise RuntimeError("selected level and delta ledgers have different row counts")
    bounds = level_rows.merge(
        delta_rows,
        on=["horizon_h", "station_id", "hour_utc"],
        how="inner",
        validate="one_to_one",
    )
    if len(bounds) != len(level_rows):
        raise RuntimeError("selected level and delta ledgers have different station-hour keys")
    current_health = bounds["actual_level"] - bounds["actual_delta"]
    lower = -current_health
    upper = 100.0 - current_health
    within_bounds = bounds["predicted_delta"].ge(lower - 1.0e-10) & bounds[
        "predicted_delta"
    ].le(upper + 1.0e-10)
    consistent = np.isclose(
        bounds["predicted_delta"].to_numpy(dtype=float),
        bounds["predicted_level"].to_numpy(dtype=float) - current_health.to_numpy(dtype=float),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    if not bool(within_bounds.all()) or not bool(consistent.all()):
        raise RuntimeError("selected delta predictions violate origin-specific physical bounds")
    transmitting_population_counts = _validate_transmitting_population_counts(run)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    master_comparison = _master_comparison_table(run.metrics)
    regression_degradation, band_degradation = _health_forecast_degradation_tables(run)
    accuracy_curve = run.band_metrics.loc[
        run.band_metrics["scope"].eq("transmitting_origin")
        & run.band_metrics["method"].eq("selected_residual_forecast"),
        ["horizon_h", "n", "accuracy"],
    ].sort_values("horizon_h", kind="mergesort").reset_index(drop=True)
    if accuracy_curve["horizon_h"].duplicated().any():
        raise RuntimeError("health forecast accuracy curve has duplicate horizons")
    output_paths = {
        "metrics": _write_frame(run.metrics, destination / "health_forecast_metrics.csv"),
        "master_comparison": _write_frame(master_comparison, destination / "health_forecast_master_comparison.csv"),
        "improvements": _write_frame(run.improvements, destination / "health_forecast_improvements.csv"),
        "conditional_metrics": _write_frame(run.conditional_metrics, destination / "health_forecast_conditional_metrics.csv"),
        "direction_metrics": _write_frame(run.direction_metrics, destination / "health_forecast_direction_metrics.csv"),
        "deterioration_metrics": _write_frame(run.deterioration_metrics, destination / "health_forecast_deterioration_metrics.csv"),
        "deterioration_confusion": _write_frame(run.deterioration_confusion, destination / "health_forecast_deterioration_confusion.csv"),
        "trajectory_metrics": _write_frame(run.trajectory_metrics, destination / "health_forecast_trajectory_metrics.csv"),
        "trajectory_confusion": _write_frame(run.trajectory_confusion, destination / "health_forecast_trajectory_confusion.csv"),
        "band_metrics": _write_frame(run.band_metrics, destination / "health_forecast_band_metrics.csv"),
        "band_confusion": _write_frame(run.band_confusion, destination / "health_forecast_band_confusion.csv"),
        "feature_importance": _write_frame(run.feature_importance, destination / "health_forecast_feature_importance.csv"),
        "calibration": _write_frame(run.calibration, destination / "health_forecast_calibration.csv"),
        "predictions": _write_frame(run.predictions, destination / "health_forecast_predictions.parquet", parquet=True),
        "selection_trace": _write_frame(run.selection_trace, destination / "health_forecast_selection_trace.csv"),
        "iteration_comparison": _write_frame(run.iteration_comparison, destination / "health_forecast_iteration_comparison.csv"),
        "feature_ablation": _write_frame(run.feature_ablation, destination / "health_forecast_feature_ablation.csv"),
        "recency_comparison": _write_frame(run.recency_comparison, destination / "health_forecast_recency_comparison.csv"),
        "model_family_comparison": _write_frame(run.model_family_comparison, destination / "health_forecast_model_family_comparison.csv"),
        "alpha_selection": _write_frame(run.alpha_selection, destination / "health_forecast_alpha_selection.csv"),
        "feature_audit": _write_frame(run.feature_audit, destination / "health_forecast_delete_future_validation.csv"),
        "feature_audit_summary": _write_frame(audit_summary, destination / "health_forecast_delete_future_summary.csv"),
        "transmitting_population_counts": _write_frame(transmitting_population_counts, destination / "health_forecast_transmitting_population_counts.csv"),
        "regression_degradation": _write_frame(regression_degradation, destination / "health_forecast_horizon_regression_degradation.csv"),
        "band_degradation": _write_frame(band_degradation, destination / "health_forecast_horizon_band_degradation.csv"),
        "accuracy_curve": _write_frame(accuracy_curve, destination / "health_forecast_accuracy_curve.csv"),
    }
    diagnostics = _legacy_part_a_diagnostics()
    for name, frame in diagnostics.items():
        output_paths[f"part_a_{name}"] = _write_frame(
            frame, destination / f"health_forecast_part_a_{name}.csv"
        )
    split_path = destination / "health_forecast_split_digests.json"
    split_path.write_text(json.dumps(run.split_digests, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_paths["split_digests"] = split_path
    manifest_path = destination / "health_forecast_model_manifest.json"
    manifest_path.write_text(
        json.dumps(_health_forecast_model_manifest(run), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_paths["model_manifest"] = manifest_path
    invariants = {
        "input_hashes_before": input_hashes_before,
        "input_hashes_after": input_hashes_after,
        "input_hashes_unchanged": True,
        "feature_delete_the_future_passed": True,
        "test_metrics_accessed_during_selection": False,
        "test_evaluations_per_model": 1,
        "post_hoc_model_improvement_evaluation": True,
        "post_hoc_extension_influenced_by_commit": "a26881d",
        "forecast_horizons_h": sorted(
            pd.to_numeric(run.metrics["horizon_h"], errors="raise").astype(int).unique().tolist()
        ),
        "primary_forecast_scope": "transmitting_origin",
        "primary_scope_rationale": "an active full outage requires intervention and recovery depends on unobserved maintenance response",
        "all_classification_metrics_scope": "transmitting_origin",
        "transmitting_regression_and_classification_row_counts_match": True,
        "part_a_legacy_source_commit": "8c7b748",
        "part_a_legacy_prediction_sha256": "db0d05bd739561c58e1014488f6fd08fec6e0b07398b2f07742df92f305b4fb5",
        "part_a_legacy_metrics_sha256": "c99a97970401907527ee56ada9b0997e5f30c973c9a5b1b8f92efc70bd644bb1",
        "discarded_implementation_validation_run": True,
        "discarded_run_reasons": [
            "ridge_feature_scope_corrected_to_core_only",
            "trajectory_and_band_scope_corrected_to_overall",
            "network_trend_corrected_to_slope_of_network_median_health",
        ],
        "discarded_run_metrics_used_for_selection": False,
        "loss": "absolute_error_for_hgb_and_mae_for_catboost",
        "automatic_early_stopping": False,
        "configuration_selection_partition": "chronological_validation_only",
        "final_model_fit_partition": "train_plus_validation",
        "prediction_bounds": "level_0_100_delta_derived_from_bounded_level",
        "station_encoding": "native_categorical_for_tree_models",
        "station_registry_policy": "predeclared_static_station_registry_with_all_validation_and_test_stations_observed_in_training",
        "alpha_grid": list(HEALTH_FORECAST_ALPHA_GRID),
        "recency_half_lives_days": list(HEALTH_FORECAST_RECENCY_HALF_LIVES_DAYS),
        "tree_iteration_grid": list(HEALTH_FORECAST_TREE_ITERATION_GRID),
        "trend_window_hours_fixed_a_priori": HEALTH_TREND_WINDOW_HOURS,
        "no_new_incident_policy": "state_continuation_with_mechanical_history_aging",
        "models": sorted(run.models),
        "model_artifact_sha256": run.model_artifact_hashes,
        "retired_model_artifacts": list(run.retired_model_artifacts),
        "forecast_model_round_trip_verified": run.forecast_model_round_trip_verified,
    }
    invariant_path = destination / "health_forecast_invariants.json"
    invariant_path.write_text(json.dumps(invariants, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_paths["invariants"] = invariant_path
    report_path = destination / "health_forecast_report.txt"
    report_path.write_text(build_health_forecast_report(run), encoding="utf-8")
    output_paths["report"] = report_path
    output_paths.update(
        generate_health_forecast_figures(
            run,
            comparison_path=comparison_figure_path,
            calibration_path=calibration_figure_path,
            degradation_path=degradation_figure_path,
        )
    )
    return output_paths
