from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.availability.build_availability_events import SENSOR_GROUP_CHANNELS, SENSOR_GROUP_ORDER
from src.availability.health_forecast import roll_forward_health_no_new_incident
from src.availability.health_score import health_band_for_values
from src.config.paths import PROJECT_ROOT, STATION_REGISTRY_PATH
from src.rules.config import PHYSICAL_LIMIT_RULES, ROLLING_VARIANCE_FLAG_THRESHOLD


JULY_START = pd.Timestamp("2026-07-01T00:00:00Z")
JULY_END = pd.Timestamp("2026-07-31T23:00:00Z")
FROZEN_STATISTICS_END = pd.Timestamp("2026-06-30T23:00:00Z")
FROZEN_STATISTICS_ROWS = 166_017
SELECTED_DETECTOR_THRESHOLD = 0.30
JULY_HEALTH_PATH = PROJECT_ROOT / "data/eval/july_2026_health/station_health_scores_through_july.parquet"
JULY_FORECAST_PATH = PROJECT_ROOT / "data/eval/july_2026_health_forecast/july_health_forecast_predictions.parquet"
JULY_DETECTION_PATH = PROJECT_ROOT / "data/eval/one_hour_candidate/july_ef_hgb_binary_predictions.parquet"
JULY_SCORES_PATH = PROJECT_ROOT / "data/eval/july_2026_features/statistical_anomaly_scores.parquet"
JULY_NEIGHBORS_PATH = PROJECT_ROOT / "data/eval/july_2026_features/spatial_neighbors.csv"
HEALTH_COMPONENT_COLUMNS = {
    "Availability": "weighted_health_availability",
    "Sensor completeness": "weighted_health_sensor_completeness",
    "Fault burden": "weighted_health_fault_burden",
    "Reference consistency": "weighted_health_reference_consistency",
    "Stability": "weighted_health_stability",
}
FROZEN_THRESHOLDS = {
    ("humidity_avg_pct", "iforest"): 0.715863,
    ("humidity_avg_pct", "zscore"): 3.678647,
    ("humidity_high_pct", "iforest"): 0.718367,
    ("humidity_high_pct", "zscore"): 3.818182,
    ("humidity_low_pct", "iforest"): 0.718854,
    ("humidity_low_pct", "zscore"): 3.75,
    ("precip_rate_mmh", "iforest"): 0.831013,
    ("precip_total_mm", "iforest"): 0.790083,
    ("pressure_max_hpa", "iforest"): 0.759184,
    ("pressure_max_hpa", "zscore"): 16.395789,
    ("pressure_min_hpa", "iforest"): 0.761348,
    ("pressure_min_hpa", "zscore"): 16.432489,
    ("pressure_trend_hpa", "iforest"): 0.759066,
    ("pressure_trend_hpa", "zscore"): 29.882353,
    ("solar_radiation_high_wm2", "iforest"): 0.712404,
    ("solar_radiation_high_wm2", "zscore"): 643.319489,
    ("temp_avg_c", "iforest"): 0.736257,
    ("temp_avg_c", "zscore"): 4.302957,
    ("temp_high_c", "iforest"): 0.736905,
    ("temp_high_c", "zscore"): 4.25,
    ("temp_low_c", "iforest"): 0.739702,
    ("temp_low_c", "zscore"): 4.25,
    ("uv_high", "iforest"): 0.734948,
    ("uv_high", "zscore"): 10.0,
    ("winddir_cos", "iforest"): 0.643084,
    ("winddir_cos", "zscore"): 4.762518,
    ("winddir_sin", "iforest"): 0.617667,
    ("winddir_sin", "zscore"): 19.342390,
    ("windgust_avg_kmh", "iforest"): 0.750250,
    ("windgust_avg_kmh", "zscore"): 7.194081,
    ("windgust_high_kmh", "iforest"): 0.753791,
    ("windgust_high_kmh", "zscore"): 7.0,
    ("windgust_low_kmh", "iforest"): 0.778738,
    ("windgust_low_kmh", "zscore"): 12.636364,
    ("windspeed_avg_kmh", "iforest"): 0.754305,
    ("windspeed_avg_kmh", "zscore"): 34.555556,
    ("windspeed_high_kmh", "iforest"): 0.751067,
    ("windspeed_high_kmh", "zscore"): 7.101124,
    ("windspeed_low_kmh", "iforest"): 0.768922,
    ("windspeed_low_kmh", "zscore"): 19.0,
}
CHANNEL_GROUP = {
    channel: group
    for group, channels in SENSOR_GROUP_CHANNELS.items()
    for channel in channels
}
CHANNEL_GROUP.update({"winddir_cos": "wind_vane", "winddir_sin": "wind_vane"})


@dataclass(frozen=True)
class ReplayBundle:
    health: pd.DataFrame
    forecasts: pd.DataFrame
    detections: pd.DataFrame
    scores: pd.DataFrame
    neighbors: pd.DataFrame
    registry: pd.DataFrame


def _utc(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], utc=True, format="mixed")
    return result


def _timestamp(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def load_replay_bundle(
    health_path: Path = JULY_HEALTH_PATH,
    forecast_path: Path = JULY_FORECAST_PATH,
    detection_path: Path = JULY_DETECTION_PATH,
    scores_path: Path = JULY_SCORES_PATH,
    neighbors_path: Path = JULY_NEIGHBORS_PATH,
    registry_path: Path = STATION_REGISTRY_PATH,
) -> ReplayBundle:
    paths = [health_path, forecast_path, detection_path, scores_path, neighbors_path, registry_path]
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing dashboard inputs:\n" + "\n".join(missing))
    full_health = _utc(pd.read_parquet(health_path), "hour_utc")
    for horizon in (1, 3, 6, 12, 24):
        full_health[f"outage_projection_{horizon}h"] = roll_forward_health_no_new_incident(
            full_health,
            horizon,
        )
    health = full_health.loc[full_health["hour_utc"].between(JULY_START, JULY_END)].copy()
    forecasts = _utc(
        pd.read_parquet(
            forecast_path,
            columns=["station_id", "hour_utc", "horizon_h", "predicted_frozen_selected_policy"],
        ),
        "hour_utc",
    )
    detections = _utc(
        pd.read_parquet(
            detection_path,
            columns=["station_id", "hour_utc", "random_probability", "random_prediction"],
        ),
        "hour_utc",
    )
    scores = _utc(
        pd.read_parquet(
            scores_path,
            columns=[
                "station_id", "hour_utc", "channel", "zscore", "rolling_variance",
                "iforest_score", "flag_zscore", "flag_stuck", "flag_iforest", "flag_physical",
            ],
        ),
        "hour_utc",
    )
    return ReplayBundle(
        health.sort_values(["hour_utc", "station_id"]).reset_index(drop=True),
        forecasts.sort_values(["hour_utc", "station_id", "horizon_h"]).reset_index(drop=True),
        detections.sort_values(["hour_utc", "station_id"]).reset_index(drop=True),
        scores.loc[scores["hour_utc"].between(JULY_START, JULY_END)].reset_index(drop=True),
        pd.read_csv(neighbors_path),
        pd.read_csv(registry_path).sort_values("station_id").reset_index(drop=True),
    )


def replay_hours(bundle: ReplayBundle) -> list[pd.Timestamp]:
    transmitting = bundle.health.groupby("hour_utc")["is_transmitting"].any()
    return list(transmitting.index[transmitting])


def segment_predicted_fault_events(
    detections: pd.DataFrame,
    reference_hour: object | None = None,
) -> pd.DataFrame:
    required = {"station_id", "hour_utc", "random_probability", "random_prediction"}
    if missing := required.difference(detections.columns):
        raise KeyError(f"Detection ledger is missing: {sorted(missing)}")
    rows = _utc(detections, "hour_utc").sort_values(["station_id", "hour_utc"])
    cutoff = _timestamp(reference_hour) if reference_hour is not None else rows["hour_utc"].max()
    positive = rows.loc[rows["hour_utc"].le(cutoff) & rows["random_prediction"].eq(1)].copy()
    if positive.empty:
        return pd.DataFrame(columns=["event_id", "station_id", "start_hour", "end_hour", "duration_hours", "peak_probability", "status"])
    starts = positive["station_id"].ne(positive["station_id"].shift()) | positive["hour_utc"].diff().ne(pd.Timedelta(hours=1))
    positive["event_number"] = starts.groupby(positive["station_id"]).cumsum()
    events = positive.groupby(["station_id", "event_number"]).agg(
        start_hour=("hour_utc", "min"), end_hour=("hour_utc", "max"),
        duration_hours=("hour_utc", "size"), peak_probability=("random_probability", "max"),
    ).reset_index()
    events["status"] = np.where(events["end_hour"].eq(cutoff), "active", "closed")
    events["event_id"] = [f"hgb_{row.station_id}_{int(row.event_number):04d}" for row in events.itertuples()]
    return events[["event_id", "station_id", "start_hour", "end_hour", "duration_hours", "peak_probability", "status"]]


def build_replay_snapshot(bundle: ReplayBundle, reference_hour: object) -> pd.DataFrame:
    hour = _timestamp(reference_hour)
    health = bundle.health.loc[bundle.health["hour_utc"].eq(hour)].copy()
    snapshot = bundle.registry.merge(health, on="station_id", how="left", validate="one_to_one")
    detection = bundle.detections.loc[
        bundle.detections["hour_utc"].eq(hour),
        ["station_id", "random_probability", "random_prediction"],
    ].rename(columns={"random_probability": "fault_probability", "random_prediction": "fault_detected"})
    snapshot = snapshot.merge(detection, on="station_id", how="left", validate="one_to_one")
    events = segment_predicted_fault_events(bundle.detections, hour)
    active = events.loc[events["status"].eq("active"), ["station_id", "duration_hours"]]
    snapshot = snapshot.merge(active.rename(columns={"duration_hours": "fault_run_hours"}), on="station_id", how="left")
    snapshot["fault_run_hours"] = snapshot["fault_run_hours"].fillna(0).astype(int)
    snapshot["status"] = np.select(
        [snapshot["availability_class"].eq("full_outage"), snapshot["availability_class"].eq("partial_outage"), snapshot["fault_detected"].eq(1).fillna(False)],
        ["Full outage", "Partial outage", "Fault alert"], default="Online",
    )
    snapshot["category"] = np.select(
        [snapshot["status"].eq("Full outage"), snapshot["status"].eq("Fault alert"), snapshot["status"].eq("Online") & snapshot["health_band"].eq("Healthy")],
        ["In outage", "Active faults", "Healthy"], default="Needs attention",
    )
    snapshot["finding"] = np.select(
        [snapshot["status"].eq("Full outage"), snapshot["status"].eq("Partial outage"), snapshot["status"].eq("Fault alert")],
        [snapshot["full_outage_run_hours"].map(lambda value: f"Full outage · {int(value)} h"), snapshot["absent_sensor_groups"].fillna("unknown group").map(lambda value: f"Partial outage · {value}"), snapshot["fault_run_hours"].map(lambda value: f"Predicted fault · {value} h")],
        default="—",
    )
    current_forecasts = bundle.forecasts.loc[bundle.forecasts["hour_utc"].eq(hour)]
    forecast_wide = current_forecasts.pivot(index="station_id", columns="horizon_h", values="predicted_frozen_selected_policy")
    for horizon in (1, 3, 6, 12, 24):
        learned = snapshot["station_id"].map(forecast_wide.get(horizon, pd.Series(dtype=float)))
        outage = snapshot["availability_class"].eq("full_outage")
        snapshot[f"forecast_{horizon}h"] = learned.where(~outage, snapshot[f"outage_projection_{horizon}h"])
        snapshot[f"forecast_source_{horizon}h"] = np.where(outage, "Continued-outage projection", "Selected model forecast")
    change = snapshot["forecast_24h"] - snapshot["health_total"]
    arrow = np.select([change.gt(0.5), change.lt(-0.5), change.notna()], ["↗", "↘", "→"], default="—")
    snapshot["health_24h"] = ["—" if pd.isna(value) else f"{direction} {value:.1f}" for direction, value in zip(arrow, snapshot["forecast_24h"], strict=True)]
    if len(snapshot) != len(bundle.registry) or snapshot["category"].isna().any():
        raise ValueError("Snapshot must assign one category to every registered station")
    return snapshot.sort_values(["health_total", "station_id"]).reset_index(drop=True)


def station_history(bundle: ReplayBundle, station_id: str, reference_hour: object) -> pd.DataFrame:
    hour = _timestamp(reference_hour)
    return bundle.health.loc[
        bundle.health["station_id"].eq(station_id) & bundle.health["hour_utc"].between(hour - pd.Timedelta(hours=71), hour),
        ["hour_utc", "health_total"],
    ].copy()


def station_sensor_states(bundle: ReplayBundle, station_id: str, reference_hour: object) -> pd.DataFrame:
    hour = _timestamp(reference_hour)
    row = bundle.health.loc[bundle.health["station_id"].eq(station_id) & bundle.health["hour_utc"].eq(hour)].iloc[0]
    return pd.DataFrame(
        {
            "Sensor group": [group.replace("_", " ").title() for group in SENSOR_GROUP_ORDER],
            "Status": [
                "Unavailable" if row["availability_class"] == "full_outage" else "Present" if bool(row[f"sensor_group_present_{group}"]) else "Missing"
                for group in SENSOR_GROUP_ORDER
            ],
        }
    )


def _physical_threshold(channel: str, value: float) -> tuple[float, float] | None:
    rule = PHYSICAL_LIMIT_RULES.get(channel)
    if rule is None or not np.isfinite(value):
        return None
    if "max_abs" in rule and abs(value) > float(rule["max_abs"]):
        threshold = float(rule["max_abs"])
        return threshold, abs(value) - threshold
    if "max" in rule and value > float(rule["max"]):
        threshold = float(rule["max"])
        return threshold, value - threshold
    if "min" in rule and value < float(rule["min"]):
        threshold = float(rule["min"])
        return threshold, threshold - value
    return None


def event_detector_evidence(bundle: ReplayBundle, event: pd.Series) -> pd.DataFrame:
    rows = bundle.scores.loc[
        bundle.scores["station_id"].eq(event["station_id"]) & bundle.scores["hour_utc"].between(event["start_hour"], event["end_hour"])
    ]
    health = bundle.health.loc[bundle.health["station_id"].eq(event["station_id"])].set_index("hour_utc")
    output: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        values = [
            ("Robust z-score", row.flag_zscore, abs(row.zscore), FROZEN_THRESHOLDS.get((row.channel, "zscore"))),
            ("Isolation forest", row.flag_iforest, row.iforest_score, FROZEN_THRESHOLDS.get((row.channel, "iforest"))),
            ("Stuck variance", row.flag_stuck, row.rolling_variance, ROLLING_VARIANCE_FLAG_THRESHOLD),
        ]
        for detector, fired, score, threshold in values:
            if bool(fired) and threshold is not None and np.isfinite(threshold):
                margin = threshold - score if detector == "Stuck variance" else score - threshold
                output.append({"Detector": detector, "Channel": row.channel, "Component": CHANNEL_GROUP.get(row.channel, "other"), "Score": score, "Threshold": threshold, "Margin": margin})
        if bool(row.flag_physical) and row.channel in health and row.hour_utc in health.index:
            score = float(health.at[row.hour_utc, row.channel]); physical = _physical_threshold(row.channel, score)
            if physical is not None:
                threshold, margin = physical
                output.append({"Detector": "Physical limit", "Channel": row.channel, "Component": CHANNEL_GROUP.get(row.channel, "other"), "Score": score, "Threshold": threshold, "Margin": margin})
    return pd.DataFrame(output, columns=["Detector", "Channel", "Component", "Score", "Threshold", "Margin"])


def event_layer_status(bundle: ReplayBundle, event: pd.Series) -> tuple[str, str]:
    event_health = bundle.health.loc[
        bundle.health["station_id"].eq(event["station_id"]) & bundle.health["hour_utc"].between(event["start_hour"], event["end_hour"])
    ]
    reference_columns = ["pressure_msl", "temperature_2m", "dew_point_2m", "wind_speed_10m", "shortwave_radiation"]
    available_hours = int(event_health[reference_columns].notna().any(axis=1).sum())
    external = f"ERA5/Open-Meteo available for {available_hours} of {len(event_health)} event hours" if available_hours else "ERA5/Open-Meteo unavailable for this event"
    neighbors = bundle.neighbors.loc[bundle.neighbors["station_id"].eq(event["station_id"]), "neighbor_id"].astype(str)
    if neighbors.empty:
        spatial = "Spatial layer unavailable: this station has no neighbours"
    else:
        transmitting = bundle.health.loc[
            bundle.health["station_id"].isin(neighbors) & bundle.health["hour_utc"].between(event["start_hour"], event["end_hour"]) & bundle.health["is_transmitting"], "station_id"
        ].nunique()
        spatial = f"Spatial layer available: {transmitting} of {len(neighbors)} neighbours transmitted during the event"
    return external, spatial
