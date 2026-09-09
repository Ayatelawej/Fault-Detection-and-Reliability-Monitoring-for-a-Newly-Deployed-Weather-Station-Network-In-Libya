from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.availability.build_availability_events import (
    AVAILABILITY_CLASS_FULL_OUTAGE,
    AVAILABILITY_CLASS_ONLINE,
    AVAILABILITY_CLASS_PARTIAL_OUTAGE,
    SENSOR_GROUP_CHANNELS,
    SENSOR_GROUP_ORDER,
)
from src.availability.risk_dataset import (
    build_causal_detector_evidence,
    build_causal_external_residuals,
)
from src.config.paths import (
    HEALTH_COMPONENT_CORRELATION_FIGURE_PATH,
    HEALTH_COMPONENT_DISTRIBUTIONS_FIGURE_PATH,
    HEALTH_DISTRIBUTION_FIGURE_PATH,
    HEALTH_OUTAGE_DURATION_TRAJECTORY_FIGURE_PATH,
    HEALTH_STATION_TIMESERIES_FIGURE_PATH,
    MEASUREMENT_COLUMNS,
    STATION_HEALTH_CAUSALITY_AUDIT_PATH,
    STATION_HEALTH_CAUSALITY_SUMMARY_PATH,
    STATION_HEALTH_INVARIANTS_PATH,
    STATION_HEALTH_OUTAGE_DURATION_CURVE_PATH,
    STATION_HEALTH_OUTAGE_TRAJECTORY_PATH,
    STATION_HEALTH_PROGRESSIVE_COMPARISON_PATH,
    STATION_HEALTH_PROGRESSIVE_RANKING_PATH,
    STATION_HEALTH_REPORT_PATH,
    STATION_HEALTH_SCORES_PATH,
    STATION_HEALTH_SUMMARY_PATH,
    ensure_directories,
)

HEALTH_WEIGHTS = {
    "health_availability": 30.0,
    "health_sensor_completeness": 20.0,
    "health_fault_burden": 25.0,
    "health_reference_consistency": 15.0,
    "health_stability": 10.0,
}
HEALTH_COMPONENT_COLUMNS = tuple(HEALTH_WEIGHTS)
HEALTH_HISTORY_HOURS = 24 * 7
STABILITY_HISTORY_HOURS = 24 * 30
STABILITY_EVENT_CAP = 20.0
FAULT_EVIDENCE_HALF_LIFE_HOURS = 24.0
FAULT_EVIDENCE_RATE_CAP = 0.20
FULL_OUTAGE_DECAY_HOURS = 24.0
PARTIAL_OUTAGE_DECAY_HOURS = 72.0
OUTAGE_DURATION_REFERENCE_HOURS = (1, 6, 24, 24 * 7)
REFERENCE_BASELINE_HOURS = 24 * 30
REFERENCE_BASELINE_MIN_HOURS = 24 * 10
REFERENCE_CURRENT_HISTORY_HOURS = 24
REFERENCE_CURRENT_MIN_HOURS = 12
REFERENCE_CURRENT_SCALES = {
    "pressure": 3.0,
    "temp": 2.0,
    "dewpoint": 2.0,
    "wind": 3.0,
}
REFERENCE_OFFSET_FLOOR_HPA = 3.0
REFERENCE_OFFSET_CAP_HPA = 6.0
REFERENCE_MIN_FLEET_STATIONS = 8
HEALTH_BANDS = ("Healthy", "Watch", "Degraded", "Critical")


def _require_columns(
    frame: pd.DataFrame,
    required: list[str],
    name: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _normalise_station_hours(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    _require_columns(frame, ["station_id", "hour_utc"], name)
    result = frame.copy(deep=True)
    result["station_id"] = result["station_id"].astype(str)
    result["hour_utc"] = pd.to_datetime(
        result["hour_utc"], utc=True, errors="coerce"
    )
    if result[["station_id", "hour_utc"]].isna().any().any():
        raise ValueError(f"{name} contains invalid station-hour keys")
    if result.duplicated(["station_id", "hour_utc"]).any():
        raise ValueError(f"{name} contains duplicate station-hour keys")
    return result.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(
        drop=True
    )


def _transmission_mask(frame: pd.DataFrame) -> pd.Series:
    if "data_present" in frame.columns:
        return pd.to_numeric(frame["data_present"], errors="coerce").fillna(0).gt(0)
    if "n_raw_records" in frame.columns:
        return pd.to_numeric(frame["n_raw_records"], errors="coerce").fillna(0).gt(0)
    raise ValueError("observations need data_present or n_raw_records")


def build_continuous_health_grid(observations: pd.DataFrame) -> pd.DataFrame:
    source = _normalise_station_hours(observations, "canonical station observations")
    if source.empty:
        return source
    source["_observed_transmission"] = _transmission_mask(source)
    for column in ["n_raw_records", "data_present", *MEASUREMENT_COLUMNS]:
        if column not in source.columns:
            source[column] = np.nan
    global_end = source["hour_utc"].max()
    grids: list[pd.DataFrame] = []
    source_columns = [
        "station_id",
        "hour_utc",
        "_observed_transmission",
        "n_raw_records",
        "data_present",
        *MEASUREMENT_COLUMNS,
    ]
    for station_id, station in source.groupby("station_id", sort=False):
        station = station.sort_values("hour_utc", kind="mergesort")
        first_hour = station["hour_utc"].min()
        clock = pd.DataFrame(
            {
                "station_id": str(station_id),
                "hour_utc": pd.date_range(first_hour, global_end, freq="h", tz="UTC"),
            }
        )
        grids.append(
            clock.merge(
                station.loc[:, source_columns],
                on=["station_id", "hour_utc"],
                how="left",
                validate="one_to_one",
            )
        )
    grid = pd.concat(grids, ignore_index=True)
    grid["is_transmitting"] = pd.to_numeric(
        grid["_observed_transmission"], errors="coerce"
    ).fillna(0.0).gt(0.0)
    grid = grid.drop(columns="_observed_transmission")
    for group in SENSOR_GROUP_ORDER:
        channels = [
            column for column in SENSOR_GROUP_CHANNELS[group] if column in grid.columns
        ]
        present = grid.loc[:, channels].notna().any(axis=1) if channels else False
        grid[f"sensor_group_present_{group}"] = (
            grid["is_transmitting"] & pd.Series(present, index=grid.index).astype(bool)
        )
        grid[f"sensor_group_absent_{group}"] = (
            grid["is_transmitting"] & ~grid[f"sensor_group_present_{group}"]
        )
    absent_columns = [f"sensor_group_absent_{group}" for group in SENSOR_GROUP_ORDER]
    grid["has_partial_outage"] = grid.loc[:, absent_columns].any(axis=1)
    grid["availability_class"] = np.select(
        [~grid["is_transmitting"], grid["has_partial_outage"]],
        [AVAILABILITY_CLASS_FULL_OUTAGE, AVAILABILITY_CLASS_PARTIAL_OUTAGE],
        default=AVAILABILITY_CLASS_ONLINE,
    )
    grid["absent_sensor_groups"] = [
        "|".join(
            group
            for group, is_absent in zip(SENSOR_GROUP_ORDER, values, strict=True)
            if bool(is_absent)
        )
        for values in grid.loc[:, absent_columns].itertuples(index=False, name=None)
    ]
    return grid.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(
        drop=True
    )


def _recency_weighted_rate(
    values: pd.Series,
    *,
    window_hours: int = HEALTH_HISTORY_HOURS,
    half_life_hours: float = FAULT_EVIDENCE_HALF_LIFE_HOURS,
) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan, dtype=float)
    if len(numeric) < window_hours:
        return result
    weights = np.exp(
        -np.log(2.0) * np.arange(window_hours, dtype=float) / float(half_life_hours)
    )
    numerator = np.convolve(numeric, weights, mode="full")[
        window_hours - 1 : window_hours - 1 + len(numeric)
    ]
    denominator = np.convolve(
        np.ones(len(numeric), dtype=float), weights, mode="full"
    )[window_hours - 1 : window_hours - 1 + len(numeric)]
    valid_count = len(numeric) - window_hours + 1
    result[window_hours - 1 :] = (
        numerator[:valid_count] / denominator[:valid_count]
    )
    return result


def _health_band(values: pd.Series) -> pd.Series:
    total = pd.to_numeric(values, errors="coerce")
    result = pd.Series("insufficient_history", index=total.index, dtype="string")
    result.loc[total.notna() & total.ge(80.0)] = "Healthy"
    result.loc[total.notna() & total.lt(80.0) & total.ge(60.0)] = "Watch"
    result.loc[total.notna() & total.lt(60.0) & total.ge(40.0)] = "Degraded"
    result.loc[total.notna() & total.lt(40.0)] = "Critical"
    return result


def health_band_for_values(values: pd.Series) -> pd.Series:
    return _health_band(values)


def _current_true_run_hours(mask: pd.Series) -> pd.Series:
    state = pd.Series(mask, index=mask.index).fillna(False).astype(bool)
    segments = state.ne(state.shift(fill_value=False)).cumsum()
    run_hours = state.astype(int).groupby(segments).cumsum()
    return run_hours.where(state, 0).astype(float)


def outage_duration_multiplier(
    full_outage_run_hours: pd.Series,
    partial_outage_run_hours: pd.Series,
) -> pd.Series:
    full_hours = pd.to_numeric(full_outage_run_hours, errors="coerce").fillna(0.0)
    partial_hours = pd.to_numeric(partial_outage_run_hours, errors="coerce").fillna(0.0)
    full_multiplier = np.exp(-full_hours / FULL_OUTAGE_DECAY_HOURS)
    partial_multiplier = np.exp(-partial_hours / PARTIAL_OUTAGE_DECAY_HOURS)
    values = np.where(
        full_hours.gt(0.0),
        full_multiplier,
        np.where(partial_hours.gt(0.0), partial_multiplier, 1.0),
    )
    return pd.Series(values, index=full_hours.index, dtype=float)


def build_outage_duration_curve() -> pd.DataFrame:
    duration = pd.Series(OUTAGE_DURATION_REFERENCE_HOURS, dtype=float)
    return pd.DataFrame(
        {
            "duration_hours": duration.astype(int),
            "full_outage_multiplier": np.exp(-duration / FULL_OUTAGE_DECAY_HOURS),
            "partial_outage_multiplier": np.exp(-duration / PARTIAL_OUTAGE_DECAY_HOURS),
        }
    )


def _causal_outage_base_score_cap(
    base_health_total: pd.Series,
    availability_class: pd.Series,
) -> pd.Series:
    totals = pd.to_numeric(base_health_total, errors="coerce")
    classes = availability_class.astype("string")
    capped = np.full(len(totals), np.nan, dtype=float)
    previous_total = np.nan
    previous_cap = np.nan
    previous_class = ""
    for position, (total, outage_class) in enumerate(
        zip(totals.to_numpy(dtype=float), classes.to_numpy(dtype=str), strict=True)
    ):
        active = outage_class in {
            AVAILABILITY_CLASS_FULL_OUTAGE,
            AVAILABILITY_CLASS_PARTIAL_OUTAGE,
        }
        if not active or not np.isfinite(total):
            capped[position] = total
        else:
            prior = previous_cap if outage_class == previous_class else previous_total
            capped[position] = (
                min(float(total), float(prior))
                if np.isfinite(prior)
                else float(total)
            )
        previous_total = total
        previous_cap = capped[position]
        previous_class = outage_class
    return pd.Series(capped, index=base_health_total.index, dtype=float)


def _outage_scores_are_nonincreasing(scores: pd.DataFrame) -> bool:
    for _, station in scores.groupby("station_id", sort=False):
        station = station.sort_values("hour_utc", kind="mergesort")
        health = pd.to_numeric(station["health_total"], errors="coerce")
        outage_class = station["availability_class"].astype("string")
        for active_class in (
            AVAILABILITY_CLASS_FULL_OUTAGE,
            AVAILABILITY_CLASS_PARTIAL_OUTAGE,
        ):
            continuing = outage_class.eq(active_class) & outage_class.shift(1).eq(
                active_class
            )
            change = health.diff()
            if change.loc[continuing & health.notna() & health.shift(1).notna()].gt(
                1e-10
            ).any():
                return False
    return True


def _station_score_components(station: pd.DataFrame) -> pd.DataFrame:
    result = station.copy(deep=True).reset_index(drop=True)
    detector = build_causal_detector_evidence(result).reset_index(drop=True)
    detector_columns = [
        f"causal_detector_{kind}_any_now"
        for kind in ("physical", "stuck", "deviation")
    ]
    result["causal_fault_evidence"] = (
        detector.loc[:, detector_columns].max(axis=1).fillna(0.0).gt(0.0)
    )
    result["stability_hard_fault_evidence"] = (
        detector.loc[:, [
            "causal_detector_physical_any_now",
            "causal_detector_stuck_any_now",
        ]]
        .max(axis=1)
        .fillna(0.0)
        .gt(0.0)
    )
    result["causal_fault_evidence_rate_7d"] = _recency_weighted_rate(
        result["causal_fault_evidence"]
    )
    result["health_fault_burden"] = 1.0 - np.clip(
        result["causal_fault_evidence_rate_7d"] / FAULT_EVIDENCE_RATE_CAP,
        0.0,
        1.0,
    )

    transmitting = result["is_transmitting"].astype(float)
    result["availability_window_hours"] = (
        transmitting.rolling(HEALTH_HISTORY_HOURS, min_periods=HEALTH_HISTORY_HOURS)
        .count()
        .astype(float)
    )
    result["availability_transmitting_hours"] = transmitting.rolling(
        HEALTH_HISTORY_HOURS, min_periods=HEALTH_HISTORY_HOURS
    ).sum()
    result["health_availability"] = transmitting.rolling(
        HEALTH_HISTORY_HOURS, min_periods=HEALTH_HISTORY_HOURS
    ).mean()
    group_components = []
    denominator = result["availability_transmitting_hours"]
    for group in SENSOR_GROUP_ORDER:
        present = result[f"sensor_group_present_{group}"].astype(float)
        numerator = present.rolling(
            HEALTH_HISTORY_HOURS, min_periods=HEALTH_HISTORY_HOURS
        ).sum()
        component = numerator / denominator.where(denominator.gt(0.0))
        component = component.mask(denominator.eq(0.0), 0.0)
        column = f"sensor_group_completeness_{group}"
        result[column] = component
        group_components.append(column)
    result["health_sensor_completeness"] = result.loc[:, group_components].mean(axis=1)

    external = build_causal_external_residuals(result).reset_index(drop=True)
    current_severities = []
    for short, scale in REFERENCE_CURRENT_SCALES.items():
        residual = pd.to_numeric(
            external[f"external_residual_now_{short}"], errors="coerce"
        )
        prior = residual.shift(1).rolling(
            REFERENCE_CURRENT_HISTORY_HOURS,
            min_periods=REFERENCE_CURRENT_MIN_HOURS,
        ).median()
        current_severities.append(((residual - prior).abs() / float(scale)).clip(0.0, 1.0))
    current_severity_frame = pd.concat(current_severities, axis=1)
    result["reference_current_channel_count"] = current_severity_frame.notna().sum(axis=1)
    result["reference_current_severity"] = current_severity_frame.mean(axis=1)
    pressure_residual = pd.to_numeric(
        external["external_residual_now_pressure"], errors="coerce"
    )
    result["reference_pressure_prior_median"] = pressure_residual.shift(1).rolling(
        REFERENCE_BASELINE_HOURS,
        min_periods=REFERENCE_BASELINE_MIN_HOURS,
    ).median()

    adverse = (
        ~result["is_transmitting"].astype(bool)
        | result["has_partial_outage"].astype(bool)
        | result["stability_hard_fault_evidence"].astype(bool)
    )
    result["stability_adverse_now"] = adverse
    starts = adverse & ~adverse.shift(1, fill_value=False)
    result["stability_event_starts_30d"] = starts.astype(float).rolling(
        STABILITY_HISTORY_HOURS, min_periods=1
    ).sum()
    last_adverse_hour = result["hour_utc"].where(adverse).ffill()
    hours_since = (
        result["hour_utc"].sub(last_adverse_hour).dt.total_seconds().div(3600.0)
    )
    result["stability_hours_since_last_adverse"] = hours_since.fillna(
        float(STABILITY_HISTORY_HOURS)
    )
    recurrence_score = 1.0 - np.clip(
        result["stability_event_starts_30d"] / STABILITY_EVENT_CAP,
        0.0,
        1.0,
    )
    recovery_score = np.clip(
        result["stability_hours_since_last_adverse"] / float(STABILITY_HISTORY_HOURS),
        0.0,
        1.0,
    )
    result["health_stability"] = 0.5 * recurrence_score + 0.5 * recovery_score
    return result


def build_station_health_scores(
    observations: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    grid = build_continuous_health_grid(observations)
    if grid.empty:
        return grid
    references = _normalise_station_hours(reference, "exact-hour external reference")
    result = grid.merge(
        references,
        on=["station_id", "hour_utc"],
        how="left",
        validate="one_to_one",
    )
    station_scores = [
        _station_score_components(station)
        for _, station in result.groupby("station_id", sort=False)
    ]
    result = pd.concat(station_scores, ignore_index=True)

    pressure_baseline = pd.to_numeric(
        result["reference_pressure_prior_median"], errors="coerce"
    )
    result["reference_fleet_support"] = pressure_baseline.notna().groupby(
        result["hour_utc"]
    ).transform("sum")
    fleet_median = pressure_baseline.groupby(result["hour_utc"]).transform("median")
    absolute_deviation = (pressure_baseline - fleet_median).abs()
    fleet_mad = absolute_deviation.groupby(result["hour_utc"]).transform("median")
    result["reference_fleet_pressure_median"] = fleet_median
    result["reference_fleet_pressure_mad"] = fleet_mad
    persistent_offset = ((absolute_deviation - REFERENCE_OFFSET_FLOOR_HPA).clip(lower=0.0)) / float(
        REFERENCE_OFFSET_CAP_HPA
    )
    result["reference_persistent_severity"] = persistent_offset.clip(0.0, 1.0).where(
        result["reference_fleet_support"].ge(REFERENCE_MIN_FLEET_STATIONS)
    )
    result["reference_severity"] = result.loc[:, [
        "reference_current_severity",
        "reference_persistent_severity",
    ]].max(axis=1)
    result["reference_evidence_mode"] = np.where(
        result["reference_severity"].notna(), "external_only", "unavailable"
    )
    result["health_reference_consistency"] = 1.0 - result["reference_severity"].fillna(0.0)

    result["health_history_hours"] = result.groupby("station_id").cumcount().add(1)
    insufficient = result["health_history_hours"].lt(HEALTH_HISTORY_HOURS)
    result["full_outage_run_hours"] = result.groupby("station_id", sort=False)[
        "availability_class"
    ].transform(
        lambda values: _current_true_run_hours(
            values.eq(AVAILABILITY_CLASS_FULL_OUTAGE)
        )
    )
    result["partial_outage_run_hours"] = result.groupby("station_id", sort=False)[
        "availability_class"
    ].transform(
        lambda values: _current_true_run_hours(
            values.eq(AVAILABILITY_CLASS_PARTIAL_OUTAGE)
        )
    )
    result["outage_duration_multiplier"] = outage_duration_multiplier(
        result["full_outage_run_hours"],
        result["partial_outage_run_hours"],
    )
    result["outage_duration_penalty"] = 1.0 - result["outage_duration_multiplier"]
    full_outage = result["availability_class"].eq(AVAILABILITY_CLASS_FULL_OUTAGE)
    for component, weight in HEALTH_WEIGHTS.items():
        base_component = pd.to_numeric(result[component], errors="coerce").clip(0.0, 1.0)
        result[f"base_{component}"] = base_component
    result["base_health_total"] = result.loc[:, [
        f"base_{component}" for component in HEALTH_COMPONENT_COLUMNS
    ]].mul(
        pd.Series(
            {f"base_{component}": weight for component, weight in HEALTH_WEIGHTS.items()}
        ),
        axis=1,
    ).sum(axis=1, min_count=len(HEALTH_COMPONENT_COLUMNS))
    outage_caps = [
        _causal_outage_base_score_cap(
            station["base_health_total"], station["availability_class"]
        )
        for _, station in result.groupby("station_id", sort=False)
    ]
    result["outage_base_score_cap"] = pd.concat(outage_caps).reindex(result.index)
    base_total = pd.to_numeric(result["base_health_total"], errors="coerce")
    capped_total = pd.to_numeric(result["outage_base_score_cap"], errors="coerce")
    result["outage_component_cap_multiplier"] = np.where(
        base_total.gt(0.0),
        (capped_total / base_total).clip(0.0, 1.0),
        1.0,
    )
    for component, weight in HEALTH_WEIGHTS.items():
        result[component] = (
            result[f"base_{component}"]
            * result["outage_component_cap_multiplier"]
            * result["outage_duration_multiplier"]
        )
        result[f"weighted_{component}"] = result[component] * float(weight)
    result["health_total"] = result.loc[:, [
        f"weighted_{component}" for component in HEALTH_COMPONENT_COLUMNS
    ]].sum(axis=1)
    result.loc[insufficient, [*HEALTH_COMPONENT_COLUMNS, *[
        f"base_{component}" for component in HEALTH_COMPONENT_COLUMNS
    ], *[
        f"weighted_{component}" for component in HEALTH_COMPONENT_COLUMNS
    ], "health_total"]] = np.nan
    result["health_status"] = np.select(
        [insufficient, full_outage],
        ["insufficient_history", "full_outage"],
        default="scored",
    )
    result["health_band"] = _health_band(result["health_total"])
    if result.duplicated(["station_id", "hour_utc"]).any():
        raise RuntimeError("health score construction produced duplicate station-hour keys")
    return result.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(
        drop=True
    )


def _numeric_summary(values: pd.Series) -> dict[str, float | int]:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return {
            "n": 0,
            "min": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "p75": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(len(series)),
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


def build_station_health_summary(scores: pd.DataFrame) -> pd.DataFrame:
    scored = scores.loc[scores["health_total"].notna()].copy()
    if scored.empty:
        return pd.DataFrame(columns=["station_id", "health_total_mean"])
    aggregations: dict[str, tuple[str, str]] = {
        "health_total_mean": ("health_total", "mean"),
        "health_total_min": ("health_total", "min"),
        "health_total_max": ("health_total", "max"),
        "health_total_median": ("health_total", "median"),
        "scored_hours": ("health_total", "size"),
    }
    for component in HEALTH_COMPONENT_COLUMNS:
        aggregations[f"{component}_mean"] = (component, "mean")
    result = scored.groupby("station_id", as_index=False).agg(**aggregations)
    return result.sort_values(
        ["health_total_mean", "station_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def _component_distribution(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for component in [*HEALTH_COMPONENT_COLUMNS, "health_total"]:
        values = pd.to_numeric(scores[component], errors="coerce").dropna()
        stats = _numeric_summary(values)
        rows.append(
            {
                "component": component,
                **stats,
                "fraction_le_0_05": float(values.le(0.05).mean()) if len(values) else np.nan,
                "fraction_ge_0_95": float(values.ge(0.95).mean()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _health_histogram(scores: pd.DataFrame) -> pd.DataFrame:
    values = pd.to_numeric(scores["health_total"], errors="coerce").dropna()
    bins = np.arange(0.0, 110.0, 10.0)
    counts, edges = np.histogram(values, bins=bins)
    return pd.DataFrame(
        {
            "score_band": [f"{int(left)}-{int(right)}" for left, right in zip(edges[:-1], edges[1:], strict=True)],
            "count": counts,
            "fraction": counts / len(values) if len(values) else np.nan,
        }
    )


def health_change_summary(scores: pd.DataFrame, horizons: tuple[int, ...] = (6, 12, 24)) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        differences: list[pd.Series] = []
        for _, station in scores.groupby("station_id", sort=False):
            future_hour = station["hour_utc"].shift(-int(horizon))
            future_score = pd.to_numeric(station["health_total"], errors="coerce").shift(-int(horizon))
            current = pd.to_numeric(station["health_total"], errors="coerce")
            exact_horizon = future_hour.sub(station["hour_utc"]).dt.total_seconds().div(3600.0).eq(horizon)
            differences.append((future_score - current).abs().where(exact_horizon))
        values = pd.concat(differences, ignore_index=True).dropna()
        rows.append(
            {
                "horizon_hours": int(horizon),
                **_numeric_summary(values),
                "p90": float(values.quantile(0.90)) if len(values) else np.nan,
                "p99": float(values.quantile(0.99)) if len(values) else np.nan,
                "fraction_gt_1": float(values.gt(1.0).mean()) if len(values) else np.nan,
                "fraction_gt_5": float(values.gt(5.0).mean()) if len(values) else np.nan,
                "fraction_gt_10": float(values.gt(10.0).mean()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def health_band_crossing_summary(
    scores: pd.DataFrame,
    horizons: tuple[int, ...] = (6, 12, 24),
) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        crossings: list[pd.Series] = []
        eligible: list[pd.Series] = []
        for _, station in scores.groupby("station_id", sort=False):
            future_hour = station["hour_utc"].shift(-int(horizon))
            current_band = station["health_band"].astype("string")
            future_band = current_band.shift(-int(horizon))
            exact_horizon = future_hour.sub(station["hour_utc"]).dt.total_seconds().div(3600.0).eq(horizon)
            valid = (
                exact_horizon
                & current_band.ne("insufficient_history")
                & future_band.ne("insufficient_history")
            )
            crossings.append(current_band.ne(future_band).where(valid))
            eligible.append(valid)
        crossing_values = pd.concat(crossings, ignore_index=True).dropna().astype(bool)
        eligible_count = int(pd.concat(eligible, ignore_index=True).sum())
        rows.append(
            {
                "horizon_hours": int(horizon),
                "eligible_station_hours": eligible_count,
                "band_crossings": int(crossing_values.sum()),
                "crossing_fraction": float(crossing_values.mean()) if len(crossing_values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_hard_zero_health_baseline(scores: pd.DataFrame) -> pd.DataFrame:
    required = [
        "availability_class",
        "health_history_hours",
        *[f"base_{component}" for component in HEALTH_COMPONENT_COLUMNS],
    ]
    _require_columns(scores, required, "progressive health scores")
    result = scores.copy(deep=True)
    insufficient = result["health_history_hours"].lt(HEALTH_HISTORY_HOURS)
    full_outage = result["availability_class"].eq(AVAILABILITY_CLASS_FULL_OUTAGE)
    for component, weight in HEALTH_WEIGHTS.items():
        result[component] = pd.to_numeric(
            result[f"base_{component}"], errors="coerce"
        ).clip(0.0, 1.0)
        result[f"weighted_{component}"] = result[component] * float(weight)
    result["health_total"] = result.loc[:, [
        f"weighted_{component}" for component in HEALTH_COMPONENT_COLUMNS
    ]].sum(axis=1)
    result.loc[full_outage & ~insufficient, list(HEALTH_COMPONENT_COLUMNS)] = 0.0
    result.loc[
        full_outage & ~insufficient,
        [f"weighted_{component}" for component in HEALTH_COMPONENT_COLUMNS],
    ] = 0.0
    result.loc[full_outage & ~insufficient, "health_total"] = 0.0
    result.loc[
        insufficient,
        [
            *HEALTH_COMPONENT_COLUMNS,
            *[f"weighted_{component}" for component in HEALTH_COMPONENT_COLUMNS],
            "health_total",
        ],
    ] = np.nan
    result["health_band"] = _health_band(result["health_total"])
    return result


def is_hard_zero_health_baseline(scores: pd.DataFrame) -> bool:
    required = {"availability_class", "health_total"}
    if not required.issubset(scores.columns):
        return False
    full_outage = scores["availability_class"].eq(AVAILABILITY_CLASS_FULL_OUTAGE)
    totals = pd.to_numeric(scores.loc[full_outage, "health_total"], errors="coerce")
    return bool(len(totals) and totals.notna().any() and totals.dropna().eq(0.0).all())


def validate_hard_zero_health_baseline(
    previous_scores: pd.DataFrame,
    reconstructed_baseline: pd.DataFrame,
) -> dict[str, object]:
    if not is_hard_zero_health_baseline(previous_scores):
        raise ValueError("previous health scores are not a hard-zero baseline")
    previous = _normalise_station_hours(previous_scores, "previous hard-zero health scores")
    reconstructed = _normalise_station_hours(
        reconstructed_baseline, "reconstructed hard-zero health scores"
    )
    fields = [
        *HEALTH_COMPONENT_COLUMNS,
        *[f"weighted_{component}" for component in HEALTH_COMPONENT_COLUMNS],
        "health_total",
    ]
    _require_columns(previous, fields, "previous hard-zero health scores")
    _require_columns(reconstructed, fields, "reconstructed hard-zero health scores")
    merged = previous.loc[:, ["station_id", "hour_utc", *fields]].merge(
        reconstructed.loc[:, ["station_id", "hour_utc", *fields]],
        on=["station_id", "hour_utc"],
        how="outer",
        suffixes=("_previous", "_reconstructed"),
        indicator=True,
        validate="one_to_one",
    )
    keys_match = bool(merged["_merge"].eq("both").all())
    comparisons = 0
    failures = 0
    if keys_match:
        for field in fields:
            left = pd.to_numeric(merged[f"{field}_previous"], errors="coerce")
            right = pd.to_numeric(merged[f"{field}_reconstructed"], errors="coerce")
            passed = np.isclose(
                left,
                right,
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            )
            comparisons += int(len(passed))
            failures += int((~passed).sum())
    return {
        "available": True,
        "keys_match": keys_match,
        "comparisons": comparisons,
        "failed_comparisons": failures,
        "passed": bool(keys_match and failures == 0),
    }


def _comparison_rows(
    section: str,
    previous: pd.DataFrame,
    progressive: pd.DataFrame,
    key_columns: list[str],
) -> pd.DataFrame:
    shared = [
        column
        for column in previous.columns
        if column in progressive.columns and column not in key_columns
    ]
    numeric_columns = [
        column
        for column in shared
        if pd.api.types.is_numeric_dtype(previous[column])
        and pd.api.types.is_numeric_dtype(progressive[column])
    ]
    if key_columns:
        merged = previous.merge(
            progressive,
            on=key_columns,
            how="outer",
            suffixes=("_hard_zero", "_progressive"),
            validate="one_to_one",
        )
    else:
        if len(previous) != 1 or len(progressive) != 1:
            raise ValueError("unkeyed health comparison inputs must each contain one row")
        merged = pd.concat(
            [
                previous.add_suffix("_hard_zero").reset_index(drop=True),
                progressive.add_suffix("_progressive").reset_index(drop=True),
            ],
            axis=1,
        )
    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        row_dict = row._asdict()
        for metric in numeric_columns:
            hard_zero = pd.to_numeric(
                pd.Series([row_dict.get(f"{metric}_hard_zero")]), errors="coerce"
            ).iloc[0]
            progressive_value = pd.to_numeric(
                pd.Series([row_dict.get(f"{metric}_progressive")]), errors="coerce"
            ).iloc[0]
            rows.append(
                {
                    "section": section,
                    **{column: row_dict.get(column) for column in key_columns},
                    "metric": metric,
                    "hard_zero_baseline": hard_zero,
                    "progressive_duration": progressive_value,
                    "difference": progressive_value - hard_zero
                    if pd.notna(hard_zero) and pd.notna(progressive_value)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _health_distribution_summary(scores: pd.DataFrame) -> pd.DataFrame:
    values = pd.to_numeric(scores["health_total"], errors="coerce").dropna()
    summary = _numeric_summary(values)
    summary["exact_zero_rows"] = int(values.eq(0.0).sum())
    summary["exact_zero_fraction"] = float(values.eq(0.0).mean()) if len(values) else np.nan
    return pd.DataFrame([summary])


def build_health_version_comparison(
    hard_zero_baseline: pd.DataFrame,
    progressive_scores: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    distribution = _comparison_rows(
        "distribution",
        _health_distribution_summary(hard_zero_baseline),
        _health_distribution_summary(progressive_scores),
        [],
    )
    histogram = _comparison_rows(
        "histogram",
        _health_histogram(hard_zero_baseline),
        _health_histogram(progressive_scores),
        ["score_band"],
    )
    changes = _comparison_rows(
        "absolute_health_change",
        health_change_summary(hard_zero_baseline),
        health_change_summary(progressive_scores),
        ["horizon_hours"],
    )
    crossings = _comparison_rows(
        "health_band_crossings",
        health_band_crossing_summary(hard_zero_baseline),
        health_band_crossing_summary(progressive_scores),
        ["horizon_hours"],
    )
    previous_summary = build_station_health_summary(hard_zero_baseline).copy()
    progressive_summary = build_station_health_summary(progressive_scores).copy()
    previous_summary["hard_zero_rank"] = np.arange(1, len(previous_summary) + 1)
    progressive_summary["progressive_rank"] = np.arange(1, len(progressive_summary) + 1)
    ranking = previous_summary.loc[:, ["station_id", "health_total_mean", "hard_zero_rank"]].merge(
        progressive_summary.loc[:, ["station_id", "health_total_mean", "progressive_rank"]],
        on="station_id",
        how="outer",
        suffixes=("_hard_zero", "_progressive"),
        validate="one_to_one",
    )
    ranking["mean_score_difference"] = (
        ranking["health_total_mean_progressive"]
        - ranking["health_total_mean_hard_zero"]
    )
    ranking["rank_difference"] = (
        ranking["progressive_rank"] - ranking["hard_zero_rank"]
    )
    ranking = ranking.sort_values(
        ["progressive_rank", "station_id"], kind="mergesort"
    ).reset_index(drop=True)
    return {
        "metrics": pd.concat(
            [distribution, histogram, changes, crossings], ignore_index=True
        ),
        "ranking": ranking,
    }


def build_outage_duration_trajectory(
    scores: pd.DataFrame,
    hard_zero_baseline: pd.DataFrame | None = None,
    context_hours: int = 72,
) -> pd.DataFrame:
    candidates: list[dict[str, object]] = []
    for station_id, station in scores.groupby("station_id", sort=False):
        station = station.sort_values("hour_utc", kind="mergesort").reset_index(drop=True)
        full = station["availability_class"].eq(AVAILABILITY_CLASS_FULL_OUTAGE)
        segments = full.ne(full.shift(fill_value=False)).cumsum()
        for _, event in station.loc[full].groupby(segments.loc[full], sort=False):
            start_position = int(event.index.min())
            end_position = int(event.index.max())
            recovered = bool(
                end_position + 1 < len(station)
                and not full.iloc[end_position + 1]
            )
            candidates.append(
                {
                    "station_id": str(station_id),
                    "start_position": start_position,
                    "end_position": end_position,
                    "duration_hours": int(len(event)),
                    "recovered": recovered,
                }
            )
    if not candidates:
        return pd.DataFrame()
    selected = sorted(
        candidates,
        key=lambda item: (
            not bool(item["recovered"]),
            -int(item["duration_hours"]),
            str(item["station_id"]),
            int(item["start_position"]),
        ),
    )[0]
    station = scores.loc[
        scores["station_id"].eq(selected["station_id"])
    ].sort_values("hour_utc", kind="mergesort").reset_index(drop=True)
    start_position = max(int(selected["start_position"]) - int(context_hours), 0)
    end_position = min(int(selected["end_position"]) + int(context_hours), len(station) - 1)
    trajectory = station.iloc[start_position : end_position + 1].loc[:, [
        "station_id",
        "hour_utc",
        "availability_class",
        "health_status",
        "health_total",
        "full_outage_run_hours",
        "partial_outage_run_hours",
        "outage_duration_multiplier",
        "outage_duration_penalty",
    ]].copy()
    trajectory["selected_event_start_utc"] = station.loc[
        int(selected["start_position"]), "hour_utc"
    ]
    trajectory["selected_event_end_utc"] = station.loc[
        int(selected["end_position"]), "hour_utc"
    ]
    trajectory["selected_event_duration_hours"] = int(selected["duration_hours"])
    trajectory["selected_event_recovered"] = bool(selected["recovered"])
    if hard_zero_baseline is not None:
        baseline = hard_zero_baseline.loc[:, ["station_id", "hour_utc", "health_total"]].rename(
            columns={"health_total": "health_total_hard_zero"}
        )
        trajectory = trajectory.merge(
            baseline,
            on=["station_id", "hour_utc"],
            how="left",
            validate="one_to_one",
        )
        trajectory = trajectory.rename(
            columns={"health_total": "health_total_progressive"}
        )
    else:
        trajectory = trajectory.rename(
            columns={"health_total": "health_total_progressive"}
        )
    return trajectory.reset_index(drop=True)


def component_correlation(scores: pd.DataFrame) -> pd.DataFrame:
    return scores.loc[:, list(HEALTH_COMPONENT_COLUMNS)].corr()


def select_health_causality_sample_keys(
    scores: pd.DataFrame,
    observations: pd.DataFrame,
    sample_size: int = 16,
) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    source = _normalise_station_hours(observations, "canonical station observations")
    observed_hours = pd.Index(source["hour_utc"].drop_duplicates())
    candidates = scores.loc[
        scores["health_total"].notna() & scores["hour_utc"].isin(observed_hours)
    ].copy()
    if candidates.empty:
        raise RuntimeError("no scored station-hours are available for health causality validation")
    choices: list[pd.DataFrame] = []
    for reason, mask in [
        ("first_scored", candidates["health_status"].eq("scored")),
        ("full_outage", candidates["health_status"].eq("full_outage")),
        ("partial_outage", candidates["availability_class"].eq(AVAILABILITY_CLASS_PARTIAL_OUTAGE)),
        ("high_reference_severity", candidates["reference_severity"].notna()),
    ]:
        subset = candidates.loc[mask].sort_values(["hour_utc", "station_id"], kind="mergesort")
        if not subset.empty:
            row = subset.iloc[[0]].loc[:, ["station_id", "hour_utc"]].copy()
            row["sample_reason"] = reason
            choices.append(row)
    remaining = max(int(sample_size) - sum(len(frame) for frame in choices), 0)
    if remaining:
        ordered = candidates.sort_values(["hour_utc", "station_id"], kind="mergesort")
        positions = np.linspace(0, len(ordered) - 1, num=remaining, dtype=int)
        sampled = ordered.iloc[np.unique(positions)].loc[:, ["station_id", "hour_utc"]].copy()
        sampled["sample_reason"] = "temporal_coverage"
        choices.append(sampled)
    result = pd.concat(choices, ignore_index=True).drop_duplicates(
        ["station_id", "hour_utc"], keep="first"
    )
    if len(result) < sample_size:
        ordered = candidates.sort_values(["hour_utc", "station_id"], kind="mergesort")
        used = set(zip(result["station_id"].astype(str), result["hour_utc"], strict=True))
        supplement = ordered.loc[
            [
                (str(row.station_id), row.hour_utc) not in used
                for row in ordered.loc[:, ["station_id", "hour_utc"]].itertuples(index=False)
            ],
            ["station_id", "hour_utc"],
        ].head(int(sample_size) - len(result)).copy()
        supplement["sample_reason"] = "supplemental_coverage"
        result = pd.concat([result, supplement], ignore_index=True)
    return result.head(int(sample_size)).reset_index(drop=True)


def validate_delete_future_health_scores(
    observations: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    full_scores: pd.DataFrame | None = None,
    sample_keys: pd.DataFrame | None = None,
    sample_size: int = 16,
) -> pd.DataFrame:
    full = (
        build_station_health_scores(observations, reference)
        if full_scores is None
        else full_scores
    )
    keys = (
        select_health_causality_sample_keys(full, observations, sample_size)
        if sample_keys is None
        else _normalise_station_hours(sample_keys, "health causality sample keys")
    )
    numeric_fields = [
        *HEALTH_COMPONENT_COLUMNS,
        "health_total",
        "health_history_hours",
        "full_outage_run_hours",
        "partial_outage_run_hours",
        "outage_duration_multiplier",
        "outage_duration_penalty",
        "base_health_total",
        "outage_base_score_cap",
        "outage_component_cap_multiplier",
        "availability_transmitting_hours",
        "causal_fault_evidence_rate_7d",
        "reference_current_severity",
        "reference_persistent_severity",
        "stability_event_starts_30d",
        "stability_hours_since_last_adverse",
    ]
    categorical_fields = [
        "health_status",
        "health_band",
        "availability_class",
        "reference_evidence_mode",
        "absent_sensor_groups",
    ]
    rows: list[dict[str, object]] = []
    source = _normalise_station_hours(observations, "canonical station observations")
    references = _normalise_station_hours(reference, "exact-hour external reference")
    for key in keys.itertuples(index=False):
        cutoff = pd.Timestamp(key.hour_utc)
        truncated = build_station_health_scores(
            source.loc[source["hour_utc"].le(cutoff)].copy(),
            references.loc[references["hour_utc"].le(cutoff)].copy(),
        )
        full_row = full.loc[
            full["station_id"].eq(str(key.station_id)) & full["hour_utc"].eq(cutoff)
        ]
        truncated_row = truncated.loc[
            truncated["station_id"].eq(str(key.station_id))
            & truncated["hour_utc"].eq(cutoff)
        ]
        if len(full_row) != 1 or len(truncated_row) != 1:
            raise RuntimeError("health delete-the-future validation could not recover a score row")
        left = full_row.iloc[0]
        right = truncated_row.iloc[0]
        for field in numeric_fields:
            left_value = pd.to_numeric(pd.Series([left[field]]), errors="coerce").iloc[0]
            right_value = pd.to_numeric(pd.Series([right[field]]), errors="coerce").iloc[0]
            passed = bool(
                np.isclose(
                    left_value,
                    right_value,
                    rtol=1e-10,
                    atol=1e-12,
                    equal_nan=True,
                )
            )
            rows.append(
                {
                    "station_id": str(key.station_id),
                    "hour_utc": cutoff,
                    "field": field,
                    "field_type": "numeric",
                    "full_value": float(left_value) if pd.notna(left_value) else np.nan,
                    "as_of_value": float(right_value) if pd.notna(right_value) else np.nan,
                    "passed": passed,
                }
            )
        for field in categorical_fields:
            left_value = "" if pd.isna(left[field]) else str(left[field])
            right_value = "" if pd.isna(right[field]) else str(right[field])
            rows.append(
                {
                    "station_id": str(key.station_id),
                    "hour_utc": cutoff,
                    "field": field,
                    "field_type": "categorical",
                    "full_value": left_value,
                    "as_of_value": right_value,
                    "passed": left_value == right_value,
                }
            )
    return pd.DataFrame(rows)


def summarize_delete_future_health_validation(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame(
            [
                {
                    "sample_rows_validated": 0,
                    "fields_validated": 0,
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
                    audit.loc[:, ["station_id", "hour_utc"]].drop_duplicates().shape[0]
                ),
                "fields_validated": int(audit["field"].nunique()),
                "comparisons": int(len(audit)),
                "failed_comparisons": int((~audit["passed"].astype(bool)).sum()),
                "all_passed": bool(audit["passed"].astype(bool).all()),
            }
        ]
    )


def select_contrasting_stations(
    scores: pd.DataFrame,
    calibration_corroboration: pd.DataFrame | None = None,
) -> dict[str, str]:
    summary = build_station_health_summary(scores)
    if summary.empty:
        return {}
    selected: dict[str, str] = {}
    selected["stable"] = str(summary.iloc[0]["station_id"])
    if calibration_corroboration is not None and {
        "station_id",
        "verdict",
    }.issubset(calibration_corroboration.columns):
        confirmed = set(
            calibration_corroboration.loc[
                calibration_corroboration["verdict"].astype(str).eq("confirmed"),
                "station_id",
            ].astype(str)
        )
        candidates = summary.loc[summary["station_id"].isin(confirmed)]
        if not candidates.empty:
            selected["confirmed_offset"] = str(candidates.iloc[-1]["station_id"])
    remaining = summary.loc[~summary["station_id"].isin(set(selected.values()))]
    heavy = remaining if not remaining.empty else summary
    selected["heavy_outage"] = str(
        heavy.sort_values(
            ["health_availability_mean", "station_id"], ascending=[True, True], kind="mergesort"
        ).iloc[0]["station_id"]
    )
    return selected


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def generate_station_health_figures(
    scores: pd.DataFrame,
    *,
    calibration_corroboration: pd.DataFrame | None = None,
    hard_zero_baseline: pd.DataFrame | None = None,
    outage_trajectory: pd.DataFrame | None = None,
    output_paths: dict[str, Path] | None = None,
) -> dict[str, Path]:
    paths = output_paths or {
        "distribution": HEALTH_DISTRIBUTION_FIGURE_PATH,
        "components": HEALTH_COMPONENT_DISTRIBUTIONS_FIGURE_PATH,
        "timeseries": HEALTH_STATION_TIMESERIES_FIGURE_PATH,
        "correlation": HEALTH_COMPONENT_CORRELATION_FIGURE_PATH,
        "outage_duration_trajectory": HEALTH_OUTAGE_DURATION_TRAJECTORY_FIGURE_PATH,
    }
    valid = scores.loc[scores["health_total"].notna()].copy()
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    axis.hist(valid["health_total"], bins=np.arange(0, 105, 5), color="#3f8f8f", edgecolor="white")
    axis.set(title="Station-health score distribution", xlabel="Health score", ylabel="Station-hours")
    axis.set_xlim(0, 100)
    _save_figure(figure, Path(paths["distribution"]))

    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    box_values = [valid[component].dropna().to_numpy() for component in HEALTH_COMPONENT_COLUMNS]
    labels = [component.replace("health_", "").replace("_", "\n") for component in HEALTH_COMPONENT_COLUMNS]
    axis.boxplot(box_values, labels=labels, showfliers=False)
    axis.set(title="Distribution of normalized health components", ylabel="Component score (0–1)", ylim=(0, 1.05))
    figure.tight_layout()
    _save_figure(figure, Path(paths["components"]))

    selections = select_contrasting_stations(valid, calibration_corroboration)
    figure, axis = plt.subplots(figsize=(11, 5))
    colors = {"stable": "#43835f", "confirmed_offset": "#c48a2c", "heavy_outage": "#b75b5b"}
    for role, station_id in selections.items():
        station = valid.loc[valid["station_id"].eq(station_id)]
        daily = station.set_index("hour_utc")["health_total"].resample("D").median()
        axis.plot(
            daily.index,
            daily,
            linewidth=1.4,
            label=f"{role.replace('_', ' ')}: {station_id}",
            color=colors.get(role, "#2f6f9f"),
        )
    axis.set(
        title="Contrasting causal station-health trajectories (daily median)",
        xlabel="UTC day",
        ylabel="Health score",
        ylim=(0, 100),
    )
    axis.legend(loc="best", frameon=False)
    figure.autofmt_xdate()
    _save_figure(figure, Path(paths["timeseries"]))

    correlation = component_correlation(valid)
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(correlation.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(correlation.columns)), [column.replace("health_", "") for column in correlation.columns], rotation=45, ha="right")
    axis.set_yticks(range(len(correlation.index)), [index.replace("health_", "") for index in correlation.index])
    for row_index, row in enumerate(correlation.to_numpy(dtype=float)):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    axis.set_title("Health-component correlation")
    figure.tight_layout()
    _save_figure(figure, Path(paths["correlation"]))

    trajectory = (
        build_outage_duration_trajectory(scores, hard_zero_baseline)
        if outage_trajectory is None
        else outage_trajectory.copy(deep=True)
    )
    if not trajectory.empty:
        figure, axis = plt.subplots(figsize=(11, 5.2))
        axis.plot(
            trajectory["hour_utc"],
            trajectory["health_total_progressive"],
            color="#2f6f9f",
            linewidth=1.7,
            label="progressive-duration health",
        )
        if "health_total_hard_zero" in trajectory.columns:
            axis.plot(
                trajectory["hour_utc"],
                trajectory["health_total_hard_zero"],
                color="#555555",
                linewidth=1.2,
                linestyle="--",
                label="previous hard-zero health",
            )
        selected_start = pd.Timestamp(trajectory.loc[0, "selected_event_start_utc"])
        selected_end = pd.Timestamp(trajectory.loc[0, "selected_event_end_utc"])
        selected_full = trajectory["hour_utc"].between(selected_start, selected_end)
        if selected_full.any():
            axis.fill_between(
                trajectory["hour_utc"],
                0,
                100,
                where=selected_full.to_numpy(dtype=bool),
                color="#b75b5b",
                alpha=0.12,
                label="full outage",
            )
        axis.set(
            title=(
                "Progressive health through a completed full outage "
                f"({trajectory.loc[0, 'station_id']})"
            ),
            xlabel="UTC hour",
            ylabel="Health score",
            ylim=(0, 100),
        )
        multiplier_axis = axis.twinx()
        multiplier_axis.plot(
            trajectory["hour_utc"],
            trajectory["outage_duration_multiplier"].where(selected_full),
            color="#c48a2c",
            linewidth=1.2,
            linestyle=":",
            label="outage-duration multiplier",
        )
        multiplier_axis.set(ylabel="Duration multiplier", ylim=(0, 1.05))
        handles, labels = axis.get_legend_handles_labels()
        second_handles, second_labels = multiplier_axis.get_legend_handles_labels()
        axis.legend(handles + second_handles, labels + second_labels, loc="best", frameon=False)
        figure.autofmt_xdate()
        figure.tight_layout()
        _save_figure(figure, Path(paths["outage_duration_trajectory"]))
    return {name: Path(path) for name, path in paths.items()}


def _format_frame(frame: pd.DataFrame) -> str:
    return "(none)" if frame.empty else frame.to_string(index=False)


def build_station_health_report(
    scores: pd.DataFrame,
    *,
    causality_summary: pd.DataFrame,
    calibration_corroboration: pd.DataFrame | None = None,
    version_comparison: dict[str, pd.DataFrame] | None = None,
    outage_duration_curve: pd.DataFrame | None = None,
    outage_trajectory: pd.DataFrame | None = None,
) -> str:
    valid = scores.loc[scores["health_total"].notna()].copy()
    summary = build_station_health_summary(scores)
    distribution = pd.DataFrame([_numeric_summary(valid["health_total"])])
    histogram = _health_histogram(scores)
    component_distribution = _component_distribution(scores)
    changes = health_change_summary(scores)
    crossings = health_band_crossing_summary(scores)
    correlations = component_correlation(scores)
    selections = select_contrasting_stations(scores, calibration_corroboration)
    status_counts = scores["health_status"].value_counts().rename_axis("health_status").reset_index(name="count")
    duration_curve = (
        build_outage_duration_curve()
        if outage_duration_curve is None
        else outage_duration_curve
    )
    progressive_metrics = (
        pd.DataFrame()
        if version_comparison is None
        else version_comparison.get("metrics", pd.DataFrame())
    )
    progressive_ranking = (
        pd.DataFrame()
        if version_comparison is None
        else version_comparison.get("ranking", pd.DataFrame())
    )
    trajectory_description = "(no full-outage event was available for a trajectory)"
    if outage_trajectory is not None and not outage_trajectory.empty:
        trajectory_description = (
            f"station={outage_trajectory.loc[0, 'station_id']}; "
            f"start={outage_trajectory.loc[0, 'selected_event_start_utc']}; "
            f"end={outage_trajectory.loc[0, 'selected_event_end_utc']}; "
            f"duration_hours={outage_trajectory.loc[0, 'selected_event_duration_hours']}; "
            f"recovered={outage_trajectory.loc[0, 'selected_event_recovered']}"
        )
    lines = [
        "STATION HEALTH SCORE",
        "",
        "This is a transparent current-state score, not a trained model and not a forecast.",
        "Every scored component uses information available at or before the completed station-hour.",
        "",
        "WEIGHTED DEFINITION",
        "health_total = 30*availability + 20*sensor_completeness + 25*fault_evidence_burden + 15*reference_consistency + 10*stability",
        "All five components are normalized to 0–1 before their fixed design weights are applied.",
        "Fault burden is a trailing, exponentially recency-weighted (24-hour half-life) rate of causally reconstructed physical-limit, stuck, or deviation evidence. It does not use reviewed episode labels. Stability counts only full/partial communication failures and hard physical/stuck evidence as recurring operational events, so isolated contextual deviations do not become a sequence of artificial event starts.",
        "Reference consistency uses exact-hour external reference residuals, prior-only residual history, and a causal fleet-relative pressure-persistence check. Completed calibration-corroboration intervals are not score inputs; where supplied, they only select a retrospective sanity-check plot.",
        "Spatial neighbour availability is not an input to this score, so stations without peers receive no penalty.",
        "Active outages apply a fixed causal duration multiplier to all five normalized components before weighting: exp(-d/24) for a full outage and exp(-d/72) for a partial outage, where d is the consecutive active duration in completed hours. These fixed 24-hour and 72-hour decay constants are engineering design choices, not fitted or tuned parameters. The multiplier is 1 for a transmitting station, preserves the weighted-component sum, and lets a newly dropped station remain distinguishable from a prolonged outage.",
        "For a partial outage, d is the consecutive period in which the station has remained partially degraded; the separate sensor-completeness component and absent-group fields retain the group-level severity evidence.",
        "Within one continuous full or partial outage, the pre-duration weighted score is causally capped at its prior active-outage value. This prevents missing telemetry from making a station appear healthier simply because fault and reference evidence are temporarily unavailable.",
        "",
        "OUTAGE-DURATION CURVE",
        _format_frame(duration_curve),
        "",
        "SELECTED OUTAGE TRAJECTORY",
        trajectory_description,
        "",
        "SCORE STATUS ACCOUNTING",
        _format_frame(status_counts),
        "",
        "1. TOTAL HEALTH DISTRIBUTION",
        _format_frame(distribution),
        "Histogram:",
        _format_frame(histogram),
        "",
        "2. COMPONENT DISTRIBUTIONS",
        _format_frame(component_distribution),
        "",
        "3. PER-STATION HEALTH RANKING",
        _format_frame(summary),
        "",
        "4. ABSOLUTE HEALTH CHANGE OVER TIME",
        _format_frame(changes),
        "",
        "5. HEALTH-BAND CROSSINGS",
        _format_frame(crossings),
        "Bands: Healthy 80–100; Watch 60–<80; Degraded 40–<60; Critical <40.",
        "",
        "6. HARD-ZERO BASELINE VERSUS PROGRESSIVE-DURATION SCORE",
        _format_frame(progressive_metrics),
        "Per-station ranking comparison:",
        _format_frame(progressive_ranking),
        "",
        "7. CONTRASTING TIME-SERIES SELECTION",
        *[f"{role}: {station_id}" for role, station_id in selections.items()],
        "A confirmed-offset case is selected from calibration corroboration only for the visual sanity check; its reviewed interval does not change the score.",
        "",
        "8. COMPONENT CORRELATION",
        correlations.to_string(),
        "",
        "DELETE-THE-FUTURE VALIDATION",
        _format_frame(causality_summary),
    ]
    return "\n".join(lines) + "\n"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> str:
    digest = sha256()
    directory = Path(path)
    for child in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(directory)).replace("\\", "/").encode("utf-8"))
        digest.update(_sha256_file(child).encode("ascii"))
    return digest.hexdigest()


def station_health_input_hashes(
    observations_path: Path,
    reference_directory: Path,
) -> dict[str, str]:
    return {
        "canonical_station_hourly_dataset": _sha256_file(Path(observations_path)),
        "exact_hour_reference_cache": _sha256_directory(Path(reference_directory)),
    }


def write_station_health_outputs(
    scores: pd.DataFrame,
    *,
    causality_audit: pd.DataFrame,
    input_hashes_before: dict[str, str],
    input_hashes_after: dict[str, str],
    calibration_corroboration: pd.DataFrame | None = None,
    previous_scores: pd.DataFrame | None = None,
    output_paths: dict[str, Path] | None = None,
    generate_figures: bool = True,
) -> dict[str, Path]:
    defaults = {
        "scores": STATION_HEALTH_SCORES_PATH,
        "summary": STATION_HEALTH_SUMMARY_PATH,
        "report": STATION_HEALTH_REPORT_PATH,
        "invariants": STATION_HEALTH_INVARIANTS_PATH,
        "causality_audit": STATION_HEALTH_CAUSALITY_AUDIT_PATH,
        "causality_summary": STATION_HEALTH_CAUSALITY_SUMMARY_PATH,
        "comparison": STATION_HEALTH_PROGRESSIVE_COMPARISON_PATH,
        "ranking_comparison": STATION_HEALTH_PROGRESSIVE_RANKING_PATH,
        "outage_duration_curve": STATION_HEALTH_OUTAGE_DURATION_CURVE_PATH,
        "outage_trajectory": STATION_HEALTH_OUTAGE_TRAJECTORY_PATH,
    }
    paths = {**defaults, **(output_paths or {})}
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("a station-health input artifact changed during score construction")
    causality_summary = summarize_delete_future_health_validation(causality_audit)
    if not bool(causality_summary.loc[0, "all_passed"]):
        raise RuntimeError("station-health delete-the-future validation failed")
    weighted_total = scores.loc[:, [
        f"weighted_{component}" for component in HEALTH_COMPONENT_COLUMNS
    ]].sum(axis=1, min_count=1)
    scored = scores["health_total"].notna()
    weighted_sum_ok = bool(
        np.isclose(
            scores.loc[scored, "health_total"],
            weighted_total.loc[scored],
            rtol=1e-10,
            atol=1e-12,
        ).all()
    )
    if not weighted_sum_ok:
        raise RuntimeError("health_total does not equal the weighted component sum")
    if scores.loc[scored, list(HEALTH_COMPONENT_COLUMNS)].lt(0.0).any().any() or scores.loc[
        scored, list(HEALTH_COMPONENT_COLUMNS)
    ].gt(1.0).any().any():
        raise RuntimeError("health components are outside the 0-1 interval")
    if not scores.loc[scores["health_status"].eq("insufficient_history"), "health_total"].isna().all():
        raise RuntimeError("insufficient-history rows must not receive a total health score")
    duration_multiplier = pd.to_numeric(
        scores["outage_duration_multiplier"], errors="coerce"
    )
    if duration_multiplier.notna().any() and (
        duration_multiplier.dropna().lt(0.0).any()
        or duration_multiplier.dropna().gt(1.0).any()
    ):
        raise RuntimeError("outage-duration multiplier is outside the 0-1 interval")
    full_duration = pd.to_numeric(scores["full_outage_run_hours"], errors="coerce")
    partial_duration = pd.to_numeric(scores["partial_outage_run_hours"], errors="coerce")
    expected_multiplier = outage_duration_multiplier(full_duration, partial_duration)
    duration_multiplier_exact = bool(
        np.isclose(
            duration_multiplier,
            expected_multiplier,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ).all()
    )
    if not duration_multiplier_exact:
        raise RuntimeError("outage-duration multiplier does not match the fixed causal curve")
    duration_curve = build_outage_duration_curve()
    duration_curve_monotonic = bool(
        duration_curve["full_outage_multiplier"].diff().dropna().lt(0.0).all()
        and duration_curve["partial_outage_multiplier"].diff().dropna().lt(0.0).all()
        and duration_curve["partial_outage_multiplier"].gt(
            duration_curve["full_outage_multiplier"]
        ).all()
    )
    if not duration_curve_monotonic:
        raise RuntimeError("outage-duration curve is not monotonic as configured")
    outage_scores_monotonic = _outage_scores_are_nonincreasing(scores)
    if not outage_scores_monotonic:
        raise RuntimeError("health score increased during a continuous active outage")
    hard_zero_baseline = build_hard_zero_health_baseline(scores)
    baseline_validation: dict[str, object] = {"available": False, "passed": True}
    if previous_scores is not None:
        baseline_validation = validate_hard_zero_health_baseline(
            previous_scores, hard_zero_baseline
        )
        if not bool(baseline_validation["passed"]):
            raise RuntimeError(
                "the reconstructed hard-zero baseline does not match the previous score artifact"
            )
    version_comparison = build_health_version_comparison(hard_zero_baseline, scores)
    outage_trajectory = build_outage_duration_trajectory(scores, hard_zero_baseline)
    ensure_directories()
    Path(paths["scores"]).parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(paths["scores"], index=False)
    build_station_health_summary(scores).to_csv(paths["summary"], index=False)
    causality_audit.to_csv(paths["causality_audit"], index=False)
    causality_summary.to_csv(paths["causality_summary"], index=False)
    version_comparison["metrics"].to_csv(paths["comparison"], index=False)
    version_comparison["ranking"].to_csv(paths["ranking_comparison"], index=False)
    duration_curve.to_csv(paths["outage_duration_curve"], index=False)
    outage_trajectory.to_csv(paths["outage_trajectory"], index=False)
    Path(paths["report"]).write_text(
        build_station_health_report(
            scores,
            causality_summary=causality_summary,
            calibration_corroboration=calibration_corroboration,
            version_comparison=version_comparison,
            outage_duration_curve=duration_curve,
            outage_trajectory=outage_trajectory,
        ),
        encoding="utf-8",
    )
    invariants = {
        "input_hashes_before": input_hashes_before,
        "input_hashes_after": input_hashes_after,
        "input_hashes_unchanged": True,
        "weighted_sum_exact": weighted_sum_ok,
        "delete_the_future_passed": True,
        "outage_duration_multiplier_exact": duration_multiplier_exact,
        "outage_duration_curve_monotonic": duration_curve_monotonic,
        "outage_scores_monotonic": outage_scores_monotonic,
        "previous_hard_zero_baseline_validation": baseline_validation,
        "score_rows": int(scored.sum()),
        "insufficient_history_rows": int(scores["health_status"].eq("insufficient_history").sum()),
        "full_outage_scored_rows": int(scores["health_status"].eq("full_outage").sum()),
        "fault_burden_source": "causal_rule_evidence",
        "reference_consistency_source": "causal_external_residuals",
    }
    Path(paths["invariants"]).write_text(
        json.dumps(invariants, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {name: Path(path) for name, path in paths.items()}
    if generate_figures:
        result.update(
            generate_station_health_figures(
                scores,
                calibration_corroboration=calibration_corroboration,
                hard_zero_baseline=hard_zero_baseline,
                outage_trajectory=outage_trajectory,
            )
        )
    return result
