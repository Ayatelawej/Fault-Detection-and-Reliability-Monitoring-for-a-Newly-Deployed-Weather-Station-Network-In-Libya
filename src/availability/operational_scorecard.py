from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

from src.availability.build_availability_events import (
    AVAILABILITY_CLASS_FULL_OUTAGE,
    AVAILABILITY_CLASS_ONLINE,
    AVAILABILITY_CLASS_PARTIAL_OUTAGE,
    SENSOR_GROUP_ORDER,
)
from src.availability.health_forecast import (
    FittedHealthForecastModel,
    build_health_forecast_features,
    health_forecast_inference_frame,
)
from src.availability.health_score import (
    HEALTH_COMPONENT_COLUMNS,
    health_band_for_values,
)
from src.availability.risk_dataset import build_causal_detector_evidence


SCORECARD_HORIZONS = (1, 3, 6, 12, 24)
NO_SPATIAL_NEIGHBOUR_STATIONS = frozenset({"IALWAH18", "IDERNA7"})
MECHANISM_BY_DETECTOR_KIND = {
    "physical": "spike_impossible",
    "stuck": "stuck_flatline",
    "deviation": "statistical_anomaly",
}
HEALTH_POINT_COLUMNS = {
    "health_availability_points": "weighted_health_availability",
    "health_sensor_completeness_points": "weighted_health_sensor_completeness",
    "health_fault_burden_points": "weighted_health_fault_burden",
    "health_reference_consistency_points": "weighted_health_reference_consistency",
    "health_stability_points": "weighted_health_stability",
}


@dataclass
class OperationalScorecardRun:
    table: pd.DataFrame
    distribution: pd.DataFrame
    inconsistencies: pd.DataFrame
    null_audit: pd.DataFrame
    metadata: dict[str, object]


def _normalise_hours(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"station_id", "hour_utc"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")
    result = frame.copy(deep=True)
    result["station_id"] = result["station_id"].astype(str)
    result["hour_utc"] = pd.to_datetime(result["hour_utc"], utc=True, errors="coerce")
    if result[["station_id", "hour_utc"]].isna().any().any():
        raise ValueError(f"{name} contains invalid station-hour keys")
    if result.duplicated(["station_id", "hour_utc"]).any():
        raise ValueError(f"{name} contains duplicate station-hour keys")
    return result.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(
        drop=True
    )


def _parse_reference_hour(value: object) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("reference hour is invalid")
    if parsed.tzinfo is None:
        raise ValueError("reference hour must include a timezone")
    parsed = parsed.tz_convert("UTC")
    if parsed != parsed.floor("h"):
        raise ValueError("reference hour must be aligned to a whole clock hour")
    return parsed


def resolve_operational_reference_hour(
    scores: pd.DataFrame,
    reference_hour: object | None = None,
) -> tuple[pd.Timestamp, str, pd.Timestamp]:
    source = _normalise_hours(scores, "station health scores")
    latest_source_hour = source["hour_utc"].max()
    if reference_hour is not None:
        resolved = _parse_reference_hour(reference_hour)
        if resolved > latest_source_hour:
            raise ValueError(
                f"reference hour {resolved} is after the latest source hour {latest_source_hour}"
            )
        return resolved, "explicit", latest_source_hour
    transmitting = source["is_transmitting"].fillna(False).astype(bool)
    candidates = source.loc[transmitting, "hour_utc"]
    if candidates.empty:
        raise ValueError("station health scores contain no transmitting station-hour")
    resolved = candidates.max()
    mode = (
        "latest_source_hour"
        if resolved == latest_source_hour
        else "latest_hour_with_observed_transmission_terminal_padding_skipped"
    )
    return resolved, mode, latest_source_hour


def load_health_forecast_models(
    model_directory: Path,
    horizons: tuple[int, ...] = SCORECARD_HORIZONS,
) -> dict[int, FittedHealthForecastModel]:
    models: dict[int, FittedHealthForecastModel] = {}
    for horizon in horizons:
        path = Path(model_directory) / (
            f"health_forecast_forecast_transmitting_origin_{int(horizon)}h.joblib"
        )
        model = joblib.load(path)
        if not isinstance(model, FittedHealthForecastModel):
            raise TypeError(f"unexpected health-forecast bundle type: {path}")
        if int(model.horizon_h or -1) != int(horizon):
            raise ValueError(f"health-forecast bundle has the wrong horizon: {path}")
        if str(model.regime) != "transmitting_origin":
            raise ValueError(f"health-forecast bundle has the wrong population: {path}")
        models[int(horizon)] = model
    return models


def _station_roster(registry: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    source = registry.copy(deep=True)
    if "station_id" not in source.columns:
        raise ValueError("station registry is missing station_id")
    source["station_id"] = source["station_id"].astype(str)
    if source.duplicated("station_id").any():
        raise ValueError("station registry contains duplicate station IDs")
    observed = set(scores["station_id"].astype(str).unique())
    registered = set(source["station_id"].unique())
    if observed.difference(registered):
        missing = ", ".join(sorted(observed.difference(registered)))
        raise ValueError(f"station registry does not cover health stations: {missing}")
    keep = [
        column
        for column in (
            "station_id",
            "station_name",
            "city",
            "country",
            "latitude",
            "longitude",
            "elevation",
        )
        if column in source.columns
    ]
    roster = source.loc[source["station_id"].isin(observed), keep].copy()
    for column in ("station_name", "city", "country"):
        if column not in roster.columns:
            roster[column] = ""
        roster[column] = roster[column].fillna("").astype(str)
    for column in ("latitude", "longitude", "elevation"):
        if column not in roster.columns:
            roster[column] = np.nan
        roster[column] = pd.to_numeric(roster[column], errors="coerce")
    roster["location"] = roster[["city", "country"]].apply(
        lambda row: ", ".join(value for value in row.astype(str) if value), axis=1
    )
    return roster.sort_values("station_id", kind="mergesort").reset_index(drop=True)


def _current_detector_evidence(history: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for station_id, station in history.groupby("station_id", sort=False):
        station = station.sort_values("hour_utc", kind="mergesort").reset_index(drop=True)
        detector = build_causal_detector_evidence(station).reset_index(drop=True)
        detector.insert(0, "hour_utc", station["hour_utc"].to_numpy())
        detector.insert(0, "station_id", str(station_id))
        rows.append(detector)
    if not rows:
        return pd.DataFrame(columns=["station_id", "hour_utc"])
    return pd.concat(rows, ignore_index=True)


def _join_current_availability_scope(
    table: pd.DataFrame,
    availability: pd.DataFrame | None,
    reference_hour: pd.Timestamp,
) -> pd.DataFrame:
    result = table.copy(deep=True)
    result["availability_evaluation_scope"] = "not_represented_in_reliability_table"
    result["availability_source_disagreement"] = False
    if availability is None or availability.empty:
        return result
    source = _normalise_hours(availability, "availability classification")
    current = source.loc[source["hour_utc"].eq(reference_hour)].copy()
    if current.empty:
        return result
    keep = [
        column
        for column in ("station_id", "availability_scope", "availability_class")
        if column in current.columns
    ]
    current = current.loc[:, keep].rename(
        columns={
            "availability_scope": "_evaluation_scope",
            "availability_class": "_evaluation_class",
        }
    )
    result = result.merge(current, on="station_id", how="left", validate="one_to_one")
    represented = result["_evaluation_scope"].notna()
    result.loc[represented, "availability_evaluation_scope"] = result.loc[
        represented, "_evaluation_scope"
    ].astype(str)
    mismatch = (
        represented
        & result["_evaluation_class"].notna()
        & result["_evaluation_class"].ne("excluded")
        & result["_evaluation_class"].ne(result["availability_class"])
    )
    result["availability_source_disagreement"] = mismatch
    return result.drop(columns=["_evaluation_scope", "_evaluation_class"])


def _run_hours(values: pd.Series) -> pd.Series:
    state = values.fillna(False).astype(bool)
    segment = state.ne(state.shift(fill_value=False)).cumsum()
    return state.astype(int).groupby(segment).cumsum().where(state, 0).astype(int)


def _history_summary(
    history: pd.DataFrame,
    reference_hour: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for station_id, station in history.groupby("station_id", sort=False):
        station = station.sort_values("hour_utc", kind="mergesort").reset_index(drop=True)
        transmitting = station["is_transmitting"].fillna(False).astype(bool)
        full = station["availability_class"].eq(AVAILABILITY_CLASS_FULL_OUTAGE)
        fault = station["causal_fault_evidence"].fillna(False).astype(bool)
        record: dict[str, object] = {"station_id": str(station_id)}
        for days, hours in ((7, 24 * 7), (30, 24 * 30)):
            start = reference_hour - pd.Timedelta(hours=hours - 1)
            selected = station["hour_utc"].ge(start)
            denominator = int(selected.sum())
            record[f"uptime_{days}d_pct"] = (
                100.0 * float(transmitting.loc[selected].mean())
                if denominator
                else np.nan
            )
            record[f"uptime_{days}d_history_hours"] = denominator
            record[f"uptime_{days}d_status"] = (
                "complete" if denominator >= hours else f"partial_history_{denominator}h"
            )
        start_30d = reference_hour - pd.Timedelta(hours=24 * 30 - 1)
        event_start = full & ~full.shift(1, fill_value=False)
        segment = event_start.cumsum().where(full)
        recent_segments = segment.loc[station["hour_utc"].ge(start_30d)].dropna().unique()
        record["full_outage_events_trailing_30d"] = int(len(recent_segments))
        recovery = ~full & full.shift(1, fill_value=False)
        ended = station.loc[recovery, "hour_utc"] - pd.Timedelta(hours=1)
        active = bool(full.iloc[-1])
        if active:
            record["hours_since_last_outage_ended"] = np.nan
            record["hours_since_last_outage_ended_status"] = "active_full_outage"
        elif len(ended):
            last_end = ended.iloc[-1]
            record["hours_since_last_outage_ended"] = float(
                (reference_hour - last_end) / pd.Timedelta(hours=1)
            )
            record["hours_since_last_outage_ended_status"] = "available"
        else:
            record["hours_since_last_outage_ended"] = np.nan
            record["hours_since_last_outage_ended_status"] = "no_completed_full_outage"
        fault_run = _run_hours(fault)
        record["current_fault_run_hours"] = int(fault_run.iloc[-1])
        for name, hours in (("24h", 24), ("7d", 24 * 7)):
            start = reference_hour - pd.Timedelta(hours=hours - 1)
            record[f"fault_hours_trailing_{name}"] = int(
                fault.loc[station["hour_utc"].ge(start)].sum()
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _forecast_columns(table: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    result = table.copy(deep=True)
    for horizon in horizons:
        result[f"forecast_health_{horizon}h"] = np.nan
        result[f"forecast_band_{horizon}h"] = "not_available"
        result[f"forecast_change_{horizon}h"] = np.nan
        result[f"forecast_status_{horizon}h"] = "not_evaluated"
    return result


def _attach_forecasts(
    table: pd.DataFrame,
    history: pd.DataFrame,
    registry: pd.DataFrame,
    reference_hour: pd.Timestamp,
    forecast_models: Mapping[int, FittedHealthForecastModel] | None,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    result = _forecast_columns(table, horizons)
    full = result["transmission_status"].eq("full_outage")
    no_history = result["health_total"].isna()
    result["forecast_scope"] = np.select(
        [full, no_history, result["transmission_status"].eq("partial_outage")],
        [
            "not_applicable_active_full_outage",
            "not_applicable_insufficient_history",
            "applicable_transmitting_partial_outage",
        ],
        default="applicable_transmitting",
    )
    if forecast_models is None:
        applicable = result["forecast_scope"].str.startswith("applicable_")
        result.loc[applicable, "forecast_scope"] = "unavailable_missing_forecast_models"
        for horizon in horizons:
            result.loc[applicable, f"forecast_status_{horizon}h"] = "missing_model"
        return result
    bundle = build_health_forecast_features(
        history,
        station_metadata=registry,
        station_ids=tuple(sorted(result["station_id"].astype(str))),
    )
    for horizon in horizons:
        model = forecast_models.get(int(horizon))
        applicable = result["forecast_scope"].str.startswith("applicable_")
        if model is None:
            result.loc[applicable, f"forecast_status_{horizon}h"] = "missing_model"
            continue
        frame = health_forecast_inference_frame(bundle, int(horizon))
        current = frame.loc[
            frame["hour_utc"].eq(reference_hour)
            & frame["is_transmitting"].fillna(False).astype(bool)
        ].copy()
        if current.empty:
            result.loc[applicable, f"forecast_status_{horizon}h"] = "missing_origin_features"
            continue
        predicted = np.asarray(model.predict_health(current), dtype=float)
        if not np.isfinite(predicted).all():
            raise RuntimeError(f"health forecast {horizon}h produced a non-finite value")
        if ((predicted < -1e-10) | (predicted > 100.0 + 1e-10)).any():
            raise RuntimeError(f"health forecast {horizon}h escaped the 0-100 bounds")
        current["_forecast"] = np.clip(predicted, 0.0, 100.0)
        current["_forecast_band"] = health_band_for_values(current["_forecast"])
        mapping = current.set_index("station_id")
        matched = result["station_id"].isin(mapping.index) & applicable
        result.loc[matched, f"forecast_health_{horizon}h"] = result.loc[
            matched, "station_id"
        ].map(mapping["_forecast"])
        result.loc[matched, f"forecast_band_{horizon}h"] = result.loc[
            matched, "station_id"
        ].map(mapping["_forecast_band"]).astype(str)
        result.loc[matched, f"forecast_change_{horizon}h"] = (
            result.loc[matched, f"forecast_health_{horizon}h"]
            - result.loc[matched, "health_total"]
        )
        result.loc[matched, f"forecast_status_{horizon}h"] = "available"
        result.loc[applicable & ~matched, f"forecast_status_{horizon}h"] = (
            "missing_origin_features"
        )
    return result


def _attach_fault_evidence(
    table: pd.DataFrame,
    detector: pd.DataFrame,
    reference_hour: pd.Timestamp,
) -> pd.DataFrame:
    result = table.copy(deep=True)
    current = detector.loc[detector["hour_utc"].eq(reference_hour)].copy()
    result = result.merge(current, on=["station_id", "hour_utc"], how="left", validate="one_to_one")
    result["fault_detected"] = result["causal_fault_evidence"].fillna(False).astype(bool)
    result["fault_signal_source"] = "causal_rule_evidence_proxy"
    result["trained_binary_detector_status"] = "not_available_causal_deployment"
    mechanisms: list[str] = []
    groups: list[str] = []
    for row in result.itertuples(index=False):
        active_mechanisms = [
            mechanism
            for kind, mechanism in MECHANISM_BY_DETECTOR_KIND.items()
            if float(getattr(row, f"causal_detector_{kind}_any_now", 0.0) or 0.0) > 0.0
        ]
        active_groups = [
            group
            for group in SENSOR_GROUP_ORDER
            if any(
                float(getattr(row, f"causal_detector_{kind}_now_{group}", 0.0) or 0.0)
                > 0.0
                for kind in MECHANISM_BY_DETECTOR_KIND
            )
        ]
        mechanisms.append("|".join(active_mechanisms))
        groups.append("|".join(active_groups))
    result["provisional_mechanism_evidence"] = mechanisms
    result["fault_evidence_sensor_groups"] = groups
    result["predicted_mechanism_reason_codes"] = ""
    result["reason_code_status"] = np.select(
        [
            result["transmission_status"].eq("full_outage"),
            ~result["fault_detected"],
        ],
        ["not_applicable_not_transmitting", "not_applicable_no_fault"],
        default="not_available_causal_deployment",
    )
    return result.drop(
        columns=[column for column in result.columns if column.startswith("causal_detector_")]
    )


def _attach_causal_offset_evidence(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy(deep=True)
    severity = pd.to_numeric(result["reference_persistent_severity"], errors="coerce")
    magnitude = pd.to_numeric(result["reference_pressure_prior_median"], errors="coerce")
    present = severity.gt(0.0) & magnitude.notna()
    result["confirmed_sustained_calibration_offset"] = False
    result["calibration_offset_channel"] = ""
    result["calibration_offset_magnitude"] = np.nan
    result["calibration_offset_status"] = "not_available_causal_live_confirmation"
    result["causal_persistent_offset_evidence"] = present
    result["causal_persistent_offset_channel"] = np.where(present, "pressure", "")
    result["causal_persistent_offset_magnitude_hpa"] = magnitude.where(present)
    return result.drop(
        columns=["reference_persistent_severity", "reference_pressure_prior_median"]
    )


def _retrospective_calibration_overlap_count(
    calibration_corroboration: pd.DataFrame | None,
    reference_hour: pd.Timestamp,
) -> int:
    if calibration_corroboration is None or calibration_corroboration.empty:
        return 0
    source = calibration_corroboration.copy(deep=True)
    required = {"verdict", "sustained_offset", "start_hour", "end_hour"}
    if required.difference(source.columns):
        return 0
    start = pd.to_datetime(source["start_hour"], utc=True, errors="coerce")
    end = pd.to_datetime(source["end_hour"], utc=True, errors="coerce")
    confirmed = source["verdict"].astype(str).eq("confirmed") & source[
        "sustained_offset"
    ].fillna(False).astype(bool)
    return int((confirmed & start.le(reference_hour) & end.ge(reference_hour)).sum())


def _distribution_table(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for value, count in table["health_band"].value_counts(dropna=False).items():
        rows.append({"category": "health_band", "value": str(value), "count": int(count)})
    for value, count in table["transmission_status"].value_counts(dropna=False).items():
        rows.append(
            {"category": "transmission_status", "value": str(value), "count": int(count)}
        )
    rows.extend(
        [
            {
                "category": "active_fault",
                "value": "yes",
                "count": int(table["fault_detected"].sum()),
            },
            {
                "category": "causal_confirmed_calibration_offset",
                "value": "yes",
                "count": int(table["confirmed_sustained_calibration_offset"].sum()),
            },
            {
                "category": "causal_persistent_offset_evidence",
                "value": "yes",
                "count": int(table["causal_persistent_offset_evidence"].sum()),
            },
        ]
    )
    return pd.DataFrame(rows)


def _inconsistency_table(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(mask: pd.Series, code: str, detail: str) -> None:
        for station_id in table.loc[mask.fillna(False), "station_id"]:
            rows.append({"station_id": str(station_id), "code": code, "detail": detail})

    add(
        table["health_total"].ge(80.0) & table["fault_detected"],
        "healthy_band_with_active_fault",
        "Current health is at least 80 while causal fault evidence is active.",
    )
    forecast_columns = [f"forecast_health_{horizon}h" for horizon in SCORECARD_HORIZONS]
    add(
        table["transmission_status"].eq("full_outage")
        & table[forecast_columns].notna().any(axis=1),
        "forecast_present_during_full_outage",
        "A transmitting-origin health forecast is present during a full outage.",
    )
    add(
        table["transmission_status"].eq("partial_outage")
        & table["absent_sensor_groups"].fillna("").eq(""),
        "partial_outage_without_absent_groups",
        "Partial-outage status has no absent sensor group.",
    )
    add(
        table["availability_source_disagreement"].fillna(False),
        "availability_source_disagreement",
        "The current health grid and reliability classification disagree.",
    )
    add(
        ~table["fault_detected"]
        & table["predicted_mechanism_reason_codes"].fillna("").ne(""),
        "reason_code_without_fault",
        "A mechanism reason code is present without a current fault signal.",
    )
    component_total = sum(
        pd.to_numeric(table[column], errors="coerce") for column in HEALTH_POINT_COLUMNS
    )
    add(
        table["health_total"].notna()
        & ~np.isclose(
            pd.to_numeric(table["health_total"], errors="coerce"),
            component_total,
            rtol=1e-9,
            atol=1e-8,
            equal_nan=True,
        ),
        "health_components_do_not_sum",
        "The five weighted health components do not reconstruct health_total.",
    )
    return pd.DataFrame(rows, columns=["station_id", "code", "detail"])


def _null_audit(table: pd.DataFrame) -> pd.DataFrame:
    expected: dict[str, tuple[pd.Series, str]] = {}
    health_missing = table["health_total"].isna()
    for column in ("health_total", *HEALTH_POINT_COLUMNS):
        expected[column] = (health_missing, "health_status explicitly reports insufficient history")
    for horizon in SCORECARD_HORIZONS:
        mask = table[f"forecast_status_{horizon}h"].ne("available")
        explanation = "forecast_status explicitly reports not applicable or unavailable"
        expected[f"forecast_health_{horizon}h"] = (mask, explanation)
        expected[f"forecast_change_{horizon}h"] = (mask, explanation)
    expected["hours_since_last_outage_ended"] = (
        table["hours_since_last_outage_ended_status"].ne("available"),
        "outage-end status reports an active outage or no completed outage",
    )
    expected["calibration_offset_magnitude"] = (
        ~table["confirmed_sustained_calibration_offset"],
        "no causal live calibration confirmation is available",
    )
    expected["causal_persistent_offset_magnitude_hpa"] = (
        ~table["causal_persistent_offset_evidence"],
        "no causal persistent pressure-offset evidence is active",
    )
    rows: list[dict[str, object]] = []
    unexplained_by_row: list[list[str]] = [[] for _ in range(len(table))]
    for column in table.columns:
        null = table[column].isna()
        if not null.any():
            continue
        expected_mask, explanation = expected.get(
            column,
            (pd.Series(False, index=table.index), "no expected-null rule"),
        )
        unexplained = null & ~expected_mask
        for index in np.flatnonzero(unexplained.to_numpy()):
            unexplained_by_row[int(index)].append(column)
        rows.append(
            {
                "column": column,
                "null_count": int(null.sum()),
                "expected_null_count": int((null & expected_mask).sum()),
                "unexplained_null_count": int(unexplained.sum()),
                "explanation": explanation,
            }
        )
    table["unexplained_null_fields"] = ["|".join(values) for values in unexplained_by_row]
    table["unexplained_null_count"] = [len(values) for values in unexplained_by_row]
    return pd.DataFrame(rows)


def build_operational_scorecard(
    scores: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    availability: pd.DataFrame | None = None,
    forecast_models: Mapping[int, FittedHealthForecastModel] | None = None,
    calibration_corroboration: pd.DataFrame | None = None,
    reference_hour: object | None = None,
    expected_station_count: int = 26,
    horizons: tuple[int, ...] = SCORECARD_HORIZONS,
) -> OperationalScorecardRun:
    source = _normalise_hours(scores, "station health scores")
    required = {
        "is_transmitting",
        "availability_class",
        "absent_sensor_groups",
        "health_total",
        "health_band",
        "health_status",
        "causal_fault_evidence",
        "reference_persistent_severity",
        "reference_pressure_prior_median",
        *HEALTH_COMPONENT_COLUMNS,
        *HEALTH_POINT_COLUMNS.values(),
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"station health scores are missing columns: {', '.join(missing)}")
    resolved, resolution, latest_source = resolve_operational_reference_hour(
        source, reference_hour
    )
    history = source.loc[source["hour_utc"].le(resolved)].copy()
    roster = _station_roster(registry, source)
    if len(roster) != int(expected_station_count):
        raise RuntimeError(
            f"scorecard roster has {len(roster)} stations; expected {int(expected_station_count)}"
        )
    current_columns = [
        "station_id",
        "hour_utc",
        "is_transmitting",
        "availability_class",
        "absent_sensor_groups",
        "full_outage_run_hours",
        "partial_outage_run_hours",
        "health_total",
        "health_band",
        "health_status",
        "health_history_hours",
        "causal_fault_evidence",
        "reference_persistent_severity",
        "reference_pressure_prior_median",
        *HEALTH_POINT_COLUMNS.values(),
    ]
    current = history.loc[
        history["hour_utc"].eq(resolved), current_columns
    ].copy()
    current = current.rename(
        columns={source: destination for destination, source in HEALTH_POINT_COLUMNS.items()}
    )
    current = current.drop(columns=[column for column in roster.columns if column != "station_id" and column in current.columns])
    table = roster.merge(current, on="station_id", how="left", validate="one_to_one")
    table["hour_utc"] = resolved
    table["reference_hour_utc"] = resolved
    table["reference_resolution"] = resolution
    table["availability_class"] = table["availability_class"].fillna("not_in_service")
    table["transmission_status"] = table["availability_class"].map(
        {
            AVAILABILITY_CLASS_ONLINE: "transmitting",
            AVAILABILITY_CLASS_PARTIAL_OUTAGE: "partial_outage",
            AVAILABILITY_CLASS_FULL_OUTAGE: "full_outage",
        }
    ).fillna("not_in_service")
    table["absent_sensor_groups"] = table["absent_sensor_groups"].fillna("").astype(str)
    full_run = pd.to_numeric(table.get("full_outage_run_hours", 0.0), errors="coerce").fillna(0.0)
    partial_run = pd.to_numeric(table.get("partial_outage_run_hours", 0.0), errors="coerce").fillna(0.0)
    table["current_outage_run_hours"] = np.where(
        table["transmission_status"].eq("full_outage"),
        full_run,
        np.where(table["transmission_status"].eq("partial_outage"), partial_run, 0.0),
    )
    table["spatial_context_status"] = np.where(
        table["station_id"].isin(NO_SPATIAL_NEIGHBOUR_STATIONS),
        "no_spatial_neighbour_not_penalised",
        "not_used_by_health_score",
    )
    table = _join_current_availability_scope(table, availability, resolved)
    detector = _current_detector_evidence(history)
    table = _attach_fault_evidence(table, detector, resolved)
    table = table.merge(
        _history_summary(history, resolved),
        on="station_id",
        how="left",
        validate="one_to_one",
    )
    table = _attach_causal_offset_evidence(table)
    table = _attach_forecasts(
        table,
        history,
        registry,
        resolved,
        forecast_models,
        tuple(int(value) for value in horizons),
    )
    table = table.sort_values(
        ["health_total", "station_id"], kind="mergesort", na_position="last"
    ).reset_index(drop=True)
    null_audit = _null_audit(table)
    inconsistencies = _inconsistency_table(table)
    distribution = _distribution_table(table)
    metadata: dict[str, object] = {
        "reference_hour_utc": resolved.isoformat(),
        "reference_resolution": resolution,
        "latest_source_hour_utc": latest_source.isoformat(),
        "terminal_hours_skipped_by_default": int(
            max(0.0, (latest_source - resolved) / pd.Timedelta(hours=1))
        ),
        "station_count": int(len(table)),
        "field_count": int(len(table.columns)),
        "unexplained_nulls": int(table["unexplained_null_count"].sum()),
        "inconsistency_count": int(len(inconsistencies)),
        "retrospective_calibration_intervals_overlapping_reference_excluded": (
            _retrospective_calibration_overlap_count(
                calibration_corroboration,
                resolved,
            )
        ),
        "fault_signal_contract": "causal_rule_evidence_proxy_not_trained_binary_model",
        "reason_code_contract": "trained_event_reason_codes_unavailable_for_causal_live_use",
        "partial_outage_forecast_contract": "included_because_partial_outage_is_transmitting",
    }
    return OperationalScorecardRun(
        table=table,
        distribution=distribution,
        inconsistencies=inconsistencies,
        null_audit=null_audit,
        metadata=metadata,
    )


def validate_delete_future_operational_scorecard(
    scores: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    availability: pd.DataFrame | None,
    forecast_models: Mapping[int, FittedHealthForecastModel],
    calibration_corroboration: pd.DataFrame | None,
    reference_hour: object,
    expected_station_count: int = 26,
) -> pd.DataFrame:
    cutoff = _parse_reference_hour(reference_hour)
    full = build_operational_scorecard(
        scores,
        registry,
        availability=availability,
        forecast_models=forecast_models,
        calibration_corroboration=calibration_corroboration,
        reference_hour=cutoff,
        expected_station_count=expected_station_count,
    ).table
    score_source = _normalise_hours(scores, "station health scores")
    score_source = score_source.loc[score_source["hour_utc"].le(cutoff)].copy()
    availability_source = None
    if availability is not None:
        availability_source = _normalise_hours(
            availability, "availability classification"
        )
        availability_source = availability_source.loc[
            availability_source["hour_utc"].le(cutoff)
        ].copy()
    truncated = build_operational_scorecard(
        score_source,
        registry,
        availability=availability_source,
        forecast_models=forecast_models,
        calibration_corroboration=calibration_corroboration,
        reference_hour=cutoff,
        expected_station_count=expected_station_count,
    ).table
    key = "station_id"
    left = full.sort_values(key, kind="mergesort").reset_index(drop=True)
    right = truncated.sort_values(key, kind="mergesort").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for column in left.columns:
        if column not in right.columns:
            raise RuntimeError(f"delete-the-future rebuild lost scorecard field: {column}")
        if pd.api.types.is_numeric_dtype(left[column]):
            passed = np.isclose(
                pd.to_numeric(left[column], errors="coerce"),
                pd.to_numeric(right[column], errors="coerce"),
                rtol=1e-9,
                atol=1e-10,
                equal_nan=True,
            )
        else:
            passed = left[column].fillna("<NA>").astype(str).eq(
                right[column].fillna("<NA>").astype(str)
            ).to_numpy()
        for station_id, value in zip(left[key], passed, strict=True):
            rows.append(
                {
                    "station_id": str(station_id),
                    "reference_hour_utc": cutoff,
                    "field": column,
                    "passed": bool(value),
                }
            )
    return pd.DataFrame(rows)


def _format_table(frame: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in frame.columns]
    return frame.loc[:, available].to_string(index=False, na_rep="N/A")


def operational_scorecard_report(run: OperationalScorecardRun) -> str:
    table = run.table
    lines = [
        "COMBINED OPERATIONAL STATION SCORECARD",
        "",
        "REFERENCE",
        f"reference_hour_utc={run.metadata['reference_hour_utc']}",
        f"reference_resolution={run.metadata['reference_resolution']}",
        f"latest_source_hour_utc={run.metadata['latest_source_hour_utc']}",
        f"terminal_hours_skipped_by_default={run.metadata['terminal_hours_skipped_by_default']}",
        "The default skips terminal-padding hours with no observed transmissions; an explicit --reference-hour is always honoured.",
        "",
        "DEPLOYMENT CONTRACT",
        "Current fault status is a causal rule-evidence proxy, not the retrospective trained binary detector.",
        "Event-level trained mechanism reason codes are unavailable in the causal live path and are never filled from held-out prediction ledgers.",
        "Health forecasts are emitted for transmitting stations, including partial outages, and suppressed during full outages.",
        "Calibration-corroboration intervals are retrospective; they are excluded from operational fields. Causal persistent pressure-offset evidence is shown separately.",
        "",
        "DISTRIBUTION SUMMARY",
        run.distribution.to_string(index=False),
        "",
        "CORE SCORECARD",
        _format_table(
            table,
            [
                "station_id",
                "station_name",
                "location",
                "transmission_status",
                "current_outage_run_hours",
                "absent_sensor_groups",
                "health_total",
                "health_band",
                "fault_detected",
                "provisional_mechanism_evidence",
                "reason_code_status",
                "fault_hours_trailing_24h",
                "fault_hours_trailing_7d",
                "current_fault_run_hours",
                "uptime_7d_pct",
                "uptime_30d_pct",
                "full_outage_events_trailing_30d",
                "hours_since_last_outage_ended",
                "causal_persistent_offset_evidence",
                "causal_persistent_offset_magnitude_hpa",
            ],
        ),
        "",
        "HEALTH COMPONENTS",
        _format_table(table, ["station_id", "health_total", *HEALTH_POINT_COLUMNS]),
        "",
        "HEALTH FORECASTS",
        _format_table(
            table,
            [
                "station_id",
                "forecast_scope",
                *[
                    column
                    for horizon in SCORECARD_HORIZONS
                    for column in (
                        f"forecast_health_{horizon}h",
                        f"forecast_band_{horizon}h",
                        f"forecast_change_{horizon}h",
                    )
                ],
            ],
        ),
        "",
        "CONSISTENCY CHECKS",
        (
            "none"
            if run.inconsistencies.empty
            else run.inconsistencies.to_string(index=False)
        ),
        "",
        "NULL AUDIT",
        (
            "no null-valued fields"
            if run.null_audit.empty
            else run.null_audit.to_string(index=False)
        ),
        "",
        "INVARIANTS",
        f"station_count={run.metadata['station_count']}",
        f"field_count={run.metadata['field_count']}",
        f"unexplained_nulls={run.metadata['unexplained_nulls']}",
        f"inconsistency_count={run.metadata['inconsistency_count']}",
        "retrospective_calibration_intervals_overlapping_reference_excluded="
        f"{run.metadata['retrospective_calibration_intervals_overlapping_reference_excluded']}",
        "",
    ]
    return "\n".join(lines)


def write_operational_scorecard_outputs(
    run: OperationalScorecardRun,
    *,
    table_path: Path,
    report_path: Path,
    invariants_path: Path,
    causality_path: Path,
    causality_audit: pd.DataFrame,
    input_hashes_before: Mapping[str, str],
    input_hashes_after: Mapping[str, str],
) -> dict[str, Path]:
    if dict(input_hashes_before) != dict(input_hashes_after):
        raise RuntimeError("an upstream scorecard artifact changed during assembly")
    if causality_audit.empty or not causality_audit["passed"].astype(bool).all():
        raise RuntimeError("operational scorecard delete-the-future validation failed")
    if int(run.metadata["unexplained_nulls"]) != 0:
        raise RuntimeError("operational scorecard contains unexplained nulls")
    outputs = {
        "table": Path(table_path),
        "report": Path(report_path),
        "invariants": Path(invariants_path),
        "causality": Path(causality_path),
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    run.table.to_csv(outputs["table"], index=False)
    outputs["report"].write_text(operational_scorecard_report(run), encoding="utf-8")
    causality_audit.to_csv(outputs["causality"], index=False)
    payload = {
        **run.metadata,
        "all_delete_future_checks_passed": bool(causality_audit["passed"].all()),
        "delete_future_comparisons": int(len(causality_audit)),
        "input_hashes_before": dict(input_hashes_before),
        "input_hashes_after": dict(input_hashes_after),
        "upstream_artifacts_unchanged": dict(input_hashes_before)
        == dict(input_hashes_after),
        "inconsistencies": run.inconsistencies.to_dict(orient="records"),
        "null_audit": run.null_audit.to_dict(orient="records"),
    }
    outputs["invariants"].write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return outputs
