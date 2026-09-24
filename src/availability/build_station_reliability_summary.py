from __future__ import annotations

import pandas as pd

from src.availability.build_availability_events import (
    AVAILABILITY_CLASS_EXCLUDED,
    SENSOR_GROUP_ORDER,
    SOURCE_KIND_OBSERVED,
    sensor_group_absence_mask,
)
from src.config.paths import (
    AVAILABILITY_CLASSIFICATION_PATH,
    AVAILABILITY_EVENTS_PATH,
    DATA_AUDIT_SUMMARY_PATH,
    PARTIAL_OUTAGE_EVENTS_PATH,
    STATION_REGISTRY_PATH,
    STATION_RELIABILITY_SUMMARY_PATH,
    ensure_directories,
)


def _require_columns(
    frame: pd.DataFrame,
    required_columns: list[str],
    frame_name: str,
) -> None:
    missing_columns = [
        column for column in required_columns
        if column not in frame.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _station_metadata(
    registry: pd.DataFrame,
    station_id: str,
) -> dict[str, object]:
    station = registry.loc[registry["station_id"].eq(station_id)]
    if station.empty:
        return {
            "city": None,
            "region": None,
            "latitude": None,
            "longitude": None,
            "elevation_m": None,
            "install_date": None,
        }
    row = station.iloc[0]
    longitude = pd.to_numeric(pd.Series([row.get("longitude")]), errors="coerce").iloc[0]
    return {
        "city": row.get("city"),
        "region": row.get(
            "region",
            "northwest" if pd.notna(longitude) and longitude < 16 else "southeast",
        ),
        "latitude": row.get("latitude"),
        "longitude": row.get("longitude"),
        "elevation_m": row.get("elevation"),
        "install_date": row.get("install_date"),
    }


def build_station_reliability_summary(
    availability_classification: pd.DataFrame,
    full_events: pd.DataFrame,
    partial_events: pd.DataFrame,
    audit: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        availability_classification,
        [
            "station_id",
            "hour_utc",
            "availability_class",
            "absent_sensor_groups",
            "is_transmitting",
            "source_kind",
        ],
        "availability_classification",
    )
    _require_columns(
        full_events,
        ["station_id", "start_utc", "end_utc", "duration_hours", "outage_class"],
        "full_events",
    )
    _require_columns(
        partial_events,
        ["station_id", "duration_hours"],
        "partial_events",
    )
    _require_columns(audit, ["station_id"], "audit")
    _require_columns(registry, ["station_id"], "registry")

    classification = availability_classification.copy(deep=True)
    classification["station_id"] = classification["station_id"].astype(str)
    classification["hour_utc"] = pd.to_datetime(
        classification["hour_utc"],
        utc=True,
        errors="coerce",
    )
    classification["is_transmitting"] = classification["is_transmitting"].fillna(False).astype(bool)

    full = full_events.copy(deep=True)
    full["station_id"] = full["station_id"].astype(str)
    full["start_utc"] = pd.to_datetime(full["start_utc"], utc=True, errors="coerce")
    full["end_utc"] = pd.to_datetime(full["end_utc"], utc=True, errors="coerce")
    full["duration_hours"] = pd.to_numeric(full["duration_hours"], errors="coerce").fillna(0)

    partial = partial_events.copy(deep=True)
    partial["station_id"] = partial["station_id"].astype(str)
    partial["duration_hours"] = pd.to_numeric(
        partial["duration_hours"],
        errors="coerce",
    ).fillna(0)

    audit = audit.copy(deep=True)
    audit["station_id"] = audit["station_id"].astype(str)
    registry = registry.copy(deep=True)
    registry["station_id"] = registry["station_id"].astype(str)

    observed_rows = classification.loc[
        classification["source_kind"].eq(SOURCE_KIND_OBSERVED)
    ]
    data_end = observed_rows["hour_utc"].max()
    station_ids = sorted(
        set(registry["station_id"])
        | set(classification["station_id"])
        | set(full["station_id"])
    )

    rows: list[dict[str, object]] = []
    for station_id in station_ids:
        station_classification = classification.loc[
            classification["station_id"].eq(station_id)
        ].copy()
        station_full = full.loc[full["station_id"].eq(station_id)].copy()
        station_partial = partial.loc[partial["station_id"].eq(station_id)].copy()
        audit_matches = audit.loc[audit["station_id"].eq(station_id)]
        audit_row = audit_matches.iloc[0] if not audit_matches.empty else None

        row = {
            "station_id": station_id,
            **_station_metadata(registry, station_id),
            "dataset_end_utc": str(data_end) if pd.notna(data_end) else None,
        }
        if audit_row is not None:
            total_rows = pd.to_numeric(
                pd.Series([audit_row.get("total_rows")]),
                errors="coerce",
            ).fillna(0).iloc[0]
            present_rows = pd.to_numeric(
                pd.Series([audit_row.get("present_rows")]),
                errors="coerce",
            ).fillna(0).iloc[0]
            row["status_class"] = audit_row.get("status_class")
            row["uptime_pct"] = (
                100.0 * present_rows / total_rows if total_rows > 0 else None
            )
            row["total_active_hours"] = int(total_rows)
            row["total_present_hours"] = int(present_rows)
            row["total_outage_hours"] = int(
                pd.to_numeric(
                    pd.Series([audit_row.get("true_outage_candidate_rows")]),
                    errors="coerce",
                ).fillna(0).iloc[0]
            )
        else:
            row["status_class"] = None
            row["uptime_pct"] = None
            row["total_active_hours"] = None
            row["total_present_hours"] = None
            row["total_outage_hours"] = None

        row["total_event_count"] = int(len(station_full))
        row["full_outage_event_count"] = int(len(station_full))
        row["full_outage_hours"] = int(station_full["duration_hours"].sum())
        row["local_event_count"] = int(
            station_full["outage_class"].eq("local").sum()
        )
        row["coordinated_event_count"] = int(
            station_full["outage_class"].eq("coordinated").sum()
        )
        row["partial_outage_event_count"] = int(len(station_partial))
        row["partial_outage_hours"] = int(station_partial["duration_hours"].sum())
        row["transmitting_hours"] = int(
            station_classification["is_transmitting"].sum()
        )
        row["structural_gap_full_outage_hours"] = int(
            station_classification["source_kind"].eq(
                "materialized_structural_gap"
            ).sum()
        )
        row["excluded_rows"] = int(
            station_classification["availability_class"].eq(
                AVAILABILITY_CLASS_EXCLUDED
            ).sum()
        )

        if not station_full.empty:
            durations = station_full["duration_hours"]
            row["median_event_duration_h"] = float(durations.median())
            row["max_event_duration_h"] = float(durations.max())
            last_event = station_full.sort_values(
                ["start_utc", "end_utc"],
                kind="mergesort",
            ).iloc[-1]
            row["last_outage_start_utc"] = str(last_event["start_utc"])
            last_end = last_event["end_utc"]
            days_since = (
                (data_end - last_end).total_seconds() / 86400.0
                if pd.notna(data_end) and pd.notna(last_end)
                else None
            )
            row["days_since_last_outage_at_data_end"] = (
                float(max(days_since, 0.0)) if days_since is not None else None
            )
        else:
            row["median_event_duration_h"] = None
            row["max_event_duration_h"] = None
            row["last_outage_start_utc"] = None
            row["days_since_last_outage_at_data_end"] = None

        transmitting = station_classification["is_transmitting"]
        denominator = int(transmitting.sum())
        for sensor_group in SENSOR_GROUP_ORDER:
            absent = transmitting & sensor_group_absence_mask(
                station_classification,
                sensor_group,
            )
            row[f"{sensor_group}_availability_pct"] = (
                100.0 * (denominator - int(absent.sum())) / denominator
                if denominator > 0
                else None
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["region", "uptime_pct", "station_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def write_station_reliability_summary(
    *,
    availability_classification: pd.DataFrame | None = None,
    full_events: pd.DataFrame | None = None,
    partial_events: pd.DataFrame | None = None,
    audit: pd.DataFrame | None = None,
    registry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    classification = (
        availability_classification
        if availability_classification is not None
        else pd.read_parquet(AVAILABILITY_CLASSIFICATION_PATH)
    )
    events = (
        full_events
        if full_events is not None
        else pd.read_parquet(AVAILABILITY_EVENTS_PATH)
    )
    partial = (
        partial_events
        if partial_events is not None
        else pd.read_parquet(PARTIAL_OUTAGE_EVENTS_PATH)
    )
    audit_frame = audit if audit is not None else pd.read_csv(DATA_AUDIT_SUMMARY_PATH)
    registry_frame = (
        registry if registry is not None else pd.read_csv(STATION_REGISTRY_PATH)
    )
    summary = build_station_reliability_summary(
        classification,
        events,
        partial,
        audit_frame,
        registry_frame,
    )
    ensure_directories()
    summary.to_csv(STATION_RELIABILITY_SUMMARY_PATH, index=False)
    return summary


def main() -> None:
    summary = write_station_reliability_summary()
    print(f"Wrote {STATION_RELIABILITY_SUMMARY_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
