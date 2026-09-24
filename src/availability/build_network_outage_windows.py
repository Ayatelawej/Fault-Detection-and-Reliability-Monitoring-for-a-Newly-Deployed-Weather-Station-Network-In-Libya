from __future__ import annotations

import pandas as pd

from src.config.paths import (
    AVAILABILITY_EVENTS_PATH,
    NETWORK_OUTAGE_WINDOWS_PATH,
    ensure_directories,
)

NETWORK_OUTAGE_MIN_STATIONS = 5
NETWORK_OUTAGE_TIME_WINDOW_HOURS = 1

WINDOW_COLUMNS = [
    "window_id",
    "outage_class",
    "window_start_utc",
    "window_end_utc",
    "backfill_start_utc",
    "backfill_end_utc",
    "station_count",
    "n_events",
    "station_ids",
    "median_duration_hours",
    "max_duration_hours",
    "total_duration_hours",
]

OUTAGE_CLASS_COORDINATED = "coordinated"
OUTAGE_CLASS_LOCAL = "local"
NETWORK_OUTAGE_CLASSES = {OUTAGE_CLASS_COORDINATED}
LEGACY_NETWORK_OUTAGE_CLASSES = {"network_midnight", "network_other"}


def _empty_windows_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": pd.Series(dtype="object"),
            "outage_class": pd.Series(dtype="object"),
            "window_start_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "window_end_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "backfill_start_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "backfill_end_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "station_count": pd.Series(dtype="int64"),
            "n_events": pd.Series(dtype="int64"),
            "station_ids": pd.Series(dtype="object"),
            "median_duration_hours": pd.Series(dtype="float64"),
            "max_duration_hours": pd.Series(dtype="float64"),
            "total_duration_hours": pd.Series(dtype="float64"),
        }
    )[WINDOW_COLUMNS]


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


def _prepared_events(events: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        events,
        ["station_id", "start_utc", "duration_hours"],
        "events",
    )

    prepared = events.copy(deep=True)
    prepared["station_id"] = prepared["station_id"].astype("string")
    prepared["start_utc"] = pd.to_datetime(
        prepared["start_utc"],
        utc=True,
        errors="coerce",
    )
    prepared["duration_hours"] = pd.to_numeric(
        prepared["duration_hours"],
        errors="coerce",
    )
    prepared = prepared.loc[
        prepared["station_id"].notna() & prepared["start_utc"].notna()
    ].copy()
    return prepared.sort_values(
        ["start_utc", "station_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _window_outage_class(window_start_utc: pd.Timestamp) -> str:
    """Shared timing is descriptive; clock hour does not establish a cause."""
    return OUTAGE_CLASS_COORDINATED


def _count_cluster_stations(
    events: pd.DataFrame,
    time_window: pd.Timedelta,
) -> list[int]:
    starts = events["start_utc"]
    station_ids = events["station_id"]

    cluster_sizes = []
    for start_utc in starts:
        in_cluster = (
            starts.ge(start_utc - time_window)
            & starts.le(start_utc + time_window)
        )
        cluster_sizes.append(int(station_ids.loc[in_cluster].nunique()))
    return cluster_sizes


def detect_network_outage_windows(
    events: pd.DataFrame,
    *,
    min_stations: int,
    time_window_hours: int,
) -> pd.DataFrame:
    prepared = _prepared_events(events)
    if prepared.empty:
        return _empty_windows_frame()

    min_stations = int(
        pd.to_numeric(pd.Series([min_stations]), errors="coerce")
        .fillna(0)
        .iloc[0]
    )
    time_window_hours = int(
        pd.to_numeric(pd.Series([time_window_hours]), errors="coerce")
        .fillna(0)
        .iloc[0]
    )
    time_window = pd.Timedelta(hours=max(time_window_hours, 0))

    prepared["cluster_size"] = _count_cluster_stations(
        prepared,
        time_window,
    )
    candidates = prepared.loc[
        pd.to_numeric(prepared["cluster_size"], errors="coerce").ge(
            min_stations,
        )
    ].copy()
    if candidates.empty:
        return _empty_windows_frame()

    candidate_gap = candidates["start_utc"].diff()
    candidates["window_segment_id"] = (
        candidate_gap.isna() | candidate_gap.gt(time_window)
    ).cumsum()

    rows = []
    for _, window_candidates in candidates.groupby(
        "window_segment_id",
        sort=False,
    ):
        window_start_utc = window_candidates["start_utc"].min()
        window_end_utc = window_candidates["start_utc"].max()
        backfill_start_utc = window_start_utc - time_window
        backfill_end_utc = window_end_utc + time_window

        members = prepared.loc[
            prepared["start_utc"].ge(backfill_start_utc)
            & prepared["start_utc"].le(backfill_end_utc)
        ].copy()
        duration_hours = pd.to_numeric(
            members["duration_hours"],
            errors="coerce",
        ).fillna(0)
        station_ids = sorted(members["station_id"].dropna().astype(str).unique())
        station_count = len(station_ids)

        rows.append(
            {
                "window_id": f"NW_{window_start_utc:%Y%m%dT%H}",
                "outage_class": _window_outage_class(window_start_utc),
                "window_start_utc": window_start_utc,
                "window_end_utc": window_end_utc,
                "backfill_start_utc": backfill_start_utc,
                "backfill_end_utc": backfill_end_utc,
                "station_count": station_count,
                "n_events": int(len(members)),
                "station_ids": ";".join(station_ids),
                "median_duration_hours": float(duration_hours.median()),
                "max_duration_hours": float(duration_hours.max()),
                "total_duration_hours": float(duration_hours.sum()),
            }
        )

    windows = pd.DataFrame(rows)
    if windows.empty:
        return _empty_windows_frame()

    station_count = pd.to_numeric(
        windows["station_count"],
        errors="coerce",
    ).fillna(0)
    windows = windows.loc[station_count.ge(min_stations)].copy()
    if windows.empty:
        return _empty_windows_frame()

    windows = windows.sort_values(
        "window_start_utc",
        kind="mergesort",
    ).reset_index(drop=True)
    return windows[WINDOW_COLUMNS]


def assign_outage_class(
    events: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(events, ["start_utc"], "events")

    classified = events.copy(deep=True)
    classified["start_utc"] = pd.to_datetime(
        classified["start_utc"],
        utc=True,
        errors="coerce",
    )
    classified["outage_class"] = OUTAGE_CLASS_LOCAL

    if windows.empty:
        return classified

    _require_columns(
        windows,
        ["backfill_start_utc", "backfill_end_utc"],
        "windows",
    )
    ranges = windows.copy(deep=True)
    ranges["backfill_start_utc"] = pd.to_datetime(
        ranges["backfill_start_utc"],
        utc=True,
        errors="coerce",
    )
    ranges["backfill_end_utc"] = pd.to_datetime(
        ranges["backfill_end_utc"],
        utc=True,
        errors="coerce",
    )
    ranges = ranges.loc[
        ranges["backfill_start_utc"].notna()
        & ranges["backfill_end_utc"].notna()
    ].copy()
    if "outage_class" not in ranges.columns:
        _require_columns(ranges, ["window_start_utc"], "windows")
        ranges["window_start_utc"] = pd.to_datetime(
            ranges["window_start_utc"],
            utc=True,
            errors="coerce",
        )
        ranges["outage_class"] = ranges["window_start_utc"].apply(
            _window_outage_class,
        )

    for row in ranges.itertuples(index=False):
        outage_class = getattr(row, "outage_class", None)
        if outage_class in LEGACY_NETWORK_OUTAGE_CLASSES:
            outage_class = OUTAGE_CLASS_COORDINATED
        if outage_class not in NETWORK_OUTAGE_CLASSES:
            continue
        in_window = (
            classified["start_utc"].ge(row.backfill_start_utc)
            & classified["start_utc"].le(row.backfill_end_utc)
        )
        classified.loc[in_window.fillna(False), "outage_class"] = outage_class

    return classified


def _print_summary(events: pd.DataFrame, windows: pd.DataFrame) -> None:
    print("Network outage windows complete.")
    print(f"min_stations: {NETWORK_OUTAGE_MIN_STATIONS}")
    print(f"time_window_hours: {NETWORK_OUTAGE_TIME_WINDOW_HOURS}")
    print(f"Total windows detected: {len(windows):,}")
    print("Outage class counts:")
    class_counts = (
        events["outage_class"]
        .value_counts()
        .reindex(
            [
                OUTAGE_CLASS_COORDINATED,
                OUTAGE_CLASS_LOCAL,
            ],
            fill_value=0,
        )
    )
    print(class_counts.to_string())
    print("10 largest windows by station_count:")
    largest_windows = windows.sort_values(
        ["station_count", "n_events", "window_start_utc"],
        ascending=[False, False, True],
        kind="mergesort",
    ).head(10)
    if largest_windows.empty:
        print("(none)")
    else:
        print(
            largest_windows[
                [
                    "window_id",
                    "window_start_utc",
                    "window_end_utc",
                    "station_count",
                    "station_ids",
                ]
            ].to_string(index=False)
        )

    print("Per-station network event count:")
    network_events = events.loc[
        events["outage_class"].isin(NETWORK_OUTAGE_CLASSES)
    ]
    station_counts = (
        network_events.groupby("station_id")
        .size()
        .rename("network_event_count")
        .reset_index()
        .sort_values(
            ["network_event_count", "station_id"],
            ascending=[False, True],
            kind="mergesort",
        )
    )
    if station_counts.empty:
        print("(none)")
    else:
        print(station_counts.to_string(index=False))


def main() -> None:
    events = pd.read_parquet(AVAILABILITY_EVENTS_PATH)
    windows = detect_network_outage_windows(
        events,
        min_stations=NETWORK_OUTAGE_MIN_STATIONS,
        time_window_hours=NETWORK_OUTAGE_TIME_WINDOW_HOURS,
    )
    classified_events = assign_outage_class(events, windows)

    ensure_directories()
    windows.to_csv(NETWORK_OUTAGE_WINDOWS_PATH, index=False)
    classified_events.to_parquet(AVAILABILITY_EVENTS_PATH, index=False)
    _print_summary(classified_events, windows)


if __name__ == "__main__":
    main()
