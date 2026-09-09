from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


FIVE_MIN_DIR = Path(
    os.environ.get(
        "MOZN_FIVE_MIN_DIR",
        Path.home() / "Desktop" / "Mozn Weather Dataset" / "per_station_weather_data",
    )
)
MIN_5MIN_OBS = 24
CONSTANCY_THRESHOLD = 0.99
VALUE_PRECISION = 2
LOWVAR_MODAL_FRACTION_FLOOR = 0.80
RANGING_SPREAD_THRESHOLD = 5.0
RANGING_DISTINCT_VALUE_THRESHOLD = 10
STUCK_CHANNEL_MAP = {
    "solar_radiation_high_wm2": "solar_radiation_high_wm2",
    "uv_high": "uv_high",
    "windspeed_avg_kmh": "windspeed_avg_kmh",
    "windspeed_high_kmh": "windspeed_high_kmh",
    "windspeed_low_kmh": "windspeed_low_kmh",
    "windgust_avg_kmh": "windgust_avg_kmh",
    "windgust_high_kmh": "windgust_high_kmh",
    "windgust_low_kmh": "windgust_low_kmh",
    "winddir_avg_deg": "winddir_avg_deg",
    "winddir_sin": "winddir_avg_deg",
    "winddir_cos": "winddir_avg_deg",
    "temp_high_c": "temp_high_c",
    "temp_low_c": "temp_low_c",
    "temp_avg_c": "temp_avg_c",
    "humidity_high_pct": "humidity_high_pct",
    "humidity_low_pct": "humidity_low_pct",
    "humidity_avg_pct": "humidity_avg_pct",
    "pressure_max_hpa": "pressure_max_hpa",
    "pressure_min_hpa": "pressure_min_hpa",
    "pressure_trend_hpa": "pressure_trend_hpa",
    "precip_rate_mmh": "precip_rate_mmh",
    "precip_total_mm": "precip_total_mm",
}
STUCK_REASON = "stuck_variance_zero"
STUCK_CHANNEL_PRIORITY = {
    "windspeed_avg_kmh": 0,
    "windgust_avg_kmh": 1,
    "windspeed_high_kmh": 2,
    "windgust_high_kmh": 3,
    "windspeed_low_kmh": 4,
    "windgust_low_kmh": 5,
    "temp_avg_c": 10,
    "temp_high_c": 11,
    "temp_low_c": 12,
    "humidity_avg_pct": 20,
    "humidity_high_pct": 21,
    "humidity_low_pct": 22,
    "pressure_max_hpa": 30,
    "pressure_min_hpa": 31,
    "pressure_trend_hpa": 32,
    "precip_rate_mmh": 40,
    "precip_total_mm": 41,
    "winddir_sin": 50,
    "winddir_cos": 51,
    "winddir_avg_deg": 52,
}
OUTPUT_COLUMNS = [
    "station_id",
    "channel",
    "start_hour",
    "end_hour",
    "duration_hours",
    "five_min_column",
    "n_5min_expected",
    "n_5min_present",
    "present_fraction",
    "n_distinct_values",
    "modal_value",
    "min_reading",
    "max_reading",
    "modal_fraction",
    "longest_constant_run_hours",
    "confirmed_stuck_5min",
    "status",
]


def _station_file(station_id: str, five_min_dir: Path) -> Path:
    return Path(five_min_dir) / f"{station_id}_complete.csv"


def _row_dict(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        return row.to_dict()

    return dict(row)


def _channel_from_row(row: dict[str, Any]) -> str:
    if "channel" in row and pd.notna(row["channel"]):
        return str(row["channel"])

    channels = [
        token
        for token in str(row.get("affected_channels", "")).split("|")
        if token
    ]
    if len(channels) == 1:
        return channels[0]

    return str(row.get("affected_channels", ""))


def _duration_hours(row: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> int:
    value = row.get("duration_hours")
    if pd.notna(value):
        return int(round(float(value)))

    return int((end - start) / pd.Timedelta(hours=1)) + 1


def _empty_result(
    row: dict[str, Any],
    channel: str,
    five_min_column: object,
    status: str,
) -> dict[str, object]:
    duration_hours = int(row.get("duration_hours", 0) or 0)
    return {
        "station_id": row.get("station_id"),
        "channel": channel,
        "start_hour": row.get("start_hour"),
        "end_hour": row.get("end_hour"),
        "duration_hours": duration_hours,
        "five_min_column": five_min_column,
        "n_5min_expected": int(round(duration_hours * 12)),
        "n_5min_present": 0,
        "present_fraction": 0.0,
        "n_distinct_values": 0,
        "modal_value": pd.NA,
        "min_reading": pd.NA,
        "max_reading": pd.NA,
        "modal_fraction": 0.0,
        "longest_constant_run_hours": 0.0,
        "confirmed_stuck_5min": False,
        "status": status,
    }


def load_five_min_window(
    station_id: str,
    start: object,
    end: object,
    five_min_dir: Path = FIVE_MIN_DIR,
) -> pd.DataFrame:
    path = _station_file(station_id, Path(five_min_dir))
    if not path.exists():
        return pd.DataFrame()

    frame = pd.read_csv(path)
    frame["expected_time_utc"] = pd.to_datetime(
        frame["expected_time_utc"],
        utc=True,
    )
    start_time = pd.to_datetime(start, utc=True)
    end_time = pd.to_datetime(end, utc=True)
    return frame.loc[
        frame["expected_time_utc"].between(start_time, end_time, inclusive="both")
    ].copy()


def _longest_constant_run_hours(values: pd.Series) -> float:
    longest = 0
    current = 0
    previous: object = None

    for value in values:
        if pd.isna(value):
            current = 0
            previous = None
            continue

        if previous is not None and value == previous:
            current += 1
        else:
            current = 1
            previous = value

        longest = max(longest, current)

    return float(longest * 5 / 60)


def confirm_episode(
    row: pd.Series | dict[str, Any],
    five_min_dir: Path = FIVE_MIN_DIR,
    min_obs: int = MIN_5MIN_OBS,
    constancy_threshold: float = CONSTANCY_THRESHOLD,
    precision: int = VALUE_PRECISION,
) -> dict[str, object]:
    data = _row_dict(row)
    channel = _channel_from_row(data)
    five_min_column = STUCK_CHANNEL_MAP.get(channel)

    if five_min_column is None:
        return _empty_result(data, channel, pd.NA, "unmapped_channel")

    start = pd.to_datetime(data["start_hour"], utc=True)
    end = pd.to_datetime(data["end_hour"], utc=True)
    duration_hours = _duration_hours(data, start, end)
    window_end = start + pd.Timedelta(hours=duration_hours) - pd.Timedelta(minutes=5)
    n_expected = int(round(duration_hours * 12))
    station_id = str(data["station_id"])
    path_exists = _station_file(station_id, Path(five_min_dir)).exists()
    window = load_five_min_window(station_id, start, window_end, Path(five_min_dir))

    if path_exists and five_min_column not in window.columns:
        return _empty_result(data, channel, five_min_column, "unmapped_channel")

    values = (
        window[five_min_column].round(precision)
        if five_min_column in window.columns
        else pd.Series(dtype="float64")
    )
    present = values.dropna()
    n_present = int(len(present))
    present_fraction = float(n_present / n_expected) if n_expected else 0.0
    n_distinct = int(present.nunique()) if n_present else 0
    modal_value: object = pd.NA
    min_reading: object = pd.NA
    max_reading: object = pd.NA
    modal_fraction = 0.0

    if n_present:
        counts = present.value_counts(sort=True)
        modal_value = counts.index[0]
        min_reading = present.min()
        max_reading = present.max()
        modal_fraction = float(counts.iloc[0] / n_present)

    longest_run = _longest_constant_run_hours(values)
    confirmed = bool(n_present >= min_obs and modal_fraction >= constancy_threshold)

    if n_present < min_obs:
        status = "insufficient_5min_data"
    elif confirmed:
        status = "confirmed_stuck"
    else:
        status = "not_constant"

    return {
        "station_id": station_id,
        "channel": channel,
        "start_hour": data["start_hour"],
        "end_hour": data["end_hour"],
        "duration_hours": duration_hours,
        "five_min_column": five_min_column,
        "n_5min_expected": n_expected,
        "n_5min_present": n_present,
        "present_fraction": present_fraction,
        "n_distinct_values": n_distinct,
        "modal_value": modal_value,
        "min_reading": min_reading,
        "max_reading": max_reading,
        "modal_fraction": modal_fraction,
        "longest_constant_run_hours": longest_run,
        "confirmed_stuck_5min": confirmed,
        "status": status,
    }


def confirm_stuck_episodes(
    episodes_df: pd.DataFrame,
    five_min_dir: Path = FIVE_MIN_DIR,
    min_obs: int = MIN_5MIN_OBS,
    constancy_threshold: float = CONSTANCY_THRESHOLD,
    precision: int = VALUE_PRECISION,
) -> pd.DataFrame:
    if episodes_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows = [
        confirm_episode(
            row,
            five_min_dir,
            min_obs,
            constancy_threshold,
            precision,
        )
        for _, row in episodes_df.iterrows()
    ]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def categorize_stuck_5min(
    row: pd.Series | dict[str, Any],
    flatline_threshold: float = CONSTANCY_THRESHOLD,
    lowvar_floor: float = LOWVAR_MODAL_FRACTION_FLOOR,
    spread_threshold: float = RANGING_SPREAD_THRESHOLD,
    distinct_threshold: int = RANGING_DISTINCT_VALUE_THRESHOLD,
) -> str:
    data = _row_dict(row)
    modal_fraction = float(data.get("modal_fraction", 0.0) or 0.0)
    n_distinct = int(data.get("n_distinct_values", 0) or 0)
    min_reading = data.get("min_reading")
    max_reading = data.get("max_reading")
    spread = 0.0

    if pd.notna(min_reading) and pd.notna(max_reading):
        spread = float(max_reading) - float(min_reading)

    if modal_fraction >= flatline_threshold:
        return "confirmed_flatline"

    if modal_fraction >= lowvar_floor:
        return "borderline_lowvar"

    if spread > spread_threshold or n_distinct > distinct_threshold:
        return "ranging_not_stuck"

    return "ambiguous_lowmodal"


def _reason_tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()

    return {token for token in str(value).split("|") if token}


def _fallback_channel(row: pd.Series) -> str:
    channels = [
        token
        for token in str(row.get("affected_channels", "")).split("|")
        if token
    ]
    if not channels:
        return str(row.get("channel", ""))

    return min(
        channels,
        key=lambda channel: STUCK_CHANNEL_PRIORITY.get(channel, 999),
    )


def _matching_stuck_events(row: pd.Series, events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    start = pd.to_datetime(row["start_hour"], utc=True)
    end = pd.to_datetime(row["end_hour"], utc=True)
    events = events_df.copy()
    events["start_hour"] = pd.to_datetime(events["start_hour"], utc=True)
    events["end_hour"] = pd.to_datetime(events["end_hour"], utc=True)
    reason_has_stuck = events["reasons"].apply(
        lambda value: STUCK_REASON in _reason_tokens(value),
    )
    return events.loc[
        events["station_id"].eq(row["station_id"])
        & reason_has_stuck
        & events["start_hour"].le(end)
        & events["end_hour"].ge(start)
    ].copy()


def _dominant_stuck_channel_details(
    episode_row: pd.Series | dict[str, Any],
    events_df: pd.DataFrame | None = None,
) -> tuple[str, str]:
    row = episode_row if isinstance(episode_row, pd.Series) else pd.Series(episode_row)

    if (
        "channel" in row
        and pd.notna(row["channel"])
        and (
            row.get("dominant_detector") == "stuck"
            or STUCK_REASON in _reason_tokens(row.get("reasons"))
        )
    ):
        return str(row["channel"]), "row_channel"

    events = events_df if events_df is not None else pd.DataFrame()
    matching_events = _matching_stuck_events(row, events)

    if matching_events.empty:
        return _fallback_channel(row), "fallback_affected_channels"

    matching_events["channel_priority"] = matching_events["channel"].map(
        STUCK_CHANNEL_PRIORITY,
    ).fillna(999)
    matching_events["_rolling_sort"] = matching_events["min_rolling_variance"].fillna(
        float("inf"),
    )
    matching_events = matching_events.sort_values(
        [
            "_rolling_sort",
            "duration_hours",
            "channel_priority",
            "channel",
        ],
        ascending=[True, False, True, True],
    )
    return str(matching_events.iloc[0]["channel"]), "stuck_event"


def dominant_stuck_channel(
    episode_row: pd.Series | dict[str, Any],
    events_df: pd.DataFrame | None = None,
) -> str:
    return _dominant_stuck_channel_details(episode_row, events_df)[0]


def select_dominant_stuck_channels(
    episodes_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    events = events_df if events_df is not None else pd.DataFrame()

    for _, row in episodes_df.iterrows():
        data = row.to_dict()
        channel, resolution = _dominant_stuck_channel_details(row, events)
        data["channel"] = channel
        data["channel_resolution"] = resolution
        rows.append(data)

    return pd.DataFrame(rows)


def expand_stuck_channels(episodes_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, row in episodes_df.iterrows():
        data = row.to_dict()
        channels = [
            token
            for token in str(data.get("affected_channels", "")).split("|")
            if token
        ]

        for channel in channels:
            expanded = data.copy()
            expanded["channel"] = channel
            expanded["channel_resolution"] = "all_channels"
            rows.append(expanded)

    return pd.DataFrame(rows)


def confirm_stuck_episodes_dominant(
    episodes_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    five_min_dir: Path = FIVE_MIN_DIR,
    min_obs: int = MIN_5MIN_OBS,
    constancy_threshold: float = CONSTANCY_THRESHOLD,
    precision: int = VALUE_PRECISION,
) -> pd.DataFrame:
    dominant = select_dominant_stuck_channels(episodes_df, events_df)
    result = confirm_stuck_episodes(
        dominant,
        five_min_dir,
        min_obs,
        constancy_threshold,
        precision,
    )

    if "channel_resolution" in dominant.columns and not result.empty:
        result["channel_resolution"] = dominant["channel_resolution"].to_numpy()

    return result
