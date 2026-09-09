from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import MERGED_DATASET_PATH
from src.rules.config import (
    EXTERNAL_BASELINE_MIN_HOURS,
    EXTERNAL_BASELINE_WINDOW_HOURS,
    EXTERNAL_CACHE_DIR,
    EXTERNAL_CHANNEL_MAP,
    EXTERNAL_CHANNEL_SOURCE,
    EXTERNAL_CLEARSKY_REF_MIN,
    EXTERNAL_HOURBIN_MIN_DAYS,
    EXTERNAL_HOURBIN_WINDOW_DAYS,
    EXTERNAL_MAD_FLOORS,
    EXTERNAL_SNAPSHOT_TOLERANCE_MIN,
    EXTERNAL_SOLAR_DAYLIGHT_MIN,
    EXTERNAL_SOLAR_MEAN_MIN_SLOTS,
    EXTERNAL_TIME_LAG_HOURS,
)
from src.rules.stuck_confirmation import FIVE_MIN_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = PROJECT_ROOT / EXTERNAL_CACHE_DIR
CHANNEL_SPECS = {
    "pressure": {
        "station_column": "pressure_max_hpa",
        "reference_column": "pressure_msl",
        "baseline_kind": "scalar",
    },
    "temp": {
        "station_column": "temp_avg_c",
        "reference_column": "temperature_2m",
        "baseline_kind": "hourbin",
    },
    "dewpoint": {
        "station_column": "dewpoint_avg_c",
        "reference_column": "dew_point_2m",
        "baseline_kind": "scalar",
    },
    "wind": {
        "station_column": "windspeed_avg_kmh",
        "reference_column": "wind_speed_10m",
        "baseline_kind": "scalar",
    },
    "solar": {
        "station_column": "solar_radiation_high_wm2",
        "reference_column": "shortwave_radiation",
        "baseline_kind": "hourbin",
    },
}
HOURLY_SOURCES = {"hourly_mean", "hourly_max", "hourly_high"}
FIVE_MIN_SOURCES = {"5min_snapshot", "5min_trailing_mean"}
LAGS = list(range(-3, 4))


def _reference_path(station_id: str, reference_dir: Path = REFERENCE_DIR) -> Path:
    return Path(reference_dir) / f"{station_id}.parquet"


def _station_file(station_id: str, five_min_dir: Path) -> Path:
    return Path(five_min_dir) / f"{station_id}_complete.csv"


def _load_reference(station_id: str, reference_dir: Path = REFERENCE_DIR) -> pd.DataFrame:
    frame = pd.read_parquet(_reference_path(station_id, reference_dir))
    if "time_utc" in frame.columns:
        frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
        frame = frame.set_index("time_utc")
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame.index.name = "time_utc"
    return frame.sort_index()


def _load_merged(merged_path: Path = MERGED_DATASET_PATH) -> pd.DataFrame:
    frame = pd.read_csv(merged_path, low_memory=False)
    frame["hour_utc"] = pd.to_datetime(frame["hour_utc"], utc=True)
    return frame


def _load_five_min_station(station_id: str, five_min_dir: Path = FIVE_MIN_DIR) -> pd.DataFrame:
    path = _station_file(station_id, Path(five_min_dir))
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    frame["expected_time_utc"] = pd.to_datetime(
        frame["expected_time_utc"],
        utc=True,
        errors="coerce",
    )
    return frame.sort_values("expected_time_utc")


def _map_spec(station_column: str) -> dict[str, object]:
    if station_column not in EXTERNAL_CHANNEL_MAP:
        raise KeyError(f"missing external channel map for {station_column}")
    spec = EXTERNAL_CHANNEL_MAP[station_column]
    if "conversion" not in spec:
        raise KeyError(f"missing conversion for {station_column}")
    return spec


def _convert(values: pd.Series, station_column: str) -> pd.Series:
    conversion = _map_spec(station_column)["conversion"]
    return pd.to_numeric(values, errors="coerce").map(conversion)


def _target_frame(target_times: object) -> pd.DataFrame:
    times = pd.to_datetime(pd.Series(target_times), utc=True)
    return pd.DataFrame({"time_utc": times, "_order": np.arange(len(times))})


def nearest_snapshot_values(
    five_min: pd.DataFrame,
    target_times: object,
    value_column: str,
    tolerance_minutes: int = EXTERNAL_SNAPSHOT_TOLERANCE_MIN,
) -> pd.Series:
    if value_column not in five_min.columns:
        raise KeyError(value_column)
    if "expected_time_utc" not in five_min.columns:
        raise KeyError("expected_time_utc")

    target = _target_frame(target_times)
    if target.empty:
        return pd.Series(dtype=float)

    source = five_min.loc[:, ["expected_time_utc", value_column]].copy()
    source["expected_time_utc"] = pd.to_datetime(
        source["expected_time_utc"],
        utc=True,
        errors="coerce",
    )
    source["value"] = pd.to_numeric(source[value_column], errors="coerce")
    source = source.dropna(subset=["expected_time_utc", "value"])
    source = source.loc[:, ["expected_time_utc", "value"]].sort_values(
        "expected_time_utc"
    )

    if source.empty:
        return pd.Series(np.nan, index=range(len(target)), dtype=float)

    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    target_sorted = target.sort_values("time_utc")
    backward = pd.merge_asof(
        target_sorted,
        source,
        left_on="time_utc",
        right_on="expected_time_utc",
        direction="backward",
        tolerance=tolerance,
    )
    forward = pd.merge_asof(
        target_sorted,
        source,
        left_on="time_utc",
        right_on="expected_time_utc",
        direction="forward",
        tolerance=tolerance,
    )
    backward_gap = (backward["time_utc"] - backward["expected_time_utc"]).abs()
    forward_gap = (forward["expected_time_utc"] - forward["time_utc"]).abs()
    backward_value = backward["value"]
    forward_value = forward["value"]
    use_backward = backward_value.notna() & (
        forward_value.isna() | backward_gap.le(forward_gap)
    )
    chosen = backward_value.where(use_backward, forward_value)
    result = pd.Series(chosen.to_numpy(dtype=float), index=target_sorted["_order"])
    return result.sort_index().reset_index(drop=True)


def trailing_mean_values(
    five_min: pd.DataFrame,
    target_times: object,
    value_column: str,
    min_slots: int = EXTERNAL_SOLAR_MEAN_MIN_SLOTS,
) -> pd.Series:
    if value_column not in five_min.columns:
        raise KeyError(value_column)
    if "expected_time_utc" not in five_min.columns:
        raise KeyError("expected_time_utc")

    target = _target_frame(target_times)
    if target.empty:
        return pd.Series(dtype=float)

    times = pd.to_datetime(five_min["expected_time_utc"], utc=True, errors="coerce")
    values = pd.to_numeric(five_min[value_column], errors="coerce")
    source = pd.Series(values.to_numpy(dtype=float), index=times)
    source = source.loc[source.index.notna()].sort_index()
    if source.empty:
        return pd.Series(np.nan, index=range(len(target)), dtype=float)
    source = source.groupby(level=0).mean()
    means = source.rolling("60min", closed="right").mean()
    counts = source.rolling("60min", closed="right").count()
    target_index = pd.DatetimeIndex(target["time_utc"])
    result = means.reindex(target_index)
    valid = counts.reindex(target_index).ge(min_slots)
    result = result.where(valid)
    return pd.Series(result.to_numpy(dtype=float), index=range(len(target)))


def _hourly_source_series(
    station: pd.DataFrame,
    target_times: object,
    station_column: str,
) -> pd.Series:
    target = _target_frame(target_times)
    if station_column not in station.columns:
        raise KeyError(station_column)
    values = _convert(station[station_column], station_column)
    index = pd.to_datetime(station["hour_utc"], utc=True)
    lookup = pd.Series(values.to_numpy(dtype=float), index=index)
    result = lookup.reindex(pd.DatetimeIndex(target["time_utc"]))
    return pd.Series(result.to_numpy(dtype=float), index=range(len(target)))


def _source_series(
    short: str,
    station: pd.DataFrame,
    target_times: object,
    station_id: str,
    five_min_dir: Path,
    five_min: pd.DataFrame | None = None,
) -> pd.Series:
    if short not in CHANNEL_SPECS:
        raise KeyError(short)
    if short not in EXTERNAL_CHANNEL_SOURCE:
        raise KeyError(f"missing external channel source for {short}")

    spec = CHANNEL_SPECS[short]
    station_column = str(spec["station_column"])
    source = EXTERNAL_CHANNEL_SOURCE[short]

    if source in HOURLY_SOURCES:
        return _hourly_source_series(station, target_times, station_column)

    if source not in FIVE_MIN_SOURCES:
        raise ValueError(f"unsupported external channel source {source}")

    source_frame = five_min
    if source_frame is None:
        source_frame = _load_five_min_station(station_id, five_min_dir)

    if source == "5min_snapshot":
        values = nearest_snapshot_values(
            source_frame,
            target_times,
            station_column,
            EXTERNAL_SNAPSHOT_TOLERANCE_MIN,
        )
    else:
        values = trailing_mean_values(
            source_frame,
            target_times,
            station_column,
            EXTERNAL_SOLAR_MEAN_MIN_SLOTS,
        )

    return _convert(values, station_column)


def _station_values(
    station: pd.DataFrame,
    station_id: str,
    target_times: object,
    five_min_dir: Path,
) -> pd.DataFrame:
    values = _target_frame(target_times).drop(columns=["_order"])
    five_min = None
    if any(EXTERNAL_CHANNEL_SOURCE[short] in FIVE_MIN_SOURCES for short in CHANNEL_SPECS):
        five_min = _load_five_min_station(station_id, five_min_dir)

    for short in CHANNEL_SPECS:
        values[f"station_{short}"] = _source_series(
            short,
            station,
            values["time_utc"],
            station_id,
            five_min_dir,
            five_min,
        )
    return values


def _reference_values(reference: pd.DataFrame, lag_hours: int) -> pd.DataFrame:
    grid = pd.DataFrame({"time_utc": reference.index})
    grid["_ref_time"] = grid["time_utc"] + pd.Timedelta(hours=lag_hours)
    ref = reference.reset_index()
    for short, spec in CHANNEL_SPECS.items():
        ref = ref.rename(columns={str(spec["reference_column"]): f"ref_{short}"})
    ref_columns = ["time_utc"] + [f"ref_{short}" for short in CHANNEL_SPECS]
    aligned = grid.merge(
        ref.loc[:, ref_columns],
        left_on="_ref_time",
        right_on="time_utc",
        how="left",
        suffixes=("", "_reference"),
    )
    return aligned.drop(columns=["_ref_time", "time_utc_reference"])


def load_joined(
    station_id: str,
    merged_path: Path = MERGED_DATASET_PATH,
    reference_dir: Path = REFERENCE_DIR,
    lag_hours: int = EXTERNAL_TIME_LAG_HOURS,
    five_min_dir: Path = FIVE_MIN_DIR,
) -> pd.DataFrame:
    merged = _load_merged(merged_path)
    station = merged.loc[merged["station_id"].eq(station_id)].copy()
    if station.duplicated(["station_id", "hour_utc"]).any():
        raise ValueError(f"duplicate station-hour keys for {station_id}")
    reference = _load_reference(station_id, reference_dir)
    joined = _reference_values(reference, lag_hours)
    station_values = _station_values(
        station,
        station_id,
        joined["time_utc"],
        Path(five_min_dir),
    )
    joined = joined.merge(station_values, on="time_utc", how="left")
    joined["station_id"] = station_id
    columns = ["station_id", "time_utc"]
    columns.extend(f"station_{short}" for short in CHANNEL_SPECS)
    columns.extend(f"ref_{short}" for short in CHANNEL_SPECS)
    return joined.loc[:, columns]


def _rolling_mad(values: np.ndarray) -> float:
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return np.nan
    median = float(np.median(clean))
    return float(np.median(np.abs(clean - median)))


def median_mad(series: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan, np.nan
    median = float(values.median())
    mad = float(np.median(np.abs(values.to_numpy(dtype=float) - median)))
    return median, mad


def rolling_median_mad(
    series: pd.Series,
    window: int,
    min_periods: int,
) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(series, errors="coerce")
    median = values.rolling(window=window, min_periods=min_periods).median()
    mad = values.rolling(window=window, min_periods=min_periods).apply(
        _rolling_mad,
        raw=True,
    )
    return median, mad


def hourbin_median_mad(
    frame: pd.DataFrame,
    residual_column: str,
    window_days: int,
    min_days: int,
) -> tuple[pd.Series, pd.Series]:
    base = pd.Series(np.nan, index=frame.index, dtype=float)
    mad = pd.Series(np.nan, index=frame.index, dtype=float)
    hours = pd.to_datetime(frame["time_utc"], utc=True).dt.hour
    for _, index in frame.groupby(hours, sort=False).groups.items():
        series = frame.loc[index, residual_column]
        group_base, group_mad = rolling_median_mad(series, window_days, min_days)
        base.loc[index] = group_base
        mad.loc[index] = group_mad
    return base, mad


def _add_residuals(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for short in CHANNEL_SPECS:
        result[f"r_{short}"] = result[f"station_{short}"] - result[f"ref_{short}"]
    solar_mask = result["ref_solar"].le(EXTERNAL_SOLAR_DAYLIGHT_MIN)
    result.loc[solar_mask, "r_solar"] = np.nan
    clear_mask = result["ref_solar"].ge(EXTERNAL_CLEARSKY_REF_MIN)
    result["clear_sky_ratio"] = np.where(
        clear_mask,
        result["station_solar"] / result["ref_solar"],
        np.nan,
    )
    return result


def _add_baselines(
    frame: pd.DataFrame,
    scalar_window_hours: int,
    scalar_min_hours: int,
    hourbin_window_days: int,
    hourbin_min_days: int,
) -> pd.DataFrame:
    result = frame.copy()
    for short, spec in CHANNEL_SPECS.items():
        residual_column = f"r_{short}"
        if spec["baseline_kind"] == "hourbin":
            base, mad = hourbin_median_mad(
                result,
                residual_column,
                hourbin_window_days,
                hourbin_min_days,
            )
        else:
            base, mad = rolling_median_mad(
                result[residual_column],
                scalar_window_hours,
                scalar_min_hours,
            )
        result[f"base_{short}"] = base
        result[f"bmad_{short}"] = mad
    return result


def _add_zscores(
    frame: pd.DataFrame,
    mad_floors: dict[str, float],
) -> pd.DataFrame:
    result = frame.copy()
    for short in CHANNEL_SPECS:
        spread = result[f"bmad_{short}"].clip(lower=mad_floors[short])
        result[f"z_{short}"] = (
            result[f"r_{short}"] - result[f"base_{short}"]
        ) / spread
    result.loc[result["ref_solar"].le(EXTERNAL_SOLAR_DAYLIGHT_MIN), "z_solar"] = np.nan
    return result


def compute_external_residuals(
    joined: pd.DataFrame,
    scalar_window_hours: int = EXTERNAL_BASELINE_WINDOW_HOURS,
    scalar_min_hours: int = EXTERNAL_BASELINE_MIN_HOURS,
    hourbin_window_days: int = EXTERNAL_HOURBIN_WINDOW_DAYS,
    hourbin_min_days: int = EXTERNAL_HOURBIN_MIN_DAYS,
    mad_floors: dict[str, float] = EXTERNAL_MAD_FLOORS,
) -> pd.DataFrame:
    result = joined.sort_values("time_utc").reset_index(drop=True)
    result = _add_residuals(result)
    result = _add_baselines(
        result,
        scalar_window_hours,
        scalar_min_hours,
        hourbin_window_days,
        hourbin_min_days,
    )
    result = _add_zscores(result, mad_floors)
    return result


def build_station(
    station_id: str,
    merged_path: Path = MERGED_DATASET_PATH,
    reference_dir: Path = REFERENCE_DIR,
    five_min_dir: Path = FIVE_MIN_DIR,
) -> pd.DataFrame:
    return compute_external_residuals(
        load_joined(station_id, merged_path, reference_dir, EXTERNAL_TIME_LAG_HOURS, five_min_dir),
    )


def build_all(
    merged_path: Path = MERGED_DATASET_PATH,
    reference_dir: Path = REFERENCE_DIR,
    five_min_dir: Path = FIVE_MIN_DIR,
) -> pd.DataFrame:
    reference_paths = sorted(Path(reference_dir).glob("*.parquet"))
    frames = [
        build_station(path.stem, merged_path, reference_dir, five_min_dir)
        for path in reference_paths
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def source_agreement(
    merged_path: Path = MERGED_DATASET_PATH,
    five_min_dir: Path = FIVE_MIN_DIR,
) -> pd.DataFrame:
    merged = _load_merged(merged_path)
    rows = []
    switched = [
        short
        for short in CHANNEL_SPECS
        if EXTERNAL_CHANNEL_SOURCE.get(short) in FIVE_MIN_SOURCES
    ]
    totals = {
        short: {
            "source_non_nan": 0,
            "hourly_non_nan": 0,
            "overlap_count": 0,
            "differences": [],
        }
        for short in switched
    }

    for station_id, station in merged.groupby("station_id", sort=True):
        five_min = _load_five_min_station(str(station_id), five_min_dir)
        target_times = station["hour_utc"]
        for short in switched:
            station_column = str(CHANNEL_SPECS[short]["station_column"])
            source_values = _source_series(
                short,
                station,
                target_times,
                str(station_id),
                five_min_dir,
                five_min,
            )
            hourly_values = _hourly_source_series(station, target_times, station_column)
            overlap = source_values.notna() & hourly_values.notna()
            totals[short]["source_non_nan"] += int(source_values.notna().sum())
            totals[short]["hourly_non_nan"] += int(hourly_values.notna().sum())
            totals[short]["overlap_count"] += int(overlap.sum())
            totals[short]["differences"].extend(
                (source_values.loc[overlap] - hourly_values.loc[overlap])
                .abs()
                .tolist()
            )

    for short in switched:
        differences = pd.Series(totals[short]["differences"], dtype=float)
        rows.append(
            {
                "channel": short,
                "source": EXTERNAL_CHANNEL_SOURCE[short],
                "source_non_nan": totals[short]["source_non_nan"],
                "hourly_non_nan": totals[short]["hourly_non_nan"],
                "overlap_count": totals[short]["overlap_count"],
                "median_abs_difference": differences.median()
                if not differences.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _corr_at_lag(
    station_hours: pd.Series,
    station_values: pd.Series,
    reference: pd.DataFrame,
    reference_column: str,
    lag_hours: int,
) -> float:
    hours = pd.Series(station_hours).reset_index(drop=True)
    values = pd.Series(station_values).reset_index(drop=True)
    left = pd.DataFrame(
        {
            "hour_utc": pd.to_datetime(hours, utc=True),
            "station_value": pd.to_numeric(values, errors="coerce"),
        }
    )
    right = reference.reset_index().loc[:, ["time_utc", reference_column]].copy()
    left["_ref_time"] = left["hour_utc"] + pd.Timedelta(hours=lag_hours)
    aligned = left.merge(right, left_on="_ref_time", right_on="time_utc", how="left")
    values = aligned.loc[
        aligned[["station_value", reference_column]].notna().all(axis=1),
        ["station_value", reference_column],
    ]
    if len(values) < 3:
        return np.nan
    if values["station_value"].nunique() < 2 or values[reference_column].nunique() < 2:
        return np.nan
    return float(values["station_value"].corr(values[reference_column]))


def temp_snapshot_lag_scan(
    merged_path: Path = MERGED_DATASET_PATH,
    reference_dir: Path = REFERENCE_DIR,
    five_min_dir: Path = FIVE_MIN_DIR,
    lags: list[int] = LAGS,
) -> pd.DataFrame:
    merged = _load_merged(merged_path)
    rows = []
    for path in sorted(Path(reference_dir).glob("*.parquet")):
        station_id = path.stem
        station = merged.loc[merged["station_id"].eq(station_id)].copy()
        reference = _load_reference(station_id, reference_dir)
        five_min = _load_five_min_station(station_id, five_min_dir)
        values = _source_series(
            "temp",
            station,
            station["hour_utc"],
            station_id,
            five_min_dir,
            five_min,
        )
        correlations = [
            (
                lag,
                _corr_at_lag(
                    station["hour_utc"],
                    values,
                    reference,
                    "temperature_2m",
                    lag,
                ),
            )
            for lag in lags
        ]
        finite = [(lag, corr) for lag, corr in correlations if pd.notna(corr)]
        if finite:
            best_lag, best_corr = max(finite, key=lambda item: item[1])
        else:
            best_lag, best_corr = None, np.nan
        rows.append(
            {
                "station_id": station_id,
                "k_temp_snapshot": best_lag,
                "r_temp_snapshot_best": best_corr,
            }
        )
    return pd.DataFrame(rows)


def lag_modal_summary(scan: pd.DataFrame) -> dict[str, object]:
    counts = scan["k_temp_snapshot"].dropna().astype(int).value_counts()
    if counts.empty:
        return {"modal_k": None, "count": 0, "share": 0.0}
    modal = int(counts.index[0])
    count = int(counts.iloc[0])
    return {
        "modal_k": modal,
        "count": count,
        "share": float(count / len(scan)) if len(scan) else 0.0,
    }
