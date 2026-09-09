from __future__ import annotations

import numpy as np
import pandas as pd

from src.rules.baselines import select_baseline
from src.rules.channel_handlers import encode_wind_direction, log_transform_precip
from src.rules.config import (
    CHANNELS_EXCLUDED_FROM_STATISTICAL_LAYER,
    CHANNELS_REQUIRING_CIRCULAR_TRANSFORM,
    CHANNELS_REQUIRING_LOG_TRANSFORM,
    COVERAGE_FLOOR_HOURS,
    ROBUST_ZSCORE_FLAG_PERCENTILE,
    STUCK_IGNORE_ZERO_CHANNELS,
    STUCK_SKIP_CHANNELS,
)
from src.rules.detectors.isolation_forest import fit_isolation_forest, score_isolation_forest
from src.rules.detectors.robust_zscore import score_robust_zscore
from src.rules.detectors.rolling_variance import detect_stuck_values
from src.rules.physical_limits import (
    REASON_PHYSICAL_LIMIT,
    REASON_PHYSICAL_SUSPECT,
    physical_limit_flags,
    physical_suspect_flags,
)


OUTPUT_COLUMNS = [
    "station_id",
    "hour_utc",
    "channel",
    "baseline_source",
    "zscore",
    "rolling_variance",
    "iforest_score",
    "flag_zscore",
    "flag_stuck",
    "flag_iforest",
    "flag_physical",
    "flag_physical_suspect",
    "flag",
    "reason",
]


def _wind_base_name(channel: str) -> str:
    if channel.endswith("_avg_deg"):
        return channel[: -len("_avg_deg")]
    return channel


def _active_channels(channels: list[str]) -> list[str]:
    return [
        channel
        for channel in channels
        if channel not in CHANNELS_EXCLUDED_FROM_STATISTICAL_LAYER
    ]


def _build_scored_wide_frame(df: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    scored = df[["station_id", "hour_utc"]].copy()

    for channel in _active_channels(channels):
        if channel in CHANNELS_REQUIRING_CIRCULAR_TRANSFORM:
            encoded = encode_wind_direction(df[channel])
            base = _wind_base_name(channel)
            scored[f"{base}_sin"] = encoded["sin"]
            scored[f"{base}_cos"] = encoded["cos"]
        elif channel in CHANNELS_REQUIRING_LOG_TRANSFORM:
            scored[channel] = log_transform_precip(df[channel])
        else:
            scored[channel] = df[channel]

    return scored


def _scored_channel_specs(channels: list[str]) -> list[tuple[str, str]]:
    scored_channels: list[tuple[str, str]] = []

    for channel in _active_channels(channels):
        if channel in CHANNELS_REQUIRING_CIRCULAR_TRANSFORM:
            base = _wind_base_name(channel)
            scored_channels.extend(
                [
                    (channel, f"{base}_sin"),
                    (channel, f"{base}_cos"),
                ]
            )
        else:
            scored_channels.append((channel, channel))

    return scored_channels


def _threshold_flags(
    values: pd.Series,
    percentile: float,
    absolute: bool,
    frozen_threshold: float | None = None,
) -> pd.Series:
    score_values = values.abs() if absolute else values
    present_scores = score_values.dropna()

    if frozen_threshold is not None:
        return score_values.ge(frozen_threshold).fillna(False)

    if present_scores.empty:
        return pd.Series(False, index=values.index)

    threshold = float(np.nanpercentile(present_scores.to_numpy(dtype=float), percentile))
    return score_values.ge(threshold).fillna(False)


def _reason(row: pd.Series) -> str:
    reasons: list[str] = []

    if bool(row["flag_zscore"]):
        reasons.append("mad_high")
    if bool(row["flag_stuck"]):
        reasons.append("stuck_variance_zero")
    if bool(row["flag_iforest"]):
        reasons.append("iforest_outlier")
    if bool(row["flag_physical"]):
        reasons.append(REASON_PHYSICAL_LIMIT)
    if bool(row["flag_physical_suspect"]):
        reasons.append(REASON_PHYSICAL_SUSPECT)

    return "|".join(reasons)


def compute_anomaly_scores(
    df: pd.DataFrame,
    channels: list[str],
    flag_percentile: float = ROBUST_ZSCORE_FLAG_PERCENTILE,
    random_state: int = 42,
    frozen_baselines: dict[tuple[str, str], dict[str, object]] | None = None,
    frozen_isolation_forests: dict[tuple[str, str], object] | None = None,
    frozen_thresholds: dict[tuple[str, str], float] | None = None,
    collect_statistics: dict[str, dict] | None = None,
) -> pd.DataFrame:
    scored_wide = _build_scored_wide_frame(df, channels)
    scored_channels = _scored_channel_specs(channels)
    result_frames: list[pd.DataFrame] = []
    collected_baselines: dict[tuple[str, str], dict[str, object]] = {}
    collected_isolation_forests: dict[tuple[str, str], object] = {}

    for original_channel, channel in scored_channels:
        for station_id, station_frame in scored_wide.groupby("station_id", sort=False):
            station_frame = station_frame.sort_values("hour_utc")
            series = station_frame[channel].astype(float)
            frozen_baseline = (
                frozen_baselines.get((str(station_id), channel))
                if frozen_baselines is not None
                else None
            )
            baseline = select_baseline(
                scored_wide,
                station_id=str(station_id),
                channel=channel,
                min_present_hours=COVERAGE_FLOOR_HOURS,
                frozen_baseline=frozen_baseline,
            )
            if collect_statistics is not None:
                collected_baselines[(str(station_id), channel)] = baseline
            zscore = score_robust_zscore(series, baseline=baseline)["score"]
            if original_channel in STUCK_SKIP_CHANNELS:
                stuck = pd.DataFrame(
                    {
                        "rolling_variance": pd.Series(np.nan, index=series.index),
                        "flag": pd.Series(False, index=series.index),
                    },
                    index=series.index,
                )
            else:
                stuck = detect_stuck_values(
                    series,
                    ignore_zero=original_channel in STUCK_IGNORE_ZERO_CHANNELS,
                )
            physical = physical_limit_flags(
                df.loc[station_frame.index, original_channel],
                original_channel,
            )
            suspect = physical_suspect_flags(
                df.loc[station_frame.index, original_channel],
                original_channel,
            )
            fitted_estimator = (
                frozen_isolation_forests.get((str(station_id), channel))
                if frozen_isolation_forests is not None
                else None
            )
            iforest_score = score_isolation_forest(
                series,
                random_state=random_state,
                fitted_estimator=fitted_estimator,
            )
            if collect_statistics is not None:
                collected_isolation_forests[(str(station_id), channel)] = (
                    fitted_estimator
                    if fitted_estimator is not None
                    else fit_isolation_forest(series, random_state=random_state)
                )

            result_frames.append(
                pd.DataFrame(
                    {
                        "station_id": station_frame["station_id"].to_numpy(),
                        "hour_utc": station_frame["hour_utc"].to_numpy(),
                        "channel": channel,
                        "baseline_source": str(baseline["source"]),
                        "zscore": zscore.to_numpy(),
                        "rolling_variance": stuck["rolling_variance"].to_numpy(),
                        "iforest_score": iforest_score.to_numpy(),
                        "flag_stuck": stuck["flag"].to_numpy(dtype=bool),
                        "flag_physical": physical.to_numpy(dtype=bool),
                        "flag_physical_suspect": suspect.to_numpy(dtype=bool),
                        "_present": series.notna().to_numpy(),
                    },
                )
            )

    if not result_frames:
        if collect_statistics is not None:
            collect_statistics["baselines"] = collected_baselines
            collect_statistics["isolation_forests"] = collected_isolation_forests
            collect_statistics["thresholds"] = {}
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    result = pd.concat(result_frames, ignore_index=True)
    result["flag_zscore"] = False
    result["flag_iforest"] = False
    collected_thresholds: dict[tuple[str, str], float] = {}

    for channel, channel_index in result.groupby("channel", sort=False).groups.items():
        channel_rows = result.loc[channel_index]
        zscore_threshold = (
            frozen_thresholds.get((str(channel), "zscore"))
            if frozen_thresholds is not None
            else None
        )
        iforest_threshold = (
            frozen_thresholds.get((str(channel), "iforest"))
            if frozen_thresholds is not None
            else None
        )
        result.loc[channel_index, "flag_zscore"] = _threshold_flags(
            channel_rows["zscore"],
            flag_percentile,
            absolute=True,
            frozen_threshold=zscore_threshold,
        ).to_numpy(dtype=bool)
        result.loc[channel_index, "flag_iforest"] = _threshold_flags(
            channel_rows["iforest_score"],
            flag_percentile,
            absolute=False,
            frozen_threshold=iforest_threshold,
        ).to_numpy(dtype=bool)
        if collect_statistics is not None:
            present_zscore = channel_rows["zscore"].abs().dropna()
            present_iforest = channel_rows["iforest_score"].dropna()
            collected_thresholds[(str(channel), "zscore")] = (
                float(np.nanpercentile(present_zscore.to_numpy(dtype=float), flag_percentile))
                if not present_zscore.empty
                else float("nan")
            )
            collected_thresholds[(str(channel), "iforest")] = (
                float(np.nanpercentile(present_iforest.to_numpy(dtype=float), flag_percentile))
                if not present_iforest.empty
                else float("nan")
            )

    if collect_statistics is not None:
        collect_statistics["baselines"] = collected_baselines
        collect_statistics["isolation_forests"] = collected_isolation_forests
        collect_statistics["thresholds"] = collected_thresholds

    result["flag"] = (
        result["flag_zscore"].astype(bool)
        | result["flag_stuck"].astype(bool)
        | result["flag_iforest"].astype(bool)
        | result["flag_physical"].astype(bool)
    )
    result["flag_physical_suspect"] = (
        result["flag_physical_suspect"].astype(bool)
        & result["flag"].astype(bool)
    )
    result["reason"] = result.apply(_reason, axis=1)
    result = result.loc[result["_present"]].drop(columns="_present")
    return result.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)
