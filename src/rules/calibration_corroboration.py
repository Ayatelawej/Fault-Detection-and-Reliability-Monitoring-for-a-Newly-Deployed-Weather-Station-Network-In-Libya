from __future__ import annotations

import numpy as np
import pandas as pd

from src.rules.channel_handlers import sensor_group_for_channel
from src.rules.config import (
    EXTERNAL_OFFSET_CHANNELS,
    EXTERNAL_OFFSET_SCORE_HIGH,
    ROLLING_VARIANCE_FLAG_THRESHOLD,
)
from src.rules.external_offsets import detect_offset_channel


CALIBRATION_CHANNEL_TO_RAW = {
    "pressure": "pressure_max_hpa",
    "temp": "temp_avg_c",
    "dewpoint": "dewpoint_avg_c",
}
RAW_TO_CALIBRATION_CHANNEL = {
    raw_channel: channel
    for channel, raw_channel in CALIBRATION_CHANNEL_TO_RAW.items()
}
CALIBRATION_CORROBORATION_COLUMNS = [
    "corroboration_run_id",
    "station_id",
    "channel",
    "verdict",
    "sustained_offset",
    "tier",
    "start_hour",
    "end_hour",
    "duration_hours",
    "n_flagged_hours",
    "full_period_covered_hours",
    "full_period_median_residual",
    "full_period_residual_mad",
    "full_period_fraction_abs_residual_z_ge_3",
    "median_offset_level",
    "offset_stability_mad",
    "mean_offset_score",
    "peak_abs_offset_score",
    "confirmed_offset_value",
    "spatial_available_hours",
    "spatial_median_residual",
    "spatial_corroboration",
    "resolved_episode_count",
    "resolved_episode_ids",
]
BORDERLINE_EVIDENCE_COLUMNS = [
    "episode_id",
    "station_id",
    "hour_utc",
    "raw_channel",
    "calibration_channel",
    "evidence_kind",
    "era5_zscore",
]
RESOLUTION_COLUMNS = [
    "episode_id",
    "station_id",
    "hour_utc",
    "raw_channel",
    "calibration_channel",
    "evidence_kind",
    "corroboration_run_id",
    "confirmed_offset_value",
]


def _normalize_times(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], utc=True, format="mixed")
    if "station_id" in result.columns:
        result["station_id"] = result["station_id"].astype(str)
    return result


def _mad(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    median = float(clean.median())
    return float((clean - median).abs().median())


def _residual_metrics(station: pd.DataFrame, channel: str) -> dict[str, float | int]:
    residual = pd.to_numeric(station.get(f"r_{channel}"), errors="coerce")
    zscore = pd.to_numeric(station.get(f"z_{channel}"), errors="coerce")
    covered = residual.notna()
    usable_zscore = zscore.dropna()
    return {
        "full_period_covered_hours": int(covered.sum()),
        "full_period_median_residual": float(residual.median()) if covered.any() else np.nan,
        "full_period_residual_mad": _mad(residual),
        "full_period_fraction_abs_residual_z_ge_3": (
            float(usable_zscore.abs().ge(EXTERNAL_OFFSET_SCORE_HIGH).mean())
            if not usable_zscore.empty
            else np.nan
        ),
    }


def _spatial_support(
    spatial_residuals: pd.DataFrame | None,
    station_id: str,
    channel: str,
    start_hour: pd.Timestamp,
    end_hour: pd.Timestamp,
    offset_value: float,
) -> dict[str, object]:
    if spatial_residuals is None or spatial_residuals.empty:
        return {
            "spatial_available_hours": 0,
            "spatial_median_residual": np.nan,
            "spatial_corroboration": "unavailable",
        }
    column = f"r_spatial_{channel}"
    if column not in spatial_residuals.columns:
        return {
            "spatial_available_hours": 0,
            "spatial_median_residual": np.nan,
            "spatial_corroboration": "unavailable",
        }
    window = spatial_residuals.loc[
        spatial_residuals["station_id"].eq(station_id)
        & spatial_residuals["time_utc"].between(start_hour, end_hour, inclusive="both"),
        column,
    ]
    values = pd.to_numeric(window, errors="coerce").dropna()
    if values.empty:
        return {
            "spatial_available_hours": 0,
            "spatial_median_residual": np.nan,
            "spatial_corroboration": "unavailable",
        }
    median = float(values.median())
    if not np.isfinite(offset_value) or median == 0:
        corroboration = "inconclusive"
    elif np.sign(median) == np.sign(offset_value):
        corroboration = "same_direction"
    else:
        corroboration = "opposite_direction"
    return {
        "spatial_available_hours": int(len(values)),
        "spatial_median_residual": median,
        "spatial_corroboration": corroboration,
    }


def build_calibration_corroboration(
    external_residuals: pd.DataFrame,
    spatial_residuals: pd.DataFrame | None = None,
    frozen_metrics: dict[tuple[str, str], dict[str, object]] | None = None,
) -> pd.DataFrame:
    external = _normalize_times(external_residuals, ["time_utc"])
    spatial = (
        _normalize_times(spatial_residuals, ["time_utc"])
        if spatial_residuals is not None
        else None
    )
    rows = []
    station_ids = sorted(external["station_id"].dropna().astype(str).unique())
    for channel in EXTERNAL_OFFSET_CHANNELS:
        statuses, episodes = detect_offset_channel(external, channel)
        status_lookup = statuses.set_index("station_id")["status"].to_dict()
        channel_episodes = (
            _normalize_times(episodes, ["start_hour", "end_hour"])
            if not episodes.empty
            else pd.DataFrame()
        )
        for station_id in station_ids:
            frozen = frozen_metrics.get((station_id, channel)) if frozen_metrics is not None else None
            if frozen is not None:
                metrics = frozen
            else:
                station = external.loc[external["station_id"].eq(station_id)]
                metrics = _residual_metrics(station, channel)
            windows = (
                channel_episodes.loc[channel_episodes["station_id"].eq(station_id)]
                .sort_values(["start_hour", "end_hour"])
                .reset_index(drop=True)
                if not channel_episodes.empty
                else pd.DataFrame()
            )
            if windows.empty:
                rows.append(
                    {
                        "corroboration_run_id": "",
                        "station_id": station_id,
                        "channel": channel,
                        "verdict": status_lookup.get(station_id, "insufficient"),
                        "sustained_offset": False,
                        "tier": "",
                        "start_hour": pd.NaT,
                        "end_hour": pd.NaT,
                        "duration_hours": np.nan,
                        "n_flagged_hours": 0,
                        **metrics,
                        "median_offset_level": np.nan,
                        "offset_stability_mad": np.nan,
                        "mean_offset_score": np.nan,
                        "peak_abs_offset_score": np.nan,
                        "confirmed_offset_value": np.nan,
                        "spatial_available_hours": 0,
                        "spatial_median_residual": np.nan,
                        "spatial_corroboration": "unavailable",
                        "resolved_episode_count": 0,
                        "resolved_episode_ids": "",
                    },
                )
                continue
            for position, (_, window) in enumerate(windows.iterrows(), start=1):
                offset_value = float(window["level_p50"])
                spatial_support = _spatial_support(
                    spatial,
                    station_id,
                    channel,
                    pd.to_datetime(window["start_hour"], utc=True),
                    pd.to_datetime(window["end_hour"], utc=True),
                    offset_value,
                )
                rows.append(
                    {
                        "corroboration_run_id": f"calibration_{channel}_{station_id}_{position:02d}",
                        "station_id": station_id,
                        "channel": channel,
                        "verdict": "confirmed",
                        "sustained_offset": True,
                        "tier": str(window["tier"]),
                        "start_hour": window["start_hour"],
                        "end_hour": window["end_hour"],
                        "duration_hours": float(window["duration_hours"]),
                        "n_flagged_hours": int(window["n_flagged_hours"]),
                        **metrics,
                        "median_offset_level": offset_value,
                        "offset_stability_mad": float(window["median_bmad"]),
                        "mean_offset_score": float(window["mean_score"]),
                        "peak_abs_offset_score": float(window["peak_abs_score"]),
                        "confirmed_offset_value": offset_value,
                        **spatial_support,
                        "resolved_episode_count": 0,
                        "resolved_episode_ids": "",
                    },
                )
    return pd.DataFrame(rows, columns=CALIBRATION_CORROBORATION_COLUMNS).sort_values(
        ["station_id", "channel", "start_hour"],
        na_position="last",
    ).reset_index(drop=True)


def ensure_label_states(labels: pd.DataFrame) -> pd.DataFrame:
    result = labels.copy()
    fault = pd.to_numeric(result["binary_fault"], errors="coerce").fillna(0).astype(int).eq(1)
    if "label_state" not in result.columns:
        result["label_state"] = np.where(fault, "fault", "benign")
    result.loc[fault, "label_state"] = "fault"
    result.loc[~fault & ~result["label_state"].isin(["benign", "borderline_review"]), "label_state"] = "benign"
    return result


def tag_borderline_review(
    labels: pd.DataFrame,
    benign_review: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = ensure_label_states(labels)
    review = benign_review.copy()
    external = pd.to_numeric(review.get("max_abs_era5_zscore"), errors="coerce").abs().ge(EXTERNAL_OFFSET_SCORE_HIGH)
    near_flatline = pd.to_numeric(review.get("near_flatline_hours"), errors="coerce").fillna(0).ge(1)
    selected = review.loc[external | near_flatline, ["episode_id", "station_id", "period"]].copy()
    selected["external_residual"] = external.loc[selected.index].to_numpy(dtype=bool)
    selected["near_flatline"] = near_flatline.loc[selected.index].to_numpy(dtype=bool)
    benign_ids = set(result.loc[result["binary_fault"].eq(0), "episode_id"].astype(str))
    selected = selected.loc[selected["episode_id"].astype(str).isin(benign_ids)].copy()
    result.loc[result["episode_id"].astype(str).isin(set(selected["episode_id"].astype(str))), "label_state"] = "borderline_review"
    return result, selected.reset_index(drop=True)


def _calibration_channel(raw_channel: object) -> str:
    return RAW_TO_CALIBRATION_CHANNEL.get(str(raw_channel), "")


def derive_borderline_evidence(
    labels: pd.DataFrame,
    scores: pd.DataFrame,
    statistical_evidence: pd.DataFrame,
) -> pd.DataFrame:
    label_frame = _normalize_times(labels, ["start_hour", "end_hour"])
    borderline = label_frame.loc[label_frame["label_state"].eq("borderline_review")]
    if borderline.empty:
        return pd.DataFrame(columns=BORDERLINE_EVIDENCE_COLUMNS)
    score_frame = _normalize_times(scores, ["hour_utc"])
    statistical = _normalize_times(statistical_evidence, ["hour_utc"])
    score_by_station = {
        station_id: frame.sort_values("hour_utc")
        for station_id, frame in score_frame.groupby("station_id", sort=False)
    }
    evidence_by_station = {
        station_id: frame.sort_values("hour_utc")
        for station_id, frame in statistical.groupby("station_id", sort=False)
    }
    rows = []
    for _, episode in borderline.iterrows():
        station_id = str(episode["station_id"])
        start = pd.to_datetime(episode["start_hour"], utc=True)
        end = pd.to_datetime(episode["end_hour"], utc=True)
        evidence_window = evidence_by_station.get(station_id, pd.DataFrame())
        if not evidence_window.empty:
            evidence_window = evidence_window.loc[
                evidence_window["hour_utc"].between(start, end, inclusive="both"),
            ]
            strong = pd.to_numeric(evidence_window.get("era5_zscore"), errors="coerce").abs().ge(EXTERNAL_OFFSET_SCORE_HIGH)
            for _, witness in evidence_window.loc[strong].iterrows():
                raw_channel = str(witness.get("raw_channel", ""))
                calibration_channel = _calibration_channel(raw_channel)
                if not calibration_channel:
                    continue
                rows.append(
                    {
                        "episode_id": episode["episode_id"],
                        "station_id": station_id,
                        "hour_utc": witness["hour_utc"],
                        "raw_channel": raw_channel,
                        "calibration_channel": calibration_channel,
                        "evidence_kind": "external_residual",
                        "era5_zscore": witness.get("era5_zscore", np.nan),
                    },
                )
        score_window = score_by_station.get(station_id, pd.DataFrame())
        if score_window.empty:
            continue
        score_window = score_window.loc[
            score_window["hour_utc"].between(start, end, inclusive="both"),
        ].copy()
        zscore = score_window.get("flag_zscore", pd.Series(False, index=score_window.index)).fillna(False).astype(bool)
        iforest = score_window.get("flag_iforest", pd.Series(False, index=score_window.index)).fillna(False).astype(bool)
        variance = pd.to_numeric(score_window.get("rolling_variance"), errors="coerce")
        near_flatline = (
            (zscore | iforest)
            & variance.gt(ROLLING_VARIANCE_FLAG_THRESHOLD)
            & variance.le(10.0 * ROLLING_VARIANCE_FLAG_THRESHOLD)
        )
        for _, witness in score_window.loc[near_flatline].iterrows():
            raw_channel = str(witness.get("channel", ""))
            calibration_channel = _calibration_channel(raw_channel)
            if not calibration_channel:
                continue
            rows.append(
                {
                    "episode_id": episode["episode_id"],
                    "station_id": station_id,
                    "hour_utc": witness["hour_utc"],
                    "raw_channel": raw_channel,
                    "calibration_channel": calibration_channel,
                    "evidence_kind": "near_flatline",
                    "era5_zscore": np.nan,
                },
            )
    if not rows:
        return pd.DataFrame(columns=BORDERLINE_EVIDENCE_COLUMNS)
    return pd.DataFrame(rows, columns=BORDERLINE_EVIDENCE_COLUMNS).drop_duplicates().sort_values(
        ["episode_id", "hour_utc", "raw_channel", "evidence_kind"],
    ).reset_index(drop=True)


def _mechanism_tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {token for token in str(value).split("|") if token}


def _fired_channels(value: object) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    if pd.isna(value):
        return values
    for token in str(value).split("|"):
        mechanism, separator, channels = token.partition("=")
        if not separator:
            continue
        values[mechanism] = {channel for channel in channels.split("+") if channel}
    return values


def _with_calibration_finding(row: pd.Series, raw_channels: set[str]) -> pd.Series:
    from src.rules.labelling import MECHANISM_ORDER

    result = row.copy()
    mechanisms = _mechanism_tokens(result.get("mechanisms", ""))
    mechanisms.add("calibration_offset")
    components = _mechanism_tokens(result.get("components", ""))
    components.update(sensor_group_for_channel(channel) for channel in raw_channels)
    fired = _fired_channels(result.get("fired_channels", ""))
    fired.setdefault("calibration_offset", set()).update(raw_channels)
    ordered = [mechanism for mechanism in MECHANISM_ORDER if mechanism in mechanisms]
    result["binary_fault"] = 1
    result["label_state"] = "fault"
    result["mechanisms"] = "|".join(ordered)
    result["components"] = "|".join(sorted(components))
    result["fired_channels"] = "|".join(
        f"{mechanism}=" + "+".join(sorted(fired.get(mechanism, set())))
        for mechanism in ordered
        if fired.get(mechanism)
    )
    return result


def resolve_borderline_labels(
    labels: pd.DataFrame,
    borderline_evidence: pd.DataFrame,
    calibration_corroboration: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = ensure_label_states(labels)
    confirmed = _normalize_times(
        calibration_corroboration.loc[
            calibration_corroboration["sustained_offset"].fillna(False).astype(bool)
        ].copy(),
        ["start_hour", "end_hour"],
    )
    evidence = _normalize_times(borderline_evidence, ["hour_utc"])
    if confirmed.empty or evidence.empty:
        return result, pd.DataFrame(columns=RESOLUTION_COLUMNS)
    candidates = evidence.merge(
        confirmed.loc[
            :,
            ["corroboration_run_id", "station_id", "channel", "start_hour", "end_hour", "confirmed_offset_value"],
        ],
        left_on=["station_id", "calibration_channel"],
        right_on=["station_id", "channel"],
        how="inner",
    )
    candidates = candidates.loc[
        candidates["hour_utc"].ge(candidates["start_hour"])
        & candidates["hour_utc"].le(candidates["end_hour"]),
    ].copy()
    if candidates.empty:
        return result, pd.DataFrame(columns=RESOLUTION_COLUMNS)
    matches = candidates.loc[:, RESOLUTION_COLUMNS].drop_duplicates().sort_values(
        ["episode_id", "hour_utc", "corroboration_run_id"],
    ).reset_index(drop=True)
    channels_by_episode = matches.groupby("episode_id")["raw_channel"].agg(lambda values: set(values))
    for index, row in result.iterrows():
        episode_id = str(row["episode_id"])
        if row["label_state"] != "borderline_review" or episode_id not in channels_by_episode.index:
            continue
        result.loc[index] = _with_calibration_finding(row, channels_by_episode.loc[episode_id])
    return result, matches


def attach_resolution_to_calibration_corroboration(
    calibration_corroboration: pd.DataFrame,
    resolutions: pd.DataFrame,
) -> pd.DataFrame:
    result = calibration_corroboration.copy()
    if resolutions.empty:
        return result.loc[:, CALIBRATION_CORROBORATION_COLUMNS]
    grouped = resolutions.groupby("corroboration_run_id")["episode_id"].agg(
        lambda values: "|".join(sorted(set(values))),
    )
    result["resolved_episode_ids"] = result["corroboration_run_id"].map(grouped).fillna("")
    result["resolved_episode_count"] = result["resolved_episode_ids"].map(
        lambda value: 0 if not value else len(str(value).split("|")),
    )
    return result.loc[:, CALIBRATION_CORROBORATION_COLUMNS]
