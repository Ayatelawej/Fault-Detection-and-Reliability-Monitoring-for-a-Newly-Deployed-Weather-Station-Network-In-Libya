from __future__ import annotations

import numpy as np
import pandas as pd

from src.rules.channel_handlers import encode_wind_direction, sensor_group_for_channel
from src.rules.config import (
    CHANNELS_REQUIRING_CIRCULAR_TRANSFORM,
    CHANNELS_REQUIRING_LOG_TRANSFORM,
    CONTEXTUAL_BASELINE_MIN_SAMPLES,
    CONTEXTUAL_MAD_FLOORS,
    CONTEXTUAL_OUTLIER_Z_THRESHOLD,
    ERA5_AGREEMENT_Z_THRESHOLD,
    EXTERNAL_OFFSET_SCORE_HIGH,
    PHYSICAL_LIMIT_RULES,
    ROLLING_VARIANCE_FLAG_THRESHOLD,
)


CONTEXT_COLUMNS = [
    "station_id",
    "hour_utc",
    "channel",
    "context_value",
    "context_month",
    "context_hour",
    "context_baseline_n",
    "context_median",
    "context_mad",
    "context_floor",
    "contextual_zscore",
    "context_available",
    "contextual_outlier",
]
EVIDENCE_COLUMNS = [
    "station_id",
    "hour_utc",
    "channel",
    "raw_channel",
    "observed_value",
    "zscore",
    "iforest_score",
    "zscore_threshold",
    "iforest_threshold",
    "flag_zscore",
    "flag_iforest",
    "both_detectors_same_channel_hour",
    "exactly_one_detector",
    "flag_physical",
    "flag_stuck",
    "physically_normal",
    "not_stuck",
    "context_value",
    "context_month",
    "context_hour",
    "context_baseline_n",
    "context_median",
    "context_mad",
    "context_floor",
    "contextual_zscore",
    "context_available",
    "contextual_outlier",
    "era5_metric",
    "era5_zscore",
    "era5_available",
    "era5_sign_matches_context",
    "era5_agrees",
    "path_b_external_comparable",
    "path_b_era5_strong",
    "direction_pair_passed",
    "evidence_path",
    "full_gate_passed",
    "gate_failure_reasons",
]
EXTERNAL_Z_COLUMNS = {
    "pressure_max_hpa": "pressure",
    "temp_avg_c": "temp",
    "windspeed_avg_kmh": "wind",
    "solar_radiation_high_wm2": "solar",
}
PATH_B_EXTERNAL_RAW_CHANNELS = {
    "pressure_max_hpa",
    "temp_avg_c",
    "windspeed_avg_kmh",
}
DIRECTION_SCORE_CHANNELS = {"winddir_sin", "winddir_cos"}


def normalize_times(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column], utc=True, format="mixed")
    if "station_id" in result.columns:
        result["station_id"] = result["station_id"].astype(str)
    return result


def raw_channel_for_score_channel(channel: str) -> str:
    if channel in DIRECTION_SCORE_CHANNELS:
        return "winddir_avg_deg"
    return channel


def _score_channels(raw: pd.DataFrame, score_channels: object = None) -> list[str]:
    if score_channels is not None:
        return sorted({str(channel) for channel in score_channels})
    channels = []
    for raw_channel in PHYSICAL_LIMIT_RULES:
        if raw_channel not in raw.columns:
            continue
        if raw_channel in CHANNELS_REQUIRING_CIRCULAR_TRANSFORM:
            channels.extend(["winddir_sin", "winddir_cos"])
        else:
            channels.append(raw_channel)
    return channels


def _values_for_score_channel(raw: pd.DataFrame, channel: str) -> pd.DataFrame:
    raw_channel = raw_channel_for_score_channel(channel)
    columns = ["station_id", "hour_utc", "channel", "context_value"]
    if raw_channel not in raw.columns:
        return pd.DataFrame(columns=columns)
    values = pd.to_numeric(raw[raw_channel], errors="coerce")
    if channel in DIRECTION_SCORE_CHANNELS:
        encoded = encode_wind_direction(values)
        values = encoded["sin"] if channel == "winddir_sin" else encoded["cos"]
    elif raw_channel in CHANNELS_REQUIRING_LOG_TRANSFORM:
        with np.errstate(invalid="ignore", divide="ignore"):
            values = pd.Series(np.log1p(values.to_numpy(dtype=float)), index=values.index)
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(
        {
            "station_id": raw["station_id"].astype(str).to_numpy(),
            "hour_utc": pd.to_datetime(raw["hour_utc"], utc=True),
            "channel": channel,
            "context_value": values.to_numpy(dtype=float),
        },
        columns=columns,
    )


def build_contextual_cohort(
    raw: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    raw_frame = normalize_times(raw, ["hour_utc"])
    score_frame = normalize_times(scores, ["hour_utc"])
    channels = _score_channels(raw_frame, score_frame.get("channel", pd.Series(dtype=object)).dropna().unique())
    frames = [_values_for_score_channel(raw_frame, channel) for channel in channels]
    if not frames:
        return pd.DataFrame(columns=["station_id", "hour_utc", "channel", "context_value", "context_month", "context_hour"])
    result = pd.concat(frames, ignore_index=True)
    result = result.loc[result["context_value"].notna()].copy()
    invalid = score_frame.copy()
    physical = invalid.get("flag_physical", pd.Series(False, index=invalid.index)).fillna(False).astype(bool)
    stuck = invalid.get("flag_stuck", pd.Series(False, index=invalid.index)).fillna(False).astype(bool)
    invalid = invalid.loc[physical | stuck, ["station_id", "hour_utc", "channel"]].drop_duplicates()
    if not invalid.empty:
        invalid["_excluded"] = True
        result = result.merge(invalid, on=["station_id", "hour_utc", "channel"], how="left")
        result = result.loc[result["_excluded"].isna()].drop(columns="_excluded")
    result["context_month"] = result["hour_utc"].dt.month.astype(int)
    result["context_hour"] = result["hour_utc"].dt.hour.astype(int)
    return result.sort_values(["station_id", "channel", "hour_utc"]).reset_index(drop=True)


def contextual_evidence(
    raw: pd.DataFrame,
    scores: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    cohort: pd.DataFrame | None = None,
    min_samples: int = CONTEXTUAL_BASELINE_MIN_SAMPLES,
    threshold: float = CONTEXTUAL_OUTLIER_Z_THRESHOLD,
    unavailable_from: pd.Timestamp | None = None,
) -> pd.DataFrame:
    required = ["station_id", "hour_utc", "channel"]
    if candidate_rows.empty:
        return pd.DataFrame(columns=CONTEXT_COLUMNS)
    query = normalize_times(candidate_rows.loc[:, required], ["hour_utc"]).drop_duplicates(required)
    raw_frame = normalize_times(raw, ["hour_utc"])
    value_frames = []
    for channel, rows in query.groupby("channel", sort=False):
        source = _values_for_score_channel(raw_frame, str(channel))
        value_frames.append(rows.merge(source, on=required, how="left"))
    result = pd.concat(value_frames, ignore_index=True) if value_frames else query.copy()
    result["context_month"] = result["hour_utc"].dt.month.astype(int)
    result["context_hour"] = result["hour_utc"].dt.hour.astype(int)
    cohort_frame = build_contextual_cohort(raw, scores) if cohort is None else cohort.copy()
    group_columns = ["station_id", "channel", "context_month", "context_hour"]
    needed = result.loc[:, group_columns].drop_duplicates()
    support = cohort_frame.merge(needed, on=group_columns, how="inner")
    groups = {
        key: frame.loc[:, ["hour_utc", "context_value"]]
        for key, frame in support.groupby(group_columns, sort=False)
    }
    summaries = []
    for row in result.itertuples(index=False):
        key = (row.station_id, row.channel, row.context_month, row.context_hour)
        values = groups.get(key, pd.DataFrame(columns=["hour_utc", "context_value"]))
        reference = values.loc[values["hour_utc"].ne(row.hour_utc), "context_value"]
        reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
        count = int(len(reference_values))
        median = float(np.median(reference_values)) if count else np.nan
        mad = float(np.median(np.abs(reference_values - median))) if count else np.nan
        floor = float(CONTEXTUAL_MAD_FLOORS.get(str(row.channel), 0.1))
        period_blocked = unavailable_from is not None and row.hour_utc >= unavailable_from
        available = bool(
            not period_blocked
            and count >= min_samples
            and pd.notna(row.context_value)
            and pd.notna(mad)
            and np.isfinite(mad)
        )
        denominator = max(mad, floor) if available else np.nan
        score = (
            float((float(row.context_value) - median) / denominator)
            if available and denominator > 0
            else np.nan
        )
        summaries.append(
            {
                "context_baseline_n": count,
                "context_median": median,
                "context_mad": mad,
                "context_floor": floor,
                "contextual_zscore": score,
                "context_available": available,
                "contextual_outlier": bool(available and abs(score) >= threshold),
            },
        )
    summary = pd.DataFrame(summaries, index=result.index)
    result = pd.concat([result, summary], axis=1)
    return result.loc[:, CONTEXT_COLUMNS]


def detector_thresholds(
    scores: pd.DataFrame,
    percentile: float = 99.7,
    frozen: dict[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    frame = normalize_times(scores, ["hour_utc"])
    rows = []
    for channel, group in frame.groupby("channel", sort=False):
        frozen_zscore = frozen.get((str(channel), "zscore")) if frozen is not None else None
        frozen_iforest = frozen.get((str(channel), "iforest")) if frozen is not None else None
        if frozen_zscore is not None and frozen_iforest is not None:
            rows.append(
                {
                    "channel": str(channel),
                    "zscore_threshold": frozen_zscore,
                    "iforest_threshold": frozen_iforest,
                },
            )
            continue
        zscore = pd.to_numeric(group.get("zscore"), errors="coerce").abs().dropna()
        iforest = pd.to_numeric(group.get("iforest_score"), errors="coerce").dropna()
        rows.append(
            {
                "channel": str(channel),
                "zscore_threshold": float(np.nanpercentile(zscore, percentile)) if not zscore.empty else np.nan,
                "iforest_threshold": float(np.nanpercentile(iforest, percentile)) if not iforest.empty else np.nan,
            },
        )
    return pd.DataFrame(rows)


def _external_values(
    frame: pd.DataFrame,
    external_residuals: pd.DataFrame | None,
) -> pd.DataFrame:
    result = frame.copy()
    result["raw_channel"] = result["channel"].map(raw_channel_for_score_channel)
    result["era5_metric"] = result["raw_channel"].map(EXTERNAL_Z_COLUMNS).fillna("")
    result["era5_zscore"] = np.nan
    result["era5_available"] = False
    result["era5_sign_matches_context"] = False
    if external_residuals is None or external_residuals.empty:
        return result
    external = normalize_times(external_residuals, ["time_utc"])
    for raw_channel, short in EXTERNAL_Z_COLUMNS.items():
        metric_columns = [
            f"station_{short}",
            f"ref_{short}",
            f"r_{short}",
            f"base_{short}",
            f"bmad_{short}",
            f"z_{short}",
        ]
        if any(column not in external.columns for column in metric_columns):
            continue
        mask = result["raw_channel"].eq(raw_channel)
        if not mask.any():
            continue
        source = external.loc[:, ["station_id", "time_utc"] + metric_columns].copy()
        source = source.drop_duplicates(["station_id", "time_utc"], keep="last")
        valid = source.loc[:, metric_columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        if short == "solar":
            valid = valid & pd.to_numeric(source["ref_solar"], errors="coerce").ge(300.0)
        lookup = source.set_index(["station_id", "time_utc"])
        keys = pd.MultiIndex.from_frame(result.loc[mask, ["station_id", "hour_utc"]])
        zscore = pd.to_numeric(lookup[f"z_{short}"].reindex(keys), errors="coerce").to_numpy(dtype=float)
        availability = valid.to_numpy(dtype=bool)
        valid_lookup = pd.Series(availability, index=lookup.index)
        present = valid_lookup.reindex(keys).fillna(False).to_numpy(dtype=bool)
        result.loc[mask, "era5_zscore"] = zscore
        result.loc[mask, "era5_available"] = present
    context = pd.to_numeric(result.get("contextual_zscore"), errors="coerce")
    external_score = pd.to_numeric(result["era5_zscore"], errors="coerce")
    result["era5_sign_matches_context"] = (
        result["era5_available"].astype(bool)
        & np.sign(context).eq(np.sign(external_score))
    ).fillna(False)
    return result


def _observed_values(raw: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    required = ["station_id", "hour_utc", "channel"]
    result = frame.loc[:, required].drop_duplicates().copy()
    result["raw_channel"] = result["channel"].map(raw_channel_for_score_channel)
    raw_frame = normalize_times(raw, ["hour_utc"])
    value_frames = []
    for raw_channel, rows in result.groupby("raw_channel", sort=False):
        if raw_channel not in raw_frame.columns:
            rows = rows.copy()
            rows["observed_value"] = np.nan
            value_frames.append(rows)
            continue
        source = raw_frame.loc[:, ["station_id", "hour_utc", raw_channel]].copy()
        source = source.rename(columns={raw_channel: "observed_value"})
        source["observed_value"] = pd.to_numeric(source["observed_value"], errors="coerce")
        value_frames.append(rows.merge(source, on=["station_id", "hour_utc"], how="left"))
    return pd.concat(value_frames, ignore_index=True) if value_frames else result.assign(observed_value=np.nan)


def _direction_pairs(result: pd.DataFrame, base_pass: pd.Series) -> pd.Series:
    passed = pd.Series(True, index=result.index, dtype=bool)
    direction = result["channel"].isin(DIRECTION_SCORE_CHANNELS)
    if not direction.any():
        return passed
    subset = result.loc[direction, ["station_id", "hour_utc", "raw_channel", "channel"]].copy()
    subset["base_pass"] = base_pass.loc[direction].to_numpy(dtype=bool)
    counts = (
        subset.loc[subset["base_pass"]]
        .groupby(["station_id", "hour_utc", "raw_channel"], sort=False)["channel"]
        .nunique()
    )
    keys = pd.MultiIndex.from_frame(subset[["station_id", "hour_utc", "raw_channel"]])
    pair_pass = counts.reindex(keys).fillna(0).to_numpy(dtype=int) == 2
    passed.loc[direction] = pair_pass
    return passed


def _failure_reasons(frame: pd.DataFrame) -> pd.Series:
    values = []
    for row in frame.itertuples(index=False):
        reasons = []
        if not bool(row.physically_normal):
            reasons.append("physical_limit")
        if not bool(row.not_stuck):
            reasons.append("stuck")
        if not bool(row.context_available):
            reasons.append("context_unavailable")
        elif not bool(row.contextual_outlier):
            reasons.append("context_not_outlier")
        base_conditions = (
            bool(row.physically_normal)
            and bool(row.not_stuck)
            and bool(row.contextual_outlier)
        )
        if bool(row.exactly_one_detector) and base_conditions and not bool(row.path_b_era5_strong):
            if not bool(row.path_b_external_comparable):
                reasons.append("path_b_external_not_comparable")
            elif not bool(row.era5_available):
                reasons.append("path_b_era5_unavailable")
            else:
                reasons.append("path_b_era5_not_strong")
        elif (
            bool(row.both_detectors_same_channel_hour)
            and not bool(row.full_gate_passed)
            and bool(row.era5_available)
            and not bool(row.era5_agrees)
        ):
            reasons.append("era5_not_agree")
        if row.channel in DIRECTION_SCORE_CHANNELS and not bool(row.direction_pair_passed):
            reasons.append("direction_pair_not_agree")
        values.append("|".join(reasons))
    return pd.Series(values, index=frame.index, dtype=object)


def build_statistical_evidence(
    raw: pd.DataFrame,
    scores: pd.DataFrame,
    external_residuals: pd.DataFrame | None = None,
    cohort: pd.DataFrame | None = None,
    frozen_thresholds: dict[tuple[str, str], float] | None = None,
    contextual_unavailable_from: pd.Timestamp | None = None,
) -> pd.DataFrame:
    score_frame = normalize_times(scores, ["hour_utc"])
    for column in ["flag_zscore", "flag_iforest", "flag_physical", "flag_stuck"]:
        if column not in score_frame.columns:
            score_frame[column] = False
    for column in ["zscore", "iforest_score"]:
        if column not in score_frame.columns:
            score_frame[column] = np.nan
    flag_zscore = score_frame.get("flag_zscore", pd.Series(False, index=score_frame.index)).fillna(False).astype(bool)
    flag_iforest = score_frame.get("flag_iforest", pd.Series(False, index=score_frame.index)).fillna(False).astype(bool)
    selected = score_frame.loc[flag_zscore | flag_iforest].copy()
    if selected.empty:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)
    selected = selected.loc[
        :,
        [
            "station_id",
            "hour_utc",
            "channel",
            "zscore",
            "iforest_score",
            "flag_zscore",
            "flag_iforest",
            "flag_physical",
            "flag_stuck",
        ],
    ]
    context = contextual_evidence(
        raw, score_frame, selected, cohort=cohort, unavailable_from=contextual_unavailable_from,
    )
    result = selected.merge(context, on=["station_id", "hour_utc", "channel"], how="left")
    observed = _observed_values(raw, result)
    result = result.merge(
        observed.loc[:, ["station_id", "hour_utc", "channel", "observed_value"]],
        on=["station_id", "hour_utc", "channel"],
        how="left",
    )
    thresholds = detector_thresholds(score_frame, frozen=frozen_thresholds)
    result = result.merge(thresholds, on="channel", how="left")
    result["both_detectors_same_channel_hour"] = (
        result["flag_zscore"].fillna(False).astype(bool)
        & result["flag_iforest"].fillna(False).astype(bool)
    )
    result["exactly_one_detector"] = (
        result["flag_zscore"].fillna(False).astype(bool)
        ^ result["flag_iforest"].fillna(False).astype(bool)
    )
    result["flag_physical"] = result["flag_physical"].fillna(False).astype(bool)
    result["flag_stuck"] = result["flag_stuck"].fillna(False).astype(bool)
    result["physically_normal"] = ~result["flag_physical"]
    result["not_stuck"] = ~result["flag_stuck"]
    result = _external_values(result, external_residuals)
    result["era5_agrees"] = (
        ~result["era5_available"].astype(bool)
        | (
            pd.to_numeric(result["era5_zscore"], errors="coerce").abs().ge(ERA5_AGREEMENT_Z_THRESHOLD)
            & result["era5_sign_matches_context"].astype(bool)
        )
    )
    result["path_b_external_comparable"] = result["raw_channel"].isin(PATH_B_EXTERNAL_RAW_CHANNELS)
    result["path_b_era5_strong"] = (
        result["path_b_external_comparable"].astype(bool)
        & result["era5_available"].astype(bool)
        & pd.to_numeric(result["era5_zscore"], errors="coerce").abs().ge(EXTERNAL_OFFSET_SCORE_HIGH)
    )
    base_conditions = (
        result["physically_normal"].astype(bool)
        & result["not_stuck"].astype(bool)
        & result["contextual_outlier"].fillna(False).astype(bool)
    )
    path_a = (
        base_conditions
        & result["both_detectors_same_channel_hour"].astype(bool)
        & result["era5_agrees"].fillna(False).astype(bool)
    )
    path_b = (
        base_conditions
        & result["exactly_one_detector"].astype(bool)
        & result["path_b_era5_strong"].astype(bool)
    )
    result["direction_pair_passed"] = _direction_pairs(result, path_a | path_b)
    path_a = path_a & result["direction_pair_passed"]
    path_b = path_b & result["direction_pair_passed"]
    result["evidence_path"] = np.select([path_a, path_b], ["A", "B"], default="")
    result["full_gate_passed"] = result["evidence_path"].ne("")
    result["gate_failure_reasons"] = _failure_reasons(result)
    return result.loc[:, EVIDENCE_COLUMNS].sort_values(
        ["station_id", "hour_utc", "channel"],
    ).reset_index(drop=True)


def statistical_channels_for_window(evidence_window: pd.DataFrame) -> set[str]:
    if evidence_window.empty or "full_gate_passed" not in evidence_window.columns:
        return set()
    passed = evidence_window.loc[evidence_window["full_gate_passed"].fillna(False).astype(bool)]
    return {raw_channel_for_score_channel(str(channel)) for channel in passed["channel"]}


STATISTICAL_REVIEW_COLUMNS = [
    "review_id",
    "episode_id",
    "station_id",
    "episode_start_hour",
    "episode_end_hour",
    "duration_hours",
    "period",
    "label_state",
    "hour_utc",
    "raw_channel",
    "scoring_channel",
    "component",
    "observed_value",
    "zscore",
    "zscore_threshold",
    "iforest_score",
    "iforest_threshold",
    "flag_zscore",
    "flag_iforest",
    "both_detectors_same_channel_hour",
    "flag_physical",
    "flag_stuck",
    "physically_normal",
    "context_month",
    "context_hour",
    "context_baseline_n",
    "context_median",
    "context_mad",
    "context_floor",
    "contextual_zscore",
    "context_available",
    "contextual_outlier",
    "era5_metric",
    "era5_zscore",
    "era5_status",
    "era5_sign_matches_context",
    "era5_agrees",
    "evidence_path",
    "full_gate_passed",
    "final_mechanism",
    "gate_failure_reasons",
]
BENIGN_REVIEW_COLUMNS = [
    "review_rank",
    "review_id",
    "episode_id",
    "station_id",
    "start_hour",
    "end_hour",
    "duration_hours",
    "period",
    "label_state",
    "training_eligible",
    "candidate_raw_channels",
    "candidate_components",
    "single_detector_channels",
    "z_only_hours",
    "iforest_only_hours",
    "max_abs_zscore",
    "max_iforest_score",
    "max_single_detector_margin",
    "context_outlier_hours",
    "max_abs_contextual_zscore",
    "era5_supported_channels",
    "era5_available_hours",
    "max_abs_era5_zscore",
    "era5_agree_hours",
    "era5_disagree_hours",
    "near_flatline_channels",
    "near_flatline_hours",
    "min_rolling_variance",
    "near_flatline_strength",
    "external_residual_score",
    "single_detector_score",
    "near_flatline_score",
    "review_priority_score",
    "review_reason",
    "review_decision",
    "review_notes",
]


def _episode_lookup(labels: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty:
        return pd.DataFrame(columns=["episode_id", "episode_start_hour", "episode_end_hour", "duration_hours", "period", "label_state", "final_mechanism"])
    label_frame = normalize_times(labels, ["start_hour", "end_hour"])
    if "label_state" not in label_frame.columns:
        binary = pd.to_numeric(label_frame.get("binary_fault", pd.Series(0, index=label_frame.index)), errors="coerce").fillna(0).astype(int)
        label_frame["label_state"] = np.where(binary.eq(1), "fault", "benign")
    rows = []
    for station_id, group in evidence.groupby("station_id", sort=False):
        candidates = label_frame.loc[label_frame["station_id"].eq(str(station_id))]
        for index, item in group.iterrows():
            matches = candidates.loc[
                candidates["start_hour"].le(item["hour_utc"])
                & candidates["end_hour"].ge(item["hour_utc"])
            ]
            if matches.empty:
                rows.append(
                    {
                        "_evidence_index": index,
                        "episode_id": "",
                        "episode_start_hour": pd.NaT,
                        "episode_end_hour": pd.NaT,
                        "duration_hours": np.nan,
                        "period": "",
                        "label_state": "",
                        "final_mechanism": "",
                    },
                )
                continue
            for _, match in matches.iterrows():
                rows.append(
                    {
                        "_evidence_index": index,
                        "episode_id": match["episode_id"],
                        "episode_start_hour": match["start_hour"],
                        "episode_end_hour": match["end_hour"],
                        "duration_hours": match["duration_hours"],
                        "period": match["period"],
                        "label_state": match["label_state"],
                        "final_mechanism": match["mechanisms"],
                    },
                )
    return pd.DataFrame(rows).set_index("_evidence_index")


def build_statistical_review(
    labels: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    if evidence.empty:
        return pd.DataFrame(columns=STATISTICAL_REVIEW_COLUMNS)
    source = normalize_times(evidence, ["hour_utc"])
    source = source.drop(
        columns=[
            "episode_id",
            "episode_start_hour",
            "episode_end_hour",
            "duration_hours",
            "period",
            "final_mechanism",
            "review_id",
        ],
        errors="ignore",
    )
    if "era5_available" not in source.columns:
        source["era5_available"] = source.get(
            "era5_status",
            pd.Series("", index=source.index),
        ).eq("available")
    if "evidence_path" not in source.columns:
        source["evidence_path"] = ""
    lookup = _episode_lookup(labels, source)
    result = source.join(lookup, how="left")
    result["scoring_channel"] = result["channel"]
    result["component"] = result["raw_channel"].map(sensor_group_for_channel)
    result["era5_status"] = np.where(
        result["era5_metric"].eq(""),
        "not_applicable",
        np.where(result["era5_available"].astype(bool), "available", "unavailable"),
    )
    result = result.sort_values(["station_id", "hour_utc", "scoring_channel"]).reset_index(drop=True)
    result["review_id"] = np.arange(1, len(result) + 1, dtype=int)
    return result.loc[:, STATISTICAL_REVIEW_COLUMNS]


def _flag_values(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.get(column, pd.Series(False, index=frame.index)).fillna(False).astype(bool)


def _max_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else np.nan


def _unique_tokens(values: pd.Series) -> str:
    return "|".join(sorted({str(value) for value in values.dropna() if str(value)}))


def _flagged_context_evidence(
    raw: pd.DataFrame,
    scores: pd.DataFrame,
    external_residuals: pd.DataFrame | None,
    cohort: pd.DataFrame,
    frozen_thresholds: dict[tuple[str, str], float] | None = None,
    contextual_unavailable_from: pd.Timestamp | None = None,
) -> pd.DataFrame:
    score_frame = normalize_times(scores, ["hour_utc"])
    if "flag" not in score_frame.columns:
        score_frame["flag"] = (
            _flag_values(score_frame, "flag_zscore")
            | _flag_values(score_frame, "flag_iforest")
            | _flag_values(score_frame, "flag_stuck")
            | _flag_values(score_frame, "flag_physical")
        )
    flagged = score_frame.loc[_flag_values(score_frame, "flag")].copy()
    if flagged.empty:
        return pd.DataFrame()
    selected = flagged.loc[
        :,
        [
            "station_id",
            "hour_utc",
            "channel",
            "zscore",
            "iforest_score",
            "flag_zscore",
            "flag_iforest",
            "flag_stuck",
            "flag_physical",
            "rolling_variance",
        ],
    ]
    context = contextual_evidence(
        raw, score_frame, selected, cohort=cohort, unavailable_from=contextual_unavailable_from,
    )
    result = selected.merge(context, on=["station_id", "hour_utc", "channel"], how="left")
    result = result.merge(detector_thresholds(score_frame, frozen=frozen_thresholds), on="channel", how="left")
    result = _external_values(result, external_residuals)
    result["era5_agrees"] = (
        result["era5_available"].astype(bool)
        & pd.to_numeric(result["era5_zscore"], errors="coerce").abs().ge(ERA5_AGREEMENT_Z_THRESHOLD)
        & result["era5_sign_matches_context"].astype(bool)
    )
    return result


def build_benign_review(
    labels: pd.DataFrame,
    raw: pd.DataFrame,
    scores: pd.DataFrame,
    external_residuals: pd.DataFrame | None = None,
    cohort: pd.DataFrame | None = None,
    frozen_thresholds: dict[tuple[str, str], float] | None = None,
    contextual_unavailable_from: pd.Timestamp | None = None,
) -> pd.DataFrame:
    label_frame = normalize_times(labels, ["start_hour", "end_hour"])
    if "label_state" not in label_frame.columns:
        binary = pd.to_numeric(label_frame.get("binary_fault", pd.Series(0, index=label_frame.index)), errors="coerce").fillna(0).astype(int)
        label_frame["label_state"] = np.where(binary.eq(1), "fault", "benign")
    benign = label_frame.loc[label_frame["binary_fault"].eq(0)].copy()
    if benign.empty:
        return pd.DataFrame(columns=BENIGN_REVIEW_COLUMNS)
    cohort_frame = build_contextual_cohort(raw, scores) if cohort is None else cohort
    evidence = _flagged_context_evidence(
        raw, scores, external_residuals, cohort_frame, frozen_thresholds, contextual_unavailable_from,
    )
    by_station = {
        station_id: frame.sort_values("hour_utc")
        for station_id, frame in evidence.groupby("station_id", sort=False)
    }
    rows = []
    for _, episode in benign.iterrows():
        station = by_station.get(str(episode["station_id"]), pd.DataFrame(columns=evidence.columns))
        window = station.loc[
            station["hour_utc"].between(episode["start_hour"], episode["end_hour"], inclusive="both"),
        ].copy()
        window["raw_channel"] = window.get("raw_channel", pd.Series(dtype=object))
        z_only = _flag_values(window, "flag_zscore") & ~_flag_values(window, "flag_iforest")
        iforest_only = ~_flag_values(window, "flag_zscore") & _flag_values(window, "flag_iforest")
        z_margin = pd.Series(np.nan, index=window.index, dtype=float)
        z_threshold = pd.to_numeric(window.get("zscore_threshold"), errors="coerce")
        z_values = pd.to_numeric(window.get("zscore"), errors="coerce").abs()
        z_margin.loc[z_only] = (z_values.loc[z_only] / z_threshold.loc[z_only]).replace([np.inf, -np.inf], np.nan)
        iforest_margin = pd.Series(np.nan, index=window.index, dtype=float)
        iforest_threshold = pd.to_numeric(window.get("iforest_threshold"), errors="coerce")
        iforest_values = pd.to_numeric(window.get("iforest_score"), errors="coerce")
        iforest_margin.loc[iforest_only] = (
            iforest_values.loc[iforest_only] / iforest_threshold.loc[iforest_only]
        ).replace([np.inf, -np.inf], np.nan)
        single_margin = pd.concat([z_margin, iforest_margin], axis=1).max(axis=1, skipna=True)
        variance = pd.to_numeric(window.get("rolling_variance"), errors="coerce")
        near = variance.gt(ROLLING_VARIANCE_FLAG_THRESHOLD) & variance.le(10.0 * ROLLING_VARIANCE_FLAG_THRESHOLD)
        near_values = variance.loc[near]
        near_strength = (
            ((10.0 * ROLLING_VARIANCE_FLAG_THRESHOLD - near_values.min()) / (9.0 * ROLLING_VARIANCE_FLAG_THRESHOLD))
            if not near_values.empty
            else 0.0
        )
        max_external = _max_or_nan(pd.to_numeric(window.get("era5_zscore"), errors="coerce").abs())
        external_score = min(float(max_external / ERA5_AGREEMENT_Z_THRESHOLD), 1.0) if pd.notna(max_external) else 0.0
        max_single = _max_or_nan(single_margin)
        single_score = min(float(max_single / 2.0), 1.0) if pd.notna(max_single) else 0.0
        reasons = []
        if external_score > 0:
            reasons.append("external_residual")
        if single_score > 0:
            reasons.append("single_detector")
        if near_strength > 0:
            reasons.append("near_flatline")
        rows.append(
            {
                "episode_id": episode["episode_id"],
                "station_id": episode["station_id"],
                "start_hour": episode["start_hour"],
                "end_hour": episode["end_hour"],
                "duration_hours": episode["duration_hours"],
                "period": episode["period"],
                "label_state": episode["label_state"],
                "training_eligible": bool(episode["label_state"] == "benign"),
                "candidate_raw_channels": _unique_tokens(window["raw_channel"]),
                "candidate_components": _unique_tokens(window["raw_channel"].map(sensor_group_for_channel)),
                "single_detector_channels": _unique_tokens(window.loc[z_only | iforest_only, "raw_channel"]),
                "z_only_hours": int(window.loc[z_only, "hour_utc"].nunique()),
                "iforest_only_hours": int(window.loc[iforest_only, "hour_utc"].nunique()),
                "max_abs_zscore": _max_or_nan(pd.to_numeric(window.get("zscore"), errors="coerce").abs()),
                "max_iforest_score": _max_or_nan(window.get("iforest_score", pd.Series(dtype=float))),
                "max_single_detector_margin": max_single,
                "context_outlier_hours": int(window.loc[window.get("contextual_outlier", pd.Series(False, index=window.index)).fillna(False).astype(bool), "hour_utc"].nunique()),
                "max_abs_contextual_zscore": _max_or_nan(pd.to_numeric(window.get("contextual_zscore"), errors="coerce").abs()),
                "era5_supported_channels": _unique_tokens(window.loc[window.get("era5_metric", pd.Series("", index=window.index)).ne(""), "raw_channel"]),
                "era5_available_hours": int(window.loc[window.get("era5_available", pd.Series(False, index=window.index)).fillna(False).astype(bool), "hour_utc"].nunique()),
                "max_abs_era5_zscore": max_external,
                "era5_agree_hours": int(window.loc[window.get("era5_agrees", pd.Series(False, index=window.index)).fillna(False).astype(bool), "hour_utc"].nunique()),
                "era5_disagree_hours": int(window.loc[window.get("era5_available", pd.Series(False, index=window.index)).fillna(False).astype(bool) & ~window.get("era5_agrees", pd.Series(False, index=window.index)).fillna(False).astype(bool), "hour_utc"].nunique()),
                "near_flatline_channels": _unique_tokens(window.loc[near, "raw_channel"]),
                "near_flatline_hours": int(window.loc[near, "hour_utc"].nunique()),
                "min_rolling_variance": float(near_values.min()) if not near_values.empty else np.nan,
                "near_flatline_strength": float(np.clip(near_strength, 0.0, 1.0)),
                "external_residual_score": external_score,
                "single_detector_score": single_score,
                "near_flatline_score": float(np.clip(near_strength, 0.0, 1.0)),
                "review_priority_score": external_score + single_score + float(np.clip(near_strength, 0.0, 1.0)),
                "review_reason": "|".join(reasons),
                "review_decision": "",
                "review_notes": "",
            },
        )
    result = pd.DataFrame(rows, columns=[column for column in BENIGN_REVIEW_COLUMNS if column not in {"review_rank", "review_id"}])
    result = result.sort_values(
        [
            "review_priority_score",
            "external_residual_score",
            "single_detector_score",
            "near_flatline_score",
            "duration_hours",
            "station_id",
            "start_hour",
        ],
        ascending=[False, False, False, False, False, True, True],
    ).reset_index(drop=True)
    result.insert(0, "review_rank", np.arange(1, len(result) + 1, dtype=int))
    result.insert(1, "review_id", np.arange(1, len(result) + 1, dtype=int))
    return result.loc[:, BENIGN_REVIEW_COLUMNS]
