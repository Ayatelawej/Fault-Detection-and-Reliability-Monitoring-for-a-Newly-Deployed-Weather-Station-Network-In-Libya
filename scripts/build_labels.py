from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import LABELS_DIR, MERGED_DATASET_PATH
from src.rules.channel_handlers import numeric_channels
from src.rules.episodes import build_episodes
from src.rules.events import build_events
from src.rules.frozen_statistics import load_frozen_rule_statistics
from src.rules.labelling import (
    build_crosswalk,
    build_episode_labels,
    crosswalk_summary,
)
from src.rules.calibration_corroboration import (
    attach_resolution_to_calibration_corroboration,
    build_calibration_corroboration,
    derive_borderline_evidence,
    resolve_borderline_labels,
    tag_borderline_review,
)
from src.rules.config import EXTERNAL_OFFSET_SCORE_HIGH
from src.rules.score import compute_anomaly_scores
from src.rules.statistical_gate import (
    build_benign_review,
    build_contextual_cohort,
    build_statistical_evidence,
    build_statistical_review,
)
from src.workflows.prerequisites import require_files


LABEL_PATH = LABELS_DIR / "episode_labels.csv"
CROSSWALK_PATH = LABELS_DIR / "label_crosswalk.csv"
STATISTICAL_REVIEW_PATH = LABELS_DIR / "statistical_anomaly_review.csv"
BENIGN_REVIEW_PATH = LABELS_DIR / "benign_review_ranked.csv"
CALIBRATION_CORROBORATION_PATH = LABELS_DIR / "calibration_offset_corroboration.csv"
FROZEN_LABELS_PATH = LABELS_DIR / "fault_episodes_labeled_FULL.csv"
EXTERNAL_RESIDUALS_PATH = PROJECT_ROOT / "data/features/external_residuals.parquet"
SPATIAL_RESIDUALS_PATH = PROJECT_ROOT / "data/features/spatial_residuals.parquet"


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "nan"
    return f"{numerator / denominator:.2%}"


def _disagreement_table(crosswalk: pd.DataFrame) -> pd.DataFrame:
    if crosswalk.empty:
        return pd.DataFrame(columns=["frozen_family", "labelled_mechanisms", "pairs"])
    result = crosswalk.loc[~crosswalk["pair_agreement"]].copy()
    result["labelled_mechanisms"] = result["labelled_mechanisms"].fillna("").replace("", "benign")
    return (
        result.groupby(["frozen_family", "labelled_mechanisms"], as_index=False)
        .size()
        .rename(columns={"size": "pairs"})
        .sort_values(["frozen_family", "pairs"], ascending=[True, False])
    )


def _period_fault_table(labels: pd.DataFrame) -> pd.DataFrame:
    result = (
        labels.groupby("period", as_index=False)
        .agg(
            candidate_episodes=("episode_id", "size"),
            fault_episodes=("binary_fault", "sum"),
        )
        .sort_values("period")
    )
    result["fault_rate"] = result["fault_episodes"] / result["candidate_episodes"]
    return result


def _period_feature_table(labels: pd.DataFrame, source: str, label: str) -> pd.DataFrame:
    result = labels.loc[:, ["period", source]].copy()
    result[label] = result[source].fillna("").str.split("|")
    result = result.explode(label)
    result = result.loc[result[label].fillna("").ne("")]
    if result.empty:
        return pd.DataFrame(columns=["period", label, "episodes"])
    return (
        result.groupby(["period", label], as_index=False)
        .size()
        .rename(columns={"size": "episodes"})
        .sort_values(["period", label])
    )


def _label_state_table(labels: pd.DataFrame) -> pd.DataFrame:
    return (
        labels.groupby(["period", "label_state"], as_index=False)
        .size()
        .rename(columns={"size": "episodes"})
        .sort_values(["period", "label_state"])
    )


def _borderline_station_table(labels: pd.DataFrame) -> pd.DataFrame:
    result = labels.loc[labels["label_state"].eq("borderline_review")]
    if result.empty:
        return pd.DataFrame(columns=["period", "station_id", "episodes"])
    return (
        result.groupby(["period", "station_id"], as_index=False)
        .size()
        .rename(columns={"size": "episodes"})
        .sort_values(["period", "station_id"])
    )


def _confirmed_calibration_table(calibration_corroboration: pd.DataFrame) -> pd.DataFrame:
    result = calibration_corroboration.loc[
        calibration_corroboration["sustained_offset"].fillna(False).astype(bool)
    ].copy()
    columns = [
        "station_id",
        "channel",
        "start_hour",
        "end_hour",
        "duration_hours",
        "confirmed_offset_value",
        "offset_stability_mad",
        "full_period_fraction_abs_residual_z_ge_3",
        "spatial_corroboration",
        "resolved_episode_count",
    ]
    return result.loc[:, columns].sort_values(["station_id", "channel", "start_hour"])


def _episode_count_for_evidence(
    review: pd.DataFrame,
    evidence: pd.DataFrame,
    selection: pd.Series,
) -> int:
    if review.empty or evidence.empty or not selection.any():
        return 0
    evidence_keys = evidence.loc[
        selection,
        ["station_id", "hour_utc", "channel"],
    ].drop_duplicates()
    review_keys = review.loc[
        :,
        ["episode_id", "station_id", "hour_utc", "scoring_channel"],
    ].rename(columns={"scoring_channel": "channel"})
    matched = review_keys.merge(
        evidence_keys,
        on=["station_id", "hour_utc", "channel"],
        how="inner",
    )
    episode_ids = matched["episode_id"].fillna("")
    return int(episode_ids.loc[episode_ids.ne("")].nunique())


def _statistical_funnel(
    evidence: pd.DataFrame,
    review: pd.DataFrame,
) -> pd.DataFrame:
    both = evidence["both_detectors_same_channel_hour"].fillna(False).astype(bool)
    single = evidence["exactly_one_detector"].fillna(False).astype(bool)
    base = (
        evidence["physically_normal"].fillna(False).astype(bool)
        & evidence["not_stuck"].fillna(False).astype(bool)
        & evidence["contextual_outlier"].fillna(False).astype(bool)
    )
    path_a = evidence["evidence_path"].eq("A")
    path_b = evidence["evidence_path"].eq("B")
    path_b_candidate = single & base
    path_b_comparable = evidence["path_b_external_comparable"].fillna(False).astype(bool)
    path_b_rejected = path_b_candidate & ~evidence["path_b_era5_strong"].fillna(False).astype(bool)
    path_b_not_comparable = path_b_candidate & ~path_b_comparable
    path_b_missing = (
        path_b_candidate
        & path_b_comparable
        & ~evidence["era5_available"].fillna(False).astype(bool)
    )
    path_b_weak = (
        path_b_candidate
        & path_b_comparable
        & evidence["era5_available"].fillna(False).astype(bool)
        & ~evidence["path_b_era5_strong"].fillna(False).astype(bool)
    )
    rows = [
        ("detector_witnesses", pd.Series(True, index=evidence.index)),
        ("both_detector_witnesses", both),
        ("single_detector_witnesses", single),
        ("path_a_passed", path_a),
        ("path_b_considered", path_b_candidate),
        ("path_b_passed", path_b),
        ("path_b_rejected_no_or_weak_era5", path_b_rejected),
        ("path_b_rejected_external_not_comparable", path_b_not_comparable),
        ("path_b_rejected_era5_unavailable", path_b_missing),
        ("path_b_rejected_era5_below_threshold", path_b_weak),
    ]
    return pd.DataFrame(
        [
            {
                "stage": stage,
                "witnesses": int(selection.sum()),
                "linked_episodes": _episode_count_for_evidence(review, evidence, selection),
            }
            for stage, selection in rows
        ],
    )


def _path_episode_table(review: pd.DataFrame) -> pd.DataFrame:
    passed = review.loc[
        review["episode_id"].fillna("").ne("") & review["evidence_path"].isin(["A", "B"]),
        ["episode_id", "evidence_path"],
    ].drop_duplicates()
    if passed.empty:
        return pd.DataFrame(columns=["evidence_paths", "episodes"])
    grouped = passed.groupby("episode_id")["evidence_path"].agg(lambda values: "".join(sorted(set(values))))
    return (
        grouped.value_counts()
        .rename_axis("evidence_paths")
        .reset_index(name="episodes")
        .sort_values("evidence_paths")
    )


def _crosswalk_family_table(crosswalk: pd.DataFrame) -> pd.DataFrame:
    if crosswalk.empty:
        return pd.DataFrame(
            columns=[
                "frozen_family",
                "matched_pairs",
                "pair_agree",
                "pair_agreement_rate",
                "matched_frozen_episodes",
                "frozen_agree",
                "frozen_agreement_rate",
            ],
        )
    pairs = (
        crosswalk.groupby("frozen_family", as_index=False)
        .agg(
            matched_pairs=("frozen_row_id", "size"),
            pair_agree=("pair_agreement", "sum"),
        )
        .sort_values("frozen_family")
    )
    pairs["pair_agreement_rate"] = pairs["pair_agree"] / pairs["matched_pairs"]
    aggregate = crosswalk.drop_duplicates("frozen_row_id")
    frozen = (
        aggregate.groupby("frozen_family", as_index=False)
        .agg(
            matched_frozen_episodes=("frozen_row_id", "size"),
            frozen_agree=("frozen_agreement", "sum"),
        )
        .sort_values("frozen_family")
    )
    frozen["frozen_agreement_rate"] = frozen["frozen_agree"] / frozen["matched_frozen_episodes"]
    return pairs.merge(frozen, on="frozen_family", how="outer")


def _config_assignment_location(name: str) -> str:
    config_path = PROJECT_ROOT / "src/rules/config.py"
    for line_number, line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip().startswith(f"{name} ="):
            return f"src/rules/config.py:{line_number}"
    raise RuntimeError(f"Missing configuration assignment for {name}")


def _rain_summary(labels: pd.DataFrame) -> tuple[int, bool]:
    rain = labels.loc[
        labels["components"].fillna("").str.split("|").map(lambda values: "rain_gauge" in values),
    ]
    if rain.empty:
        return 0, False
    all_spike = bool(
        rain["mechanisms"].fillna("").str.split("|").map(lambda values: "spike_impossible" in values).all(),
    )
    return int(len(rain)), all_spike


def _write_labels_and_optional_crosswalk(
    labels: pd.DataFrame,
    label_path: Path,
    crosswalk_path: Path,
) -> tuple[pd.DataFrame | None, dict[str, object] | None]:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(label_path, index=False)
    if not FROZEN_LABELS_PATH.exists():
        return None, None
    frozen = pd.read_csv(FROZEN_LABELS_PATH)
    crosswalk = build_crosswalk(frozen, labels)
    crosswalk.to_csv(crosswalk_path, index=False)
    return crosswalk, crosswalk_summary(frozen, labels, crosswalk)


def _require_label_inputs(source: Path) -> None:
    require_files(
        "Label construction",
        {
            "canonical merged dataset": source,
            "external residual evidence": EXTERNAL_RESIDUALS_PATH,
            "spatial residual evidence": SPATIAL_RESIDUALS_PATH,
        },
        "Run scripts/rebuild_detection_features.py after supplying its public reference and observation inputs.",
    )


def _inuqat9_path_b_rows(review: pd.DataFrame) -> pd.DataFrame:
    return review.loc[
        review["station_id"].eq("INUQAT9")
        & review["period"].eq("june")
        & review["component"].eq("anemometer")
        & review["evidence_path"].eq("B"),
        [
            "episode_id",
            "episode_start_hour",
            "episode_end_hour",
            "duration_hours",
            "hour_utc",
            "raw_channel",
            "era5_zscore",
            "contextual_zscore",
        ],
    ].drop_duplicates()


def _format_rates(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = result[column].map(
            lambda value: f"{float(value):.2%}" if pd.notna(value) else "nan",
        )
    return result


def _prepare_scores(scores: pd.DataFrame) -> pd.DataFrame:
    result = scores.copy()
    if "flag_physical_suspect" in result.columns:
        result["flag_physical_suspect"] = False
    if "reason" in result.columns:
        result["reason"] = result["reason"].fillna("").map(
            lambda value: "|".join(
                token
                for token in str(value).split("|")
                if token and token != "physical_suspect_breach"
            ),
        )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=MERGED_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=LABELS_DIR)
    parser.add_argument("--frozen-statistics-dir", type=Path, default=None)
    parser.add_argument("--contextual-unavailable-from", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source = args.source
    output_dir = args.output_dir
    label_path = output_dir / "episode_labels.csv"
    crosswalk_path = output_dir / "label_crosswalk.csv"
    statistical_review_path = output_dir / "statistical_anomaly_review.csv"
    benign_review_path = output_dir / "benign_review_ranked.csv"
    calibration_corroboration_path = output_dir / "calibration_offset_corroboration.csv"
    frozen_statistics = (
        load_frozen_rule_statistics(args.frozen_statistics_dir)
        if args.frozen_statistics_dir is not None
        else None
    )
    contextual_unavailable_from = (
        pd.Timestamp(args.contextual_unavailable_from, tz="UTC")
        if args.contextual_unavailable_from is not None
        else None
    )
    _require_label_inputs(source)
    raw = pd.read_csv(source, parse_dates=["hour_utc"], low_memory=False)
    raw["hour_utc"] = pd.to_datetime(raw["hour_utc"], utc=True)
    channels = numeric_channels(raw)
    scores = _prepare_scores(
        compute_anomaly_scores(
            raw,
            channels=channels,
            frozen_baselines=None if frozen_statistics is None else frozen_statistics.baselines,
            frozen_isolation_forests=None if frozen_statistics is None else frozen_statistics.isolation_forests,
            frozen_thresholds=None if frozen_statistics is None else frozen_statistics.thresholds,
        )
    )
    events = build_events(scores)
    episodes = build_episodes(events)
    external_residuals = pd.read_parquet(EXTERNAL_RESIDUALS_PATH)
    cohort = build_contextual_cohort(raw, scores)
    statistical_evidence = build_statistical_evidence(
        raw,
        scores,
        external_residuals,
        cohort,
        frozen_thresholds=None if frozen_statistics is None else frozen_statistics.evidence_thresholds,
        contextual_unavailable_from=contextual_unavailable_from,
    )
    labels = build_episode_labels(
        episodes,
        raw,
        scores,
        statistical_evidence,
    )
    initial_benign_review = build_benign_review(
        labels,
        raw,
        scores,
        external_residuals,
        cohort,
        frozen_thresholds=None if frozen_statistics is None else frozen_statistics.evidence_thresholds,
        contextual_unavailable_from=contextual_unavailable_from,
    )
    labels, initial_borderline = tag_borderline_review(
        labels,
        initial_benign_review,
    )
    spatial_residuals = pd.read_parquet(SPATIAL_RESIDUALS_PATH)
    calibration_corroboration = build_calibration_corroboration(
        external_residuals,
        spatial_residuals,
        frozen_metrics=(
            None
            if frozen_statistics is None
            else frozen_statistics.calibration_corroboration_metrics
        ),
    )
    borderline_evidence = derive_borderline_evidence(
        labels,
        scores,
        statistical_evidence,
    )
    labels, resolutions = resolve_borderline_labels(
        labels,
        borderline_evidence,
        calibration_corroboration,
    )
    calibration_corroboration = attach_resolution_to_calibration_corroboration(
        calibration_corroboration,
        resolutions,
    )
    crosswalk, summary = _write_labels_and_optional_crosswalk(labels, label_path, crosswalk_path)
    statistical_review = build_statistical_review(labels, statistical_evidence)
    benign_review = build_benign_review(
        labels,
        raw,
        scores,
        external_residuals,
        cohort,
        frozen_thresholds=None if frozen_statistics is None else frozen_statistics.evidence_thresholds,
        contextual_unavailable_from=contextual_unavailable_from,
    )
    period_summary = _period_fault_table(labels)
    label_state_summary = _label_state_table(labels)
    mechanism_summary = _period_feature_table(labels, "mechanisms", "mechanism")
    component_summary = _period_feature_table(labels, "components", "component")
    funnel = _statistical_funnel(statistical_evidence, statistical_review)
    path_episodes = _path_episode_table(statistical_review)
    family_summary = _crosswalk_family_table(crosswalk) if crosswalk is not None else None
    solar_passed = statistical_review.loc[
        statistical_review["raw_channel"].eq("solar_radiation_high_wm2")
        & statistical_review["evidence_path"].isin(["A", "B"])
        & statistical_review["episode_id"].fillna("").ne(""),
        "episode_id",
    ].nunique()
    rain_episodes, rain_all_spike = _rain_summary(labels)
    inuqat9_path_b = _inuqat9_path_b_rows(statistical_review)
    threshold_location = _config_assignment_location("EXTERNAL_OFFSET_SCORE_HIGH")
    confirmed_calibration = _confirmed_calibration_table(calibration_corroboration)
    remaining_borderline = _borderline_station_table(labels)
    initial_borderline_ids = set(initial_borderline["episode_id"].astype(str))
    resolved_ids = set(resolutions["episode_id"].astype(str)) if not resolutions.empty else set()
    calibration_eligible_ids = set(borderline_evidence["episode_id"].astype(str)) if not borderline_evidence.empty else set()
    solar_calibration_labels = int(
        labels.loc[
            labels["mechanisms"].fillna("").str.split("|").map(lambda values: "calibration_offset" in values)
            & labels["components"].fillna("").str.split("|").map(lambda values: "light_uv" in values),
        ].shape[0],
    )
    statistical_review.to_csv(statistical_review_path, index=False)
    benign_review.to_csv(benign_review_path, index=False)
    calibration_corroboration.to_csv(calibration_corroboration_path, index=False)

    print("LABEL PREVIEW")
    print(f"hourly_rows={len(raw)}")
    print(f"hourly_start={raw['hour_utc'].min()}")
    print(f"hourly_end={raw['hour_utc'].max()}")
    print(f"scored_channels={len(scores['channel'].unique())}")
    print(f"candidate_episodes={len(labels)}")
    print("period_fault_rates=")
    print(_format_rates(period_summary, ["fault_rate"]).to_string(index=False))
    print("label_state_counts_by_period=")
    print(label_state_summary.to_string(index=False))
    print("mechanism_counts_by_period=")
    print(mechanism_summary.to_string(index=False))
    print("component_counts_by_period=")
    print(component_summary.to_string(index=False))
    print("confirmed_calibration_offsets=")
    print(confirmed_calibration.to_string(index=False))
    print("borderline_resolution=")
    print(f"initial_borderline_review={len(initial_borderline_ids)}")
    print(f"calibration_eligible_borderline={len(calibration_eligible_ids)}")
    print(f"resolved_to_calibration_offset={len(resolved_ids)}")
    print(f"remaining_borderline_review={int(labels['label_state'].eq('borderline_review').sum())}")
    print("remaining_borderline_by_period_and_station=")
    print(remaining_borderline.to_string(index=False))
    print(f"solar_calibration_offset_labels={solar_calibration_labels}")
    print("statistical_gate_funnel=")
    print(funnel.to_string(index=False))
    print("statistical_gate_episode_paths=")
    print(path_episodes.to_string(index=False))
    print("reused_stage5_numeric_threshold=")
    print(f"source={threshold_location}")
    print(f"value={EXTERNAL_OFFSET_SCORE_HIGH}")
    print(f"path_b_comparison=abs(era5_zscore)>={EXTERNAL_OFFSET_SCORE_HIGH}")
    print("stage5_uses_offset_abs_score; path_b_reuses_only_the_numeric_cutoff_for_hourly_residual_zscores")
    print(f"solar_gate_passing_episodes={solar_passed}")
    print(f"rain_gauge_episodes={rain_episodes}")
    print(f"rain_gauge_all_include_spike_impossible={rain_all_spike}")
    print(f"inuqat9_june_anemometer_path_b_episodes={inuqat9_path_b['episode_id'].nunique()}")
    print("inuqat9_june_anemometer_path_b_witnesses=")
    print(inuqat9_path_b.to_string(index=False))
    if crosswalk is None or summary is None or family_summary is None:
        print(f"legacy_crosswalk=skipped_missing_input:{FROZEN_LABELS_PATH}")
    else:
        print("crosswalk=")
        print(f"matched_pairs={summary['pair_count']}")
        print(f"pair_agreement={summary['pair_agree']}")
        print(f"pair_agreement_rate={_rate(summary['pair_agree'], summary['pair_count'])}")
        print(f"matched_frozen_episodes={summary['frozen_matched']}")
        print(f"frozen_aggregate_agreement={summary['frozen_agree']}")
        print(
            "frozen_aggregate_agreement_rate="
            + _rate(summary["frozen_agree"], summary["frozen_matched"]),
        )
        print(f"frozen_without_label_match={summary['frozen_unmatched']}")
        print(f"labels_without_frozen_match={summary['labelled_unmatched']}")
        print("crosswalk_by_frozen_family=")
        print(_format_rates(family_summary, ["pair_agreement_rate", "frozen_agreement_rate"]).to_string(index=False))
        print("crosswalk_disagreements=")
        print(_disagreement_table(crosswalk).to_string(index=False))
    print(f"labels_path={label_path}")
    print(f"preview_rows={len(labels)}")
    if crosswalk is not None:
        print(f"crosswalk_path={crosswalk_path}")
        print(f"crosswalk_rows={len(crosswalk)}")
    print(f"statistical_review_path={statistical_review_path}")
    print(f"statistical_review_rows={len(statistical_review)}")
    print(f"benign_review_path={benign_review_path}")
    print(f"benign_review_rows={len(benign_review)}")
    print(f"calibration_corroboration_path={calibration_corroboration_path}")
    print(f"calibration_corroboration_rows={len(calibration_corroboration)}")


if __name__ == "__main__":
    main()
