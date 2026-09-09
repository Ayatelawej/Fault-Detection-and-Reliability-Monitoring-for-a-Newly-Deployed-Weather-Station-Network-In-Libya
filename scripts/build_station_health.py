from __future__ import annotations

from argparse import ArgumentParser, Namespace
from hashlib import sha256
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.availability.health_score import (
    build_station_health_scores,
    is_hard_zero_health_baseline,
    station_health_input_hashes,
    validate_delete_future_health_scores,
    write_station_health_outputs,
)
from src.availability.health_forecast import (
    HEALTH_FORECAST_FEATURE_SETS,
    HEALTH_FORECAST_HORIZONS,
    HEALTH_FORECAST_LONG_FEATURE_SETS,
    HEALTH_FORECAST_LONG_HORIZONS,
    run_health_forecast,
    write_health_forecast_outputs,
)
from src.availability.operational_scorecard import (
    SCORECARD_HORIZONS,
    build_operational_scorecard,
    load_health_forecast_models,
    validate_delete_future_operational_scorecard,
    write_operational_scorecard_outputs,
)
from src.availability.risk_dataset import load_exact_hour_reference
from src.config.paths import (
    HEALTH_COMPONENT_CORRELATION_FIGURE_PATH,
    HEALTH_COMPONENT_DISTRIBUTIONS_FIGURE_PATH,
    HEALTH_DISTRIBUTION_FIGURE_PATH,
    HEALTH_FORECAST_BASELINE_COMPARISON_FIGURE_PATH,
    HEALTH_FORECAST_CALIBRATION_FIGURE_PATH,
    HEALTH_FORECAST_DIR,
    HEALTH_FORECAST_HORIZON_DEGRADATION_FIGURE_PATH,
    HEALTH_FORECAST_LONG_HORIZON_BASELINE_COMPARISON_FIGURE_PATH,
    HEALTH_FORECAST_LONG_HORIZON_CALIBRATION_FIGURE_PATH,
    HEALTH_FORECAST_LONG_HORIZON_DEGRADATION_FIGURE_PATH,
    HEALTH_FORECAST_LONG_HORIZON_DIR,
    HEALTH_FORECAST_LONG_HORIZON_MODEL_DIR,
    HEALTH_FORECAST_MODEL_DIR,
    HEALTH_OUTAGE_DURATION_TRAJECTORY_FIGURE_PATH,
    HEALTH_STATION_TIMESERIES_FIGURE_PATH,
    MERGED_DATASET_PATH,
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
    AVAILABILITY_CLASSIFICATION_PATH,
    STATION_OPERATIONAL_SCORECARD_CAUSALITY_PATH,
    STATION_OPERATIONAL_SCORECARD_INVARIANTS_PATH,
    STATION_OPERATIONAL_SCORECARD_PATH,
    STATION_OPERATIONAL_SCORECARD_REPORT_PATH,
    STATION_REGISTRY_PATH,
)
from src.rules.config import EXTERNAL_CACHE_DIR
from src.workflows.prerequisites import require_files

CALIBRATION_CORROBORATION_PATH = (
    PROJECT_ROOT / "data" / "labels" / "calibration_offset_corroboration.csv"
)


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(
        description="Build the causal, transparent station health score."
    )
    parser.add_argument("--observations", type=Path, default=MERGED_DATASET_PATH)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=PROJECT_ROOT / EXTERNAL_CACHE_DIR,
    )
    parser.add_argument("--scores-output", type=Path, default=STATION_HEALTH_SCORES_PATH)
    parser.add_argument("--summary-output", type=Path, default=STATION_HEALTH_SUMMARY_PATH)
    parser.add_argument("--report-output", type=Path, default=STATION_HEALTH_REPORT_PATH)
    parser.add_argument("--invariants-output", type=Path, default=STATION_HEALTH_INVARIANTS_PATH)
    parser.add_argument(
        "--causality-audit-output",
        type=Path,
        default=STATION_HEALTH_CAUSALITY_AUDIT_PATH,
    )
    parser.add_argument(
        "--causality-summary-output",
        type=Path,
        default=STATION_HEALTH_CAUSALITY_SUMMARY_PATH,
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=STATION_HEALTH_PROGRESSIVE_COMPARISON_PATH,
    )
    parser.add_argument(
        "--ranking-comparison-output",
        type=Path,
        default=STATION_HEALTH_PROGRESSIVE_RANKING_PATH,
    )
    parser.add_argument(
        "--outage-duration-curve-output",
        type=Path,
        default=STATION_HEALTH_OUTAGE_DURATION_CURVE_PATH,
    )
    parser.add_argument(
        "--outage-trajectory-output",
        type=Path,
        default=STATION_HEALTH_OUTAGE_TRAJECTORY_PATH,
    )
    parser.add_argument(
        "--previous-scores",
        type=Path,
        default=None,
        help="Optional prior hard-zero score artifact to validate the migration comparison.",
    )
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--causality-samples", type=int, default=16)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--forecast",
        action="store_true",
        help="Train and evaluate the health-score forecaster from the frozen current-health artifact.",
    )
    mode.add_argument(
        "--scorecard",
        action="store_true",
        help="Assemble the causal per-station operational snapshot without training models.",
    )
    mode.add_argument(
        "--long-horizon-forecast",
        action="store_true",
        help="Train and evaluate the separate 2-to-7-day health forecast comparison.",
    )
    parser.add_argument(
        "--forecast-output-dir",
        type=Path,
        default=HEALTH_FORECAST_DIR,
    )
    parser.add_argument(
        "--forecast-model-dir",
        type=Path,
        default=HEALTH_FORECAST_MODEL_DIR,
    )
    parser.add_argument("--forecast-causality-samples", type=int, default=8)
    parser.add_argument(
        "--long-horizon-features",
        choices=("current", "extended"),
        default="extended",
        help="Feature comparison used only with --long-horizon-forecast.",
    )
    parser.add_argument(
        "--reference-hour",
        default=None,
        help=(
            "Timezone-aware whole-hour scorecard reference. By default, use the latest "
            "canonical hour containing at least one observed transmission, skipping terminal padding."
        ),
    )
    parser.add_argument("--station-registry", type=Path, default=STATION_REGISTRY_PATH)
    parser.add_argument(
        "--availability-classification",
        type=Path,
        default=AVAILABILITY_CLASSIFICATION_PATH,
    )
    parser.add_argument(
        "--scorecard-output",
        type=Path,
        default=STATION_OPERATIONAL_SCORECARD_PATH,
    )
    parser.add_argument(
        "--scorecard-report-output",
        type=Path,
        default=STATION_OPERATIONAL_SCORECARD_REPORT_PATH,
    )
    parser.add_argument(
        "--scorecard-invariants-output",
        type=Path,
        default=STATION_OPERATIONAL_SCORECARD_INVARIANTS_PATH,
    )
    parser.add_argument(
        "--scorecard-causality-output",
        type=Path,
        default=STATION_OPERATIONAL_SCORECARD_CAUSALITY_PATH,
    )
    return parser.parse_args(argv)


def _require_reference_cache(reference_directory: Path) -> None:
    if not reference_directory.is_dir() or not list(reference_directory.glob("*.parquet")):
        raise FileNotFoundError(
            "Station-health construction cannot start because the exact-hour "
            f"reference cache is missing or empty: {reference_directory}"
        )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_existing_tree(directory: Path, *, excluded: Path | None = None) -> str:
    digest = sha256()
    root = Path(directory)
    excluded_path = Path(excluded).resolve() if excluded is not None else None
    if not root.exists():
        return digest.hexdigest()
    for child in sorted(item for item in root.rglob("*") if item.is_file()):
        if excluded_path is not None and excluded_path in child.resolve().parents:
            continue
        digest.update(str(child.relative_to(root)).replace("\\", "/").encode("utf-8"))
        digest.update(_sha256_file(child).encode("ascii"))
    return digest.hexdigest()


def _forecast_input_hashes(args: Namespace) -> dict[str, str]:
    labels_path = PROJECT_ROOT / "data" / "labels" / "episode_labels.csv"
    model_root = PROJECT_ROOT / "data" / "model"
    paths = {
        "station_health_scores": args.scores_output,
        "canonical_station_hourly_dataset": args.observations,
        "availability_classification": AVAILABILITY_CLASSIFICATION_PATH,
        "live_episode_labels": labels_path,
    }
    hashes = {
        name: _sha256_file(path)
        for name, path in paths.items()
        if Path(path).is_file()
    }
    hashes["preexisting_model_artifacts"] = _sha256_existing_tree(
        model_root,
        excluded=args.forecast_model_dir,
    )
    hashes["all_processed_availability_and_health_artifacts"] = (
        _sha256_existing_tree(PROJECT_ROOT / "data" / "processed")
    )
    hashes["all_label_artifacts"] = _sha256_existing_tree(
        PROJECT_ROOT / "data" / "labels"
    )
    hashes["hourly_detection_and_reason_code_models"] = _sha256_existing_tree(
        PROJECT_ROOT / "data" / "hourly_detection" / "models"
    )
    return hashes


def _run_forecast(args: Namespace) -> None:
    require_files(
        "Health-score forecasting",
        {
            "causal station-health scores": args.scores_output,
            "canonical merged dataset": args.observations,
        },
        "Run the station-health score stage before forecasting.",
    )
    if args.forecast_causality_samples <= 0:
        raise ValueError("--forecast-causality-samples must be positive")
    long_horizon = bool(args.long_horizon_forecast)
    horizons = HEALTH_FORECAST_LONG_HORIZONS if long_horizon else HEALTH_FORECAST_HORIZONS
    long_variant = str(args.long_horizon_features)
    output_directory = (
        HEALTH_FORECAST_LONG_HORIZON_DIR / long_variant
        if long_horizon
        else args.forecast_output_dir
    )
    model_directory = (
        HEALTH_FORECAST_LONG_HORIZON_MODEL_DIR / long_variant
        if long_horizon
        else args.forecast_model_dir
    )
    feature_sets = (
        HEALTH_FORECAST_LONG_FEATURE_SETS
        if long_horizon and long_variant == "extended"
        else HEALTH_FORECAST_FEATURE_SETS
    )
    comparison_figure = (
        HEALTH_FORECAST_LONG_HORIZON_BASELINE_COMPARISON_FIGURE_PATH
        if long_horizon
        else HEALTH_FORECAST_BASELINE_COMPARISON_FIGURE_PATH
    )
    calibration_figure = (
        HEALTH_FORECAST_LONG_HORIZON_CALIBRATION_FIGURE_PATH
        if long_horizon
        else HEALTH_FORECAST_CALIBRATION_FIGURE_PATH
    )
    degradation_figure = (
        HEALTH_FORECAST_LONG_HORIZON_DEGRADATION_FIGURE_PATH
        if long_horizon
        else HEALTH_FORECAST_HORIZON_DEGRADATION_FIGURE_PATH
    )
    if long_horizon:
        comparison_figure = comparison_figure.with_name(
            f"{comparison_figure.stem}_{long_variant}{comparison_figure.suffix}"
        )
        calibration_figure = calibration_figure.with_name(
            f"{calibration_figure.stem}_{long_variant}{calibration_figure.suffix}"
        )
        degradation_figure = degradation_figure.with_name(
            f"{degradation_figure.stem}_{long_variant}{degradation_figure.suffix}"
        )
    original_model_directory = args.forecast_model_dir
    args.forecast_model_dir = model_directory
    before = _forecast_input_hashes(args)
    scores = pd.read_parquet(args.scores_output)
    metadata = pd.read_csv(args.observations, usecols=["station_id", "elevation"])
    run = run_health_forecast(
        scores,
        station_metadata=metadata,
        horizons=horizons,
        feature_audit_samples=args.forecast_causality_samples,
        model_directory=model_directory,
        feature_sets=feature_sets,
    )
    after = _forecast_input_hashes(args)
    args.forecast_model_dir = original_model_directory
    outputs = write_health_forecast_outputs(
        run,
        output_directory=output_directory,
        comparison_figure_path=comparison_figure,
        calibration_figure_path=calibration_figure,
        degradation_figure_path=degradation_figure,
        input_hashes_before=before,
        input_hashes_after=after,
    )
    if long_horizon and long_variant == "extended":
        current_accuracy_path = (
            HEALTH_FORECAST_LONG_HORIZON_DIR
            / "current"
            / "health_forecast_accuracy_curve.csv"
        )
        if current_accuracy_path.is_file():
            current_accuracy = pd.read_csv(current_accuracy_path).rename(
                columns={
                    "n": "current_n",
                    "accuracy": "current_feature_accuracy",
                }
            )
            extended_accuracy = pd.read_csv(outputs["accuracy_curve"]).rename(
                columns={
                    "n": "extended_n",
                    "accuracy": "validation_selected_extended_accuracy",
                }
            )
            accuracy_comparison = current_accuracy.merge(
                extended_accuracy,
                on="horizon_h",
                how="outer",
                validate="one_to_one",
            ).sort_values("horizon_h", kind="mergesort")
            accuracy_comparison["accuracy_change"] = (
                accuracy_comparison["validation_selected_extended_accuracy"]
                - accuracy_comparison["current_feature_accuracy"]
            )
            comparison_path = (
                HEALTH_FORECAST_LONG_HORIZON_DIR
                / "health_forecast_accuracy_comparison.csv"
            )
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            accuracy_comparison.to_csv(comparison_path, index=False)
            outputs["accuracy_comparison"] = comparison_path
    print(Path(outputs["report"]).read_text(encoding="utf-8"))
    print("OUTPUTS")
    for name, path in outputs.items():
        print(f"{name}={path}")


def _scorecard_input_hashes(args: Namespace) -> dict[str, str]:
    hourly_directory = PROJECT_ROOT / "data" / "hourly_detection"
    paths = {
        "canonical_station_hourly_dataset": args.observations,
        "station_registry": args.station_registry,
        "station_health_scores": args.scores_output,
        "availability_classification": args.availability_classification,
        "live_episode_labels": PROJECT_ROOT / "data" / "labels" / "episode_labels.csv",
        "calibration_corroboration_evidence": CALIBRATION_CORROBORATION_PATH,
        "hourly_short_tensor": hourly_directory / "hourly_detection_short.npz",
        "binary_metrics": hourly_directory / "hourly_short_calibration_metrics.json",
        "reason_code_metrics": hourly_directory / "reason_code_method_comparison_metrics.json",
    }
    hashes = {
        name: _sha256_file(path)
        for name, path in paths.items()
        if Path(path).is_file()
    }
    hashes["binary_and_reason_code_model_tree"] = _sha256_existing_tree(
        hourly_directory / "models"
    )
    hashes["health_forecast_model_tree"] = _sha256_existing_tree(
        args.forecast_model_dir
    )
    return hashes


def _run_scorecard(args: Namespace) -> None:
    model_paths = {
        f"health forecast {horizon}h model": args.forecast_model_dir
        / f"health_forecast_forecast_transmitting_origin_{horizon}h.joblib"
        for horizon in SCORECARD_HORIZONS
    }
    require_files(
        "Combined operational scorecard",
        {
            "causal station-health scores": args.scores_output,
            "station registry": args.station_registry,
            "availability classification": args.availability_classification,
            "calibration corroboration evidence": CALIBRATION_CORROBORATION_PATH,
            **model_paths,
        },
        "Run station health and the five-horizon health forecast before assembling the scorecard.",
    )
    before = _scorecard_input_hashes(args)
    scores = pd.read_parquet(args.scores_output)
    registry = pd.read_csv(args.station_registry)
    availability = pd.read_parquet(args.availability_classification)
    calibration_corroboration = pd.read_csv(CALIBRATION_CORROBORATION_PATH)
    models = load_health_forecast_models(args.forecast_model_dir)
    run = build_operational_scorecard(
        scores,
        registry,
        availability=availability,
        forecast_models=models,
        calibration_corroboration=calibration_corroboration,
        reference_hour=args.reference_hour,
    )
    causality = validate_delete_future_operational_scorecard(
        scores,
        registry,
        availability=availability,
        forecast_models=models,
        calibration_corroboration=calibration_corroboration,
        reference_hour=run.metadata["reference_hour_utc"],
    )
    after = _scorecard_input_hashes(args)
    outputs = write_operational_scorecard_outputs(
        run,
        table_path=args.scorecard_output,
        report_path=args.scorecard_report_output,
        invariants_path=args.scorecard_invariants_output,
        causality_path=args.scorecard_causality_output,
        causality_audit=causality,
        input_hashes_before=before,
        input_hashes_after=after,
    )
    print(Path(outputs["report"]).read_text(encoding="utf-8"))
    print("OUTPUTS")
    for name, path in outputs.items():
        print(f"{name}={path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.forecast or args.long_horizon_forecast:
        _run_forecast(args)
        return
    if args.scorecard:
        _run_scorecard(args)
        return
    require_files(
        "Station-health construction",
        {"canonical merged dataset": args.observations},
        "Start from the frozen hourly dataset and run the public reference fetch first.",
    )
    _require_reference_cache(args.reference_dir)
    if args.causality_samples <= 0:
        raise ValueError("--causality-samples must be positive")
    before = station_health_input_hashes(args.observations, args.reference_dir)
    previous_scores = None
    previous_path = args.previous_scores or args.scores_output
    if previous_path.is_file():
        candidate = pd.read_parquet(previous_path)
        if is_hard_zero_health_baseline(candidate):
            previous_scores = candidate
        elif args.previous_scores is not None:
            raise ValueError(
                "--previous-scores must point to the prior hard-zero health-score artifact"
            )
    observations = pd.read_csv(args.observations, low_memory=False)
    reference = load_exact_hour_reference(args.reference_dir)
    calibration_corroboration = (
        pd.read_csv(CALIBRATION_CORROBORATION_PATH)
        if CALIBRATION_CORROBORATION_PATH.is_file()
        else None
    )
    scores = build_station_health_scores(observations, reference)
    causality_audit = validate_delete_future_health_scores(
        observations,
        reference,
        full_scores=scores,
        sample_size=args.causality_samples,
    )
    after = station_health_input_hashes(args.observations, args.reference_dir)
    outputs = write_station_health_outputs(
        scores,
        causality_audit=causality_audit,
        input_hashes_before=before,
        input_hashes_after=after,
        calibration_corroboration=calibration_corroboration,
        previous_scores=previous_scores,
        output_paths={
            "scores": args.scores_output,
            "summary": args.summary_output,
            "report": args.report_output,
            "invariants": args.invariants_output,
            "causality_audit": args.causality_audit_output,
            "causality_summary": args.causality_summary_output,
            "comparison": args.comparison_output,
            "ranking_comparison": args.ranking_comparison_output,
            "outage_duration_curve": args.outage_duration_curve_output,
            "outage_trajectory": args.outage_trajectory_output,
        },
        generate_figures=not args.skip_figures,
    )
    print(Path(args.report_output).read_text(encoding="utf-8"))
    print("OUTPUTS")
    for name, path in outputs.items():
        print(f"{name}={path}")
    if not args.skip_figures:
        print(f"distribution_figure={HEALTH_DISTRIBUTION_FIGURE_PATH}")
        print(f"component_figure={HEALTH_COMPONENT_DISTRIBUTIONS_FIGURE_PATH}")
        print(f"timeseries_figure={HEALTH_STATION_TIMESERIES_FIGURE_PATH}")
        print(f"correlation_figure={HEALTH_COMPONENT_CORRELATION_FIGURE_PATH}")
        print(f"outage_duration_trajectory_figure={HEALTH_OUTAGE_DURATION_TRAJECTORY_FIGURE_PATH}")


if __name__ == "__main__":
    main()
