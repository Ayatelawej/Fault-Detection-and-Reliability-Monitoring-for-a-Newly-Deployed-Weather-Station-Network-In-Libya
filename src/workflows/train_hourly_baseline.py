from __future__ import annotations

import gc
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.hourly_baseline import HourlyBaselineConfig, assert_matching_metadata, baseline_report, filter_eligible_examples, flatten_hourly_features, grouped_permutation_importance, load_hourly_metadata, load_hourly_tensor, make_split_manifest, reason_code_aggregated_metric_rows, resolve_fault_class_weight, run_baseline_matrix, save_model_bundle, validation_importance_reference, write_metrics_json
from src.model.hourly_detection import LONG_TENSOR_PATH, MASK_MODE_PER_HOUR, MASK_MODES, SHORT_TENSOR_PATH
from src.workflows.prerequisites import require_files


OUTPUT_DIR = PROJECT_ROOT / "data" / "hourly_detection"
METRICS_PATH = OUTPUT_DIR / "hourly_baseline_metrics.json"
REPORT_PATH = OUTPUT_DIR / "hourly_baseline_report.txt"
IMPORTANCE_PATH = OUTPUT_DIR / "hourly_baseline_feature_importance.csv"
MANIFEST_PATH = OUTPUT_DIR / "hourly_baseline_split_manifest.csv"


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--short-tensor", type=Path, default=SHORT_TENSOR_PATH)
    parser.add_argument("--long-tensor", type=Path, default=LONG_TENSOR_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--fault-class-weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mask-mode", choices=MASK_MODES, default=MASK_MODE_PER_HOUR)
    return parser


def _matching_eligible_metadata(short_path: Path, long_path: Path) -> tuple[dict[str, object], int]:
    short = load_hourly_metadata(short_path)
    long = load_hourly_metadata(long_path)
    assert_matching_metadata(short, long)
    short_examples = {
        "X_cont": short["y_binary"].reshape(-1, 1, 1).astype("float32"),
        "mask": short["y_binary"].reshape(-1, 1, 1).astype("float32"),
        "time_since_last": short["y_binary"].reshape(-1, 1, 1).astype("float32"),
        "static": short["y_binary"].reshape(-1, 1).astype("float32"),
        "rule_evidence": short["y_binary"].reshape(-1, 1).astype("float32"),
        "y_binary": short["y_binary"],
        "station_id": short["station_id"],
        "hour": short["hour"],
        "display_state": short["display_state"],
        "source_episode_ids": short["source_episode_ids"],
        "continuous_feature_names": ["metadata"],
        "static_feature_names": ["metadata"],
        "rule_evidence_feature_names": ["metadata"],
    }
    eligible, excluded = filter_eligible_examples(short_examples)
    return eligible, excluded


def _load_filtered_tensor(path: Path) -> tuple[dict[str, object], int]:
    examples = load_hourly_tensor(path)
    return filter_eligible_examples(examples)


def _reason_code_station_hour_keys(stations: np.ndarray, hours: np.ndarray) -> np.ndarray:
    station_values = np.asarray(stations, dtype=object).astype(str)
    parsed = pd.to_datetime(pd.Series(hours), utc=True, format="mixed")
    ticks = parsed.astype("int64").to_numpy(dtype=np.int64)
    return np.asarray(
        [f"{station}\x1f{int(tick)}" for station, tick in zip(station_values, ticks)],
        dtype=object,
    )


def _verify_saved_prediction_artifact(
    saved_rows: pd.DataFrame,
    inferred_rows: pd.DataFrame,
    probability_tolerance: float = 2e-6,
    threshold_tolerance: float = 1e-10,
) -> dict[str, dict[str, object]]:
    identity = ["method", "split_scheme", "axis", "label"]
    required = {
        *identity,
        "station_id",
        "hour",
        "source_episode_ids",
        "truth",
        "probability",
        "threshold",
        "hourly_prediction",
    }
    missing_saved = sorted(required.difference(saved_rows.columns))
    missing_inferred = sorted(required.difference(inferred_rows.columns))
    if missing_saved or missing_inferred:
        raise KeyError(
            f"saved/inferred reason-code rows lack fields: saved={missing_saved}, inferred={missing_inferred}"
        )
    left = saved_rows.loc[:, sorted(required)].copy()
    right = inferred_rows.loc[:, sorted(required)].copy()
    for frame in (left, right):
        frame["_sample_key"] = _reason_code_station_hour_keys(
            frame["station_id"].to_numpy(), frame["hour"].to_numpy()
        ).astype(str)
    keys = [*identity, "_sample_key"]
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("saved/inferred reason-code prediction rows duplicate a label-level station-hour key")
    compared = left.merge(right, on=keys, how="outer", suffixes=("_saved", "_inferred"), indicator=True)
    if not compared["_merge"].eq("both").all():
        raise ValueError("saved reason-code predictions and loaded-model inference have different row membership")
    details: dict[str, dict[str, object]] = {}
    for method, frame in compared.groupby("method", sort=True):
        same_source = frame["source_episode_ids_saved"].astype(str).eq(
            frame["source_episode_ids_inferred"].astype(str)
        )
        same_truth = pd.to_numeric(frame["truth_saved"], errors="raise").eq(
            pd.to_numeric(frame["truth_inferred"], errors="raise")
        )
        same_prediction = pd.to_numeric(frame["hourly_prediction_saved"], errors="raise").eq(
            pd.to_numeric(frame["hourly_prediction_inferred"], errors="raise")
        )
        probability_delta = np.abs(
            pd.to_numeric(frame["probability_saved"], errors="raise").to_numpy(dtype=float)
            - pd.to_numeric(frame["probability_inferred"], errors="raise").to_numpy(dtype=float)
        )
        threshold_delta = np.abs(
            pd.to_numeric(frame["threshold_saved"], errors="raise").to_numpy(dtype=float)
            - pd.to_numeric(frame["threshold_inferred"], errors="raise").to_numpy(dtype=float)
        )
        max_probability_delta = float(np.max(probability_delta)) if len(probability_delta) else 0.0
        max_threshold_delta = float(np.max(threshold_delta)) if len(threshold_delta) else 0.0
        if (
            not same_source.all()
            or not same_truth.all()
            or not same_prediction.all()
            or max_probability_delta > float(probability_tolerance)
            or max_threshold_delta > float(threshold_tolerance)
        ):
            raise ValueError(
                f"saved {method} reason-code predictions do not match inference from the loaded artifact"
            )
        details[str(method)] = {
            "rows": int(len(frame)),
            "source_episode_ids_match": True,
            "truth_match": True,
            "binary_prediction_match": True,
            "max_abs_probability_delta": max_probability_delta,
            "probability_tolerance": float(probability_tolerance),
            "max_abs_threshold_delta": max_threshold_delta,
            "threshold_tolerance": float(threshold_tolerance),
        }
    return details


def _frozen_selection_test_metrics(
    aggregated_test_rows: pd.DataFrame,
    mechanism_selection: dict[str, object],
) -> pd.DataFrame:
    required = {
        "selected_method",
        "selected_aggregation_rule",
        "selection_split_scheme",
        "axis",
        "selection_labels",
    }
    missing = sorted(required.difference(mechanism_selection))
    if missing:
        raise KeyError(f"reason-code selection trace lacks fields: {missing}")
    if str(mechanism_selection["selection_split_scheme"]) != "spaced":
        raise ValueError("frozen reason-code test evaluation requires the spaced selection split")
    if str(mechanism_selection["axis"]) != "mechanism":
        raise ValueError("frozen reason-code test evaluation is limited to mechanism labels")
    labels = [str(value) for value in mechanism_selection["selection_labels"]]
    if not labels:
        raise ValueError("frozen reason-code test evaluation has no selected labels")
    selected = aggregated_test_rows.loc[
        aggregated_test_rows["method"].eq(str(mechanism_selection["selected_method"]))
        & aggregated_test_rows["split_scheme"].eq("spaced")
        & aggregated_test_rows["aggregation_rule"].eq(
            str(mechanism_selection["selected_aggregation_rule"])
        )
        & aggregated_test_rows["axis"].eq("mechanism")
        & aggregated_test_rows["label"].isin(labels)
    ].copy()
    if set(selected["label"].astype(str)) != set(labels):
        raise ValueError("frozen reason-code test evaluation is missing a selected mechanism label")
    metrics = pd.DataFrame(reason_code_aggregated_metric_rows(selected))
    per_label = metrics.loc[metrics["average"].eq("per_label")]
    supported_labels = sorted(
        per_label.loc[per_label["estimated"].astype(bool), "label"].astype(str).tolist()
    )
    metrics["selection_evaluation_scope"] = (
        "descriptive_heldout_test_of_validation_frozen_configuration"
    )
    metrics["selected_method"] = str(mechanism_selection["selected_method"])
    metrics["selected_aggregation_rule"] = str(
        mechanism_selection["selected_aggregation_rule"]
    )
    metrics["frozen_selection_labels"] = "|".join(labels)
    metrics["frozen_selection_label_count"] = int(len(labels))
    metrics["frozen_selection_labels_with_test_support"] = int(len(supported_labels))
    metrics["frozen_selection_labels_without_test_support"] = "|".join(
        label for label in labels if label not in set(supported_labels)
    )
    metrics["test_metrics_used_for_selection"] = False
    return metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args().parse_args(argv)
    require_files(
        "Hourly baseline training",
        {
            "short hourly tensor": args.short_tensor,
            "long hourly tensor": args.long_tensor,
        },
        "Run scripts/build_hourly_dataset.py before training.",
    )
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    metadata, metadata_excluded = _matching_eligible_metadata(args.short_tensor, args.long_tensor)
    labels = metadata["y_binary"]
    resolved_weight = resolve_fault_class_weight(labels, args.fault_class_weight)
    config = HourlyBaselineConfig(seed=int(args.seed), fault_class_weight=resolved_weight)
    tensors = (("short", args.short_tensor), ("long", args.long_tensor))
    all_rows = []
    all_split_maps = None
    spaced_detail = None
    feature_names_by_window: dict[str, list[str]] = {}
    model_paths: dict[str, str] = {}
    for window_name, tensor_path in tensors:
        examples, excluded = _load_filtered_tensor(tensor_path)
        if excluded != metadata_excluded:
            raise ValueError("hourly tensors disagree on eligible example count")
        assert_matching_metadata(metadata, examples)
        values, feature_names, _ = flatten_hourly_features(examples, mask_mode=args.mask_mode)
        rows, models, split_maps, current_spaced_detail = run_baseline_matrix(
            values,
            examples["y_binary"],
            examples["station_id"],
            examples["hour"],
            examples["display_state"],
            examples["source_episode_ids"],
            window_name,
            config,
        )
        if all_split_maps is None:
            all_split_maps = split_maps
            spaced_detail = current_spaced_detail
        else:
            for split_name in ("random", "spaced"):
                for partition in ("train", "validation", "test"):
                    if not (all_split_maps[split_name][partition] == split_maps[split_name][partition]).all():
                        raise ValueError("short and long tensors received different split membership")
        for row in rows:
            run_name = str(row["run"])
            path = model_dir / f"{run_name.replace('-', '_')}.joblib"
            save_model_bundle(
                models[run_name],
                path,
                feature_names,
                int(examples["X_cont"].shape[1]),
                config,
                str(row["split_scheme"]),
            )
            model_paths[run_name] = str(path)
        feature_names_by_window[window_name] = feature_names
        all_rows.extend(rows)
        del examples
        del values
        del models
        gc.collect()
    if all_split_maps is None or spaced_detail is None:
        raise RuntimeError("no baseline runs were completed")
    importance_reference = validation_importance_reference(all_rows)
    reference_window = str(importance_reference["window"])
    reference_path = args.short_tensor if reference_window == "short" else args.long_tensor
    reference_examples, _ = _load_filtered_tensor(reference_path)
    reference_values, _, reference_groups = flatten_hourly_features(
        reference_examples,
        mask_mode=args.mask_mode,
    )
    reference_splits = all_split_maps[str(importance_reference["split_scheme"])]
    reference_model = joblib.load(model_paths[str(importance_reference["run"])])["estimator"]
    importance = grouped_permutation_importance(
        reference_model,
        reference_values[reference_splits["validation"]],
        reference_examples["y_binary"][reference_splits["validation"]],
        reference_groups,
        config.threshold,
        config.seed,
    )
    manifest = make_split_manifest(
        all_split_maps,
        metadata["y_binary"],
        metadata["station_id"],
        metadata["hour"],
        metadata["display_state"],
        metadata["source_episode_ids"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_PATH.name
    metrics_path = output_dir / METRICS_PATH.name
    report_path = output_dir / REPORT_PATH.name
    importance_path = output_dir / IMPORTANCE_PATH.name
    manifest.to_csv(manifest_path, index=False)
    pd.DataFrame(importance).to_csv(importance_path, index=False)
    report = baseline_report(
        all_rows,
        importance,
        config,
        spaced_detail,
        importance_reference,
    )
    report_path.write_text(report + "\n", encoding="utf-8")
    payload = {
        "configuration": asdict(config),
        "mask_mode": args.mask_mode,
        "excluded_examples_removed": int(metadata_excluded),
        "runs": all_rows,
        "importance_reference": importance_reference,
        "importance_reference_selection": "validation_f1_then_recall_then_precision",
        "grouped_permutation_importance": importance,
        "spaced_split": spaced_detail,
        "model_paths": model_paths,
        "manifest_path": str(manifest_path),
        "feature_dimensions": {name: len(names) for name, names in feature_names_by_window.items()},
    }
    write_metrics_json(payload, metrics_path)
    print(report)
    print()
    print("OUTPUTS")
    print(f"metrics_json={metrics_path}")
    print(f"report={report_path}")
    print(f"feature_importance={importance_path}")
    print(f"split_manifest={manifest_path}")
    for run_name in sorted(model_paths):
        print(f"model_{run_name}={model_paths[run_name]}")


if __name__ == "__main__":
    main()
