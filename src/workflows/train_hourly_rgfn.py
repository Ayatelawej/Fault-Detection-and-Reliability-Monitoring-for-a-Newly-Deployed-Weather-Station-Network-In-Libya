from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.hourly_baseline import (
    EvidenceFusedHgbClassifier,
    HourlyBaselineConfig,
    binary_metrics,
    filter_eligible_examples,
    flatten_hourly_features,
    load_hourly_tensor,
    make_classifier,
    save_model_bundle,
    write_metrics_json,
)
from src.model.hourly_detection import MASK_MODE_PER_HOUR, MASK_MODES, SHORT_TENSOR_PATH
from src.model.hourly_calibration import CALIBRATION_THRESHOLDS, CALIBRATION_WEIGHTS
from src.model.hourly_rgfn import ENCODER_CONV, ENCODER_GRU, ENCODER_MLP
from src.model.hourly_rgfn_training import (
    DEVICE,
    RGFN_SEEDS,
    RGFN_THRESHOLDS,
    RGFN_WEIGHTS,
    HourlyRgfnTrainingConfig,
    comparison_report,
    load_calibrated_baseline,
    load_manifest_splits,
    master_comparison_frame,
    spaced_gap_frame,
    target_frame,
    train_hourly_rgfn_variant,
)
from src.workflows.prerequisites import require_files


OUTPUT_DIR = PROJECT_ROOT / "data" / "hourly_detection"
MANIFEST_PATH = OUTPUT_DIR / "hourly_baseline_split_manifest.csv"
BASELINE_METRICS_PATH = OUTPUT_DIR / "hourly_short_calibration_metrics.json"
METRICS_PATH = OUTPUT_DIR / "hourly_rgfn_comparison_metrics.json"
REPORT_PATH = OUTPUT_DIR / "hourly_rgfn_comparison_report.txt"
SWEEP_PATH = OUTPUT_DIR / "hourly_rgfn_comparison_validation_sweep.csv"
ONE_HOUR_OUTPUT_DIR = OUTPUT_DIR / "one_hour_final"
ONE_HOUR_JULY_OUTPUT_DIR = PROJECT_ROOT / "data" / "eval" / "one_hour_candidate"
FUSION_WEIGHTS = tuple(value / 10.0 for value in range(11))


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--tensor", type=Path, default=SHORT_TENSOR_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--baseline-metrics", type=Path, default=BASELINE_METRICS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--mask-mode", choices=MASK_MODES, default=MASK_MODE_PER_HOUR)
    return parser


def parse_one_hour_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ONE_HOUR_OUTPUT_DIR)
    parser.add_argument("--mask-mode", choices=MASK_MODES, default=MASK_MODE_PER_HOUR)
    return parser


def parse_one_hour_july_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ONE_HOUR_JULY_OUTPUT_DIR)
    parser.add_argument("--mask-mode", choices=MASK_MODES, default=MASK_MODE_PER_HOUR)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _station_hour_keys(stations: np.ndarray, hours: np.ndarray) -> np.ndarray:
    parsed = pd.to_datetime(pd.Series(hours), utc=True, format="mixed")
    ticks = parsed.astype("int64").to_numpy(dtype=np.int64)
    return np.asarray(
        [f"{station}\x1f{int(tick)}" for station, tick in zip(np.asarray(stations).astype(str), ticks)],
        dtype=object,
    )


def _comparison_metric_row(
    model: str,
    split: str,
    validation: dict[str, object],
    test: dict[str, object],
    validation_std: dict[str, object] | None = None,
    test_std: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {"model": model, "split": split}
    for partition, values, deviations in (
        ("validation", validation, validation_std),
        ("test", test, test_std),
    ):
        for metric in ("precision", "recall", "f1", "accuracy"):
            row[f"{partition}_{metric}"] = float(values[metric])
            row[f"{partition}_{metric}_std"] = None if deviations is None else float(deviations[metric])
    row["validation_minimum_metric"] = min(
        float(validation[name]) for name in ("precision", "recall", "f1")
    )
    return row


def _model_selection_key(row: dict[str, object]) -> tuple[float, ...]:
    return (
        float(row["validation_minimum_metric"]),
        float(row["validation_f1"]),
        float(row["validation_precision"]),
        float(row["validation_recall"]),
        1.0 if str(row["model"]) == "HGB" else 0.0,
    )


def select_one_hour_model_by_validation(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("one-hour model selection requires validation results")
    return max(rows, key=_model_selection_key).copy()


def evidence_fusion_feature_views(groups: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    context = sorted(
        int(index)
        for name, indices in groups.items()
        if name.startswith(("continuous:", "mask:", "time_since_last:", "static:"))
        for index in np.asarray(indices, dtype=np.int64)
    )
    rules = sorted(
        int(index)
        for name, indices in groups.items()
        if name.startswith(("rule:", "static:"))
        for index in np.asarray(indices, dtype=np.int64)
    )
    if not context or not rules:
        raise ValueError("evidence fusion requires non-empty context and rule feature views")
    return np.asarray(context, dtype=np.int64), np.asarray(rules, dtype=np.int64)


def select_evidence_fusion_by_validation(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("evidence-fusion selection requires validation results")
    return max(
        rows,
        key=lambda row: (
            float(row["validation_minimum_metric"]),
            float(row["validation_f1"]),
            float(row["validation_precision"]),
            float(row["validation_recall"]),
            float(row["full_weight"]),
        ),
    ).copy()


def _one_hour_report(
    comparison: pd.DataFrame,
    selections: dict[str, dict[str, object]],
    rgfn: dict[str, dict[str, object]],
) -> str:
    columns = [
        "model",
        "split",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_accuracy",
    ]
    parts = [
        "ONE-HOUR HGB AND RGFN COMPARISON",
        "",
        "Both models use the same current-hour engineered input population and exact split manifest.",
        "RGFN uses an MLP sensor encoder and retains its independent evidence branch and learned reliability gate.",
        "Class weights and thresholds are selected from validation only. Test metrics do not enter model selection.",
        "",
        "FULL COMPARISON",
        comparison.loc[:, columns].to_string(index=False),
        "",
        "VALIDATION-ONLY MODEL SELECTION",
    ]
    for split in sorted(selections):
        selected = selections[split]
        parts.append(
            f"{split}: {selected['model']} "
            f"(validation minimum={float(selected['validation_minimum_metric']):.6f}, "
            f"F1={float(selected['validation_f1']):.6f})"
        )
    parts.extend(["", "RGFN SELECTED OPERATING POINTS"])
    for split in ("random", "spaced"):
        point = rgfn[split]["best_balanced"]
        parts.append(
            f"{split}: class_weight={float(point['fault_class_weight']):.1f}, "
            f"threshold={float(point['threshold']):.2f}, "
            f"seeds={int(rgfn[split]['seed_count'])}"
        )
    return "\n".join(parts)


def one_hour_main(argv: list[str] | None = None) -> None:
    args = parse_one_hour_args().parse_args(argv)
    require_files(
        "One-hour HGB/RGFN comparison",
        {
            "one-hour tensor": args.tensor,
            "split manifest": args.manifest,
            "one-hour HGB metrics": args.baseline_metrics,
        },
        "Build the one-hour tensor and run calibration with saved models first.",
    )
    examples, excluded_removed = filter_eligible_examples(load_hourly_tensor(Path(args.tensor)))
    window_hours = int(np.asarray(examples["window_hours"]).reshape(-1)[0])
    if window_hours != 1:
        raise ValueError("one-hour comparison requires a tensor with window_hours=1")
    splits = load_manifest_splits(examples, Path(args.manifest))
    with Path(args.baseline_metrics).open("r", encoding="utf-8") as handle:
        baseline_payload = json.load(handle)
    baseline_results = baseline_payload.get("results")
    baseline_models = baseline_payload.get("selected_model_paths")
    if not isinstance(baseline_results, dict) or not isinstance(baseline_models, dict):
        raise KeyError("one-hour HGB metrics lack results or saved model paths")
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models" / "rgfn"
    training_config = HourlyRgfnTrainingConfig(mask_mode=args.mask_mode)
    rgfn: dict[str, dict[str, object]] = {}

    def progress(item: dict[str, object]) -> None:
        print(
            "candidate_complete="
            f"RGFN:{item['split']}:seed={item['seed']}:"
            f"weight={float(item['fault_class_weight']):.1f}:"
            f"epochs={item['epochs_completed']}:best_epoch={item['best_epoch']}:"
            f"validation_loss={float(item['best_validation_loss']):.6f}",
            flush=True,
        )

    print(f"device={DEVICE}", flush=True)
    for split in ("random", "spaced"):
        rgfn[split] = train_hourly_rgfn_variant(
            examples=examples,
            splits=splits[split],
            encoder=ENCODER_MLP,
            split_name=split,
            model_dir=model_dir,
            seeds=RGFN_SEEDS,
            weights=CALIBRATION_WEIGHTS,
            thresholds=RGFN_THRESHOLDS,
            base_config=training_config,
            progress=progress,
        )
    rows: list[dict[str, object]] = []
    for split in ("random", "spaced"):
        baseline = baseline_results[split]
        rows.append(
            _comparison_metric_row(
                "HGB", split, baseline["best_balanced"]["validation"], baseline["final_test"]
            )
        )
        validation_summary = rgfn[split]["selected_validation_summary"]
        test_summary = rgfn[split]["test_summary"]
        rows.append(
            _comparison_metric_row(
                "RGFN",
                split,
                validation_summary,
                test_summary,
                {name: validation_summary[f"{name}_std"] for name in ("precision", "recall", "f1", "accuracy")},
                {name: test_summary[f"{name}_std"] for name in ("precision", "recall", "f1", "accuracy")},
            )
        )
    comparison = pd.DataFrame(rows).sort_values(["split", "model"]).reset_index(drop=True)
    selections = {
        split: select_one_hour_model_by_validation(
            comparison.loc[comparison["split"].eq(split)].to_dict("records")
        )
        for split in ("random", "spaced")
    }
    comparison["selected_by_validation"] = comparison.apply(
        lambda row: str(row["model"]) == str(selections[str(row["split"])]["model"]),
        axis=1,
    )
    report = _one_hour_report(comparison, selections, rgfn)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "one_hour_model_comparison.csv"
    metrics_path = output_dir / "one_hour_model_comparison_metrics.json"
    report_path = output_dir / "one_hour_model_comparison_report.txt"
    sweep_path = output_dir / "one_hour_rgfn_validation_sweep.csv"
    comparison.to_csv(comparison_path, index=False)
    _sweep_frame({"RGFN": rgfn}).to_csv(sweep_path, index=False)
    report_path.write_text(report + "\n", encoding="utf-8")
    write_metrics_json(
        _jsonable(
            {
                "configuration": {
                    "task": "one-hour binary fault detection",
                    "rgfn_encoder": ENCODER_MLP,
                    "training": asdict(training_config),
                    "weights": list(CALIBRATION_WEIGHTS),
                    "thresholds": list(RGFN_THRESHOLDS),
                    "seeds": list(RGFN_SEEDS),
                    "selection_source": "validation_only",
                },
                "inputs": {
                    "tensor": str(args.tensor),
                    "manifest": str(args.manifest),
                    "baseline_metrics": str(args.baseline_metrics),
                    "baseline_models": baseline_models,
                    "eligible_examples": int(len(examples["y_binary"])),
                    "excluded_examples_removed": int(excluded_removed),
                },
                "comparison": comparison.to_dict("records"),
                "selected_models": selections,
                "rgfn": rgfn,
            }
        ),
        metrics_path,
    )
    print(report, flush=True)
    print(f"comparison={comparison_path}", flush=True)
    print(f"metrics={metrics_path}", flush=True)
    print(f"report={report_path}", flush=True)
    print(f"validation_sweep={sweep_path}", flush=True)
    print(f"models={model_dir}", flush=True)


def evidence_fusion_main(argv: list[str] | None = None) -> None:
    args = parse_one_hour_args().parse_args(argv)
    require_files(
        "One-hour evidence-fused HGB",
        {
            "one-hour tensor": args.tensor,
            "split manifest": args.manifest,
            "one-hour HGB metrics": args.baseline_metrics,
        },
        "Build the one-hour tensor and run calibration with saved HGB models first.",
    )
    examples, excluded_removed = filter_eligible_examples(load_hourly_tensor(Path(args.tensor)))
    window_hours = int(np.asarray(examples["window_hours"]).reshape(-1)[0])
    if window_hours != 1:
        raise ValueError("evidence-fused HGB requires a one-hour tensor")
    values, feature_names, groups = flatten_hourly_features(examples, mask_mode=args.mask_mode)
    labels = np.asarray(examples["y_binary"], dtype=np.int64)
    splits = load_manifest_splits(examples, Path(args.manifest))
    context_indices, rule_indices = evidence_fusion_feature_views(groups)
    with Path(args.baseline_metrics).open("r", encoding="utf-8") as handle:
        baseline_payload = json.load(handle)
    baseline_results = baseline_payload.get("results")
    baseline_paths = baseline_payload.get("selected_model_paths")
    if not isinstance(baseline_results, dict) or not isinstance(baseline_paths, dict):
        raise KeyError("one-hour HGB metrics lack results or saved model paths")
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models" / "evidence_fusion"
    comparison_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    selections: dict[str, dict[str, object]] = {}
    model_paths: dict[str, str] = {}

    for split_name in ("random", "spaced"):
        baseline_bundle = joblib.load(Path(baseline_paths[split_name]))
        if list(baseline_bundle["feature_names"]) != feature_names:
            raise ValueError(f"{split_name} HGB feature order differs from the one-hour tensor")
        baseline_config = HourlyBaselineConfig(**baseline_bundle["config"])
        split = splits[split_name]
        context_estimator = make_classifier(baseline_config)
        rule_estimator = make_classifier(baseline_config)
        context_estimator.fit(values[split["train"]][:, context_indices], labels[split["train"]])
        rule_estimator.fit(values[split["train"]][:, rule_indices], labels[split["train"]])
        full_estimator = baseline_bundle["estimator"]
        validation_probabilities = {
            "full": full_estimator.predict_proba(values[split["validation"]])[:, 1],
            "context": context_estimator.predict_proba(
                values[split["validation"]][:, context_indices]
            )[:, 1],
            "rules": rule_estimator.predict_proba(values[split["validation"]][:, rule_indices])[:, 1],
        }
        candidates: list[dict[str, object]] = []
        for full_weight in FUSION_WEIGHTS:
            for context_weight in FUSION_WEIGHTS:
                if full_weight + context_weight > 1.0:
                    continue
                rule_weight = 1.0 - full_weight - context_weight
                probability = (
                    full_weight * validation_probabilities["full"]
                    + context_weight * validation_probabilities["context"]
                    + rule_weight * validation_probabilities["rules"]
                )
                for threshold in CALIBRATION_THRESHOLDS:
                    metric = binary_metrics(labels[split["validation"]], probability, threshold)
                    row = {
                        "split": split_name,
                        "full_weight": float(full_weight),
                        "context_weight": float(context_weight),
                        "rule_weight": float(rule_weight),
                        "threshold": float(threshold),
                        **{f"validation_{name}": value for name, value in metric.items()},
                    }
                    row["validation_minimum_metric"] = min(
                        float(metric[name]) for name in ("precision", "recall", "f1")
                    )
                    candidates.append(row)
        selected = select_evidence_fusion_by_validation(candidates)
        selections[split_name] = selected
        for row in candidates:
            row["selected"] = all(
                np.isclose(float(row[name]), float(selected[name]))
                for name in ("full_weight", "context_weight", "rule_weight", "threshold")
            )
        validation_rows.extend(candidates)
        ensemble = EvidenceFusedHgbClassifier(
            full_estimator=full_estimator,
            context_estimator=context_estimator,
            rule_estimator=rule_estimator,
            context_indices=context_indices,
            rule_indices=rule_indices,
            full_weight=float(selected["full_weight"]),
            context_weight=float(selected["context_weight"]),
            rule_weight=float(selected["rule_weight"]),
        )
        selected_config = replace(baseline_config, threshold=float(selected["threshold"]))
        model_path = model_dir / f"selected_ef_hgb_{split_name}_01h.joblib"
        save_model_bundle(
            ensemble,
            model_path,
            feature_names,
            1,
            selected_config,
            split_name,
            extra_metadata={
                "model_name": "EF-HGB",
                "fusion_weights": {
                    "full": float(selected["full_weight"]),
                    "context": float(selected["context_weight"]),
                    "rules": float(selected["rule_weight"]),
                },
                "context_feature_names": [feature_names[index] for index in context_indices],
                "rule_feature_names": [feature_names[index] for index in rule_indices],
                "selection_source": "validation_only",
            },
        )
        model_paths[split_name] = str(model_path)
        test_probability = ensemble.predict_proba(values[split["test"]])[:, 1]
        fusion_test = binary_metrics(labels[split["test"]], test_probability, selected["threshold"])
        baseline = baseline_results[split_name]
        comparison_rows.append(
            _comparison_metric_row(
                "HGB", split_name, baseline["best_balanced"]["validation"], baseline["final_test"]
            )
        )
        comparison_rows.append(
            _comparison_metric_row(
                "EF-HGB",
                split_name,
                {name: selected[f"validation_{name}"] for name in ("precision", "recall", "f1", "accuracy")},
                fusion_test,
            )
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(["split", "model"]).reset_index(drop=True)
    selected_models = {
        split: select_one_hour_model_by_validation(
            comparison.loc[comparison["split"].eq(split)].to_dict("records")
        )
        for split in ("random", "spaced")
    }
    comparison["selected_by_validation"] = comparison.apply(
        lambda row: str(row["model"]) == str(selected_models[str(row["split"])]["model"]),
        axis=1,
    )
    validation_grid = pd.DataFrame(validation_rows).sort_values(
        ["split", "full_weight", "context_weight", "threshold"], kind="stable"
    )
    display = comparison.loc[
        :,
        [
            "model",
            "split",
            "validation_precision",
            "validation_recall",
            "validation_f1",
            "validation_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_accuracy",
            "selected_by_validation",
        ],
    ]
    report = "\n".join(
        [
            "ONE-HOUR EVIDENCE-FUSED HGB",
            "",
            "The full, context, and rule HGB branches use the same training partition.",
            "Fusion weights and the operating threshold are selected from validation only.",
            "Test is evaluated once after the fusion is frozen.",
            "",
            display.to_string(index=False),
            "",
            "VALIDATION-SELECTED FUSION",
            *[
                f"{split}: full={float(selections[split]['full_weight']):.1f}, "
                f"context={float(selections[split]['context_weight']):.1f}, "
                f"rules={float(selections[split]['rule_weight']):.1f}, "
                f"threshold={float(selections[split]['threshold']):.2f}"
                for split in ("random", "spaced")
            ],
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "evidence_fusion_comparison.csv"
    grid_path = output_dir / "evidence_fusion_validation_grid.csv"
    metrics_path = output_dir / "evidence_fusion_metrics.json"
    report_path = output_dir / "evidence_fusion_report.txt"
    display.to_csv(comparison_path, index=False)
    validation_grid.to_csv(grid_path, index=False)
    report_path.write_text(report + "\n", encoding="utf-8")
    write_metrics_json(
        _jsonable(
            {
                "model": "EF-HGB",
                "tensor": str(args.tensor),
                "manifest": str(args.manifest),
                "baseline_metrics": str(args.baseline_metrics),
                "eligible_examples": int(len(labels)),
                "excluded_examples_removed": int(excluded_removed),
                "feature_views": {
                    "full": feature_names,
                    "context": [feature_names[index] for index in context_indices],
                    "rules": [feature_names[index] for index in rule_indices],
                },
                "selection_source": "validation_only",
                "selection_objective": "maximum validation minimum(precision, recall, f1)",
                "selected_fusions": selections,
                "selected_models": selected_models,
                "model_paths": model_paths,
                "comparison": display.to_dict("records"),
                "test_evaluations_per_split": 1,
            }
        ),
        metrics_path,
    )
    print(report)
    print(f"comparison={comparison_path}")
    print(f"validation_grid={grid_path}")
    print(f"metrics={metrics_path}")
    print(f"report={report_path}")
    for split, path in model_paths.items():
        print(f"selected_{split}_model={path}")


def one_hour_july_main(argv: list[str] | None = None) -> None:
    args = parse_one_hour_july_args().parse_args(argv)
    require_files(
        "Frozen one-hour July evaluation",
        {
            "combined one-hour tensor": args.tensor,
            "validation-selected detector model": args.model,
            "frozen July evaluation ledger": args.reference_ledger,
        },
        "Build the combined one-hour tensor and complete validation-only model selection first.",
    )
    examples, excluded_removed = filter_eligible_examples(load_hourly_tensor(Path(args.tensor)))
    window_hours = int(np.asarray(examples["window_hours"]).reshape(-1)[0])
    if window_hours != 1:
        raise ValueError("frozen one-hour July evaluation requires window_hours=1")
    values, feature_names, _ = flatten_hourly_features(examples, mask_mode=args.mask_mode)
    bundle = joblib.load(args.model)
    if int(bundle["window_hours"]) != 1 or str(bundle["split_scheme"]) != "random":
        raise ValueError("July evaluation requires a validation-selected random one-hour detector")
    if list(bundle["feature_names"]) != list(feature_names):
        raise ValueError("July tensor feature order differs from the saved one-hour HGB")
    ledger = pd.read_parquet(args.reference_ledger)
    required = {"station_id", "hour_utc", "truth_fault", "display_state", "source_episode_ids"}
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise KeyError(f"frozen July evaluation ledger lacks fields: {missing}")
    tensor_keys = _station_hour_keys(examples["station_id"], examples["hour"])
    if len(np.unique(tensor_keys)) != len(tensor_keys):
        raise ValueError("combined one-hour tensor has duplicate station-hour keys")
    tensor_index = {str(key): index for index, key in enumerate(tensor_keys)}
    ledger_keys = _station_hour_keys(
        ledger["station_id"].to_numpy(), ledger["hour_utc"].to_numpy()
    )
    mapped = np.asarray([tensor_index.get(str(key), -1) for key in ledger_keys], dtype=np.int64)
    if np.any(mapped < 0) or len(np.unique(mapped)) != len(mapped):
        raise ValueError("one-hour tensor does not exactly cover the frozen July ledger keys")
    truth = pd.to_numeric(ledger["truth_fault"], errors="raise").to_numpy(dtype=np.int64)
    tensor_truth = np.asarray(examples["y_binary"], dtype=np.int64)[mapped]
    if not np.array_equal(truth, tensor_truth):
        raise ValueError("one-hour tensor truth disagrees with the frozen July ledger")
    threshold = float(bundle["config"]["threshold"])
    model_name = str(bundle.get("model_name", "HGB"))
    model_slug = model_name.lower().replace("-", "_")
    probabilities = bundle["estimator"].predict_proba(values[mapped])[:, 1]
    predictions = np.greater_equal(probabilities, threshold).astype(np.int64)
    metric = binary_metrics(truth, probabilities, threshold)
    output = ledger.loc[:, ["station_id", "hour_utc", "truth_fault", "display_state", "source_episode_ids"]].copy()
    output["random_probability"] = probabilities
    output["random_prediction"] = predictions
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"july_{model_slug}_binary_predictions.parquet"
    metrics_path = output_dir / f"july_{model_slug}_binary_metrics.csv"
    manifest_path = output_dir / f"july_{model_slug}_scoring_manifest.json"
    report_path = output_dir / f"july_{model_slug}_scoring_report.txt"
    output.to_parquet(predictions_path, index=False)
    metric_row = {
        "configuration": f"random_one_hour_{model_slug}",
        "threshold": threshold,
        "rows": int(len(output)),
        "fault_support": int(truth.sum()),
        "not_fault_support": int(len(truth) - truth.sum()),
        **metric,
        "auroc": float(roc_auc_score(truth, probabilities)),
        "auprc": float(average_precision_score(truth, probabilities)),
        "model_sha256": _sha256(args.model),
    }
    pd.DataFrame([metric_row]).to_csv(metrics_path, index=False)
    report = "\n".join(
        [
            f"FROZEN ONE-HOUR {model_name} JULY EVALUATION",
            "",
            "The model family, class weight, and threshold were selected from development validation before this July evaluation.",
            "July was scored once without fitting, calibration, or threshold changes.",
            "",
            pd.DataFrame([metric_row]).to_string(index=False),
        ]
    )
    report_path.write_text(report + "\n", encoding="utf-8")
    write_metrics_json(
        {
            "model": str(args.model),
            "model_name": model_name,
            "model_sha256": _sha256(args.model),
            "tensor": str(args.tensor),
            "tensor_sha256": _sha256(args.tensor),
            "reference_ledger": str(args.reference_ledger),
            "reference_ledger_sha256": _sha256(args.reference_ledger),
            "mask_mode": args.mask_mode,
            "window_hours": 1,
            "feature_dimension": int(values.shape[1]),
            "eligible_combined_examples": int(len(examples["y_binary"])),
            "excluded_examples_removed": int(excluded_removed),
            "july_keys_exactly_match_frozen_ledger": True,
            "truth_exactly_matches_frozen_ledger": True,
            "selection_source": "development_validation_only",
            "july_used_for_selection_or_tuning": False,
            "metrics": metric_row,
        },
        manifest_path,
    )
    print(report)
    print(f"predictions={predictions_path}")
    print(f"metrics={metrics_path}")
    print(f"manifest={manifest_path}")
    print(f"report={report_path}")


def _sweep_frame(results: dict[str, dict[str, dict[str, object]]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, by_split in results.items():
        for split_name, result in by_split.items():
            selected = result["best_balanced"]
            for item in result["validation_seed_rows"]:
                metric = item["validation"]
                rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "row_type": "seed",
                        "seed": item["seed"],
                        "fault_class_weight": item["fault_class_weight"],
                        "threshold": item["threshold"],
                        "precision": metric["precision"],
                        "recall": metric["recall"],
                        "f1": metric["f1"],
                        "accuracy": metric["accuracy"],
                        "minimum_metric": min(metric["precision"], metric["recall"], metric["f1"]),
                        "precision_std": None,
                        "recall_std": None,
                        "f1_std": None,
                        "best_epoch": item["best_epoch"],
                        "best_validation_loss": item["best_validation_loss"],
                        "selected": bool(
                            float(item["fault_class_weight"]) == float(selected["fault_class_weight"])
                            and float(item["threshold"]) == float(selected["threshold"])
                        ),
                    }
                )
            for item in result["validation_grid"]:
                metric = item["validation"]
                deviation = item["validation_std"]
                rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "row_type": "aggregate",
                        "seed": None,
                        "fault_class_weight": item["fault_class_weight"],
                        "threshold": item["threshold"],
                        "precision": metric["precision"],
                        "recall": metric["recall"],
                        "f1": metric["f1"],
                        "accuracy": metric["accuracy"],
                        "minimum_metric": item["validation_minimum_metric"],
                        "precision_std": deviation["precision"],
                        "recall_std": deviation["recall"],
                        "f1_std": deviation["f1"],
                        "best_epoch": None,
                        "best_validation_loss": None,
                        "selected": bool(
                            float(item["fault_class_weight"]) == float(selected["fault_class_weight"])
                            and float(item["threshold"]) == float(selected["threshold"])
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["model", "split", "row_type", "fault_class_weight", "threshold", "seed"],
        kind="stable",
    ).reset_index(drop=True)


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(name): _jsonable(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main(argv: list[str] | None = None) -> None:
    args = parse_args().parse_args(argv)
    require_files(
        "Hourly RGFN training",
        {
            "short hourly tensor": args.tensor,
            "baseline split manifest": args.manifest,
            "calibrated baseline metrics": args.baseline_metrics,
        },
        "Run baseline and calibration through scripts/train_hourly_detection.py first.",
    )
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models" / "rgfn_hourly"
    examples, excluded_removed = filter_eligible_examples(load_hourly_tensor(Path(args.tensor)))
    splits = load_manifest_splits(examples, Path(args.manifest))
    baseline = load_calibrated_baseline(Path(args.baseline_metrics))
    training_config = HourlyRgfnTrainingConfig(mask_mode=args.mask_mode)
    print(f"device={DEVICE}", flush=True)
    print(f"eligible_hourly_examples={len(examples['y_binary'])}", flush=True)
    print(f"excluded_examples_removed={excluded_removed}", flush=True)
    results: dict[str, dict[str, dict[str, object]]] = {"RGFN-GRU": {}, "RGFN-CONV": {}}

    def progress(item: dict[str, object]) -> None:
        print(
            "candidate_complete="
            f"{item['encoder']}:{item['split']}:seed={item['seed']}:"
            f"weight={float(item['fault_class_weight']):.1f}:"
            f"epochs={item['epochs_completed']}:best_epoch={item['best_epoch']}:"
            f"validation_loss={float(item['best_validation_loss']):.6f}",
            flush=True,
        )

    for model_name, encoder in (("RGFN-GRU", ENCODER_GRU), ("RGFN-CONV", ENCODER_CONV)):
        for split_name in ("random", "spaced"):
            print(f"training={model_name} split={split_name} seeds={len(RGFN_SEEDS)} weights={len(RGFN_WEIGHTS)}", flush=True)
            results[model_name][split_name] = train_hourly_rgfn_variant(
                examples=examples,
                splits=splits[split_name],
                encoder=encoder,
                split_name=split_name,
                model_dir=model_dir,
                base_config=training_config,
                progress=progress,
            )
    gru = results["RGFN-GRU"]
    conv = results["RGFN-CONV"]
    payload = {
        "configuration": {
            "task": "hour-level binary fault detection",
            "window_hours": 7,
            "device": str(DEVICE),
            "training": asdict(training_config),
            "weights": list(RGFN_WEIGHTS),
            "thresholds": list(RGFN_THRESHOLDS),
            "seeds": list(RGFN_SEEDS),
            "selection": "aggregate validation maximum of minimum precision, recall, and f1",
            "test_evaluation": "one evaluation per selected seed model",
        },
        "inputs": {
            "tensor": str(Path(args.tensor)),
            "split_manifest": str(Path(args.manifest)),
            "baseline_metrics": str(Path(args.baseline_metrics)),
            "excluded_examples_removed": int(excluded_removed),
            "eligible_hourly_examples": int(len(examples["y_binary"])),
            "mask_mode": args.mask_mode,
        },
        "baseline": baseline,
        "rgfn_gru": gru,
        "rgfn_conv": conv,
        "master_test_comparison": master_comparison_frame(baseline, gru, conv).to_dict(orient="records"),
        "spaced_f1_answer": spaced_gap_frame(baseline, gru, conv).to_dict(orient="records"),
        "test_target_check": target_frame(baseline, gru, conv).to_dict(orient="records"),
    }
    report = comparison_report(baseline, gru, conv)
    metrics_path = output_dir / METRICS_PATH.name
    report_path = output_dir / REPORT_PATH.name
    sweep_path = output_dir / SWEEP_PATH.name
    write_metrics_json(_jsonable(payload), metrics_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    _sweep_frame(results).to_csv(sweep_path, index=False)
    print(report, flush=True)
    print(f"metrics={metrics_path}", flush=True)
    print(f"report={report_path}", flush=True)
    print(f"validation_sweep={sweep_path}", flush=True)
    print(f"models={model_dir}", flush=True)


if __name__ == "__main__":
    main()
