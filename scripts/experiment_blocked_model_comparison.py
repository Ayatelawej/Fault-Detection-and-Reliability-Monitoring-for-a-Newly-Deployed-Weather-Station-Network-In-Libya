"""Compare HGB, RGFN, and EF-HGB on the frozen blocked-period split."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import argparse
import json
import os
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from scripts.experiment_blocked_fault_detection import blocked_split, validate_split
from src.model.hourly_baseline import (
    EvidenceFusedHgbClassifier,
    HourlyBaselineConfig,
    _fault_groups,
    binary_metrics,
    filter_eligible_examples,
    flatten_hourly_features,
    load_hourly_tensor,
    make_classifier,
)
from src.model.hourly_calibration import (
    CALIBRATION_THRESHOLDS,
    CALIBRATION_WEIGHTS,
    calibrate_split,
)
from src.model.hourly_rgfn import ENCODER_MLP
from src.model.hourly_rgfn_training import (
    DEVICE,
    HourlyRgfnTrainingConfig,
    train_hourly_rgfn_variant,
)
from src.workflows.train_hourly_rgfn import (
    FUSION_WEIGHTS,
    evidence_fusion_feature_views,
    select_evidence_fusion_by_validation,
)


def _json(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _row(model, validation, test, **extra):
    return {
        "model": model,
        **extra,
        **{f"validation_{key}": float(validation[key]) for key in ("precision", "recall", "f1", "accuracy")},
        **{f"test_{key}": float(test[key]) for key in ("precision", "recall", "f1", "accuracy")},
        "validation_minimum_metric": min(float(validation[key]) for key in ("precision", "recall", "f1")),
    }


def run(out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    tensor_path = ROOT / "data/hourly_detection/one_hour_final/hourly_detection_01h.npz"
    examples, excluded = filter_eligible_examples(load_hourly_tensor(tensor_path))
    values, feature_names, feature_groups = flatten_hourly_features(examples)
    labels = np.asarray(examples["y_binary"], dtype=np.int64)
    hours = pd.to_datetime(examples["hour"], utc=True)
    groups = np.asarray(
        [f"normal:{station}:{str(hour)[:10]}" for station, hour in zip(examples["station_id"], hours)],
        dtype=object,
    )
    for key, indices in _fault_groups(labels, examples["source_episode_ids"]).items():
        groups[indices] = "event:" + key
    splits = blocked_split(hours, groups)
    validate_split(hours, groups, splits)
    # Calibration utilities require a complete partition of their input. The
    # blocked design intentionally omits embargo/crossing-group rows, so remap
    # the retained population without changing any partition membership.
    retained = np.concatenate([splits[name] for name in ("train", "validation", "test")])
    remap = np.full(len(labels), -1, dtype=np.int64)
    remap[retained] = np.arange(len(retained), dtype=np.int64)
    splits = {name: remap[indices] for name, indices in splits.items()}
    examples = {
        key: (np.asarray(value)[retained] if np.asarray(value).ndim and len(np.asarray(value)) == len(labels) else value)
        for key, value in examples.items()
    }
    values = values[retained]
    labels = labels[retained]
    hours = hours[retained]
    groups = groups[retained]
    out.mkdir(parents=True)
    (out / "models").mkdir()

    # Plain HGB: repeat the established class-weight/threshold calibration grid.
    hgb = calibrate_split(values, labels, splits, HourlyBaselineConfig())
    hgb_choice = hgb["best_balanced"]
    hgb_config = replace(
        HourlyBaselineConfig(),
        fault_class_weight=float(hgb_choice["fault_class_weight"]),
        threshold=float(hgb_choice["threshold"]),
    )
    hgb_model = make_classifier(hgb_config)
    hgb_model.fit(values[splits["train"]], labels[splits["train"]])
    joblib.dump({"estimator": hgb_model, "config": asdict(hgb_config), "feature_names": feature_names}, out / "models/hgb.joblib")

    # EF-HGB: use the selected HGB full arm, then repeat the complete convex fusion grid.
    context_indices, rule_indices = evidence_fusion_feature_views(feature_groups)
    context_model = make_classifier(hgb_config)
    rule_model = make_classifier(hgb_config)
    context_model.fit(values[splits["train"]][:, context_indices], labels[splits["train"]])
    rule_model.fit(values[splits["train"]][:, rule_indices], labels[splits["train"]])
    valid = splits["validation"]
    branch_valid = np.column_stack(
        [
            hgb_model.predict_proba(values[valid])[:, 1],
            context_model.predict_proba(values[valid][:, context_indices])[:, 1],
            rule_model.predict_proba(values[valid][:, rule_indices])[:, 1],
        ]
    )
    fusion_grid = []
    for full_weight in FUSION_WEIGHTS:
        for context_weight in FUSION_WEIGHTS:
            if full_weight + context_weight > 1.0:
                continue
            rule_weight = 1.0 - full_weight - context_weight
            probability = branch_valid @ np.asarray([full_weight, context_weight, rule_weight])
            for threshold in CALIBRATION_THRESHOLDS:
                metric = binary_metrics(labels[valid], probability, threshold)
                fusion_grid.append(
                    {
                        "full_weight": full_weight,
                        "context_weight": context_weight,
                        "rule_weight": rule_weight,
                        "threshold": threshold,
                        **{f"validation_{key}": metric[key] for key in metric},
                        "validation_minimum_metric": min(metric[key] for key in ("precision", "recall", "f1")),
                    }
                )
    ef_choice = select_evidence_fusion_by_validation(fusion_grid)
    ef_model = EvidenceFusedHgbClassifier(
        hgb_model,
        context_model,
        rule_model,
        context_indices,
        rule_indices,
        float(ef_choice["full_weight"]),
        float(ef_choice["context_weight"]),
        float(ef_choice["rule_weight"]),
    )
    joblib.dump(
        {"estimator": ef_model, "config": asdict(hgb_config), "selection": ef_choice, "feature_names": feature_names},
        out / "models/ef_hgb.joblib",
    )
    pd.DataFrame(fusion_grid).to_csv(out / "ef_hgb_validation_grid.csv", index=False)

    # RGFN: repeat the established five weights x five seeds and threshold grid.
    rgfn = train_hourly_rgfn_variant(
        examples=examples,
        splits=splits,
        encoder=ENCODER_MLP,
        split_name="blocked",
        model_dir=out / "models/rgfn",
        base_config=HourlyRgfnTrainingConfig(),
        progress=lambda item: print(
            f"RGFN seed={item['seed']} weight={item['fault_class_weight']:.1f} epochs={item['epochs_completed']}",
            flush=True,
        ),
    )

    test = splits["test"]
    hgb_valid = hgb_model.predict_proba(values[valid])[:, 1]
    hgb_test = hgb_model.predict_proba(values[test])[:, 1]
    ef_valid = ef_model.predict_proba(values[valid])[:, 1]
    ef_test = ef_model.predict_proba(values[test])[:, 1]
    rows = [
        _row(
            "HGB",
            binary_metrics(labels[valid], hgb_valid, float(hgb_choice["threshold"])),
            binary_metrics(labels[test], hgb_test, float(hgb_choice["threshold"])),
            threshold=float(hgb_choice["threshold"]),
            class_weight=float(hgb_choice["fault_class_weight"]),
        ),
        _row(
            "EF-HGB",
            {key: ef_choice[f"validation_{key}"] for key in ("precision", "recall", "f1", "accuracy")},
            binary_metrics(labels[test], ef_test, float(ef_choice["threshold"])),
            threshold=float(ef_choice["threshold"]),
            class_weight=float(hgb_choice["fault_class_weight"]),
            full_weight=float(ef_choice["full_weight"]),
            context_weight=float(ef_choice["context_weight"]),
            rule_weight=float(ef_choice["rule_weight"]),
        ),
        _row(
            "RGFN",
            rgfn["selected_validation_summary"],
            rgfn["test_summary"],
            threshold=float(rgfn["best_balanced"]["threshold"]),
            class_weight=float(rgfn["best_balanced"]["fault_class_weight"]),
            validation_f1_std=float(rgfn["selected_validation_summary"]["f1_std"]),
            test_f1_std=float(rgfn["test_summary"]["f1_std"]),
        ),
    ]
    comparison = pd.DataFrame(rows).sort_values("model")
    selected = max(
        rows,
        key=lambda row: (
            row["validation_minimum_metric"],
            row["validation_f1"],
            row["validation_precision"],
            row["validation_recall"],
            row["model"] == "EF-HGB",
        ),
    )["model"]
    comparison["selected_by_validation"] = comparison.model.eq(selected)
    comparison.to_csv(out / "comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "HGB",
                "test_auroc": roc_auc_score(labels[test], hgb_test),
                "test_auprc": average_precision_score(labels[test], hgb_test),
            },
            {
                "model": "EF-HGB",
                "test_auroc": roc_auc_score(labels[test], ef_test),
                "test_auprc": average_precision_score(labels[test], ef_test),
            },
        ]
    ).to_csv(out / "ranking_metrics.csv", index=False)
    payload = {
        "design": {
            "train": "June 2025-January 2026 plus May-June 2026",
            "validation": "February 8-28 2026 after embargo/crossing-group removal",
            "test": "March 8-April 30 2026 after embargo/crossing-group removal",
            "selection_source": "February validation only",
            "test_role": "comparison only; no threshold, class-weight, fusion, or model selection",
            "excluded_examples": int(excluded),
            "device": str(DEVICE),
        },
        "selected_by_validation": selected,
        "comparison": comparison.to_dict("records"),
        "hgb": hgb,
        "ef_hgb_selection": ef_choice,
        "rgfn": rgfn,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=_json), encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/eval/blocked_model_comparison_20260916")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)
