from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.hourly_baseline import (
    HourlyBaselineConfig,
    filter_eligible_examples,
    flatten_hourly_features,
    load_hourly_tensor,
    make_classifier,
    random_split_indices,
    resolve_fault_class_weight,
    save_model_bundle,
    spaced_split_indices,
    write_metrics_json,
)
from src.model.hourly_calibration import (
    CALIBRATION_THRESHOLDS,
    CALIBRATION_WEIGHTS,
    calibrate_split,
    calibration_report,
    validation_grid_frame,
)
from src.model.hourly_detection import MASK_MODE_PER_HOUR, MASK_MODES, SHORT_TENSOR_PATH
from src.workflows.prerequisites import require_files


OUTPUT_DIR = PROJECT_ROOT / "data" / "hourly_detection"
METRICS_PATH = OUTPUT_DIR / "hourly_short_calibration_metrics.json"
REPORT_PATH = OUTPUT_DIR / "hourly_short_calibration_report.txt"
SWEEP_PATH = OUTPUT_DIR / "hourly_short_calibration_validation_sweep.csv"


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--tensor", type=Path, default=SHORT_TENSOR_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mask-mode", choices=MASK_MODES, default=MASK_MODE_PER_HOUR)
    parser.add_argument("--save-selected-models", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = parse_args().parse_args(argv)
    require_files(
        "Hourly calibration",
        {"short hourly tensor": args.tensor},
        "Run scripts/build_hourly_dataset.py before calibration.",
    )
    examples, excluded = filter_eligible_examples(load_hourly_tensor(args.tensor))
    values, feature_names, _ = flatten_hourly_features(examples, mask_mode=args.mask_mode)
    labels = np.asarray(examples["y_binary"], dtype=int)
    base_weight = resolve_fault_class_weight(labels)
    config = HourlyBaselineConfig(seed=int(args.seed), fault_class_weight=base_weight)
    splits = {
        "random": random_split_indices(labels, config.seed),
        "spaced": spaced_split_indices(
            labels,
            examples["station_id"],
            examples["hour"],
            examples["display_state"],
            examples["source_episode_ids"],
        )[0],
    }
    results = {
        split_name: calibrate_split(
            values,
            labels,
            split,
            config,
            CALIBRATION_WEIGHTS,
            CALIBRATION_THRESHOLDS,
        )
        for split_name, split in splits.items()
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / METRICS_PATH.name
    report_path = output_dir / REPORT_PATH.name
    sweep_path = output_dir / SWEEP_PATH.name
    sweep = pd.concat(
        [
            validation_grid_frame(result).assign(split=split_name)
            for split_name, result in results.items()
        ],
        ignore_index=True,
    )
    sweep.to_csv(sweep_path, index=False)
    report = calibration_report(results)
    report_path.write_text(report + "\n", encoding="utf-8")
    model_paths: dict[str, str] = {}
    if args.save_selected_models:
        window_hours = int(np.asarray(examples["window_hours"]).reshape(-1)[0])
        model_dir = output_dir / "models"
        for split_name, split in splits.items():
            selected = results[split_name]["best_balanced"]
            selected_config = replace(
                config,
                fault_class_weight=float(selected["fault_class_weight"]),
                threshold=float(selected["threshold"]),
            )
            model = make_classifier(selected_config)
            model.fit(values[split["train"]], labels[split["train"]])
            path = model_dir / f"selected_hgb_{split_name}_{window_hours:02d}h.joblib"
            save_model_bundle(
                model,
                path,
                feature_names,
                window_hours,
                selected_config,
                split_name,
            )
            model_paths[split_name] = str(path)
    payload = {
        "configuration": asdict(config),
        "mask_mode": args.mask_mode,
        "weights": list(CALIBRATION_WEIGHTS),
        "thresholds": list(CALIBRATION_THRESHOLDS),
        "excluded_examples_removed": int(excluded),
        "results": results,
        "tensor": str(args.tensor),
        "selected_model_paths": model_paths,
    }
    write_metrics_json(payload, metrics_path)
    print(report)
    print()
    print("OUTPUTS")
    print(f"metrics_json={metrics_path}")
    print(f"report={report_path}")
    print(f"validation_sweep={sweep_path}")
    for split_name, path in sorted(model_paths.items()):
        print(f"selected_{split_name}_model={path}")


if __name__ == "__main__":
    main()
