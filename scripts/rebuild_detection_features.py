from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.rebuild import (
    DEFAULT_FIVE_MIN_DIR,
    DEFAULT_REFERENCE_DIR,
    rebuild_detection_features,
    resolve_rebuild_outputs,
)
from src.config.paths import MERGED_DATASET_PATH, STATION_REGISTRY_PATH
from src.rules.frozen_statistics import load_frozen_rule_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, default=MERGED_DATASET_PATH)
    parser.add_argument("--registry", type=Path, default=STATION_REGISTRY_PATH)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--five-min-dir", type=Path, default=DEFAULT_FIVE_MIN_DIR)
    parser.add_argument("--frozen-statistics-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = resolve_rebuild_outputs(args.merged, args.output_dir)
    print("OUTPUT MAP")
    for name, path in outputs.items():
        print(f"{name}={path}")
    frozen_statistics = (
        load_frozen_rule_statistics(args.frozen_statistics_dir)
        if args.frozen_statistics_dir is not None
        else None
    )
    rows = rebuild_detection_features(
        merged_path=args.merged,
        registry_path=args.registry,
        reference_dir=args.reference_dir,
        five_min_dir=args.five_min_dir,
        output_dir=args.output_dir,
        frozen_statistics=frozen_statistics,
    )
    print("DETECTION FEATURES REBUILT")
    for name, value in rows.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
