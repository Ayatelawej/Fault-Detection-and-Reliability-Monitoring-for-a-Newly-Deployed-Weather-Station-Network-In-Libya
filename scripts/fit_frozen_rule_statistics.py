from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import FROZEN_RULE_STATISTICS_DIR, MERGED_DATASET_PATH, PROJECT_ROOT
from src.rules.config import EXTERNAL_RESIDUALS_PATH
from src.rules.frozen_statistics import fit_frozen_rule_statistics, save_frozen_rule_statistics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=MERGED_DATASET_PATH)
    parser.add_argument(
        "--external-residuals",
        type=Path,
        default=PROJECT_ROOT / EXTERNAL_RESIDUALS_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=FROZEN_RULE_STATISTICS_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    statistics = fit_frozen_rule_statistics(args.source, args.external_residuals)
    destination = save_frozen_rule_statistics(statistics, args.output_dir)
    print("FROZEN RULE STATISTICS FITTED")
    print(f"source_path={statistics.source_path}")
    print(f"source_sha256={statistics.source_sha256}")
    print(f"source_rows={statistics.source_rows}")
    print(f"source_start={statistics.source_start}")
    print(f"source_end={statistics.source_end}")
    print(f"baselines={len(statistics.baselines)}")
    print(f"isolation_forests={len(statistics.isolation_forests)}")
    print(f"thresholds={len(statistics.thresholds)}")
    print(
        "calibration_corroboration_metrics="
        f"{len(statistics.calibration_corroboration_metrics)}"
    )
    print(f"output_path={destination}")


if __name__ == "__main__":
    main()
