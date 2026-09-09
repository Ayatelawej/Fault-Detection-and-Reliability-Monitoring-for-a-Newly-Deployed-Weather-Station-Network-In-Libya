from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.workflows.train_hourly_baseline import main as run_baseline
from src.workflows.train_hourly_baseline import reason_codes_main as run_reason_codes
from src.workflows.train_hourly_calibration import main as run_calibration
from src.workflows.train_hourly_rgfn import main as run_rgfn
from src.workflows.train_hourly_rgfn import evidence_fusion_main as run_evidence_fusion
from src.workflows.train_hourly_rgfn import one_hour_main as run_one_hour_comparison
from src.workflows.train_hourly_rgfn import one_hour_july_main as run_one_hour_july
from src.workflows.train_hourly_split_comparison import main as run_split_comparison


RUNNERS: dict[str, Callable[[list[str] | None], None]] = {
    "baseline": run_baseline,
    "split-comparison": run_split_comparison,
    "calibration": run_calibration,
    "rgfn": run_rgfn,
    "evidence-fusion": run_evidence_fusion,
    "one-hour-comparison": run_one_hour_comparison,
    "one-hour-july": run_one_hour_july,
    "reason-codes": run_reason_codes,
}


def parse_args(argv: list[str] | None = None) -> tuple[str, list[str]]:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = ArgumentParser(
        description="Run an hour-level detection training workflow.",
    )
    parser.add_argument("mode", choices=tuple(RUNNERS))
    if not values or values[0] in {"-h", "--help"}:
        parser.print_help()
        raise SystemExit(0)
    args = parser.parse_args(values[:1])
    return str(args.mode), values[1:]


def main(argv: list[str] | None = None) -> None:
    mode, remaining = parse_args(argv)
    RUNNERS[mode](remaining)


if __name__ == "__main__":
    main()
