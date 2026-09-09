from __future__ import annotations

import gc
from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.hourly_detection import (
    FEATURE_PATH,
    HOURLY_DATA_DIR,
    HOURLY_LABEL_PATH,
    LONG_TENSOR_PATH,
    LONG_WINDOW_HOURS,
    MASK_MODE_PER_FEATURE,
    MASK_MODE_PER_HOUR,
    MASK_MODES,
    SHORT_TENSOR_PATH,
    SHORT_WINDOW_HOURS,
    SOURCE_PATH,
    LABEL_PATH,
    build_hourly_examples,
    build_hourly_labels,
    hourly_report,
    load_hourly_frame,
    load_labelled_episodes,
    write_hourly_labels,
    write_hourly_tensor,
)
from src.workflows.prerequisites import require_files


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--features", type=Path, default=FEATURE_PATH)
    parser.add_argument("--labels", type=Path, default=LABEL_PATH)
    parser.add_argument("--labels-output", type=Path, default=HOURLY_LABEL_PATH)
    parser.add_argument("--mask-mode", choices=MASK_MODES, default=MASK_MODE_PER_HOUR)
    parser.add_argument("--short-output", type=Path)
    parser.add_argument("--long-output", type=Path)
    parser.add_argument("--window-hours", type=int)
    parser.add_argument("--window-output", type=Path)
    return parser.parse_args(argv)


def _build_window_tensor(
    hourly: pd.DataFrame,
    labels: pd.DataFrame,
    window_hours: int,
    output: Path,
    mask_mode: str,
) -> None:
    if int(window_hours) < 1:
        raise ValueError("window hours must be positive")
    examples = build_hourly_examples(
        hourly,
        labels,
        int(window_hours),
        mask_mode=mask_mode,
    )
    tensor_path = write_hourly_tensor(examples, Path(output))
    print("WINDOW OUTPUT")
    print(f"mask_mode={mask_mode}")
    print(f"window_hours={int(window_hours)}")
    print(f"tensor={tensor_path}")
    print(f"X_cont_shape={examples['X_cont'].shape}")
    print(f"mask_shape={examples['mask'].shape}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if (args.window_hours is None) != (args.window_output is None):
        raise ValueError("--window-hours and --window-output must be provided together")
    require_files(
        "Hourly dataset construction",
        {
            "canonical merged dataset": args.source,
            "feature matrix": args.features,
            "live episode labels": args.labels,
        },
        "Run scripts/rebuild_detection_features.py and scripts/build_labels.py first.",
    )
    hourly = load_hourly_frame(args.source, args.features)
    episodes = load_labelled_episodes(args.labels)
    labels = build_hourly_labels(hourly, episodes)
    if args.window_hours is not None:
        _build_window_tensor(
            hourly,
            labels,
            args.window_hours,
            args.window_output,
            args.mask_mode,
        )
        return
    labels_path = write_hourly_labels(labels, args.labels_output)
    if args.mask_mode == MASK_MODE_PER_FEATURE:
        default_short_output = HOURLY_DATA_DIR / "hourly_detection_short_per_feature_mask.npz"
        default_long_output = HOURLY_DATA_DIR / "hourly_detection_long_per_feature_mask.npz"
    else:
        default_short_output = SHORT_TENSOR_PATH
        default_long_output = LONG_TENSOR_PATH
    short_output = args.short_output or default_short_output
    long_output = args.long_output or default_long_output
    short_examples = build_hourly_examples(
        hourly,
        labels,
        SHORT_WINDOW_HOURS,
        mask_mode=args.mask_mode,
    )
    short_path = write_hourly_tensor(short_examples, short_output)
    short_shape = short_examples["X_cont"].shape
    short_mask_shape = short_examples["mask"].shape
    del short_examples
    gc.collect()
    long_examples = build_hourly_examples(
        hourly,
        labels,
        LONG_WINDOW_HOURS,
        mask_mode=args.mask_mode,
    )
    long_path = write_hourly_tensor(long_examples, long_output)
    long_shape = long_examples["X_cont"].shape
    long_mask_shape = long_examples["mask"].shape
    del long_examples
    gc.collect()
    print(hourly_report(labels, episodes))
    print()
    print("OUTPUTS")
    print(f"mask_mode={args.mask_mode}")
    print(f"hourly_labels={labels_path}")
    print(f"short_tensor={short_path}")
    print(f"short_X_cont_shape={short_shape}")
    print(f"short_mask_shape={short_mask_shape}")
    print(f"long_tensor={long_path}")
    print(f"long_X_cont_shape={long_shape}")
    print(f"long_mask_shape={long_mask_shape}")


if __name__ == "__main__":
    main()
