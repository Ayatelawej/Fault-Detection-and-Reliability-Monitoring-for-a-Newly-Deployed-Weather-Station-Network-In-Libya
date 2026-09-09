from __future__ import annotations

import numpy as np
import pandas as pd

from src.rules.config import COVERAGE_FLOOR_HOURS


def _median_and_mad(values: pd.Series) -> tuple[float, float]:
    cleaned = values.dropna()
    if cleaned.empty:
        return float("nan"), float("nan")

    median = float(cleaned.median())
    mad = float(np.median(np.abs(cleaned.to_numpy(dtype=float) - median)))
    return median, mad


def select_baseline(
    frame: pd.DataFrame,
    station_id: str,
    channel: str,
    min_present_hours: int = COVERAGE_FLOOR_HOURS,
    frozen_baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    if frozen_baseline is not None:
        return frozen_baseline
    station_values = frame.loc[frame["station_id"].eq(station_id), channel].dropna()
    n_present = int(station_values.size)

    if n_present >= min_present_hours:
        baseline_value, baseline_spread = _median_and_mad(station_values)
        source = "station"
    else:
        present_counts = frame.groupby("station_id")[channel].count()
        qualifying_stations = present_counts[
            present_counts.ge(min_present_hours)
        ].index
        pooled_values = frame.loc[
            frame["station_id"].isin(qualifying_stations),
            channel,
        ].dropna()
        baseline_value, baseline_spread = _median_and_mad(pooled_values)
        source = "network_pooled"

    return {
        "source": source,
        "baseline_value": baseline_value,
        "baseline_spread": baseline_spread,
        "n_present": n_present,
    }
