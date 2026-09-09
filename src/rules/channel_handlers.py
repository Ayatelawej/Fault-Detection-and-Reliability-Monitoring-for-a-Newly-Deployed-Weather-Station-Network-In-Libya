from __future__ import annotations

import numpy as np
import pandas as pd

from src.rules.config import SENSOR_GROUP_PREFIXES


def encode_wind_direction(series: pd.Series) -> pd.DataFrame:
    radians = np.deg2rad(series.astype(float))
    return pd.DataFrame(
        {
            "sin": np.sin(radians),
            "cos": np.cos(radians),
        },
        index=series.index,
    )


def log_transform_precip(series: pd.Series) -> pd.Series:
    return pd.Series(
        np.log1p(series.astype(float)),
        index=series.index,
        name=series.name,
    )


def sensor_group_for_channel(channel: str) -> str:
    matches = [
        prefix
        for prefix in SENSOR_GROUP_PREFIXES
        if channel.startswith(prefix)
    ]

    if not matches:
        return "other"

    prefix = max(matches, key=len)
    return SENSOR_GROUP_PREFIXES[prefix]


METADATA_NUMERIC_COLUMNS = {
    "n_raw_records",
    "latitude",
    "longitude",
    "qc_status",
    "epoch",
    "data_present",
    "elevation",
}


def numeric_channels(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in METADATA_NUMERIC_COLUMNS
    ]
