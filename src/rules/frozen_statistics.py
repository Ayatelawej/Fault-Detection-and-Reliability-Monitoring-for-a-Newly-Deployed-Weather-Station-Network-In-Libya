from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config.paths import FROZEN_RULE_STATISTICS_DIR, MERGED_DATASET_PATH, PROJECT_ROOT
from src.rules.channel_handlers import numeric_channels
from src.rules.config import EXTERNAL_OFFSET_CHANNELS, EXTERNAL_RESIDUALS_PATH
from src.rules.calibration_corroboration import _residual_metrics
from src.rules.score import compute_anomaly_scores
from src.rules.statistical_gate import detector_thresholds

FROZEN_STATISTICS_FILENAME = "frozen_rule_statistics.joblib"


@dataclass(frozen=True)
class FrozenRuleStatistics:
    baselines: dict[tuple[str, str], dict[str, object]]
    isolation_forests: dict[tuple[str, str], IsolationForest]
    thresholds: dict[tuple[str, str], float]
    evidence_thresholds: dict[tuple[str, str], float]
    calibration_corroboration_metrics: dict[tuple[str, str], dict[str, object]]
    source_path: str
    source_sha256: str
    source_rows: int
    source_start: str
    source_end: str


def _evidence_thresholds_from_scores(scores: pd.DataFrame) -> dict[tuple[str, str], float]:
    table = detector_thresholds(scores)
    result: dict[tuple[str, str], float] = {}
    for row in table.itertuples(index=False):
        result[(str(row.channel), "zscore")] = float(row.zscore_threshold)
        result[(str(row.channel), "iforest")] = float(row.iforest_threshold)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_frozen_rule_statistics(
    source: Path = MERGED_DATASET_PATH,
    external_residuals_path: Path = PROJECT_ROOT / EXTERNAL_RESIDUALS_PATH,
) -> FrozenRuleStatistics:
    source = Path(source)
    raw = pd.read_csv(source, parse_dates=["hour_utc"], low_memory=False)
    raw["hour_utc"] = pd.to_datetime(raw["hour_utc"], utc=True)
    channels = numeric_channels(raw)
    collected: dict[str, dict] = {}
    scores = compute_anomaly_scores(raw, channels=channels, collect_statistics=collected)
    evidence_thresholds = _evidence_thresholds_from_scores(scores)

    external_residuals = pd.read_parquet(Path(external_residuals_path))
    external_residuals["station_id"] = external_residuals["station_id"].astype(str)
    calibration_corroboration_metrics: dict[tuple[str, str], dict[str, object]] = {}
    for channel in EXTERNAL_OFFSET_CHANNELS:
        station_ids = sorted(external_residuals["station_id"].dropna().unique())
        for station_id in station_ids:
            station = external_residuals.loc[external_residuals["station_id"].eq(station_id)]
            calibration_corroboration_metrics[(station_id, channel)] = _residual_metrics(
                station,
                channel,
            )

    return FrozenRuleStatistics(
        baselines=collected["baselines"],
        isolation_forests=collected["isolation_forests"],
        thresholds=collected["thresholds"],
        evidence_thresholds=evidence_thresholds,
        calibration_corroboration_metrics=calibration_corroboration_metrics,
        source_path=str(source),
        source_sha256=sha256_file(source),
        source_rows=int(len(raw)),
        source_start=str(raw["hour_utc"].min()),
        source_end=str(raw["hour_utc"].max()),
    )


def save_frozen_rule_statistics(
    statistics: FrozenRuleStatistics,
    output_dir: Path = FROZEN_RULE_STATISTICS_DIR,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / FROZEN_STATISTICS_FILENAME
    joblib.dump(statistics, destination)
    return destination


def load_frozen_rule_statistics(input_dir: Path = FROZEN_RULE_STATISTICS_DIR) -> FrozenRuleStatistics:
    statistics = joblib.load(Path(input_dir) / FROZEN_STATISTICS_FILENAME)
    if hasattr(statistics, "calibration_corroboration_metrics"):
        return statistics
    legacy_metrics = vars(statistics).get("layer2_metrics")
    if legacy_metrics is None:
        raise ValueError("Frozen statistics do not contain calibration corroboration metrics")
    return FrozenRuleStatistics(
        baselines=statistics.baselines,
        isolation_forests=statistics.isolation_forests,
        thresholds=statistics.thresholds,
        evidence_thresholds=statistics.evidence_thresholds,
        calibration_corroboration_metrics=legacy_metrics,
        source_path=statistics.source_path,
        source_sha256=statistics.source_sha256,
        source_rows=statistics.source_rows,
        source_start=statistics.source_start,
        source_end=statistics.source_end,
    )
