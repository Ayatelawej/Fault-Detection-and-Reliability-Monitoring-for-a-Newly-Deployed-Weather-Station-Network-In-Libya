from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from src.model.hourly_detection import (
    DETECTOR_GROUPS,
    MASK_MODE_PER_FEATURE,
    MASK_MODE_PER_HOUR,
    MASK_MODES,
    detector_columns_by_group,
)
from src.rules.channel_handlers import sensor_group_for_channel


SPLIT_NAMES = ("train", "validation", "test")
SPLIT_FRACTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}
SPLIT_FRACTIONS_80_20 = {
    "train": 0.80,
    "test": 0.20,
}
SPACED_SPLIT_VERSION = "hourly-spaced-v1"
REASON_CODE_CV_REQUESTED_FOLDS = 5
REASON_CODE_MIN_VALIDATION_EPISODE_GROUPS = 5
REQUIRED_TENSOR_KEYS = (
    "X_cont",
    "mask",
    "time_since_last",
    "static",
    "rule_evidence",
    "y_binary",
    "station_id",
    "hour",
    "display_state",
    "source_episode_ids",
    "continuous_feature_names",
    "static_feature_names",
    "rule_evidence_feature_names",
)


@dataclass(frozen=True)
class HourlyBaselineConfig:
    seed: int = 2026
    fault_class_weight: float = 13.0
    threshold: float = 0.5
    learning_rate: float = 0.05
    max_iter: int = 100
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0


@dataclass
class EvidenceFusedHgbClassifier:
    full_estimator: object
    context_estimator: object
    rule_estimator: object
    context_indices: np.ndarray
    rule_indices: np.ndarray
    full_weight: float
    context_weight: float
    rule_weight: float

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values)
        full = self.full_estimator.predict_proba(matrix)[:, 1]
        context = self.context_estimator.predict_proba(matrix[:, self.context_indices])[:, 1]
        rules = self.rule_estimator.predict_proba(matrix[:, self.rule_indices])[:, 1]
        probability = (
            float(self.full_weight) * full
            + float(self.context_weight) * context
            + float(self.rule_weight) * rules
        )
        return np.column_stack([1.0 - probability, probability])


@dataclass(frozen=True)
class HourlyReasonCodeConfig:
    seed: int = 2026
    learning_rate: float = 0.05
    max_iter: int = 100
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    threshold_policy: str = "training_oof_cv_mean_f1"


@dataclass(frozen=True)
class HourlyReasonCodeLogisticConfig:
    seed: int = 2026
    max_iter: int = 2000
    c_value: float = 1.0
    tolerance: float = 1e-4
    threshold_policy: str = "training_oof_cv_mean_f1"


@dataclass
class HourlyLogisticScaler:
    median: np.ndarray
    iqr: np.ndarray

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        train_indices: np.ndarray,
    ) -> "HourlyLogisticScaler":
        selected = np.asarray(train_indices, dtype=np.int64)
        if not len(selected):
            raise ValueError("reason-code logistic preprocessing requires training examples")
        train_values = np.asarray(values, dtype=np.float32)[selected]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanpercentile(train_values, 50, axis=0).astype(np.float32)
            q25 = np.nanpercentile(train_values, 25, axis=0).astype(np.float32)
            q75 = np.nanpercentile(train_values, 75, axis=0).astype(np.float32)
        iqr = (q75 - q25).astype(np.float32)
        median[~np.isfinite(median)] = 0.0
        iqr[~np.isfinite(iqr)] = 1.0
        iqr[iqr == 0.0] = 1.0
        return cls(median=median, iqr=iqr)

    def transform(self, values: np.ndarray) -> np.ndarray:
        scaled = (
            np.asarray(values, dtype=np.float32) - self.median.reshape(1, -1)
        ) / self.iqr.reshape(1, -1)
        scaled = np.clip(scaled, -5.0, 5.0)
        return np.nan_to_num(scaled, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)

    def payload(self) -> dict[str, np.ndarray]:
        return {"median": self.median, "iqr": self.iqr}


@dataclass
class FittedReasonCodeModel:
    method: str
    split_scheme: str
    axis: str
    label_name: str
    label_index: int
    estimator: Any
    positive_class_weight: float
    selected_threshold: float
    train_support: int
    validation_support: int
    validation_metrics: dict[str, object]
    selection_trace: dict[str, object]
    scaler: HourlyLogisticScaler | None = None


def load_hourly_tensor(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_hourly_metadata(path: Path) -> dict[str, np.ndarray]:
    keys = (
        "y_binary",
        "station_id",
        "hour",
        "display_state",
        "source_episode_ids",
        "window_hours",
    )
    with np.load(path, allow_pickle=True) as data:
        missing = sorted(set(keys).difference(data.files))
        if missing:
            raise KeyError(f"hourly tensor lacks metadata fields: {missing}")
        return {key: data[key] for key in keys}


def validate_tensor(examples: dict[str, np.ndarray]) -> None:
    missing = sorted(set(REQUIRED_TENSOR_KEYS).difference(examples))
    if missing:
        raise KeyError(f"hourly tensor fields missing: {missing}")
    count = len(examples["y_binary"])
    if count == 0:
        raise ValueError("hourly tensor has no examples")
    sample_keys = (
        "X_cont",
        "mask",
        "time_since_last",
        "static",
        "rule_evidence",
        "station_id",
        "hour",
        "display_state",
        "source_episode_ids",
    )
    for key in sample_keys:
        value = np.asarray(examples[key])
        if value.ndim == 0 or value.shape[0] != count:
            raise ValueError(f"hourly tensor field has an unexpected sample dimension: {key}")
    labels = np.asarray(examples["y_binary"])
    valid = np.isin(labels, [0, 1])
    states = np.asarray(examples["display_state"], dtype=object).astype(str)
    invalid = ~valid & ~np.equal(states, "excluded")
    if invalid.any():
        raise ValueError("hourly tensor has non-binary labels outside excluded rows")


def filter_eligible_examples(examples: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    validate_tensor(examples)
    count = len(examples["y_binary"])
    labels = np.asarray(examples["y_binary"])
    states = np.asarray(examples["display_state"], dtype=object).astype(str)
    eligible = np.not_equal(states, "excluded") & np.isin(labels, [0, 1])
    result: dict[str, np.ndarray] = {}
    for key, value in examples.items():
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == count:
            result[key] = value[eligible]
        else:
            result[key] = value
    if not len(result["y_binary"]):
        raise ValueError("no eligible hourly examples remain")
    result_states = np.asarray(result["display_state"], dtype=object).astype(str)
    if np.equal(result_states, "excluded").any():
        raise RuntimeError("excluded examples remained after eligibility filtering")
    if not np.isin(result["y_binary"], [0, 1]).all():
        raise RuntimeError("eligible examples have non-binary labels")
    return result, int((~eligible).sum())


def assert_matching_metadata(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> None:
    keys = ("y_binary", "station_id", "hour", "display_state", "source_episode_ids")
    for key in keys:
        if key not in left or key not in right:
            raise KeyError(f"metadata field missing for tensor comparison: {key}")
        if not np.array_equal(left[key], right[key]):
            raise ValueError(f"hourly tensors disagree on {key}")


def resolve_fault_class_weight(
    labels: np.ndarray,
    requested_weight: float | None = None,
) -> float:
    if requested_weight is not None:
        value = float(requested_weight)
        if value <= 0.0:
            raise ValueError("fault class weight must be positive")
        return value
    labels = np.asarray(labels, dtype=int)
    fault_count = int(np.equal(labels, 1).sum())
    not_fault_count = int(np.equal(labels, 0).sum())
    if fault_count == 0 or not_fault_count == 0:
        raise ValueError("both binary classes are required to resolve fault class weight")
    return float(not_fault_count / fault_count)


def _feature_names(examples: dict[str, np.ndarray]) -> tuple[list[str], dict[str, np.ndarray]]:
    continuous = [str(value) for value in examples["continuous_feature_names"]]
    static = [str(value) for value in examples["static_feature_names"]]
    rules = [str(value) for value in examples["rule_evidence_feature_names"]]
    window_hours = int(np.asarray(examples["X_cont"]).shape[1])
    per_hour_width = len(continuous) + 2
    names: list[str] = []
    groups: dict[str, list[int]] = {f"continuous:{name}": [] for name in continuous}
    groups["mask:all_lags"] = []
    groups["time_since_last:all_lags"] = []
    position = 0
    for hour_index in range(window_hours):
        hours_ago = window_hours - hour_index - 1
        suffix = "t0" if hours_ago == 0 else f"tminus_{hours_ago}h"
        for feature_index, name in enumerate(continuous):
            names.append(f"{name}_{suffix}")
            groups[f"continuous:{name}"].append(position + feature_index)
        names.append(f"row_present_{suffix}")
        groups["mask:all_lags"].append(position + len(continuous))
        names.append(f"time_since_last_{suffix}")
        groups["time_since_last:all_lags"].append(position + len(continuous) + 1)
        position += per_hour_width
    for name in rules:
        names.append(name)
        groups[f"rule:{name}"] = [position]
        position += 1
    for name in static:
        names.append(name)
        groups[f"static:{name}"] = [position]
        position += 1
    return names, {
        name: np.asarray(indices, dtype=np.int64)
        for name, indices in groups.items()
    }


def _flatten_hourly_features_per_hour(
    examples: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    validate_tensor(examples)
    continuous = np.asarray(examples["X_cont"], dtype=np.float32)
    mask = np.asarray(examples["mask"], dtype=np.float32)
    elapsed = np.asarray(examples["time_since_last"], dtype=np.float32)
    static = np.asarray(examples["static"], dtype=np.float32)
    rules = np.asarray(examples["rule_evidence"], dtype=np.float32)
    if continuous.ndim != 3 or mask.shape[:2] != continuous.shape[:2] or elapsed.shape[:2] != continuous.shape[:2]:
        raise ValueError("hourly temporal arrays do not share a sample and window shape")
    if mask.shape[2] != 1 or elapsed.shape[2] != 1:
        raise ValueError("hourly mask and elapsed arrays must each have one channel")
    count, window_hours, continuous_width = continuous.shape
    dimension = window_hours * (continuous_width + 2) + rules.shape[1] + static.shape[1]
    values = np.empty((count, dimension), dtype=np.float32)
    cursor = 0
    for hour_index in range(window_hours):
        values[:, cursor:cursor + continuous_width] = continuous[:, hour_index, :]
        cursor += continuous_width
        values[:, cursor] = mask[:, hour_index, 0]
        cursor += 1
        values[:, cursor] = elapsed[:, hour_index, 0]
        cursor += 1
    values[:, cursor:cursor + rules.shape[1]] = rules
    cursor += rules.shape[1]
    values[:, cursor:cursor + static.shape[1]] = static
    cursor += static.shape[1]
    if cursor != dimension:
        raise RuntimeError("hourly feature flattening ended at an unexpected dimension")
    values[~np.isfinite(values)] = np.nan
    names, groups = _feature_names(examples)
    if len(names) != dimension:
        raise RuntimeError("hourly feature names do not match flattened dimension")
    return values, names, groups


def _feature_names_per_feature(
    examples: dict[str, np.ndarray],
) -> tuple[list[str], dict[str, np.ndarray]]:
    continuous = [str(value) for value in examples["continuous_feature_names"]]
    static = [str(value) for value in examples["static_feature_names"]]
    rules = [str(value) for value in examples["rule_evidence_feature_names"]]
    window_hours = int(np.asarray(examples["X_cont"]).shape[1])
    n_continuous = len(continuous)
    per_hour_width = n_continuous * 2 + 1
    names: list[str] = []
    groups: dict[str, list[int]] = {f"continuous:{name}": [] for name in continuous}
    groups.update({f"feature_mask:{name}": [] for name in continuous})
    groups["time_since_last:all_lags"] = []
    position = 0
    for hour_index in range(window_hours):
        hours_ago = window_hours - hour_index - 1
        suffix = "t0" if hours_ago == 0 else f"tminus_{hours_ago}h"
        for feature_index, name in enumerate(continuous):
            names.append(f"{name}_{suffix}")
            groups[f"continuous:{name}"].append(position + feature_index)
        for feature_index, name in enumerate(continuous):
            names.append(f"{name}_present_{suffix}")
            groups[f"feature_mask:{name}"].append(position + n_continuous + feature_index)
        names.append(f"time_since_last_{suffix}")
        groups["time_since_last:all_lags"].append(position + n_continuous * 2)
        position += per_hour_width
    for name in rules:
        names.append(name)
        groups[f"rule:{name}"] = [position]
        position += 1
    for name in static:
        names.append(name)
        groups[f"static:{name}"] = [position]
        position += 1
    return names, {
        name: np.asarray(indices, dtype=np.int64)
        for name, indices in groups.items()
    }


def _flatten_hourly_features_per_feature(
    examples: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    validate_tensor(examples)
    continuous = np.asarray(examples["X_cont"], dtype=np.float32)
    mask = np.asarray(examples["mask"], dtype=np.float32)
    elapsed = np.asarray(examples["time_since_last"], dtype=np.float32)
    static = np.asarray(examples["static"], dtype=np.float32)
    rules = np.asarray(examples["rule_evidence"], dtype=np.float32)
    if continuous.ndim != 3 or mask.shape[:2] != continuous.shape[:2] or elapsed.shape[:2] != continuous.shape[:2]:
        raise ValueError("hourly temporal arrays do not share a sample and window shape")
    if mask.shape[2] != continuous.shape[2] or elapsed.shape[2] != 1:
        raise ValueError("hourly per-feature mask and elapsed arrays must match the expected channel widths")
    count, window_hours, continuous_width = continuous.shape
    dimension = window_hours * (continuous_width * 2 + 1) + rules.shape[1] + static.shape[1]
    values = np.empty((count, dimension), dtype=np.float32)
    cursor = 0
    for hour_index in range(window_hours):
        values[:, cursor:cursor + continuous_width] = continuous[:, hour_index, :]
        cursor += continuous_width
        values[:, cursor:cursor + continuous_width] = mask[:, hour_index, :]
        cursor += continuous_width
        values[:, cursor] = elapsed[:, hour_index, 0]
        cursor += 1
    values[:, cursor:cursor + rules.shape[1]] = rules
    cursor += rules.shape[1]
    values[:, cursor:cursor + static.shape[1]] = static
    cursor += static.shape[1]
    if cursor != dimension:
        raise RuntimeError("hourly feature flattening ended at an unexpected dimension")
    values[~np.isfinite(values)] = np.nan
    names, groups = _feature_names_per_feature(examples)
    if len(names) != dimension:
        raise RuntimeError("hourly feature names do not match flattened dimension")
    return values, names, groups


def flatten_hourly_features(
    examples: dict[str, np.ndarray],
    mask_mode: str = MASK_MODE_PER_HOUR,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    resolved_mask_mode = str(mask_mode)
    if resolved_mask_mode == MASK_MODE_PER_HOUR:
        return _flatten_hourly_features_per_hour(examples)
    if resolved_mask_mode == MASK_MODE_PER_FEATURE:
        return _flatten_hourly_features_per_feature(examples)
    raise ValueError(f"unknown hourly mask mode: {mask_mode}; expected one of {MASK_MODES}")


def _sample_keys(station_ids: np.ndarray, hours: np.ndarray) -> np.ndarray:
    stations = np.asarray(station_ids, dtype=object).astype(str)
    parsed = pd.to_datetime(pd.Series(hours), utc=True, format="mixed")
    values = parsed.astype("int64").to_numpy(dtype=np.int64)
    return np.asarray(
        [f"{station}\x1f{int(hour)}" for station, hour in zip(stations, values)],
        dtype=object,
    )


def load_reason_code_manifest_splits(
    examples: dict[str, np.ndarray],
    path: Path,
) -> dict[str, dict[str, np.ndarray]]:
    """Load and validate the binary detector's exact full-population partitions."""
    validate_tensor(examples)
    manifest = pd.read_csv(path, keep_default_na=False)
    required = {"split_scheme", "split", "station_id", "hour", "fault_hour", "display_state"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise KeyError(f"split manifest fields missing: {missing}")
    tensor_keys = _sample_keys(examples["station_id"], examples["hour"])
    if len(np.unique(tensor_keys)) != len(tensor_keys):
        raise ValueError("hourly tensor station-hour keys are not unique")
    tensor_index = {str(key): index for index, key in enumerate(tensor_keys)}
    labels = np.asarray(examples["y_binary"], dtype=int)
    display_states = np.asarray(examples["display_state"], dtype=object).astype(str)
    result: dict[str, dict[str, np.ndarray]] = {}
    for scheme in ("random", "spaced"):
        frame = manifest.loc[manifest["split_scheme"].eq(scheme)].copy()
        if frame.empty:
            raise ValueError(f"split manifest lacks {scheme} membership")
        frame_keys = _sample_keys(frame["station_id"].to_numpy(), frame["hour"].to_numpy())
        if len(np.unique(frame_keys)) != len(frame_keys):
            raise ValueError(f"split manifest duplicates a {scheme} station-hour key")
        mapped = np.asarray([tensor_index.get(str(key), -1) for key in frame_keys], dtype=np.int64)
        if np.any(mapped < 0):
            raise ValueError(f"split manifest contains unknown {scheme} station-hour keys")
        if len(mapped) != len(labels) or len(np.unique(mapped)) != len(labels):
            raise ValueError(f"split manifest does not cover every {scheme} hourly example once")
        manifest_labels = pd.to_numeric(frame["fault_hour"], errors="raise").to_numpy(dtype=int)
        if not np.array_equal(labels[mapped], manifest_labels):
            raise ValueError(f"split manifest labels disagree with the {scheme} hourly tensor")
        manifest_states = frame["display_state"].astype(str).to_numpy(dtype=object)
        if not np.array_equal(display_states[mapped], manifest_states):
            raise ValueError(f"split manifest display states disagree with the {scheme} hourly tensor")
        partitions: dict[str, np.ndarray] = {}
        for name in ("train", "validation", "test"):
            indices = np.sort(mapped[frame["split"].eq(name).to_numpy()]).astype(np.int64)
            if not len(indices):
                raise ValueError(f"split manifest has an empty {scheme} {name} partition")
            partitions[name] = indices
        combined = np.concatenate([partitions[name] for name in ("train", "validation", "test")])
        if len(np.unique(combined)) != len(labels) or set(combined.tolist()) != set(range(len(labels))):
            raise ValueError(f"split manifest membership is invalid for {scheme}")
        result[scheme] = partitions
    return result


def _subset_examples(examples: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    count = len(examples["y_binary"])
    selected = np.asarray(indices, dtype=np.int64)
    result: dict[str, np.ndarray] = {}
    for key, value in examples.items():
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == count:
            result[key] = value[selected]
        else:
            result[key] = value
    return result


def _reason_axis_specs(examples: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    required = {
        "y_mechanism",
        "y_component",
        "mechanism_label_names",
        "component_label_names",
    }
    missing = sorted(required.difference(examples))
    if missing:
        raise KeyError(f"hourly tensor lacks reason-code fields: {missing}")
    count = len(examples["y_binary"])
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for axis, target_key, names_key in (
        ("mechanism", "y_mechanism", "mechanism_label_names"),
        ("component", "y_component", "component_label_names"),
    ):
        targets = np.asarray(examples[target_key], dtype=np.int64)
        names = np.asarray(examples[names_key], dtype=object).astype(str)
        if targets.ndim != 2 or targets.shape[0] != count or targets.shape[1] != len(names):
            raise ValueError(f"hourly tensor has an invalid {axis} target shape")
        if not np.isin(targets, [0, 1]).all():
            raise ValueError(f"hourly tensor has non-binary {axis} targets")
        result[axis] = (targets, names)
    return result


def prepare_reason_code_population(
    examples: dict[str, np.ndarray],
    full_splits: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], list[dict[str, object]]]:
    """Restrict validated binary partitions to their fault-hour members without re-splitting."""
    validate_tensor(examples)
    axes = _reason_axis_specs(examples)
    labels = np.asarray(examples["y_binary"], dtype=int)
    states = np.asarray(examples["display_state"], dtype=object).astype(str)
    fault_indices = np.flatnonzero(np.equal(labels, 1)).astype(np.int64)
    if not len(fault_indices):
        raise ValueError("no fault hours are available for reason-code modelling")
    if not np.equal(states[fault_indices], "fault").all():
        raise ValueError("fault reason-code population contains a non-fault display state")
    for axis, (targets, _) in axes.items():
        if not targets[fault_indices].astype(bool).any(axis=1).all():
            raise ValueError(f"a fault hour lacks a {axis} target")
    fault_examples = _subset_examples(examples, fault_indices)
    full_to_fault = np.full(len(labels), -1, dtype=np.int64)
    full_to_fault[fault_indices] = np.arange(len(fault_indices), dtype=np.int64)
    splits: dict[str, dict[str, np.ndarray]] = {}
    trace: list[dict[str, object]] = []
    keys = _sample_keys(examples["station_id"], examples["hour"])
    for scheme in ("random", "spaced"):
        if scheme not in full_splits:
            raise KeyError(f"full split map lacks {scheme}")
        partitions: dict[str, np.ndarray] = {}
        for partition in ("train", "validation", "test"):
            full_indices = np.asarray(full_splits[scheme][partition], dtype=np.int64)
            selected_full = full_indices[np.equal(labels[full_indices], 1)]
            local_indices = full_to_fault[selected_full]
            if np.any(local_indices < 0):
                raise RuntimeError("fault split restriction produced an invalid local index")
            if not len(local_indices):
                raise ValueError(f"{scheme} {partition} has no fault hours")
            partitions[partition] = np.sort(local_indices).astype(np.int64)
            partition_keys = np.sort(keys[selected_full].astype(str))
            trace.append(
                {
                    "split_scheme": scheme,
                    "partition": partition,
                    "full_partition_hours": int(len(full_indices)),
                    "fault_hours": int(len(selected_full)),
                    "fault_key_sha256": hashlib.sha256(
                        "\n".join(partition_keys.tolist()).encode("utf-8")
                    ).hexdigest(),
                    "selection_source": "existing_binary_split_manifest",
                }
            )
        combined = np.concatenate([partitions[name] for name in ("train", "validation", "test")])
        if len(np.unique(combined)) != len(fault_indices) or set(combined.tolist()) != set(range(len(fault_indices))):
            raise ValueError(f"fault-only {scheme} partitions are not complete and disjoint")
        splits[scheme] = partitions
    return fault_examples, splits, trace


def resolve_reason_code_class_weight(labels: np.ndarray) -> float:
    values = np.asarray(labels, dtype=int)
    positives = int(np.equal(values, 1).sum())
    negatives = int(np.equal(values, 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("each reason-code training label needs both classes")
    return float(negatives / positives)


def make_reason_code_classifier(
    config: HourlyReasonCodeConfig,
    positive_class_weight: float,
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=float(config.learning_rate),
        max_iter=int(config.max_iter),
        max_leaf_nodes=int(config.max_leaf_nodes),
        min_samples_leaf=int(config.min_samples_leaf),
        l2_regularization=float(config.l2_regularization),
        early_stopping=False,
        random_state=int(config.seed),
        class_weight={0: 1.0, 1: float(positive_class_weight)},
    )


def make_reason_code_logistic_classifier(
    config: HourlyReasonCodeLogisticConfig,
    positive_class_weight: float,
) -> LogisticRegression:
    if int(config.max_iter) < 1 or float(config.c_value) <= 0.0 or float(config.tolerance) <= 0.0:
        raise ValueError("reason-code logistic configuration values must be positive")
    return LogisticRegression(
        class_weight={0: 1.0, 1: float(positive_class_weight)},
        max_iter=int(config.max_iter),
        C=float(config.c_value),
        tol=float(config.tolerance),
        solver="lbfgs",
        random_state=int(config.seed),
    )


def reason_code_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    values = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    predictions = np.asarray(scores >= float(threshold), dtype=int)
    matrix = confusion_matrix(values, predictions, labels=[0, 1])
    support = int(np.equal(values, 1).sum())
    metrics: dict[str, object] = {
        "support": support,
        "accuracy": float(accuracy_score(values, predictions)),
        "tp": int(matrix[1, 1]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tn": int(matrix[0, 0]),
        "estimated": bool(support > 0),
    }
    if support == 0:
        metrics.update({"precision": None, "recall": None, "f1": None})
        return metrics
    metrics.update(
        {
            "precision": float(precision_score(values, predictions, zero_division=0)),
            "recall": float(recall_score(values, predictions, zero_division=0)),
            "f1": float(f1_score(values, predictions, zero_division=0)),
        }
    )
    return metrics


def select_reason_code_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, object], int]:
    values = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if not np.equal(values, 1).any() or not np.equal(values, 0).any():
        raise ValueError("validation threshold selection needs both classes")
    candidates = np.unique(scores)
    best_threshold = float(candidates[0])
    best_metrics = reason_code_metrics(values, scores, best_threshold)
    for threshold in np.sort(candidates)[::-1]:
        current = reason_code_metrics(values, scores, float(threshold))
        current_f1 = float(current["f1"])
        best_f1 = float(best_metrics["f1"])
        if current_f1 > best_f1 or (current_f1 == best_f1 and float(threshold) > best_threshold):
            best_threshold = float(threshold)
            best_metrics = current
    return best_threshold, best_metrics, int(len(candidates))


def _reason_code_cv_seed(
    seed: int,
    split_scheme: str,
    axis: str,
    label_name: str,
) -> int:
    del split_scheme, axis, label_name
    return int(seed)


def reason_code_cv_group_ids(source_episode_ids: np.ndarray) -> np.ndarray:
    """Give all connected source episodes one deterministic inner-CV group identity."""
    values = np.asarray(source_episode_ids, dtype=object).astype(str)
    parent: dict[str, str] = {}

    def find(token: str) -> str:
        parent.setdefault(token, token)
        if parent[token] != token:
            parent[token] = find(parent[token])
        return parent[token]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    tokens_by_row: list[list[str]] = []
    for index, value in enumerate(values):
        tokens = [token for token in str(value).split("|") if token]
        if not tokens:
            tokens = [f"unlinked_fault_hour_{index}"]
        for token in tokens[1:]:
            union(tokens[0], token)
        find(tokens[0])
        tokens_by_row.append(tokens)
    return np.asarray([find(tokens[0]) for tokens in tokens_by_row], dtype=object)


def reason_code_connected_group_metadata(
    source_episode_ids: np.ndarray,
    splits: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build stable connected fault-event identities and partition diagnostics."""
    values = np.asarray(source_episode_ids, dtype=object).astype(str)
    roots = reason_code_cv_group_ids(values)
    members_by_root: dict[str, set[str]] = {}
    for root, value in zip(roots, values):
        members = members_by_root.setdefault(str(root), set())
        tokens = [token for token in str(value).split("|") if token]
        members.update(tokens or [str(root)])
    stable_by_root = {
        root: "event_" + hashlib.sha256("|".join(sorted(members)).encode("utf-8")).hexdigest()[:16]
        for root, members in members_by_root.items()
    }
    group_ids = np.asarray([stable_by_root[str(root)] for root in roots], dtype=object)
    raw_ids = {
        stable_by_root[root]: "|".join(sorted(members))
        for root, members in members_by_root.items()
    }
    count = len(values)
    rows: list[dict[str, object]] = []
    for scheme in ("random", "spaced"):
        if scheme not in splits:
            raise KeyError(f"reason-code partitions lack {scheme}")
        membership = np.full(count, "", dtype=object)
        for partition in ("train", "validation", "test"):
            if partition not in splits[scheme]:
                raise KeyError(f"reason-code {scheme} partitions lack {partition}")
            indices = np.asarray(splits[scheme][partition], dtype=np.int64)
            if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= count):
                raise ValueError(f"reason-code {scheme} {partition} indices are invalid")
            if np.any(membership[indices] != ""):
                raise ValueError(f"reason-code {scheme} partitions overlap")
            membership[indices] = partition
        if np.any(membership == ""):
            raise ValueError(f"reason-code {scheme} partitions do not cover every fault hour")
        for group_id in sorted(set(group_ids.tolist())):
            group_membership = membership[group_ids == group_id]
            counts = {
                partition: int(np.equal(group_membership, partition).sum())
                for partition in ("train", "validation", "test")
            }
            present = [partition for partition in ("train", "validation", "test") if counts[partition]]
            complete_by_partition = {
                partition: counts[partition] == int(len(group_membership))
                for partition in ("train", "validation", "test")
            }
            rows.append(
                {
                    "split_scheme": scheme,
                    "connected_episode_group_id": group_id,
                    "raw_episode_ids": raw_ids[group_id],
                    "total_fault_hours": int(len(group_membership)),
                    "train_fault_hours": counts["train"],
                    "validation_fault_hours": counts["validation"],
                    "test_fault_hours": counts["test"],
                    "partitions_present": "|".join(present),
                    "is_complete_within_train": bool(complete_by_partition["train"]),
                    "is_complete_within_validation": bool(complete_by_partition["validation"]),
                    "is_complete_within_test": bool(complete_by_partition["test"]),
                    "fragmented_across_partitions": bool(len(present) > 1),
                    "evaluation_unit": (
                        "complete_connected_event_group"
                        if complete_by_partition["test"]
                        else "observed_test_event_fragment"
                    ),
                }
            )
    return group_ids, pd.DataFrame(rows)


def _reason_code_evaluation_group_diagnostics(
    group_diagnostics: pd.DataFrame,
    evaluation_partition: str,
) -> pd.DataFrame:
    """Describe connected groups for one pre-existing partition without sibling rows."""
    partition = str(evaluation_partition)
    if partition not in {"validation", "test"}:
        raise ValueError("reason-code event aggregation is limited to validation or test partitions")
    count_column = f"{partition}_fault_hours"
    complete_column = f"is_complete_within_{partition}"
    required = {
        "split_scheme",
        "connected_episode_group_id",
        "raw_episode_ids",
        "total_fault_hours",
        count_column,
        complete_column,
        "partitions_present",
        "fragmented_across_partitions",
    }
    missing = sorted(required.difference(group_diagnostics.columns))
    if missing:
        raise KeyError(f"reason-code group diagnostics lack fields: {missing}")
    result = group_diagnostics.copy()
    result["evaluation_partition"] = partition
    result["evaluation_fault_hours"] = pd.to_numeric(
        result[count_column], errors="raise"
    ).astype(int)
    result["is_complete_within_evaluation_partition"] = result[complete_column].astype(bool)
    result["evaluation_unit"] = np.where(
        result["is_complete_within_evaluation_partition"],
        "complete_connected_event_group",
        f"observed_{partition}_event_fragment",
    )
    return result


def aggregate_reason_code_prediction_rows(
    rows: pd.DataFrame,
    group_diagnostics: pd.DataFrame,
    evaluation_partition: str = "test",
) -> pd.DataFrame:
    """Aggregate one held-out partition without adding sibling hours."""
    required = {
        "method",
        "split_scheme",
        "axis",
        "label",
        "connected_episode_group_id",
        "truth",
        "probability",
        "threshold",
        "hourly_prediction",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise KeyError(f"reason-code aggregation rows lack fields: {missing}")
    source = rows.copy()
    if "split" in source.columns and not source["split"].eq(str(evaluation_partition)).all():
        raise ValueError("reason-code aggregation rows do not all match the requested partition")
    diagnostics = _reason_code_evaluation_group_diagnostics(
        group_diagnostics, evaluation_partition
    )
    diagnostic_columns = {
        "split_scheme",
        "connected_episode_group_id",
        "raw_episode_ids",
        "total_fault_hours",
        "evaluation_partition",
        "evaluation_fault_hours",
        "partitions_present",
        "is_complete_within_evaluation_partition",
        "fragmented_across_partitions",
        "evaluation_unit",
    }
    missing_diagnostics = sorted(diagnostic_columns.difference(diagnostics.columns))
    if missing_diagnostics:
        raise KeyError(f"reason-code group diagnostics lack fields: {missing_diagnostics}")
    source["truth"] = pd.to_numeric(source["truth"], errors="raise").astype(int)
    source["hourly_prediction"] = pd.to_numeric(
        source["hourly_prediction"], errors="raise"
    ).astype(int)
    source["probability"] = pd.to_numeric(source["probability"], errors="raise").astype(float)
    source["threshold"] = pd.to_numeric(source["threshold"], errors="raise").astype(float)
    if not np.isin(source["truth"].to_numpy(), [0, 1]).all():
        raise ValueError("reason-code aggregation truth must be binary")
    if not np.isin(source["hourly_prediction"].to_numpy(), [0, 1]).all():
        raise ValueError("reason-code aggregation predictions must be binary")
    if not np.isfinite(source[["probability", "threshold"]].to_numpy(dtype=float)).all():
        raise ValueError("reason-code aggregation probabilities and thresholds must be finite")
    join = diagnostics.loc[:, sorted(diagnostic_columns)].drop_duplicates(
        ["split_scheme", "connected_episode_group_id"]
    )
    source = source.merge(
        join,
        on=["split_scheme", "connected_episode_group_id"],
        how="left",
        validate="many_to_one",
    )
    if source["evaluation_unit"].isna().any():
        raise ValueError("reason-code aggregation contains a group without diagnostics")
    identities = [
        "method",
        "split_scheme",
        "axis",
        "label",
        "connected_episode_group_id",
    ]
    result: list[dict[str, object]] = []
    for identity, frame in source.groupby(identities, sort=True, dropna=False):
        method, scheme, axis, label, group_id = identity
        thresholds = frame["threshold"].to_numpy(dtype=float)
        if not np.allclose(thresholds, thresholds[0], rtol=0.0, atol=1e-12):
            raise ValueError(
                f"reason-code aggregation has inconsistent frozen thresholds for {method}/{scheme}/{axis}/{label}"
            )
        probabilities = frame["probability"].to_numpy(dtype=float)
        hourly_predictions = frame["hourly_prediction"].to_numpy(dtype=int)
        truth = int(frame["truth"].max())
        common = {
            "method": str(method),
            "split_scheme": str(scheme),
            "axis": str(axis),
            "label": str(label),
            "connected_episode_group_id": str(group_id),
            "raw_episode_ids": str(frame["raw_episode_ids"].iloc[0]),
            "truth": truth,
            "frozen_hourly_threshold": float(thresholds[0]),
            "evaluation_partition": str(frame["evaluation_partition"].iloc[0]),
            "observed_partition_hours": int(len(frame)),
            "total_fault_hours": int(frame["total_fault_hours"].iloc[0]),
            "evaluation_fault_hours": int(frame["evaluation_fault_hours"].iloc[0]),
            "partitions_present": str(frame["partitions_present"].iloc[0]),
            "is_complete_within_evaluation_partition": bool(
                frame["is_complete_within_evaluation_partition"].iloc[0]
            ),
            "fragmented_across_partitions": bool(frame["fragmented_across_partitions"].iloc[0]),
            "evaluation_unit": str(frame["evaluation_unit"].iloc[0]),
            "mean_probability": float(np.mean(probabilities)),
            "max_probability": float(np.max(probabilities)),
        }
        predictions = {
            "any": int(np.any(hourly_predictions == 1)),
            "majority": int(np.sum(hourly_predictions) > len(hourly_predictions) / 2.0),
            "mean_probability": int(np.mean(probabilities) >= float(thresholds[0])),
        }
        for rule, prediction in predictions.items():
            result.append(
                {
                    **common,
                    "aggregation_rule": rule,
                    "prediction": int(prediction),
                }
            )
    return pd.DataFrame(result)


def reason_code_aggregated_metric_rows(aggregated_rows: pd.DataFrame) -> list[dict[str, object]]:
    """Evaluate fixed aggregation rules; none is selected from held-out results."""
    required = {
        "method",
        "split_scheme",
        "axis",
        "label",
        "aggregation_rule",
        "truth",
        "prediction",
        "evaluation_partition",
        "evaluation_unit",
        "is_complete_within_evaluation_partition",
    }
    missing = sorted(required.difference(aggregated_rows.columns))
    if missing:
        raise KeyError(f"reason-code aggregate metric rows lack fields: {missing}")
    def summary_unit(frame: pd.DataFrame) -> str:
        partition = str(frame["evaluation_partition"].iloc[0])
        complete = int(frame["is_complete_within_evaluation_partition"].sum())
        total = int(len(frame))
        if complete == total:
            return "complete_connected_event_group"
        if complete == 0:
            return f"observed_{partition}_event_fragment"
        return f"mixed_complete_and_fragmented_{partition}_event_groups"

    records: list[dict[str, object]] = []
    identities = ["method", "split_scheme", "aggregation_rule", "axis"]
    for identity, axis_frame in aggregated_rows.groupby(identities, sort=True, dropna=False):
        method, scheme, rule, axis = identity
        partitions = axis_frame["evaluation_partition"].astype(str).unique()
        if len(partitions) != 1:
            raise ValueError("reason-code aggregated metrics mix evaluation partitions")
        evaluation_partition = str(partitions[0])
        label_rows: list[dict[str, object]] = []
        for label, label_frame in axis_frame.groupby("label", sort=True, dropna=False):
            truth = label_frame["truth"].to_numpy(dtype=int)
            predictions = label_frame["prediction"].to_numpy(dtype=int)
            metrics = reason_code_metrics(truth, predictions.astype(float), 0.5)
            row = {
                "method": str(method),
                "split_scheme": str(scheme),
                "aggregation_rule": str(rule),
                "axis": str(axis),
                "average": "per_label",
                "label": str(label),
                "evaluation_partition": evaluation_partition,
                "evaluation_unit": summary_unit(label_frame),
                "episode_groups": int(len(label_frame)),
                "complete_evaluation_groups": int(
                    label_frame["is_complete_within_evaluation_partition"].sum()
                ),
                "fragmented_evaluation_groups": int(
                    (~label_frame["is_complete_within_evaluation_partition"]).sum()
                ),
                **metrics,
            }
            label_rows.append(row)
            records.append(row)
        estimable = [row for row in label_rows if bool(row["estimated"])]
        macro = {
            key: (float(np.mean([float(row[key]) for row in estimable])) if estimable else None)
            for key in ("precision", "recall", "f1", "accuracy")
        }
        truths = axis_frame["truth"].to_numpy(dtype=int)
        predictions = axis_frame["prediction"].to_numpy(dtype=int)
        micro = reason_code_metrics(truths, predictions.astype(float), 0.5)
        group_reference = axis_frame.loc[
            axis_frame["label"].eq(axis_frame["label"].iloc[0])
        ]
        records.append(
            {
                "method": str(method),
                "split_scheme": str(scheme),
                "aggregation_rule": str(rule),
                "axis": str(axis),
                "average": "macro",
                "label": "",
                "evaluation_partition": evaluation_partition,
                "evaluation_unit": summary_unit(group_reference),
                "episode_groups": int(axis_frame["connected_episode_group_id"].nunique()),
                "complete_evaluation_groups": int(
                    group_reference["is_complete_within_evaluation_partition"].sum()
                ),
                "fragmented_evaluation_groups": int(
                    (~group_reference["is_complete_within_evaluation_partition"]).sum()
                ),
                **macro,
                "support": int(sum(int(row["support"]) for row in label_rows)),
                "labels_total": int(len(label_rows)),
                "labels_with_evaluation_support": int(len(estimable)),
                "unsupported_labels": "|".join(
                    str(row["label"]) for row in label_rows if not bool(row["estimated"])
                ),
            }
        )
        records.append(
            {
                "method": str(method),
                "split_scheme": str(scheme),
                "aggregation_rule": str(rule),
                "axis": str(axis),
                "average": "micro",
                "label": "",
                "evaluation_partition": evaluation_partition,
                "evaluation_unit": summary_unit(group_reference),
                "episode_groups": int(axis_frame["connected_episode_group_id"].nunique()),
                "complete_evaluation_groups": int(
                    group_reference["is_complete_within_evaluation_partition"].sum()
                ),
                "fragmented_evaluation_groups": int(
                    (~group_reference["is_complete_within_evaluation_partition"]).sum()
                ),
                **micro,
                "labels_total": int(len(label_rows)),
                "labels_with_evaluation_support": int(len(estimable)),
                "unsupported_labels": "|".join(
                    str(row["label"]) for row in label_rows if not bool(row["estimated"])
                ),
            }
        )
    return records


def select_reason_code_event_configuration(
    validation_metric_rows: pd.DataFrame,
    split_scheme: str = "spaced",
    axis: str = "mechanism",
    minimum_positive_validation_event_groups: int = REASON_CODE_MIN_VALIDATION_EPISODE_GROUPS,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Choose one complete-event reason-code configuration from validation metrics only."""
    required = {
        "method",
        "split_scheme",
        "aggregation_rule",
        "axis",
        "average",
        "evaluation_partition",
        "evaluation_unit",
        "complete_evaluation_groups",
        "fragmented_evaluation_groups",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "labels_total",
        "labels_with_evaluation_support",
        "unsupported_labels",
        "estimated",
        "support",
    }
    missing = sorted(required.difference(validation_metric_rows.columns))
    if missing:
        raise KeyError(f"reason-code event selection metrics lack fields: {missing}")
    source = validation_metric_rows.copy()
    if not source["evaluation_partition"].eq("validation").all():
        raise ValueError("reason-code event configuration selection accepts validation metrics only")
    if str(split_scheme) != "spaced":
        raise ValueError("reason-code event configuration selection is limited to the spaced split")
    if str(axis) != "mechanism":
        raise ValueError("reason-code event configuration selection is limited to mechanism labels")
    if int(minimum_positive_validation_event_groups) < 1:
        raise ValueError("reason-code event selection needs a positive validation-event support minimum")
    candidates = source.loc[
        source["split_scheme"].eq(str(split_scheme))
        & source["axis"].eq(str(axis))
        & source["average"].eq("macro")
    ].copy()
    expected = {
        (method, rule)
        for method in ("logistic", "gradient_boosted", "rgfn")
        for rule in ("any", "majority", "mean_probability")
    }
    actual = {
        (str(row.method), str(row.aggregation_rule))
        for row in candidates.loc[:, ["method", "aggregation_rule"]].itertuples(index=False)
    }
    if actual != expected or len(candidates) != len(expected):
        missing_candidates = sorted(expected.difference(actual))
        unexpected_candidates = sorted(actual.difference(expected))
        raise ValueError(
            "reason-code event selection requires all predefined validation candidates; "
            f"missing={missing_candidates}, unexpected={unexpected_candidates}"
        )
    if not candidates["evaluation_unit"].eq("complete_connected_event_group").all():
        raise ValueError("reason-code event selection requires complete validation connected-event groups")
    if pd.to_numeric(candidates["fragmented_evaluation_groups"], errors="raise").ne(0).any():
        raise ValueError("reason-code event selection cannot use fragmented validation groups")
    score_columns = ["precision", "recall", "f1", "accuracy"]
    if not np.isfinite(candidates.loc[:, score_columns].to_numpy(dtype=float)).all():
        raise ValueError("reason-code event selection requires estimable validation macro metrics")
    label_rows = source.loc[
        source["split_scheme"].eq(str(split_scheme))
        & source["axis"].eq(str(axis))
        & source["average"].eq("per_label")
    ].copy()
    label_supports = {
        (str(method), str(rule)): {
            str(row.label): int(row.support)
            for row in frame.itertuples(index=False)
        }
        for (method, rule), frame in label_rows.groupby(["method", "aggregation_rule"], sort=True)
    }
    if set(label_supports) != expected or len({tuple(sorted(value.items())) for value in label_supports.values()}) != 1:
        raise ValueError("reason-code event selection candidates have inconsistent evaluable labels")
    support_by_label = next(iter(label_supports.values()))
    all_labels = sorted(support_by_label)
    labels_with_positive_validation_support = sorted(
        label for label, support in support_by_label.items() if support > 0
    )
    evaluable_labels = sorted(
        label
        for label, support in support_by_label.items()
        if support >= int(minimum_positive_validation_event_groups)
    )
    excluded_low_support_labels = sorted(
        label
        for label, support in support_by_label.items()
        if support < int(minimum_positive_validation_event_groups)
    )
    if not evaluable_labels:
        raise ValueError("reason-code event selection has no labels meeting the validation-event support minimum")
    zero_support_labels = sorted(
        label for label, support in support_by_label.items() if support == 0
    )
    support_detail = "|".join(
        f"{label}:{support_by_label[label]}" for label in all_labels
    )
    candidate_rows: list[dict[str, object]] = []
    for macro_row in candidates.itertuples(index=False):
        method = str(macro_row.method)
        rule = str(macro_row.aggregation_rule)
        labels = label_rows.loc[
            label_rows["method"].eq(method)
            & label_rows["aggregation_rule"].eq(rule)
            & label_rows["label"].isin(evaluable_labels)
        ].copy()
        if set(labels["label"].astype(str)) != set(evaluable_labels):
            raise ValueError("reason-code event selection candidate is missing a support-qualified label")
        values = {
            metric: float(pd.to_numeric(labels[metric], errors="raise").mean())
            for metric in ("precision", "recall", "f1", "accuracy")
        }
        candidate_rows.append(
            {
                "method": method,
                "split_scheme": str(macro_row.split_scheme),
                "aggregation_rule": rule,
                "axis": str(macro_row.axis),
                "complete_validation_event_groups": int(
                    macro_row.complete_evaluation_groups
                ),
                "all_positive_validation_labels": "|".join(
                    labels_with_positive_validation_support
                ),
                "zero_support_validation_labels": "|".join(zero_support_labels),
                "validation_event_support_by_label": support_detail,
                "all_positive_validation_macro_precision": float(macro_row.precision),
                "all_positive_validation_macro_recall": float(macro_row.recall),
                "all_positive_validation_macro_f1": float(macro_row.f1),
                "all_positive_validation_macro_accuracy": float(macro_row.accuracy),
                "selection_macro_precision": values["precision"],
                "selection_macro_recall": values["recall"],
                "selection_macro_f1": values["f1"],
                "selection_macro_accuracy": values["accuracy"],
                "selection_labels": "|".join(evaluable_labels),
                "excluded_low_support_labels": "|".join(excluded_low_support_labels),
                "minimum_positive_validation_event_groups": int(
                    minimum_positive_validation_event_groups
                ),
            }
        )
    candidates = pd.DataFrame(candidate_rows)
    candidates["selection_eligible"] = True
    candidates["selection_criterion"] = "validation_macro_f1"
    candidates["selection_tie_break_order"] = (
        "macro_f1_desc|macro_precision_desc|macro_accuracy_desc|macro_recall_desc|"
        "method_name_asc|aggregation_rule_name_asc"
    )
    ordered = candidates.sort_values(
        [
            "selection_macro_f1",
            "selection_macro_precision",
            "selection_macro_accuracy",
            "selection_macro_recall",
            "method",
            "aggregation_rule",
        ],
        ascending=[False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered["validation_selection_rank"] = np.arange(1, len(ordered) + 1, dtype=int)
    selected_method = str(ordered.loc[0, "method"])
    selected_rule = str(ordered.loc[0, "aggregation_rule"])
    ordered["selected"] = (
        ordered["method"].eq(selected_method)
        & ordered["aggregation_rule"].eq(selected_rule)
    )
    selected = ordered.loc[ordered["selected"]].iloc[0]
    trace = {
        "selection_source": "spaced_validation_complete_connected_event_groups",
        "selection_partition": "validation",
        "selection_split_scheme": str(split_scheme),
        "axis": str(axis),
        "evaluation_unit": "complete_connected_event_group",
        "selection_criterion": "validation_macro_f1_over_support_qualified_labels",
        "selection_tie_break_order": str(selected["selection_tie_break_order"]),
        "predefined_candidate_count": int(len(ordered)),
        "selected_method": selected_method,
        "selected_aggregation_rule": selected_rule,
        "selected_validation_metrics": {
            "precision": float(selected["selection_macro_precision"]),
            "recall": float(selected["selection_macro_recall"]),
            "f1": float(selected["selection_macro_f1"]),
            "accuracy": float(selected["selection_macro_accuracy"]),
        },
        "selection_labels": evaluable_labels,
        "minimum_positive_validation_event_groups": int(
            minimum_positive_validation_event_groups
        ),
        "labels_total": int(len(all_labels)),
        "labels_with_positive_validation_support": int(
            len(labels_with_positive_validation_support)
        ),
        "labels_support_qualified": int(len(evaluable_labels)),
        "zero_support_validation_labels": zero_support_labels,
        "excluded_low_support_labels": excluded_low_support_labels,
        "test_metrics_read_during_configuration_selection": False,
        "test_metrics_available_to_selector": False,
        "retrospective_ground_truth_event_grouping": True,
    }
    return trace, ordered


def build_reason_code_cv_folds(
    target: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
    requested_folds: int = REASON_CODE_CV_REQUESTED_FOLDS,
    groups: np.ndarray | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    """Build deterministic, training-only stratified folds for one reason-code label."""
    selected = np.asarray(train_indices, dtype=np.int64)
    values = np.asarray(target, dtype=int)[selected]
    if selected.ndim != 1 or not len(selected):
        raise ValueError("reason-code cross-validation requires non-empty training indices")
    if not np.isin(values, [0, 1]).all():
        raise ValueError("reason-code cross-validation targets must be binary")
    if int(requested_folds) < 2:
        raise ValueError("reason-code cross-validation requires at least two requested folds")
    positives = int(np.equal(values, 1).sum())
    negatives = int(np.equal(values, 0).sum())
    selected_groups: np.ndarray | None = None
    if groups is not None:
        all_groups = np.asarray(groups, dtype=object).astype(str)
        if all_groups.shape != np.asarray(target).shape:
            raise ValueError("reason-code CV groups must align with the full target array")
        selected_groups = all_groups[selected]
        positive_group_count = int(len(np.unique(selected_groups[values == 1])))
        negative_group_count = int(len(np.unique(selected_groups[values == 0])))
        effective_folds = min(int(requested_folds), positive_group_count, negative_group_count)
        grouping = "connected_source_episode"
    else:
        positive_group_count = positives
        negative_group_count = negatives
        effective_folds = min(int(requested_folds), positives, negatives)
        grouping = "row"
    base = {
        "cv_requested_folds": int(requested_folds),
        "cv_effective_folds": int(effective_folds) if effective_folds >= 2 else 0,
        "cv_train_support": positives,
        "cv_train_negatives": negatives,
        "cv_positive_group_count": positive_group_count,
        "cv_negative_group_count": negative_group_count,
        "cv_grouping": grouping,
        "cv_seed": int(seed),
    }
    if effective_folds < 2:
        return [], {
            **base,
            "cv_status": "not_possible_insufficient_minority_support",
            "cv_fold_details": [],
        }
    for candidate_folds in range(int(effective_folds), 1, -1):
        if selected_groups is None:
            splitter = StratifiedKFold(
                n_splits=int(candidate_folds),
                shuffle=True,
                random_state=int(seed),
            )
            raw_folds = splitter.split(np.zeros(len(values)), values)
        else:
            splitter = StratifiedGroupKFold(
                n_splits=int(candidate_folds),
                shuffle=True,
                random_state=int(seed),
            )
            raw_folds = splitter.split(np.zeros(len(values)), values, selected_groups)
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        details: list[dict[str, object]] = []
        valid = True
        for fold_index, (fit_local, oof_local) in enumerate(raw_folds, start=1):
            fit_indices = selected[np.asarray(fit_local, dtype=np.int64)]
            oof_indices = selected[np.asarray(oof_local, dtype=np.int64)]
            fit_target = np.asarray(target, dtype=int)[fit_indices]
            oof_target = np.asarray(target, dtype=int)[oof_indices]
            if (
                not np.equal(fit_target, 1).any()
                or not np.equal(fit_target, 0).any()
                or not np.equal(oof_target, 1).any()
                or not np.equal(oof_target, 0).any()
            ):
                valid = False
                break
            detail = {
                "fold": int(fold_index),
                "fit_examples": int(len(fit_indices)),
                "oof_examples": int(len(oof_indices)),
                "fit_support": int(fit_target.sum()),
                "oof_support": int(oof_target.sum()),
                "fit_key_sha256": hashlib.sha256(
                    np.sort(fit_indices).astype("<i8", copy=False).tobytes()
                ).hexdigest(),
                "oof_key_sha256": hashlib.sha256(
                    np.sort(oof_indices).astype("<i8", copy=False).tobytes()
                ).hexdigest(),
            }
            if selected_groups is not None:
                fit_groups = set(selected_groups[np.asarray(fit_local, dtype=np.int64)].tolist())
                oof_groups = set(selected_groups[np.asarray(oof_local, dtype=np.int64)].tolist())
                if fit_groups.intersection(oof_groups):
                    raise RuntimeError("reason-code grouped CV split an episode group across folds")
                detail.update(
                    {
                        "fit_group_count": int(len(fit_groups)),
                        "oof_group_count": int(len(oof_groups)),
                    }
                )
            folds.append((fit_indices, oof_indices))
            details.append(detail)
        if valid:
            reduced = int(candidate_folds) != int(requested_folds)
            return folds, {
                **base,
                "cv_effective_folds": int(candidate_folds),
                "cv_status": "reduced_to_group_support" if reduced else "completed",
                "cv_fold_details": details,
            }
    return [], {
        **base,
        "cv_effective_folds": 0,
        "cv_status": "not_possible_after_grouped_fold_validation",
        "cv_fold_details": [],
    }


def select_reason_code_threshold_mean_fold_f1(
    labels: np.ndarray,
    probabilities: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[float, dict[str, object], int]:
    """Select an exact threshold by mean per-fold OOF F1, with higher-threshold ties."""
    values = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    folds = np.asarray(fold_ids, dtype=int)
    if values.ndim != 1 or scores.ndim != 1 or folds.ndim != 1:
        raise ValueError("reason-code OOF threshold inputs must be one-dimensional")
    if len(values) != len(scores) or len(values) != len(folds) or not len(values):
        raise ValueError("reason-code OOF threshold inputs have incompatible lengths")
    if not np.isin(values, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("reason-code OOF threshold inputs must be finite binary labels and scores")
    if not np.equal(values, 1).any() or not np.equal(values, 0).any():
        raise ValueError("reason-code OOF threshold selection needs both classes")
    fold_values = np.unique(folds)
    if len(fold_values) < 2:
        raise ValueError("reason-code OOF threshold selection needs at least two folds")
    fold_lookup = {int(value): index for index, value in enumerate(fold_values.tolist())}
    compact_folds = np.asarray([fold_lookup[int(value)] for value in folds], dtype=np.int64)
    fold_support = np.bincount(
        compact_folds,
        weights=np.equal(values, 1).astype(np.int64),
        minlength=len(fold_values),
    ).astype(np.int64)
    fold_count = np.bincount(compact_folds, minlength=len(fold_values)).astype(np.int64)
    if np.any(fold_support == 0) or np.any(fold_support == fold_count):
        raise ValueError("each OOF fold needs both reason-code classes")

    order = np.argsort(-scores, kind="mergesort")
    true_positive = np.zeros(len(fold_values), dtype=np.int64)
    false_positive = np.zeros(len(fold_values), dtype=np.int64)
    false_negative = fold_support.copy()
    best_threshold: float | None = None
    best_mean_f1 = -1.0
    candidate_count = 0
    cursor = 0
    while cursor < len(order):
        score = float(scores[order[cursor]])
        stop = cursor + 1
        while stop < len(order) and float(scores[order[stop]]) == score:
            stop += 1
        group = order[cursor:stop]
        group_folds = compact_folds[group]
        group_positive = values[group] == 1
        np.add.at(true_positive, group_folds[group_positive], 1)
        np.add.at(false_negative, group_folds[group_positive], -1)
        np.add.at(false_positive, group_folds[~group_positive], 1)
        denominator = 2 * true_positive + false_positive + false_negative
        fold_f1 = np.divide(
            2.0 * true_positive,
            denominator,
            out=np.zeros(len(fold_values), dtype=float),
            where=denominator > 0,
        )
        mean_f1 = float(np.mean(fold_f1))
        candidate_count += 1
        if mean_f1 > best_mean_f1:
            best_threshold = score
            best_mean_f1 = mean_f1
        cursor = stop
    if best_threshold is None:
        raise RuntimeError("reason-code OOF threshold selection produced no candidates")
    fold_metrics = [
        {
            "fold": int(fold_value),
            **reason_code_metrics(
                values[folds == fold_value],
                scores[folds == fold_value],
                float(best_threshold),
            ),
        }
        for fold_value in fold_values
    ]
    return float(best_threshold), {
        "mean_oof_fold_f1": float(best_mean_f1),
        "pooled_oof_metrics": reason_code_metrics(values, scores, float(best_threshold)),
        "fold_metrics": fold_metrics,
    }, int(candidate_count)


def select_reason_code_threshold_from_training_oof(
    target: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
    fit_predict_fold: Any,
    requested_folds: int = REASON_CODE_CV_REQUESTED_FOLDS,
    groups: np.ndarray | None = None,
) -> tuple[float, dict[str, object]]:
    """Fit fold-local models and freeze a threshold before outer validation or test use."""
    selected = np.asarray(train_indices, dtype=np.int64)
    folds, plan = build_reason_code_cv_folds(
        target,
        selected,
        seed=int(seed),
        requested_folds=int(requested_folds),
        groups=groups,
    )
    if not folds:
        return 0.5, {
            **plan,
            "selection_source": "predeclared_fixed_0_5",
            "threshold_policy": "predeclared_fixed_0_5",
            "selected_threshold": 0.5,
            "oof_candidate_count": 0,
            "oof_coverage_complete": False,
            "oof_selection_metrics": None,
        }
    local_position = {int(index): position for position, index in enumerate(selected.tolist())}
    oof_probabilities = np.full(len(selected), np.nan, dtype=float)
    fold_ids = np.full(len(selected), -1, dtype=np.int64)
    details = [dict(value) for value in plan["cv_fold_details"]]
    for fold_number, (fit_indices, oof_indices) in enumerate(folds, start=1):
        result = fit_predict_fold(fit_indices, oof_indices, int(fold_number))
        if isinstance(result, tuple):
            probabilities, extra = result
        else:
            probabilities, extra = result, {}
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(oof_indices),):
            raise ValueError("reason-code CV fold predictions have an unexpected shape")
        positions = np.asarray([local_position[int(index)] for index in oof_indices], dtype=np.int64)
        if np.isfinite(oof_probabilities[positions]).any():
            raise RuntimeError("reason-code CV generated duplicate OOF predictions")
        oof_probabilities[positions] = probabilities
        fold_ids[positions] = int(fold_number)
        if extra:
            details[fold_number - 1].update(dict(extra))
    if not np.isfinite(oof_probabilities).all() or np.any(fold_ids < 0):
        raise RuntimeError("reason-code CV did not cover every outer-training row exactly once")
    threshold, oof_metrics, candidate_count = select_reason_code_threshold_mean_fold_f1(
        np.asarray(target, dtype=int)[selected],
        oof_probabilities,
        fold_ids,
    )
    return float(threshold), {
        **plan,
        "cv_fold_details": details,
        "selection_source": "training_oof_cross_validation",
        "threshold_policy": "training_oof_cv_mean_f1",
        "selected_threshold": float(threshold),
        "oof_candidate_count": int(candidate_count),
        "oof_coverage_complete": True,
        "oof_selection_metrics": oof_metrics,
    }


def fit_reason_code_models(
    values: np.ndarray,
    examples: dict[str, np.ndarray],
    splits: dict[str, dict[str, np.ndarray]],
    config: HourlyReasonCodeConfig,
    model_factory: Any = make_reason_code_classifier,
) -> tuple[list[FittedReasonCodeModel], list[dict[str, object]]]:
    """Fit final reason-code estimators after training-only OOF threshold selection."""
    axes = _reason_axis_specs(examples)
    groups = reason_code_cv_group_ids(examples["source_episode_ids"])
    fitted: list[FittedReasonCodeModel] = []
    trace: list[dict[str, object]] = []
    for scheme in ("random", "spaced"):
        if scheme not in splits:
            raise KeyError(f"reason-code splits lack {scheme}")
        train_indices = np.asarray(splits[scheme]["train"], dtype=np.int64)
        validation_indices = np.asarray(splits[scheme]["validation"], dtype=np.int64)
        for axis, (targets, names) in axes.items():
            for label_index, label_name in enumerate(names):
                target = targets[:, label_index]
                train_target = target[train_indices]
                validation_target = target[validation_indices]

                def fit_predict_fold(
                    fold_train_indices: np.ndarray,
                    fold_oof_indices: np.ndarray,
                    _fold_number: int,
                ) -> tuple[np.ndarray, dict[str, object]]:
                    fold_weight = resolve_reason_code_class_weight(target[fold_train_indices])
                    fold_estimator = model_factory(config, fold_weight)
                    fold_estimator.fit(values[fold_train_indices], target[fold_train_indices])
                    probabilities = fold_estimator.predict_proba(values[fold_oof_indices])[:, 1]
                    return probabilities, {"positive_class_weight": float(fold_weight)}

                threshold, cv_trace = select_reason_code_threshold_from_training_oof(
                    target,
                    train_indices,
                    seed=_reason_code_cv_seed(int(config.seed), scheme, axis, str(label_name)),
                    fit_predict_fold=fit_predict_fold,
                    groups=groups,
                )
                class_weight = resolve_reason_code_class_weight(train_target)
                estimator = model_factory(config, class_weight)
                estimator.fit(values[train_indices], train_target)
                validation_probabilities = estimator.predict_proba(values[validation_indices])[:, 1]
                validation_metrics = reason_code_metrics(
                    validation_target,
                    validation_probabilities,
                    threshold,
                )
                (
                    validation_reference_threshold,
                    validation_reference_metrics,
                    validation_reference_candidate_count,
                ) = select_reason_code_threshold(
                    validation_target,
                    validation_probabilities,
                )
                selection_trace = {
                    "method": "gradient_boosted",
                    "split_scheme": scheme,
                    "axis": axis,
                    "label": str(label_name),
                    **cv_trace,
                    "train_support": int(train_target.sum()),
                    "validation_support": int(validation_target.sum()),
                    "positive_class_weight": float(class_weight),
                    "selected_threshold": float(threshold),
                    "validation_sanity_metrics": validation_metrics,
                    "validation_metrics": validation_metrics,
                    "single_validation_reference_threshold": float(validation_reference_threshold),
                    "single_validation_reference_metrics": validation_reference_metrics,
                    "single_validation_reference_candidate_count": int(validation_reference_candidate_count),
                    "single_validation_reference_is_diagnostic_only": True,
                    "outer_validation_used_for_threshold": False,
                    "outer_test_used_for_threshold": False,
                    "test_metrics_read_during_selection": False,
                }
                fitted.append(
                    FittedReasonCodeModel(
                        method="gradient_boosted",
                        split_scheme=scheme,
                        axis=axis,
                        label_name=str(label_name),
                        label_index=int(label_index),
                        estimator=estimator,
                        positive_class_weight=class_weight,
                        selected_threshold=float(threshold),
                        train_support=int(train_target.sum()),
                        validation_support=int(validation_target.sum()),
                        validation_metrics=validation_metrics,
                        selection_trace=selection_trace,
                    )
                )
                trace.append(selection_trace)
    return fitted, trace


def fit_reason_code_logistic_models(
    values: np.ndarray,
    examples: dict[str, np.ndarray],
    splits: dict[str, dict[str, np.ndarray]],
    config: HourlyReasonCodeLogisticConfig,
    model_factory: Any = make_reason_code_logistic_classifier,
) -> tuple[list[FittedReasonCodeModel], list[dict[str, object]]]:
    axes = _reason_axis_specs(examples)
    groups = reason_code_cv_group_ids(examples["source_episode_ids"])
    fitted: list[FittedReasonCodeModel] = []
    trace: list[dict[str, object]] = []
    for scheme in ("random", "spaced"):
        if scheme not in splits:
            raise KeyError(f"reason-code splits lack {scheme}")
        train_indices = np.asarray(splits[scheme]["train"], dtype=np.int64)
        validation_indices = np.asarray(splits[scheme]["validation"], dtype=np.int64)
        scaler = HourlyLogisticScaler.fit(values, train_indices)
        scaled_values = scaler.transform(values)
        for axis, (targets, names) in axes.items():
            for label_index, label_name in enumerate(names):
                target = targets[:, label_index]
                train_target = target[train_indices]
                validation_target = target[validation_indices]

                def fit_predict_fold(
                    fold_train_indices: np.ndarray,
                    fold_oof_indices: np.ndarray,
                    _fold_number: int,
                ) -> tuple[np.ndarray, dict[str, object]]:
                    fold_scaler = HourlyLogisticScaler.fit(values, fold_train_indices)
                    fold_weight = resolve_reason_code_class_weight(target[fold_train_indices])
                    fold_estimator = model_factory(config, fold_weight)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=ConvergenceWarning)
                        fold_estimator.fit(
                            fold_scaler.transform(values[fold_train_indices]),
                            target[fold_train_indices],
                        )
                    probabilities = fold_estimator.predict_proba(
                        fold_scaler.transform(values[fold_oof_indices])
                    )[:, 1]
                    return probabilities, {
                        "positive_class_weight": float(fold_weight),
                        "scaler_fit_examples": int(len(fold_train_indices)),
                    }

                threshold, cv_trace = select_reason_code_threshold_from_training_oof(
                    target,
                    train_indices,
                    seed=_reason_code_cv_seed(int(config.seed), scheme, axis, str(label_name)),
                    fit_predict_fold=fit_predict_fold,
                    groups=groups,
                )
                class_weight = resolve_reason_code_class_weight(train_target)
                estimator = model_factory(config, class_weight)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    estimator.fit(scaled_values[train_indices], train_target)
                validation_probabilities = estimator.predict_proba(scaled_values[validation_indices])[:, 1]
                validation_metrics = reason_code_metrics(
                    validation_target,
                    validation_probabilities,
                    threshold,
                )
                (
                    validation_reference_threshold,
                    validation_reference_metrics,
                    validation_reference_candidate_count,
                ) = select_reason_code_threshold(
                    validation_target,
                    validation_probabilities,
                )
                selection_trace = {
                    "method": "logistic",
                    "split_scheme": scheme,
                    "axis": axis,
                    "label": str(label_name),
                    **cv_trace,
                    "train_support": int(train_target.sum()),
                    "validation_support": int(validation_target.sum()),
                    "positive_class_weight": float(class_weight),
                    "selected_threshold": float(threshold),
                    "validation_sanity_metrics": validation_metrics,
                    "validation_metrics": validation_metrics,
                    "single_validation_reference_threshold": float(validation_reference_threshold),
                    "single_validation_reference_metrics": validation_reference_metrics,
                    "single_validation_reference_candidate_count": int(validation_reference_candidate_count),
                    "single_validation_reference_is_diagnostic_only": True,
                    "outer_validation_used_for_threshold": False,
                    "outer_test_used_for_threshold": False,
                    "test_metrics_read_during_selection": False,
                }
                fitted.append(
                    FittedReasonCodeModel(
                        method="logistic",
                        split_scheme=scheme,
                        axis=axis,
                        label_name=str(label_name),
                        label_index=int(label_index),
                        estimator=estimator,
                        positive_class_weight=class_weight,
                        selected_threshold=float(threshold),
                        train_support=int(train_target.sum()),
                        validation_support=int(validation_target.sum()),
                        validation_metrics=validation_metrics,
                        selection_trace=selection_trace,
                        scaler=scaler,
                    )
                )
                trace.append(selection_trace)
    return fitted, trace


def reason_code_axis_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    names: np.ndarray,
) -> dict[str, object]:
    predictions = np.asarray(probabilities >= thresholds.reshape(1, -1), dtype=int)
    rows = []
    for index, name in enumerate(names):
        metric = reason_code_metrics(labels[:, index], probabilities[:, index], float(thresholds[index]))
        rows.append({"label": str(name), **metric})
    estimable = [row for row in rows if bool(row["estimated"])]
    macro = {
        key: (float(np.mean([float(row[key]) for row in estimable])) if estimable else None)
        for key in ("precision", "recall", "f1", "accuracy")
    }
    flat_truth = np.asarray(labels, dtype=int).reshape(-1)
    flat_predictions = predictions.reshape(-1)
    matrix = confusion_matrix(flat_truth, flat_predictions, labels=[0, 1])
    micro = {
        "precision": float(precision_score(flat_truth, flat_predictions, zero_division=0)),
        "recall": float(recall_score(flat_truth, flat_predictions, zero_division=0)),
        "f1": float(f1_score(flat_truth, flat_predictions, zero_division=0)),
        "accuracy": float(accuracy_score(flat_truth, flat_predictions)),
        "support": int(flat_truth.sum()),
        "tp": int(matrix[1, 1]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tn": int(matrix[0, 0]),
    }
    return {
        "per_label": rows,
        "macro": {
            **macro,
            "labels_total": int(len(rows)),
            "labels_with_test_support": int(len(estimable)),
            "unsupported_labels": [str(row["label"]) for row in rows if not bool(row["estimated"])],
        },
        "micro": micro,
        "predictions": predictions,
    }


def _reason_code_model_probabilities(
    fitted: FittedReasonCodeModel,
    values: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    selected = np.asarray(values, dtype=np.float32)[np.asarray(indices, dtype=np.int64)]
    if fitted.scaler is not None:
        selected = fitted.scaler.transform(selected)
    return np.asarray(fitted.estimator.predict_proba(selected)[:, 1], dtype=float)


def evaluate_reason_code_models(
    fitted: list[FittedReasonCodeModel],
    values: np.ndarray,
    examples: dict[str, np.ndarray],
    splits: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, dict[str, dict[str, object]]], list[dict[str, object]]]:
    """Evaluate already-selected reason-code models once on held-out fault-hours."""
    axes = _reason_axis_specs(examples)
    by_identity = {(item.split_scheme, item.axis, item.label_name): item for item in fitted}
    evaluations: dict[str, dict[str, dict[str, object]]] = {}
    rows: list[dict[str, object]] = []
    for scheme in ("random", "spaced"):
        test_indices = np.asarray(splits[scheme]["test"], dtype=np.int64)
        evaluations[scheme] = {}
        for axis, (targets, names) in axes.items():
            probabilities = np.zeros((len(test_indices), len(names)), dtype=float)
            thresholds = np.zeros(len(names), dtype=float)
            for label_index, label_name in enumerate(names):
                item = by_identity.get((scheme, axis, str(label_name)))
                if item is None:
                    raise KeyError(f"missing fitted {scheme} {axis} model for {label_name}")
                probabilities[:, label_index] = _reason_code_model_probabilities(
                    item,
                    values,
                    test_indices,
                )
                thresholds[label_index] = item.selected_threshold
            summary = reason_code_axis_summary(targets[test_indices], probabilities, thresholds, names)
            for label_index, metric in enumerate(summary["per_label"]):
                item = by_identity[(scheme, axis, str(names[label_index]))]
                rows.append(
                    {
                        "method": item.method,
                        "split_scheme": scheme,
                        "axis": axis,
                        "label": str(names[label_index]),
                        "threshold": float(item.selected_threshold),
                        "threshold_source": str(item.selection_trace["selection_source"]),
                        "threshold_policy": str(item.selection_trace["threshold_policy"]),
                        "train_support": int(item.train_support),
                        "validation_support": int(item.validation_support),
                        **metric,
                    }
                )
            evaluations[scheme][axis] = {
                "test_indices": test_indices,
                "thresholds": thresholds,
                "probabilities": probabilities,
                "truth": targets[test_indices].astype(np.int64),
                **summary,
            }
    return evaluations, rows


def save_reason_code_model(
    fitted: FittedReasonCodeModel,
    path: Path,
    feature_names: list[str],
    window_hours: int,
    config: HourlyReasonCodeConfig | HourlyReasonCodeLogisticConfig,
    manifest_path: Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "estimator": fitted.estimator,
            "feature_names": feature_names,
            "window_hours": int(window_hours),
            "mode": "reason_codes",
            "method": fitted.method,
            "axis": fitted.axis,
            "label": fitted.label_name,
            "split_scheme": fitted.split_scheme,
            "selected_threshold": float(fitted.selected_threshold),
            "threshold_source": str(fitted.selection_trace["selection_source"]),
            "threshold_policy": str(fitted.selection_trace["threshold_policy"]),
            "positive_class_weight": float(fitted.positive_class_weight),
            "train_support": int(fitted.train_support),
            "validation_support": int(fitted.validation_support),
            "selection_trace": fitted.selection_trace,
            "scaler_payload": None if fitted.scaler is None else fitted.scaler.payload(),
            "manifest_path": str(manifest_path),
            "config": asdict(config),
        },
        destination,
    )
    return destination


def _detector_channel(column: str) -> str:
    for _, prefix in DETECTOR_GROUPS:
        if column.startswith(prefix):
            return column[len(prefix):]
    return column


def detector_channel_evidence_for_rows(
    feature_frame: pd.DataFrame,
    rows: pd.DataFrame,
) -> pd.DataFrame:
    required = {"station_id", "hour"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise KeyError(f"reason-code rows lack evidence keys: {missing}")
    source = feature_frame.copy()
    if "station_id" not in source.columns:
        raise KeyError("feature matrix lacks station_id for detector evidence")
    time_column = next(
        (name for name in ("hour", "hour_utc", "time_utc", "timestamp") if name in source.columns),
        None,
    )
    if time_column is None:
        raise KeyError("feature matrix lacks a timestamp for detector evidence")
    source["station_id"] = source["station_id"].astype(str)
    source["hour"] = pd.to_datetime(source[time_column], utc=True, format="mixed").astype(str)
    if source.duplicated(["station_id", "hour"]).any():
        raise ValueError("feature matrix has duplicate station-hour evidence keys")
    grouped = detector_columns_by_group(list(source.columns))
    detector_columns = [column for _, columns in grouped.items() for column in columns]
    source = source.loc[:, ["station_id", "hour", *detector_columns]]
    keys = rows.loc[:, ["station_id", "hour"]].copy()
    keys["station_id"] = keys["station_id"].astype(str)
    keys["hour"] = pd.to_datetime(keys["hour"], utc=True, format="mixed").astype(str)
    joined = keys.merge(source, on=["station_id", "hour"], how="left", validate="many_to_one")
    groups: list[str] = []
    channels: list[str] = []
    components: list[str] = []
    for _, row in joined.iterrows():
        fired_groups: list[str] = []
        fired_channels: list[str] = []
        for group, columns in grouped.items():
            active = [
                column
                for column in columns
                if pd.notna(row[column]) and float(row[column]) > 0.0
            ]
            if active:
                fired_groups.append(group)
                fired_channels.extend(_detector_channel(column) for column in active)
        groups.append("|".join(fired_groups))
        channels.append("|".join(dict.fromkeys(fired_channels)))
        components.append(
            "|".join(
                dict.fromkeys(sensor_group_for_channel(channel) for channel in fired_channels)
            )
        )
    return pd.DataFrame(
        {
            "detector_groups_fired": groups,
            "detector_channels_fired": channels,
            "detector_component_evidence": components,
        },
        index=rows.index,
    )


def _serialise_target_row(values: np.ndarray, names: np.ndarray) -> str:
    return "|".join(str(name) for name, active in zip(names, values) if int(active) == 1)


def reason_code_prediction_frame(
    examples: dict[str, np.ndarray],
    evaluations: dict[str, dict[str, dict[str, object]]],
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    axes = _reason_axis_specs(examples)
    rows: list[pd.DataFrame] = []
    for scheme in ("random", "spaced"):
        if scheme not in evaluations:
            continue
        mechanism = evaluations[scheme]["mechanism"]
        component = evaluations[scheme]["component"]
        mechanism_indices = np.asarray(mechanism["test_indices"], dtype=np.int64)
        component_indices = np.asarray(component["test_indices"], dtype=np.int64)
        if not np.array_equal(mechanism_indices, component_indices):
            raise ValueError(f"{scheme} mechanism and component test membership differ")
        mechanism_targets, mechanism_names = axes["mechanism"]
        component_targets, component_names = axes["component"]
        result = pd.DataFrame(
            {
                "split_scheme": scheme,
                "split": "test",
                "station_id": np.asarray(examples["station_id"], dtype=object)[mechanism_indices].astype(str),
                "hour": np.asarray(examples["hour"], dtype=object)[mechanism_indices].astype(str),
                "source_episode_ids": np.asarray(examples["source_episode_ids"], dtype=object)[mechanism_indices].astype(str),
                "tensor_detector_groups": np.asarray(examples["detectors_fired"], dtype=object)[mechanism_indices].astype(str),
                "true_mechanisms": [
                    _serialise_target_row(row, mechanism_names)
                    for row in mechanism_targets[mechanism_indices]
                ],
                "true_components": [
                    _serialise_target_row(row, component_names)
                    for row in component_targets[mechanism_indices]
                ],
                "predicted_mechanisms": [
                    _serialise_target_row(row, mechanism_names)
                    for row in np.asarray(mechanism["predictions"], dtype=int)
                ],
                "predicted_components": [
                    _serialise_target_row(row, component_names)
                    for row in np.asarray(component["predictions"], dtype=int)
                ],
            }
        )
        for axis, summary, names in (
            ("mechanism", mechanism, mechanism_names),
            ("component", component, component_names),
        ):
            probabilities = np.asarray(summary["probabilities"], dtype=float)
            thresholds = np.asarray(summary["thresholds"], dtype=float)
            predictions = np.asarray(summary["predictions"], dtype=int)
            for index, name in enumerate(names):
                result[f"{axis}_probability_{name}"] = probabilities[:, index]
                result[f"{axis}_threshold_{name}"] = thresholds[index]
                result[f"{axis}_prediction_{name}"] = predictions[:, index]
        evidence = detector_channel_evidence_for_rows(feature_frame, result)
        rows.append(pd.concat([result, evidence], axis=1))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["split_scheme", "station_id", "hour"]
    ).reset_index(drop=True)


def reason_code_summary_rows(
    evaluations: dict[str, dict[str, dict[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scheme in ("random", "spaced"):
        for axis in ("mechanism", "component"):
            summary = evaluations[scheme][axis]
            rows.append(
                {
                    "split_scheme": scheme,
                    "axis": axis,
                    "average": "macro",
                    **dict(summary["macro"]),
                }
            )
            rows.append(
                {
                    "split_scheme": scheme,
                    "axis": axis,
                    "average": "micro",
                    **dict(summary["micro"]),
                }
            )
    return rows


def reason_code_report(
    partition_trace: list[dict[str, object]],
    selection_trace: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> str:
    parts = [
        "HOURLY CONDITIONAL REASON-CODE MODELS",
        "",
        "TASK",
        "Each model is conditioned on an already-labelled fault hour and predicts independent multi-label mechanism and component reason codes.",
        "The binary detector, binary tensors, binary model artifacts, and binary metrics are read-only inputs to this workflow.",
        "",
        "SPLIT REUSE",
        pd.DataFrame(partition_trace).to_string(index=False),
        "",
        "SELECTION TRACE",
        "Thresholds are selected from grouped, training-only out-of-fold predictions. Outer validation is reported only as a frozen-threshold sanity check; no held-out test metric participates in selection.",
        pd.DataFrame(
            [
                {
                    key: value
                    for key, value in row.items()
                    if key != "validation_metrics"
                }
                for row in selection_trace
            ]
        ).to_string(index=False),
        "",
    ]
    metrics = pd.DataFrame(metric_rows)
    summaries = pd.DataFrame(summary_rows)
    for scheme in ("random", "spaced"):
        parts.extend([f"{scheme.upper()} HELD-OUT TEST RESULTS", ""])
        for axis in ("mechanism", "component"):
            frame = metrics.loc[
                metrics["split_scheme"].eq(scheme) & metrics["axis"].eq(axis),
                [
                    "label",
                    "train_support",
                    "validation_support",
                    "support",
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "accuracy",
                    "estimated",
                ],
            ].copy()
            for column in ("precision", "recall", "f1"):
                frame[column] = frame[column].map(lambda value: "N/A" if pd.isna(value) else value)
            parts.extend(
                [
                    f"{axis.upper()} PER-LABEL METRICS",
                    frame.to_string(index=False),
                    f"{axis.upper()} MACRO AND MICRO AVERAGES",
                    summaries.loc[
                        summaries["split_scheme"].eq(scheme) & summaries["axis"].eq(axis)
                    ].to_string(index=False),
                    "",
                ]
            )
    parts.extend(
        [
            "REPORTING NOTE",
            "Labels with zero positive held-out support retain their rows with precision, recall, and F1 marked N/A; they are excluded from the corresponding macro estimate and named in unsupported_labels.",
            "Reason-code labels are derived from the rule-and-evidence workflow, so these retrospective scores measure recovery of that taxonomy rather than independent field-verified root-cause accuracy.",
            "Held-out metrics are retrospective comparisons and do not select a configuration or operating point.",
        ]
    )
    return "\n".join(parts)


def reason_code_comparison_rows(
    evaluations_by_method: dict[str, dict[str, dict[str, dict[str, object]]]],
    status_by_method: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in ("logistic", "gradient_boosted", "rgfn"):
        method_evaluations = evaluations_by_method.get(method, {})
        method_status = status_by_method.get(method, {})
        for scheme in ("random", "spaced"):
            resolved_status = str(method_status.get(scheme, "completed"))
            for axis in ("mechanism", "component"):
                summary = method_evaluations.get(scheme, {}).get(axis)
                if summary is None:
                    rows.append(
                        {
                            "method": method,
                            "split_scheme": scheme,
                            "axis": axis,
                            "macro_precision": None,
                            "macro_recall": None,
                            "macro_f1": None,
                            "macro_accuracy": None,
                            "macro_labels_evaluable": 0,
                            "macro_unsupported_labels": [],
                            "micro_precision": None,
                            "micro_recall": None,
                            "micro_f1": None,
                            "micro_accuracy": None,
                            "micro_support": 0,
                            "status": resolved_status,
                        }
                    )
                    continue
                macro = dict(summary["macro"])
                micro = dict(summary["micro"])
                rows.append(
                    {
                        "method": method,
                        "split_scheme": scheme,
                        "axis": axis,
                        "macro_precision": macro["precision"],
                        "macro_recall": macro["recall"],
                        "macro_f1": macro["f1"],
                        "macro_accuracy": macro["accuracy"],
                        "macro_labels_evaluable": int(macro["labels_with_test_support"]),
                        "macro_unsupported_labels": list(macro["unsupported_labels"]),
                        "micro_precision": micro["precision"],
                        "micro_recall": micro["recall"],
                        "micro_f1": micro["f1"],
                        "micro_accuracy": micro["accuracy"],
                        "micro_support": int(micro["support"]),
                        "status": resolved_status,
                    }
                )
    return rows


def reason_code_threshold_robustness_rows(
    evaluations_by_method: dict[str, dict[str, dict[str, dict[str, object]]]],
    metric_rows_by_method: dict[str, list[dict[str, object]]],
    selection_trace: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Compare the frozen OOF threshold with a diagnostic-only old validation rule."""
    traces = {
        (
            str(row["method"]),
            str(row["split_scheme"]),
            str(row["axis"]),
            str(row["label"]),
        ): row
        for row in selection_trace
    }
    new_rows = {
        (
            str(row["method"]),
            str(row["split_scheme"]),
            str(row["axis"]),
            str(row["label"]),
        ): row
        for rows in metric_rows_by_method.values()
        for row in rows
    }
    result: list[dict[str, object]] = []
    for method in ("logistic", "gradient_boosted", "rgfn"):
        for scheme in ("random", "spaced"):
            evaluations = evaluations_by_method.get(method, {}).get(scheme, {})
            for axis in ("mechanism", "component"):
                summary = evaluations.get(axis)
                if summary is None:
                    continue
                probabilities = np.asarray(summary["probabilities"], dtype=float)
                truth = np.asarray(summary["truth"], dtype=int)
                names = [str(row["label"]) for row in summary["per_label"]]
                for label_index, label_name in enumerate(names):
                    identity = (method, scheme, axis, label_name)
                    trace = traces.get(identity)
                    new = new_rows.get(identity)
                    if trace is None or new is None:
                        raise KeyError(f"reason-code threshold trace is missing {identity}")
                    old_threshold = trace.get("single_validation_reference_threshold")
                    old_metrics = (
                        reason_code_metrics(
                            truth[:, label_index],
                            probabilities[:, label_index],
                            float(old_threshold),
                        )
                        if old_threshold is not None
                        else None
                    )
                    row: dict[str, object] = {
                        "method": method,
                        "split_scheme": scheme,
                        "axis": axis,
                        "label": label_name,
                        "train_support": int(new["train_support"]),
                        "validation_support": int(new["validation_support"]),
                        "test_support": int(new["support"]),
                        "old_threshold": old_threshold,
                        "new_threshold": float(new["threshold"]),
                        "old_threshold_source": "single_validation_diagnostic_only",
                        "new_threshold_source": str(new["threshold_source"]),
                        "old_test_metrics": old_metrics,
                        "new_test_metrics": {
                            key: new[key]
                            for key in ("precision", "recall", "f1", "accuracy", "estimated")
                        },
                    }
                    for prefix, metrics in (("old", old_metrics), ("new", row["new_test_metrics"])):
                        for metric_name in ("precision", "recall", "f1", "accuracy"):
                            value = None if metrics is None else metrics[metric_name]
                            row[f"{prefix}_test_{metric_name}"] = value
                    for metric_name in ("precision", "recall", "f1", "accuracy"):
                        old_value = row[f"old_test_{metric_name}"]
                        new_value = row[f"new_test_{metric_name}"]
                        row[f"delta_test_{metric_name}"] = (
                            None
                            if old_value is None or new_value is None
                            else float(new_value) - float(old_value)
                        )
                    result.append(row)
    return result


def _comparison_display_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    for name in (
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_accuracy",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "micro_accuracy",
    ):
        frame[name] = frame[name].map(lambda value: "N/A" if pd.isna(value) else value)
    frame["macro_unsupported_labels"] = frame["macro_unsupported_labels"].map(
        lambda values: "|".join(values) if values else ""
    )
    return frame


def _threshold_robustness_display_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "method",
        "split_scheme",
        "axis",
        "label",
        "train_support",
        "validation_support",
        "test_support",
        "old_threshold",
        "new_threshold",
        "old_test_precision",
        "new_test_precision",
        "delta_test_precision",
        "old_test_recall",
        "new_test_recall",
        "delta_test_recall",
        "old_test_f1",
        "new_test_f1",
        "delta_test_f1",
        "old_test_accuracy",
        "new_test_accuracy",
        "delta_test_accuracy",
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame.loc[:, columns].copy()
    for column in (
        "old_test_precision",
        "new_test_precision",
        "delta_test_precision",
        "old_test_recall",
        "new_test_recall",
        "delta_test_recall",
        "old_test_f1",
        "new_test_f1",
        "delta_test_f1",
        "old_test_accuracy",
        "new_test_accuracy",
        "delta_test_accuracy",
    ):
        frame[column] = frame[column].map(lambda value: "N/A" if pd.isna(value) else value)
    return frame


def _spaced_reason_code_diagnostic(metric_rows: list[dict[str, object]]) -> list[str]:
    frame = pd.DataFrame(metric_rows)
    if frame.empty:
        return ["No completed held-out reason-code metrics are available."]
    spaced = frame.loc[
        frame["split_scheme"].eq("spaced")
        & frame["estimated"].astype(bool)
        & frame["f1"].notna()
    ].copy()
    if spaced.empty:
        return ["No spaced label has positive held-out support."]
    pivot = spaced.pivot_table(
        index=["axis", "label"],
        columns="method",
        values="f1",
        aggfunc="first",
    )
    completed = [method for method in ("logistic", "gradient_boosted", "rgfn") if method in pivot.columns]
    if not completed:
        return ["No completed method has spaced per-label metrics."]
    weak = {
        method: sorted(
            f"{axis}:{label}"
            for (axis, label), value in pivot[method].items()
            if float(value) < 0.50
        )
        for method in completed
    }
    lines = [
        "The following descriptive comparison uses only already-finalized held-out metrics; it did not select a model, threshold, or deployment configuration.",
    ]
    for method in completed:
        labels = ", ".join(weak[method]) if weak[method] else "none"
        lines.append(f"{method} spaced labels below F1 0.50: {labels}.")
    if len(completed) == 3:
        shared = sorted(set.intersection(*(set(weak[method]) for method in completed)))
        same_sets = len({tuple(weak[method]) for method in completed}) == 1
        if same_sets:
            lines.append(
                "All three methods have the same sub-0.50 spaced labels, consistent with a label-support and partition limitation rather than a model-family-only effect."
            )
        else:
            lines.append(
                "The methods do not have identical weak-label sets under the spaced partition; inspect the per-label table and support column rather than treating any aggregate as a winner."
            )
        lines.append(
            "Labels weak for all three completed methods: "
            + (", ".join(shared) if shared else "none")
            + "."
        )
        comparable = pivot.dropna(subset=["rgfn", "logistic", "gradient_boosted"])
        if not comparable.empty:
            rgfn_higher = sorted(
                f"{axis}:{label}"
                for (axis, label), row in comparable.iterrows()
                if float(row["rgfn"]) > max(float(row["logistic"]), float(row["gradient_boosted"])) + 0.05
            )
            if rgfn_higher:
                lines.append(
                    "RGFN has a held-out spaced F1 more than 0.05 above both comparators for: "
                    + ", ".join(rgfn_higher)
                    + "."
                )
            else:
                lines.append(
                    "RGFN has no held-out spaced per-label F1 more than 0.05 above both comparators."
                )
    return lines


def reason_code_method_comparison_report(
    partition_trace: list[dict[str, object]],
    selection_trace: list[dict[str, object]],
    metric_rows_by_method: dict[str, list[dict[str, object]]],
    comparison_rows: list[dict[str, object]],
    rgfn_status: dict[str, dict[str, object]],
    threshold_robustness_rows: list[dict[str, object]] | None = None,
) -> str:
    parts = [
        "HOURLY CONDITIONAL REASON-CODE METHOD COMPARISON",
        "",
        "TASK",
        "Each method receives only already-labelled fault-hours. It predicts independent multi-label mechanism and component reason codes; it does not alter the frozen binary detector.",
        "",
        "METHODS",
        "logistic: one-vs-rest logistic regression on the same flattened 223-feature, seven-hour input.",
        "gradient_boosted: one-vs-rest HistGradientBoosting classifiers on the same flattened input.",
        "rgfn: one shared GRU sensor stream, evidence MLP, and per-label Evidence Gate with four mechanism and six component outputs.",
        "",
        "SPLIT REUSE",
        pd.DataFrame(partition_trace).to_string(index=False),
        "",
        "SELECTION TRACE",
        "All class weights are fitted from training fault-hours only. The deployed threshold is selected from grouped training-only out-of-fold predictions; validation is a frozen-threshold sanity check and test is evaluated only after selection is complete.",
        pd.DataFrame(
            [
                {key: value for key, value in row.items() if key != "validation_metrics"}
                for row in selection_trace
            ]
        ).to_string(index=False),
        "",
        "COMPARISON TABLE",
        "The table is in fixed method/split/axis order. It has no winner, rank, or test-selected configuration.",
        _comparison_display_frame(comparison_rows).to_string(index=False),
        "",
    ]
    if threshold_robustness_rows is not None:
        parts.extend(
            [
                "THRESHOLD ROBUSTNESS: SINGLE-VALIDATION REFERENCE VS TRAINING-ONLY OOF CV",
                "The old columns are a diagnostic recreation of the former single-validation rule. They are reported after the CV threshold is frozen and never select a model, threshold, or configuration.",
                _threshold_robustness_display_frame(threshold_robustness_rows).to_string(index=False),
                "",
            ]
        )
    for method in ("logistic", "gradient_boosted", "rgfn"):
        metrics = pd.DataFrame(metric_rows_by_method.get(method, []))
        parts.extend([f"{method.upper()} PER-LABEL HELD-OUT RESULTS", ""])
        if metrics.empty:
            parts.append("No completed held-out result is available for this method.")
            parts.append("")
            continue
        for scheme in ("random", "spaced"):
            for axis in ("mechanism", "component"):
                frame = metrics.loc[
                    metrics["split_scheme"].eq(scheme) & metrics["axis"].eq(axis),
                    [
                        "label",
                        "train_support",
                        "validation_support",
                        "support",
                        "threshold",
                        "precision",
                        "recall",
                        "f1",
                        "accuracy",
                        "estimated",
                    ],
                ].copy()
                if frame.empty:
                    continue
                for column in ("precision", "recall", "f1"):
                    frame[column] = frame[column].map(
                        lambda value: "N/A" if pd.isna(value) else value
                    )
                parts.extend(
                    [
                        f"{scheme.upper()} {axis.upper()}S",
                        frame.to_string(index=False),
                        "",
                    ]
                )
    parts.extend(
        [
            "RGFN RUN STATUS",
            pd.DataFrame(list(rgfn_status.values())).to_string(index=False)
            if rgfn_status
            else "RGFN was not requested.",
            "",
            "DIAGNOSTIC",
            *_spaced_reason_code_diagnostic(
                [row for rows in metric_rows_by_method.values() for row in rows]
            ),
            "",
            "REPORTING NOTE",
            "Labels with zero positive held-out support retain their rows with precision, recall, and F1 marked N/A; they are excluded from the corresponding macro estimate and named in macro_unsupported_labels.",
            "This is a retrospective comparison of recovery of the project rule-and-evidence taxonomy, not independent field-verified root-cause accuracy.",
            "Held-out metrics are reported after training-only OOF threshold selection and do not select a method, threshold, or deployment configuration.",
        ]
    )
    return "\n".join(parts)


def allocation_counts(
    total: int,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
) -> dict[str, int]:
    if total < 0:
        raise ValueError("allocation total must be non-negative")
    names = tuple(fractions)
    raw = {name: total * fraction for name, fraction in fractions.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    ranked = sorted(
        names,
        key=lambda name: (raw[name] - counts[name], -names.index(name)),
        reverse=True,
    )
    for name in ranked[:remaining]:
        counts[name] += 1
    return counts


def _split_indices_from_assignments(
    assignments: np.ndarray,
    names: tuple[str, ...] = SPLIT_NAMES,
) -> dict[str, np.ndarray]:
    return {
        name: np.flatnonzero(assignments == name).astype(np.int64)
        for name in names
    }


def validate_splits(
    splits: dict[str, np.ndarray],
    labels: np.ndarray,
) -> None:
    labels = np.asarray(labels, dtype=int)
    count = len(labels)
    names = tuple(splits)
    pieces = [np.asarray(splits[name], dtype=np.int64) for name in names]
    combined = np.concatenate(pieces)
    if len(combined) != count or len(np.unique(combined)) != count or set(combined.tolist()) != set(range(count)):
        raise ValueError("split membership is not disjoint and complete")
    for name, indices in zip(names, pieces):
        if not len(indices):
            raise ValueError(f"split is empty: {name}")
        values = labels[indices]
        if not np.isin(values, [0, 1]).all() or not np.equal(values, 0).any() or not np.equal(values, 1).any():
            raise ValueError(f"split does not contain both binary classes: {name}")


def random_split_indices(
    labels: np.ndarray,
    seed: int,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    source = np.arange(len(labels), dtype=np.int64)
    names = tuple(fractions)
    if names == ("train", "test"):
        train, test = train_test_split(
            source,
            test_size=float(fractions["test"]),
            random_state=int(seed),
            stratify=labels,
        )
        result = {
            "train": np.sort(train.astype(np.int64)),
            "test": np.sort(test.astype(np.int64)),
        }
    elif names == ("train", "validation", "test"):
        train, remainder = train_test_split(
            source,
            test_size=float(fractions["validation"] + fractions["test"]),
            random_state=int(seed),
            stratify=labels,
        )
        validation, test = train_test_split(
            remainder,
            test_size=float(fractions["test"] / (fractions["validation"] + fractions["test"])),
            random_state=int(seed) + 1,
            stratify=labels[remainder],
        )
        result = {
            "train": np.sort(train.astype(np.int64)),
            "validation": np.sort(validation.astype(np.int64)),
            "test": np.sort(test.astype(np.int64)),
        }
    else:
        raise ValueError(f"unsupported random split names: {names}")
    validate_splits(result, labels)
    return result


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fault_groups(
    labels: np.ndarray,
    source_episode_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    positive = np.flatnonzero(np.equal(labels, 1)).astype(np.int64)
    parent: dict[str, str] = {}

    def find(token: str) -> str:
        parent.setdefault(token, token)
        if parent[token] != token:
            parent[token] = find(parent[token])
        return parent[token]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    row_tokens: dict[int, list[str]] = {}
    for index in positive:
        tokens = [token for token in str(source_episode_ids[index]).split("|") if token]
        if not tokens:
            tokens = [f"unlinked_{index}"]
        row_tokens[int(index)] = tokens
        for token in tokens[1:]:
            union(tokens[0], token)
        find(tokens[0])
    groups: dict[str, list[int]] = {}
    for index in positive:
        root = find(row_tokens[int(index)][0])
        groups.setdefault(root, []).append(int(index))
    return {
        "|".join(sorted({token for index in indices for token in row_tokens[index]})): np.asarray(indices, dtype=np.int64)
        for _, indices in groups.items()
    }


def _fault_strata(
    groups: dict[str, np.ndarray],
    hours: np.ndarray,
    partition_count: int = len(SPLIT_NAMES),
) -> list[tuple[str, list[tuple[str, np.ndarray]]]]:
    parsed = pd.to_datetime(pd.Series(hours), utc=True).dt.tz_localize(None)
    by_month: dict[pd.Period, list[tuple[str, np.ndarray]]] = {}
    for group_id, indices in groups.items():
        month = parsed.iloc[indices].min().to_period("M")
        by_month.setdefault(month, []).append((group_id, indices))
    if not by_month:
        return []
    months = pd.period_range(min(by_month), max(by_month), freq="M")
    result: list[tuple[str, list[tuple[str, np.ndarray]]]] = []
    pending: list[tuple[str, np.ndarray]] = []
    pending_start: pd.Period | None = None
    for month in months:
        if pending_start is None:
            pending_start = month
        pending.extend(by_month.get(month, []))
        if len(pending) >= partition_count:
            result.append((f"{pending_start.strftime('%Y-%m')}_to_{month.strftime('%Y-%m')}", pending))
            pending = []
            pending_start = None
    if pending:
        if result:
            previous_name, previous_groups = result[-1]
            result[-1] = (f"{previous_name}_plus_tail", previous_groups + pending)
        else:
            end = months[-1]
            result.append((f"{pending_start.strftime('%Y-%m')}_to_{end.strftime('%Y-%m')}", pending))
    return result


def _assign_fault_stratum(
    groups: list[tuple[str, np.ndarray]],
    fractions: dict[str, float] = SPLIT_FRACTIONS,
) -> dict[str, np.ndarray]:
    names = tuple(fractions)
    quotas = {name: 0.0 for name in names}
    total = float(sum(len(indices) for _, indices in groups))
    for name, fraction in fractions.items():
        quotas[name] = total * fraction
    weights = {name: 0.0 for name in names}
    assigned_groups = {name: 0 for name in names}
    assignment: dict[str, np.ndarray] = {}
    ordered = sorted(groups, key=lambda item: (-len(item[1]), _digest(f"{SPACED_SPLIT_VERSION}|{item[0]}")))
    for position, (group_id, indices) in enumerate(ordered):
        remaining_groups = len(ordered) - position
        empty = [name for name in names if assigned_groups[name] == 0]
        candidates = empty if empty and remaining_groups == len(empty) else list(names)
        scores: dict[str, float] = {}
        for name in candidates:
            proposed = dict(weights)
            proposed[name] += len(indices)
            scores[name] = float(
                sum(
                    ((proposed[split_name] - quotas[split_name]) / max(quotas[split_name], 1.0)) ** 2
                    for split_name in names
                )
            )
        selected = min(names, key=lambda name: (scores.get(name, np.inf), names.index(name)))
        weights[selected] += len(indices)
        assigned_groups[selected] += 1
        assignment[group_id] = np.full(len(indices), selected, dtype=object)
    return assignment


def _assign_non_fault_rows(
    assignments: np.ndarray,
    labels: np.ndarray,
    station_ids: np.ndarray,
    hours: np.ndarray,
    display_states: np.ndarray,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
) -> None:
    parsed = pd.to_datetime(pd.Series(hours), utc=True).dt.tz_localize(None)
    strata: dict[tuple[str, str], list[int]] = {}
    for index in np.flatnonzero(np.equal(labels, 0)):
        key = (str(parsed.iloc[index].to_period("M")), str(display_states[index]))
        strata.setdefault(key, []).append(int(index))
    for key, indices in strata.items():
        ordered = sorted(
            indices,
            key=lambda index: _digest(
                f"{SPACED_SPLIT_VERSION}|{key[0]}|{key[1]}|{station_ids[index]}|{hours[index]}"
            ),
        )
        counts = allocation_counts(len(ordered), fractions)
        start = 0
        for name in fractions:
            end = start + counts[name]
            assignments[np.asarray(ordered[start:end], dtype=np.int64)] = name
            start = end


def spaced_split_indices(
    labels: np.ndarray,
    station_ids: np.ndarray,
    hours: np.ndarray,
    display_states: np.ndarray,
    source_episode_ids: np.ndarray,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    labels = np.asarray(labels, dtype=int)
    station_ids = np.asarray(station_ids, dtype=object).astype(str)
    hours = np.asarray(hours, dtype=object).astype(str)
    display_states = np.asarray(display_states, dtype=object).astype(str)
    source_episode_ids = np.asarray(source_episode_ids, dtype=object).astype(str)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("spaced split requires only eligible binary labels")
    assignments = np.full(len(labels), "", dtype=object)
    fault_groups = _fault_groups(labels, source_episode_ids)
    names = tuple(fractions)
    strata = _fault_strata(fault_groups, hours, partition_count=len(names))
    fault_group_assignment: dict[str, str] = {}
    stratum_rows = []
    for stratum_name, groups in strata:
        local = _assign_fault_stratum(groups, fractions)
        counts = {name: 0 for name in names}
        groups_by_split = {name: 0 for name in names}
        ordered = sorted(groups, key=lambda item: (-len(item[1]), _digest(f"{SPACED_SPLIT_VERSION}|{item[0]}")))
        for group_id, indices in ordered:
            values = local[group_id]
            split_name = str(values[0])
            assignments[indices] = split_name
            fault_group_assignment[group_id] = split_name
            counts[split_name] += len(indices)
            groups_by_split[split_name] += 1
        stratum_rows.append(
            {
                "stratum": stratum_name,
                **{f"{name}_fault_hours": int(counts[name]) for name in names},
                **{f"{name}_fault_groups": int(groups_by_split[name]) for name in names},
            }
        )
    _assign_non_fault_rows(assignments, labels, station_ids, hours, display_states, fractions)
    if np.equal(assignments, "").any():
        raise RuntimeError("spaced split left examples without a partition")
    result = _split_indices_from_assignments(assignments, names)
    validate_splits(result, labels)
    detail = {
        "version": SPACED_SPLIT_VERSION,
        "fault_group_count": int(len(fault_groups)),
        "fault_strata": stratum_rows,
        "fault_group_assignment": fault_group_assignment,
    }
    return result, detail


def split_summary(
    splits: dict[str, np.ndarray],
    labels: np.ndarray,
    hours: np.ndarray,
    display_states: np.ndarray,
) -> list[dict[str, object]]:
    parsed = pd.to_datetime(pd.Series(hours), utc=True)
    rows = []
    for name in splits:
        indices = np.asarray(splits[name], dtype=np.int64)
        subset_labels = np.asarray(labels, dtype=int)[indices]
        subset_states = np.asarray(display_states, dtype=object).astype(str)[indices]
        faults = indices[np.equal(subset_labels, 1)]
        rows.append(
            {
                "split": name,
                "hours": int(len(indices)),
                "fault_hours": int(np.equal(subset_labels, 1).sum()),
                "not_fault_hours": int(np.equal(subset_labels, 0).sum()),
                "benign_hours": int(np.equal(subset_states, "benign").sum()),
                "clean_hours": int(np.equal(subset_states, "clean").sum()),
                "first_fault_hour": "" if not len(faults) else str(parsed.iloc[faults].min()),
                "last_fault_hour": "" if not len(faults) else str(parsed.iloc[faults].max()),
            }
        )
    return rows


def make_classifier(config: HourlyBaselineConfig) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=float(config.learning_rate),
        max_iter=int(config.max_iter),
        max_leaf_nodes=int(config.max_leaf_nodes),
        min_samples_leaf=int(config.min_samples_leaf),
        l2_regularization=float(config.l2_regularization),
        early_stopping=False,
        random_state=int(config.seed),
        class_weight={0: 1.0, 1: float(config.fault_class_weight)},
    )


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(probabilities >= float(threshold), dtype=int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "tp": int(matrix[1, 1]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tn": int(matrix[0, 0]),
    }


def train_baseline_run(
    values: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    window_name: str,
    split_name: str,
    config: HourlyBaselineConfig,
    split_configuration: str | None = None,
) -> tuple[dict[str, object], HistGradientBoostingClassifier]:
    labels = np.asarray(labels, dtype=int)
    validate_splits(splits, labels)
    model = make_classifier(config)
    model.fit(values[splits["train"]], labels[splits["train"]])
    test_probabilities = model.predict_proba(values[splits["test"]])[:, 1]
    validation = None
    if "validation" in splits:
        validation_probabilities = model.predict_proba(values[splits["validation"]])[:, 1]
        validation = binary_metrics(labels[splits["validation"]], validation_probabilities, config.threshold)
    suffix = "" if split_configuration is None else f"-{split_configuration}"
    return {
        "run": f"{window_name}-{split_name}{suffix}",
        "window": window_name,
        "split_scheme": split_name,
        "split_configuration": "70_15_15" if split_configuration is None else split_configuration,
        "feature_dimension": int(values.shape[1]),
        "fault_class_weight": float(config.fault_class_weight),
        "threshold": float(config.threshold),
        "model_iterations": int(model.n_iter_),
        "validation": validation,
        "test": binary_metrics(labels[splits["test"]], test_probabilities, config.threshold),
    }, model


def run_baseline_matrix(
    values: np.ndarray,
    labels: np.ndarray,
    station_ids: np.ndarray,
    hours: np.ndarray,
    display_states: np.ndarray,
    source_episode_ids: np.ndarray,
    window_name: str,
    config: HourlyBaselineConfig,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
    split_configuration: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, HistGradientBoostingClassifier], dict[str, dict[str, np.ndarray]], dict[str, object]]:
    labels = np.asarray(labels, dtype=int)
    random_splits = random_split_indices(labels, config.seed, fractions)
    spaced_splits, spaced_detail = spaced_split_indices(
        labels,
        station_ids,
        hours,
        display_states,
        source_episode_ids,
        fractions,
    )
    split_map = {
        "random": random_splits,
        "spaced": spaced_splits,
    }
    rows = []
    models: dict[str, HistGradientBoostingClassifier] = {}
    for split_name, splits in split_map.items():
        row, model = train_baseline_run(
            values,
            labels,
            splits,
            window_name,
            split_name,
            config,
            split_configuration,
        )
        row["split_summary"] = split_summary(splits, labels, hours, display_states)
        rows.append(row)
        models[str(row["run"])] = model
    return rows, models, split_map, spaced_detail


def make_split_manifest(
    splits_by_scheme: dict[str, dict[str, np.ndarray]],
    labels: np.ndarray,
    station_ids: np.ndarray,
    hours: np.ndarray,
    display_states: np.ndarray,
    source_episode_ids: np.ndarray,
    split_configuration: str = "70_15_15",
) -> pd.DataFrame:
    rows = []
    for scheme, splits in splits_by_scheme.items():
        for split_name, indices in splits.items():
            rows.append(
                pd.DataFrame(
                    {
                        "split_configuration": split_configuration,
                        "split_scheme": scheme,
                        "split": split_name,
                        "station_id": np.asarray(station_ids, dtype=object)[indices].astype(str),
                        "hour": np.asarray(hours, dtype=object)[indices].astype(str),
                        "fault_hour": np.asarray(labels, dtype=int)[indices],
                        "display_state": np.asarray(display_states, dtype=object)[indices].astype(str),
                        "source_episode_ids": np.asarray(source_episode_ids, dtype=object)[indices].astype(str),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["split_configuration", "split_scheme", "split", "station_id", "hour"],
    ).reset_index(drop=True)


def save_model_bundle(
    model: object,
    path: Path,
    feature_names: list[str],
    window_hours: int,
    config: HourlyBaselineConfig,
    split_name: str,
    split_configuration: str = "70_15_15",
    split_fractions: dict[str, float] | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "estimator": model,
        "feature_names": feature_names,
        "window_hours": int(window_hours),
        "config": asdict(config),
        "split_scheme": split_name,
        "split_configuration": split_configuration,
        "split_fractions": dict(SPLIT_FRACTIONS if split_fractions is None else split_fractions),
        "class_weight": {0: 1.0, 1: float(config.fault_class_weight)},
    }
    if extra_metadata:
        bundle.update(extra_metadata)
    joblib.dump(bundle, destination)
    return destination


def stratified_sample_indices(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if len(labels) <= maximum:
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    faults = np.flatnonzero(np.equal(labels, 1))
    not_faults = np.flatnonzero(np.equal(labels, 0))
    fault_count = max(1, min(len(faults), int(round(maximum * len(faults) / len(labels)))))
    not_fault_count = maximum - fault_count
    if not_fault_count > len(not_faults):
        not_fault_count = len(not_faults)
        fault_count = maximum - not_fault_count
    chosen = np.concatenate(
        [
            rng.choice(faults, size=fault_count, replace=False),
            rng.choice(not_faults, size=not_fault_count, replace=False),
        ]
    )
    return np.sort(chosen.astype(np.int64))


def grouped_permutation_importance(
    model: HistGradientBoostingClassifier,
    values: np.ndarray,
    labels: np.ndarray,
    groups: dict[str, np.ndarray],
    threshold: float,
    seed: int,
    maximum_rows: int = 2000,
) -> list[dict[str, object]]:
    selected = stratified_sample_indices(labels, maximum_rows, seed)
    sample_values = values[selected].copy()
    sample_labels = np.asarray(labels, dtype=int)[selected]
    base_probabilities = model.predict_proba(sample_values)[:, 1]
    base_f1 = float(f1_score(sample_labels, base_probabilities >= threshold, zero_division=0))
    rng = np.random.default_rng(seed)
    rows = []
    for name, columns in groups.items():
        permuted = sample_values.copy()
        permutation = rng.permutation(len(sample_values))
        permuted[:, columns] = permuted[permutation][:, columns]
        probabilities = model.predict_proba(permuted)[:, 1]
        score = float(f1_score(sample_labels, probabilities >= threshold, zero_division=0))
        rows.append(
            {
                "feature_group": name,
                "importance_f1_drop": float(base_f1 - score),
                "base_validation_f1": base_f1,
                "permuted_validation_f1": score,
                "sample_rows": int(len(selected)),
            }
        )
    return sorted(rows, key=lambda row: (-float(row["importance_f1_drop"]), str(row["feature_group"])))


def validation_importance_reference(rows: list[dict[str, object]]) -> dict[str, object]:
    candidates = [row for row in rows if isinstance(row.get("validation"), dict)]
    if not candidates:
        raise ValueError("cannot choose an importance reference without validation results")
    return max(
        candidates,
        key=lambda row: (
            float(row["validation"]["f1"]),
            float(row["validation"]["recall"]),
            float(row["validation"]["precision"]),
        ),
    )


def baseline_report(
    rows: list[dict[str, object]],
    top_importance: list[dict[str, object]],
    config: HourlyBaselineConfig,
    spaced_detail: dict[str, object],
    importance_reference: dict[str, object],
) -> str:
    parts = [
        "HOURLY BINARY GRADIENT-BOOSTED BASELINE",
        "",
        "CONFIGURATION",
        f"estimator=HistGradientBoostingClassifier",
        f"fault_class_weight={config.fault_class_weight:.6f}",
        f"decision_threshold={config.threshold:.3f}",
        f"max_iter={config.max_iter}",
        f"learning_rate={config.learning_rate:.3f}",
        f"max_leaf_nodes={config.max_leaf_nodes}",
        f"min_samples_leaf={config.min_samples_leaf}",
        "",
    ]
    for row in rows:
        test = row["test"]
        validation = row["validation"]
        parts.extend(
            [
                f"RUN {row['run']}",
                f"feature_dimension={row['feature_dimension']}",
                f"model_iterations={row['model_iterations']}",
                "TEST METRICS",
                pd.DataFrame(
                    [
                        {
                            "precision": test["precision"],
                            "recall": test["recall"],
                            "f1": test["f1"],
                            "accuracy": test["accuracy"],
                        }
                    ]
                ).to_string(index=False),
                "TEST CONFUSION MATRIX",
                pd.DataFrame(
                    [{"tp": test["tp"], "fp": test["fp"], "fn": test["fn"], "tn": test["tn"]}]
                ).to_string(index=False),
                "VALIDATION METRICS",
            ]
        )
        if validation is None:
            parts.append("not available for this split configuration")
        else:
            parts.append(
                pd.DataFrame(
                    [
                        {
                            "precision": validation["precision"],
                            "recall": validation["recall"],
                            "f1": validation["f1"],
                            "accuracy": validation["accuracy"],
                        }
                    ]
                ).to_string(index=False)
            )
        parts.append("")
    summary_rows = []
    for row in rows:
        validation = row["validation"]
        test = row["test"]
        summary_rows.append(
            {
                "run": row["run"],
                "validation_precision": validation["precision"] if validation is not None else np.nan,
                "validation_recall": validation["recall"] if validation is not None else np.nan,
                "validation_f1": validation["f1"] if validation is not None else np.nan,
                "test_precision": test["precision"],
                "test_recall": test["recall"],
                "test_f1": test["f1"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    positive_total = sum(max(0.0, float(row["importance_f1_drop"])) for row in top_importance)
    dominant = bool(top_importance) and positive_total > 0.0 and float(top_importance[0]["importance_f1_drop"]) / positive_total >= 0.50
    parts.extend(
        [
            "SUMMARY COMPARISON",
            summary.to_string(index=False),
            "",
            "REPORTING POLICY",
            "All predefined configurations are reported. Held-out test metrics are descriptive comparisons only and do not select or name a winner.",
            f"importance_reference_run={importance_reference['run']}",
            "Feature importance uses the validation-selected reference above, with validation F1, recall, and precision as its selection order.",
            "",
            "RANDOM SPLIT CONSTRUCTION",
            "Rows are assigned with a deterministic seed and binary stratification at 70/15/15.",
            "",
            "SPACED SPLIT CONSTRUCTION",
            "Fault hours are grouped by connected source episode identifiers, placed within chronological strata, and assigned against 70/15/15 positive-hour targets.",
            "Benign and clean hours are assigned separately within calendar-month and display-state strata using deterministic hashes and exact largest-remainder quotas.",
            f"spaced_fault_group_count={spaced_detail.get('fault_group_count', 0)}",
            "",
            "VALIDATION-SELECTED GROUPED PERMUTATION IMPORTANCE",
            pd.DataFrame(top_importance[:20]).to_string(index=False),
            f"single_feature_dominance_flag={dominant}",
            "",
            "EVALUATION NOTE",
            "The windows end at the scored hour, but the existing feature snapshot includes retrospective detector calculations. These are offline comparison metrics, not future-facing deployment metrics.",
        ]
    )
    return "\n".join(parts)


def write_metrics_json(payload: dict[str, object], path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return destination
