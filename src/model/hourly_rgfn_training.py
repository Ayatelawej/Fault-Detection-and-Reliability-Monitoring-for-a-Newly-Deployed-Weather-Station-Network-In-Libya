from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
from typing import Callable, Mapping
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.model.hourly_baseline import (
    binary_metrics,
    build_reason_code_cv_folds,
    reason_code_axis_summary,
    reason_code_cv_group_ids,
    reason_code_metrics,
    resolve_reason_code_class_weight,
    select_reason_code_threshold,
    select_reason_code_threshold_mean_fold_f1,
)
from src.model.hourly_calibration import CALIBRATION_THRESHOLDS, select_operating_point, target_check
from src.model.hourly_rgfn import (
    ENCODER_CONV,
    ENCODER_GRU,
    MASK_MODE_PER_HOUR,
    HourlyRgfnConfig,
    build_hourly_rgfn,
    set_hourly_rgfn_seed,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RGFN_WEIGHTS = (1.0, 2.0, 4.0, 6.0, 8.0)
RGFN_THRESHOLDS = CALIBRATION_THRESHOLDS
RGFN_SEEDS = (0, 1, 2, 3, 4)
FEATURE_KEYS = ("X_cont", "mask", "time_since_last", "static", "rule_evidence")
METRIC_NAMES = ("precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn")


@dataclass(frozen=True)
class HourlyRgfnTrainingConfig:
    seed: int = 0
    fault_class_weight: float = 1.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    mask_mode: str = MASK_MODE_PER_HOUR


@dataclass(frozen=True)
class HourlyReasonCodeRgfnConfig:
    seed: int = 0
    cv_seed: int = 2026
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 15
    min_delta: float = 1e-4
    sensor_hidden_size: int = 48
    evidence_hidden_size: int = 16
    evidence_embed_size: int = 16
    gate_hidden_size: int = 8
    dropout: float = 0.3
    gate_l1: float = 0.0
    encoder: str = ENCODER_GRU
    mask_mode: str = MASK_MODE_PER_HOUR


@dataclass
class FittedReasonCodeRgfn:
    split_scheme: str
    model: nn.Module
    scaler: "HourlyRgfnScaler"
    config: HourlyReasonCodeRgfnConfig
    mechanism_count: int
    label_names: np.ndarray
    thresholds: np.ndarray
    positive_class_weights: np.ndarray
    train_support: np.ndarray
    validation_support: np.ndarray
    validation_metrics: list[dict[str, object]]
    selection_trace: list[dict[str, object]]
    training: dict[str, object]


def _center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanpercentile(values, 50, axis=0).astype(np.float32)
        q25 = np.nanpercentile(values, 25, axis=0).astype(np.float32)
        q75 = np.nanpercentile(values, 75, axis=0).astype(np.float32)
    scale = (q75 - q25).astype(np.float32)
    median[~np.isfinite(median)] = 0.0
    scale[~np.isfinite(scale)] = 1.0
    scale[scale == 0.0] = 1.0
    return median, scale


def _scale(values: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scaled = (np.asarray(values, dtype=np.float32) - median) / scale
    scaled = np.clip(scaled, -5.0, 5.0)
    return np.nan_to_num(scaled, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)


@dataclass
class HourlyRgfnScaler:
    continuous_median: np.ndarray
    continuous_iqr: np.ndarray
    static_median: np.ndarray
    static_iqr: np.ndarray
    rule_median: np.ndarray
    rule_iqr: np.ndarray

    @classmethod
    def fit(cls, examples: dict[str, np.ndarray], train_indices: np.ndarray) -> "HourlyRgfnScaler":
        selected = np.asarray(train_indices, dtype=np.int64)
        continuous = np.asarray(examples["X_cont"], dtype=np.float32)[selected]
        cont_median, cont_iqr = _center_scale(continuous.reshape(-1, continuous.shape[-1]))
        static_median, static_iqr = _center_scale(np.asarray(examples["static"], dtype=np.float32)[selected])
        rule_median, rule_iqr = _center_scale(np.asarray(examples["rule_evidence"], dtype=np.float32)[selected])
        return cls(
            continuous_median=cont_median,
            continuous_iqr=cont_iqr,
            static_median=static_median,
            static_iqr=static_iqr,
            rule_median=rule_median,
            rule_iqr=rule_iqr,
        )

    def transform(self, examples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            "X_cont": _scale(
                np.asarray(examples["X_cont"], dtype=np.float32),
                self.continuous_median.reshape(1, 1, -1),
                self.continuous_iqr.reshape(1, 1, -1),
            ),
            "mask": np.nan_to_num(
                np.asarray(examples["mask"], dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32),
            "time_since_last": np.nan_to_num(
                np.asarray(examples["time_since_last"], dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32),
            "static": _scale(
                np.asarray(examples["static"], dtype=np.float32),
                self.static_median.reshape(1, -1),
                self.static_iqr.reshape(1, -1),
            ),
            "rule_evidence": _scale(
                np.asarray(examples["rule_evidence"], dtype=np.float32),
                self.rule_median.reshape(1, -1),
                self.rule_iqr.reshape(1, -1),
            ),
        }

    def payload(self) -> dict[str, np.ndarray]:
        return {
            "continuous_median": self.continuous_median,
            "continuous_iqr": self.continuous_iqr,
            "static_median": self.static_median,
            "static_iqr": self.static_iqr,
            "rule_median": self.rule_median,
            "rule_iqr": self.rule_iqr,
        }


def _sample_keys(station_ids: np.ndarray, hours: np.ndarray) -> np.ndarray:
    stations = np.asarray(station_ids, dtype=object).astype(str)
    parsed = pd.to_datetime(pd.Series(hours), utc=True, format="mixed")
    values = parsed.astype("int64").to_numpy(dtype=np.int64)
    return np.asarray([f"{station}\x1f{int(hour)}" for station, hour in zip(stations, values)], dtype=object)


def load_manifest_splits(examples: dict[str, np.ndarray], path: Path) -> dict[str, dict[str, np.ndarray]]:
    manifest = pd.read_csv(path, keep_default_na=False)
    required = {"split_scheme", "split", "station_id", "hour", "fault_hour"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise KeyError(f"split manifest fields missing: {missing}")
    tensor_keys = _sample_keys(examples["station_id"], examples["hour"])
    if len(np.unique(tensor_keys)) != len(tensor_keys):
        raise ValueError("hourly tensor station-hour keys are not unique")
    tensor_index = {str(key): index for index, key in enumerate(tensor_keys)}
    labels = np.asarray(examples["y_binary"], dtype=int)
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
        partitions: dict[str, np.ndarray] = {}
        for name in ("train", "validation", "test"):
            indices = np.sort(mapped[frame["split"].eq(name).to_numpy()]).astype(np.int64)
            if not len(indices):
                raise ValueError(f"split manifest has an empty {scheme} {name} partition")
            partitions[name] = indices
        combined = np.concatenate([partitions[name] for name in ("train", "validation", "test")])
        if len(np.unique(combined)) != len(labels) or set(combined.tolist()) != set(range(len(labels))):
            raise ValueError(f"split manifest membership is invalid for {scheme}")
        for name in ("train", "validation"):
            values = labels[partitions[name]]
            if not np.equal(values, 0).any() or not np.equal(values, 1).any():
                raise ValueError(f"split manifest {scheme} {name} lacks a binary class")
        result[scheme] = partitions
    return result


@dataclass
class TensorPartition:
    features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    labels: torch.Tensor | None

    @property
    def count(self) -> int:
        return int(self.features[0].shape[0])


@dataclass
class PreparedHourlyRgfnSplit:
    train: TensorPartition
    validation: TensorPartition
    test: TensorPartition
    scaler: HourlyRgfnScaler


def _make_partition(
    values: dict[str, np.ndarray],
    labels: np.ndarray,
    indices: np.ndarray,
    include_labels: bool,
) -> TensorPartition:
    selected = np.asarray(indices, dtype=np.int64)
    features = tuple(
        torch.as_tensor(
            np.ascontiguousarray(np.asarray(values[name], dtype=np.float32)[selected]),
            dtype=torch.float32,
            device=DEVICE,
        )
        for name in FEATURE_KEYS
    )
    target = None
    if include_labels:
        target = torch.as_tensor(
            np.ascontiguousarray(np.asarray(labels, dtype=np.float32)[selected]),
            dtype=torch.float32,
            device=DEVICE,
        )
    return TensorPartition(features=features, labels=target)


def reason_code_rgfn_config_from_arm2(
    values: Mapping[str, object],
    seed: int = 0,
) -> HourlyReasonCodeRgfnConfig:
    required = {
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
        "min_delta",
        "sensor_hidden_size",
        "evidence_hidden_size",
        "evidence_embed_size",
        "gate_hidden_size",
        "dropout",
        "gate_l1",
        "encoder",
    }
    missing = sorted(required.difference(values))
    if missing:
        raise KeyError(f"Arm 2 configuration lacks required settings: {missing}")
    return HourlyReasonCodeRgfnConfig(
        seed=int(seed),
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        batch_size=int(values["batch_size"]),
        max_epochs=int(values["max_epochs"]),
        patience=int(values["patience"]),
        min_delta=float(values["min_delta"]),
        sensor_hidden_size=int(values["sensor_hidden_size"]),
        evidence_hidden_size=int(values["evidence_hidden_size"]),
        evidence_embed_size=int(values["evidence_embed_size"]),
        gate_hidden_size=int(values["gate_hidden_size"]),
        dropout=float(values["dropout"]),
        gate_l1=float(values["gate_l1"]),
        encoder=str(values["encoder"]),
        mask_mode=MASK_MODE_PER_HOUR,
    )


def _reason_code_targets(
    examples: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    required = {
        "y_mechanism",
        "y_component",
        "mechanism_label_names",
        "component_label_names",
    }
    missing = sorted(required.difference(examples))
    if missing:
        raise KeyError(f"reason-code RGFN examples lack fields: {missing}")
    mechanisms = np.asarray(examples["y_mechanism"], dtype=np.int64)
    components = np.asarray(examples["y_component"], dtype=np.int64)
    mechanism_names = np.asarray(examples["mechanism_label_names"], dtype=object).astype(str)
    component_names = np.asarray(examples["component_label_names"], dtype=object).astype(str)
    if mechanisms.ndim != 2 or components.ndim != 2 or len(mechanisms) != len(components):
        raise ValueError("reason-code RGFN target arrays have incompatible shapes")
    if mechanisms.shape[1] != len(mechanism_names) or components.shape[1] != len(component_names):
        raise ValueError("reason-code RGFN target names do not match target widths")
    targets = np.concatenate((mechanisms, components), axis=1)
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("reason-code RGFN targets must be binary")
    return targets.astype(np.float32), np.concatenate((mechanism_names, component_names)), int(len(mechanism_names))


def _reason_code_rgfn_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    mechanism_count: int,
    mechanism_criterion: nn.Module,
    component_criterion: nn.Module,
    gate_l1: float,
) -> torch.Tensor:
    if labels.ndim != 2:
        raise ValueError("reason-code RGFN labels must be two-dimensional")
    mechanism_loss = mechanism_criterion(
        output["mechanism_logits"],
        labels[:, :mechanism_count],
    )
    component_loss = component_criterion(
        output["component_logits"],
        labels[:, mechanism_count:],
    )
    value = 0.5 * (mechanism_loss + component_loss)
    if float(gate_l1) > 0.0:
        value = value + float(gate_l1) * torch.mean(torch.abs(output["alpha"]))
    return value


def _reason_code_rgfn_validation_loss(
    model: nn.Module,
    partition: TensorPartition,
    mechanism_count: int,
    mechanism_criterion: nn.Module,
    component_criterion: nn.Module,
    gate_l1: float,
    batch_size: int,
) -> float:
    if partition.labels is None:
        raise ValueError("reason-code RGFN validation labels are required")
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for start, stop in _batch_ranges(partition.count, batch_size):
            output = model(*[value[start:stop] for value in partition.features])
            value = _reason_code_rgfn_loss(
                output,
                partition.labels[start:stop],
                mechanism_count,
                mechanism_criterion,
                component_criterion,
                gate_l1,
            )
            total += float(value.detach().cpu()) * (stop - start)
    return float(total / max(partition.count, 1))


def predict_reason_code_rgfn(
    model: nn.Module,
    partition: TensorPartition,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start, stop in _batch_ranges(partition.count, batch_size):
            output = model(*[value[start:stop] for value in partition.features])
            values.append(
                output["reason_code_probabilities"].detach().cpu().numpy().astype(np.float32)
            )
    width = int(getattr(model, "output_dim", 0))
    return np.concatenate(values, axis=0) if values else np.empty((0, width), dtype=np.float32)


def _reason_code_rgfn_model(
    config: HourlyReasonCodeRgfnConfig,
    train_partition: TensorPartition,
    output_dim: int,
    mechanism_count: int,
) -> nn.Module:
    if train_partition.labels is None:
        raise ValueError("reason-code RGFN training labels are required")
    architecture = HourlyRgfnConfig(
        n_continuous=int(train_partition.features[0].shape[-1]),
        n_static=int(train_partition.features[3].shape[-1]),
        n_rule_evidence=int(train_partition.features[4].shape[-1]),
        window_hours=int(train_partition.features[0].shape[1]),
        sensor_hidden_size=int(config.sensor_hidden_size),
        evidence_hidden_size=int(config.evidence_hidden_size),
        evidence_embed_size=int(config.evidence_embed_size),
        fusion_hidden_size=int(config.gate_hidden_size),
        dropout=float(config.dropout),
        mask_mode=str(config.mask_mode),
    )
    return build_hourly_rgfn(
        str(config.encoder),
        config=architecture,
        output_dim=int(output_dim),
        mechanism_count=int(mechanism_count),
    ).to(DEVICE)


def _reason_code_rgfn_shared_cv_plan(
    examples: dict[str, np.ndarray],
    targets: np.ndarray,
    label_names: np.ndarray,
    train_indices: np.ndarray,
    config: HourlyReasonCodeRgfnConfig,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    if "source_episode_ids" not in examples:
        raise KeyError("reason-code RGFN cross-validation requires source_episode_ids")
    source_episode_ids = np.asarray(examples["source_episode_ids"], dtype=object).astype(str)
    if source_episode_ids.shape != (len(targets),):
        raise ValueError("reason-code RGFN source_episode_ids do not align with targets")
    selected = np.asarray(train_indices, dtype=np.int64)
    group_ids = reason_code_cv_group_ids(source_episode_ids)
    train_targets = np.asarray(targets, dtype=np.int64)[selected]
    train_groups = group_ids[selected]
    support = train_targets.sum(axis=0).astype(np.int64)
    positive_group_count = np.asarray(
        [
            len(np.unique(train_groups[train_targets[:, index] == 1]))
            for index in range(targets.shape[1])
        ],
        dtype=np.int64,
    )
    proxy_index = min(
        range(targets.shape[1]),
        key=lambda index: (
            int(positive_group_count[index]),
            int(support[index]),
            str(label_names[index]),
        ),
    )
    folds, plan = build_reason_code_cv_folds(
        np.asarray(targets, dtype=np.int64)[:, proxy_index],
        selected,
        seed=int(config.cv_seed),
        requested_folds=5,
        groups=group_ids,
    )
    return folds, {
        **plan,
        "cv_shared_fold_plan": True,
        "cv_proxy_label": str(label_names[proxy_index]),
        "cv_proxy_label_index": int(proxy_index),
        "cv_shared_label_train_support": {
            str(label_names[index]): int(support[index]) for index in range(targets.shape[1])
        },
        "cv_shared_label_positive_group_count": {
            str(label_names[index]): int(positive_group_count[index])
            for index in range(targets.shape[1])
        },
    }


def _fit_reason_code_rgfn_cv_fold(
    examples: dict[str, np.ndarray],
    targets: np.ndarray,
    fit_indices: np.ndarray,
    oof_indices: np.ndarray,
    config: HourlyReasonCodeRgfnConfig,
    fold_number: int,
    mechanism_count: int,
    deadline: float,
) -> tuple[np.ndarray | None, dict[str, object]]:
    started = time.monotonic()
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    oof_indices = np.asarray(oof_indices, dtype=np.int64)
    if not len(fit_indices) or not len(oof_indices):
        raise ValueError("reason-code RGFN CV folds must have fit and OOF examples")
    fit_targets = np.asarray(targets, dtype=np.float32)[fit_indices]
    positive_weights = np.asarray(
        [resolve_reason_code_class_weight(fit_targets[:, index]) for index in range(targets.shape[1])],
        dtype=np.float32,
    )
    fold_config = replace(config, seed=int(config.cv_seed) + int(fold_number) - 1)
    scaler = HourlyRgfnScaler.fit(examples, fit_indices)
    scaled = scaler.transform(examples)
    train_partition = _make_partition(scaled, targets, fit_indices, include_labels=True)
    oof_partition = _make_partition(scaled, targets, oof_indices, include_labels=False)
    if train_partition.labels is None:
        raise RuntimeError("reason-code RGFN CV failed to prepare a labelled fit partition")
    set_hourly_rgfn_seed(int(fold_config.seed))
    model = _reason_code_rgfn_model(
        fold_config,
        train_partition,
        targets.shape[1],
        mechanism_count,
    )
    if time.monotonic() >= deadline:
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return None, {
            "fold": int(fold_number),
            "status": "timebox_exceeded",
            "elapsed_seconds": float(time.monotonic() - started),
            "epochs_completed": 0,
        }
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(fold_config.learning_rate),
        weight_decay=float(fold_config.weight_decay),
    )
    mechanism_weight = torch.as_tensor(
        positive_weights[:mechanism_count],
        dtype=torch.float32,
        device=DEVICE,
    )
    component_weight = torch.as_tensor(
        positive_weights[mechanism_count:],
        dtype=torch.float32,
        device=DEVICE,
    )
    mechanism_criterion = nn.BCEWithLogitsLoss(pos_weight=mechanism_weight)
    component_criterion = nn.BCEWithLogitsLoss(pos_weight=component_weight)
    generator = torch.Generator(device=DEVICE.type)
    generator.manual_seed(int(fold_config.seed))
    completed_epochs = 0
    timed_out = False
    for epoch in range(1, int(fold_config.max_epochs) + 1):
        if time.monotonic() >= deadline:
            timed_out = True
            break
        model.train()
        permutation = torch.randperm(train_partition.count, generator=generator, device=DEVICE)
        for start, stop in _batch_ranges(train_partition.count, int(fold_config.batch_size)):
            if time.monotonic() >= deadline:
                timed_out = True
                break
            batch_indices = permutation[start:stop]
            features = tuple(value.index_select(0, batch_indices) for value in train_partition.features)
            labels = train_partition.labels.index_select(0, batch_indices)
            optimizer.zero_grad(set_to_none=True)
            output = model(*features)
            loss = _reason_code_rgfn_loss(
                output,
                labels,
                mechanism_count,
                mechanism_criterion,
                component_criterion,
                float(fold_config.gate_l1),
            )
            loss.backward()
            optimizer.step()
        if timed_out:
            break
        completed_epochs = int(epoch)
    elapsed = float(time.monotonic() - started)
    if timed_out or time.monotonic() >= deadline:
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return None, {
            "fold": int(fold_number),
            "status": "timebox_exceeded",
            "elapsed_seconds": elapsed,
            "epochs_completed": int(completed_epochs),
        }
    probabilities = predict_reason_code_rgfn(
        model,
        oof_partition,
        max(int(fold_config.batch_size), 512),
    )
    if probabilities.shape != (len(oof_indices), targets.shape[1]):
        raise RuntimeError("reason-code RGFN CV OOF probabilities have an unexpected shape")
    detail = {
        "fold": int(fold_number),
        "status": "completed",
        "elapsed_seconds": elapsed,
        "epochs_completed": int(completed_epochs),
        "fold_seed": int(fold_config.seed),
        "fit_support": fit_targets.sum(axis=0).astype(np.int64).tolist(),
        "oof_support": np.asarray(targets, dtype=np.int64)[oof_indices].sum(axis=0).astype(np.int64).tolist(),
        "positive_class_weights": positive_weights.tolist(),
        "outer_validation_used": False,
        "fixed_epoch_training": True,
    }
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return probabilities, detail


def _reason_code_rgfn_oof_thresholds(
    examples: dict[str, np.ndarray],
    targets: np.ndarray,
    label_names: np.ndarray,
    mechanism_count: int,
    train_indices: np.ndarray,
    config: HourlyReasonCodeRgfnConfig,
    deadline: float,
) -> tuple[np.ndarray | None, list[dict[str, object]], dict[str, object]]:
    started = time.monotonic()
    selected = np.asarray(train_indices, dtype=np.int64)
    folds, plan = _reason_code_rgfn_shared_cv_plan(
        examples,
        targets,
        label_names,
        selected,
        config,
    )
    if not folds:
        return None, [], {
            "status": "cv_not_possible",
            "elapsed_seconds": float(time.monotonic() - started),
            "cv_plan": plan,
        }
    local_position = {int(index): position for position, index in enumerate(selected.tolist())}
    oof_probabilities = np.full((len(selected), targets.shape[1]), np.nan, dtype=np.float32)
    oof_fold_ids = np.full(len(selected), -1, dtype=np.int64)
    fold_details = [dict(value) for value in plan["cv_fold_details"]]
    for fold_number, (fit_indices, oof_indices) in enumerate(folds, start=1):
        probabilities, detail = _fit_reason_code_rgfn_cv_fold(
            examples,
            targets,
            fit_indices,
            oof_indices,
            config,
            fold_number,
            mechanism_count,
            deadline,
        )
        fold_details[fold_number - 1]["training"] = detail
        if probabilities is None:
            return None, [], {
                "status": "timebox_exceeded",
                "elapsed_seconds": float(time.monotonic() - started),
                "cv_plan": {**plan, "cv_fold_details": fold_details},
            }
        positions = np.asarray([local_position[int(index)] for index in oof_indices], dtype=np.int64)
        if np.isfinite(oof_probabilities[positions]).any():
            raise RuntimeError("reason-code RGFN CV generated duplicate OOF probabilities")
        oof_probabilities[positions] = probabilities
        oof_fold_ids[positions] = int(fold_number)
    if not np.isfinite(oof_probabilities).all() or np.any(oof_fold_ids < 0):
        raise RuntimeError("reason-code RGFN CV did not cover all outer-training examples exactly once")
    selected_targets = np.asarray(targets, dtype=np.int64)[selected]
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    records: list[dict[str, object]] = []
    for axis, start, stop in (
        ("mechanism", 0, mechanism_count),
        ("component", mechanism_count, targets.shape[1]),
    ):
        for index in range(start, stop):
            valid_folds = []
            for fold_number, (fit_indices, oof_indices) in enumerate(folds, start=1):
                fit_target = np.asarray(targets, dtype=np.int64)[fit_indices, index]
                oof_target = np.asarray(targets, dtype=np.int64)[oof_indices, index]
                if (
                    np.equal(fit_target, 1).any()
                    and np.equal(fit_target, 0).any()
                    and np.equal(oof_target, 1).any()
                    and np.equal(oof_target, 0).any()
                ):
                    valid_folds.append(int(fold_number))
            selection_mask = np.isin(oof_fold_ids, np.asarray(valid_folds, dtype=np.int64))
            if len(valid_folds) >= 2:
                threshold, oof_metrics, candidate_count = select_reason_code_threshold_mean_fold_f1(
                    selected_targets[selection_mask, index],
                    oof_probabilities[selection_mask, index],
                    oof_fold_ids[selection_mask],
                )
                selection_source = "training_oof_cross_validation"
                threshold_policy = "training_oof_cv_mean_f1"
                cross_validation_possible = True
            else:
                threshold = 0.5
                oof_metrics = None
                candidate_count = 0
                selection_source = "predeclared_fixed_0_5"
                threshold_policy = "predeclared_fixed_0_5"
                cross_validation_possible = False
            thresholds[index] = float(threshold)
            records.append(
                {
                    "method": "rgfn",
                    "axis": axis,
                    "label": str(label_names[index]),
                    "label_index": int(index),
                    "selection_source": selection_source,
                    "threshold_policy": threshold_policy,
                    "selected_threshold": float(threshold),
                    "oof_candidate_count": int(candidate_count),
                    "oof_selection_metrics": oof_metrics,
                    "cv_requested_folds": int(plan["cv_requested_folds"]),
                    "cv_effective_folds": int(len(valid_folds)),
                    "cv_status": str(plan["cv_status"]),
                    "cv_grouping": str(plan["cv_grouping"]),
                    "cv_seed": int(plan["cv_seed"]),
                    "cv_positive_group_count": int(
                        plan["cv_shared_label_positive_group_count"][str(label_names[index])]
                    ),
                    "cv_shared_plan_effective_folds": int(plan["cv_effective_folds"]),
                    "cv_possible": bool(cross_validation_possible),
                    "cv_valid_fold_numbers": valid_folds,
                    "test_metrics_read_during_selection": False,
                }
            )
    return thresholds, records, {
        "status": "completed",
        "elapsed_seconds": float(time.monotonic() - started),
        "cv_plan": {**plan, "cv_fold_details": fold_details},
        "oof_coverage_complete": True,
    }


def _fit_reason_code_rgfn_split(
    examples: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    split_scheme: str,
    config: HourlyReasonCodeRgfnConfig,
    deadline: float,
) -> tuple[FittedReasonCodeRgfn | None, dict[str, object]]:
    split_started = time.monotonic()
    required = {"train", "validation", "test"}
    if set(splits) != required:
        raise ValueError("reason-code RGFN requires train, validation, and test partitions")
    if int(config.batch_size) < 1 or int(config.max_epochs) < 1 or int(config.patience) < 1:
        raise ValueError("reason-code RGFN training values must be positive")
    targets, label_names, mechanism_count = _reason_code_targets(examples)
    train_indices = np.asarray(splits["train"], dtype=np.int64)
    validation_indices = np.asarray(splits["validation"], dtype=np.int64)
    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise ValueError("reason-code RGFN train and validation partitions must be non-empty")
    thresholds, cv_records, cv_status = _reason_code_rgfn_oof_thresholds(
        examples,
        targets,
        label_names,
        mechanism_count,
        train_indices,
        config,
        deadline,
    )
    if thresholds is None:
        return None, {
            "split_scheme": split_scheme,
            "status": str(cv_status["status"]),
            "elapsed_seconds": float(time.monotonic() - split_started),
            "epochs_completed": 0,
            "best_epoch": 0,
            "best_validation_loss": None,
            "test_evaluated": False,
            "threshold_selection": cv_status,
        }
    train_targets = targets[train_indices]
    validation_targets = targets[validation_indices]
    positive_weights = np.asarray(
        [resolve_reason_code_class_weight(train_targets[:, index]) for index in range(targets.shape[1])],
        dtype=np.float32,
    )
    scaler = HourlyRgfnScaler.fit(examples, train_indices)
    scaled = scaler.transform(examples)
    train_partition = _make_partition(scaled, targets, train_indices, include_labels=True)
    validation_partition = _make_partition(scaled, targets, validation_indices, include_labels=True)
    if train_partition.labels is None or validation_partition.labels is None:
        raise RuntimeError("reason-code RGFN failed to prepare labelled partitions")
    set_hourly_rgfn_seed(int(config.seed))
    model = _reason_code_rgfn_model(
        config,
        train_partition,
        targets.shape[1],
        mechanism_count,
    )
    if time.monotonic() >= deadline:
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return None, {
            "split_scheme": split_scheme,
            "status": "timebox_exceeded",
            "elapsed_seconds": float(time.monotonic() - split_started),
            "epochs_completed": 0,
            "best_epoch": 0,
            "best_validation_loss": None,
            "test_evaluated": False,
            "threshold_selection": cv_status,
        }
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    mechanism_weight = torch.as_tensor(
        positive_weights[:mechanism_count],
        dtype=torch.float32,
        device=DEVICE,
    )
    component_weight = torch.as_tensor(
        positive_weights[mechanism_count:],
        dtype=torch.float32,
        device=DEVICE,
    )
    mechanism_criterion = nn.BCEWithLogitsLoss(pos_weight=mechanism_weight)
    component_criterion = nn.BCEWithLogitsLoss(pos_weight=component_weight)
    generator = torch.Generator(device=DEVICE.type)
    generator.manual_seed(int(config.seed))
    best_state = _copy_state(model)
    best_loss = float("inf")
    best_epoch = 0
    completed_epochs = 0
    stale = 0
    timed_out = False
    for epoch in range(1, int(config.max_epochs) + 1):
        if time.monotonic() >= deadline:
            timed_out = True
            break
        model.train()
        permutation = torch.randperm(train_partition.count, generator=generator, device=DEVICE)
        for start, stop in _batch_ranges(train_partition.count, int(config.batch_size)):
            if time.monotonic() >= deadline:
                timed_out = True
                break
            batch_indices = permutation[start:stop]
            features = tuple(value.index_select(0, batch_indices) for value in train_partition.features)
            labels = train_partition.labels.index_select(0, batch_indices)
            optimizer.zero_grad(set_to_none=True)
            output = model(*features)
            loss = _reason_code_rgfn_loss(
                output,
                labels,
                mechanism_count,
                mechanism_criterion,
                component_criterion,
                float(config.gate_l1),
            )
            loss.backward()
            optimizer.step()
        if timed_out:
            break
        validation_loss = _reason_code_rgfn_validation_loss(
            model,
            validation_partition,
            mechanism_count,
            mechanism_criterion,
            component_criterion,
            float(config.gate_l1),
            max(int(config.batch_size), 512),
        )
        completed_epochs = epoch
        if validation_loss < best_loss - float(config.min_delta):
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _copy_state(model)
            stale = 0
        else:
            stale += 1
        if stale >= int(config.patience):
            break
    elapsed = float(time.monotonic() - split_started)
    if time.monotonic() >= deadline:
        timed_out = True
    if timed_out:
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return None, {
            "split_scheme": split_scheme,
            "status": "timebox_exceeded",
            "elapsed_seconds": elapsed,
            "epochs_completed": int(completed_epochs),
            "best_epoch": int(best_epoch),
            "best_validation_loss": None if not np.isfinite(best_loss) else float(best_loss),
            "test_evaluated": False,
            "threshold_selection": cv_status,
        }
    model.load_state_dict(best_state)
    validation_probabilities = predict_reason_code_rgfn(
        model,
        validation_partition,
        max(int(config.batch_size), 512),
    )
    validation_metrics: list[dict[str, object]] = []
    selection_trace: list[dict[str, object]] = []
    if len(cv_records) != targets.shape[1]:
        raise RuntimeError("reason-code RGFN cross-validation did not produce one threshold per label")
    for record in cv_records:
        index = int(record["label_index"])
        threshold = float(thresholds[index])
        metrics = reason_code_metrics(
            validation_targets[:, index],
            validation_probabilities[:, index],
            threshold,
        )
        reference_threshold, reference_metrics, reference_candidate_count = select_reason_code_threshold(
            validation_targets[:, index],
            validation_probabilities[:, index],
        )
        validation_metrics.append(
            {
                "axis": str(record["axis"]),
                "label": str(record["label"]),
                **metrics,
            }
        )
        selection_trace.append(
            {
                **record,
                "split_scheme": split_scheme,
                "train_support": int(train_targets[:, index].sum()),
                "validation_support": int(validation_targets[:, index].sum()),
                "positive_class_weight": float(positive_weights[index]),
                "validation_metrics": metrics,
                "single_validation_reference_threshold": float(reference_threshold),
                "single_validation_reference_metrics": reference_metrics,
                "single_validation_reference_candidate_count": int(reference_candidate_count),
                "single_validation_reference_is_diagnostic_only": True,
                "outer_validation_used_for_threshold_selection": False,
                "test_metrics_read_during_selection": False,
            }
        )
    training = {
        "status": "completed",
        "elapsed_seconds": elapsed,
        "epochs_completed": int(completed_epochs),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "loss": "0.5 * mechanism_BCE + 0.5 * component_BCE + optional_gate_l1",
        "threshold_selection": cv_status,
        "outer_validation_used_for_early_stopping": True,
        "outer_validation_used_for_threshold_selection": False,
        "test_evaluated": False,
    }
    return FittedReasonCodeRgfn(
        split_scheme=split_scheme,
        model=model,
        scaler=scaler,
        config=config,
        mechanism_count=mechanism_count,
        label_names=label_names,
        thresholds=thresholds,
        positive_class_weights=positive_weights,
        train_support=train_targets.sum(axis=0).astype(np.int64),
        validation_support=validation_targets.sum(axis=0).astype(np.int64),
        validation_metrics=validation_metrics,
        selection_trace=selection_trace,
        training=training,
    ), training


def fit_reason_code_rgfn_models(
    examples: dict[str, np.ndarray],
    splits: dict[str, dict[str, np.ndarray]],
    configs: Mapping[str, HourlyReasonCodeRgfnConfig],
    timebox_seconds: float = 45.0 * 60.0,
    cv_seed: int | None = None,
) -> tuple[dict[str, FittedReasonCodeRgfn], list[dict[str, object]], dict[str, dict[str, object]]]:
    if float(timebox_seconds) <= 0.0:
        raise ValueError("reason-code RGFN timebox must be positive")
    started = time.monotonic()
    deadline = started + float(timebox_seconds)
    fitted: dict[str, FittedReasonCodeRgfn] = {}
    trace: list[dict[str, object]] = []
    status: dict[str, dict[str, object]] = {}
    for scheme in ("random", "spaced"):
        if scheme not in splits:
            raise KeyError(f"reason-code RGFN splits lack {scheme}")
        if scheme not in configs:
            raise KeyError(f"reason-code RGFN configurations lack {scheme}")
        if time.monotonic() >= deadline:
            status[scheme] = {
                "split_scheme": scheme,
                "status": "timebox_exceeded",
                "elapsed_seconds": float(time.monotonic() - started),
                "epochs_completed": 0,
                "best_epoch": 0,
                "best_validation_loss": None,
                "test_evaluated": False,
            }
            continue
        model, detail = _fit_reason_code_rgfn_split(
            examples,
            splits[scheme],
            scheme,
            (
                replace(configs[scheme], cv_seed=int(cv_seed))
                if cv_seed is not None
                else configs[scheme]
            ),
            deadline,
        )
        status[scheme] = detail
        if model is None:
            continue
        fitted[scheme] = model
        trace.extend(model.selection_trace)
    for scheme, detail in status.items():
        detail["total_rgfn_elapsed_seconds"] = float(time.monotonic() - started)
    return fitted, trace, status


def _reason_code_rgfn_test_partition(
    examples: dict[str, np.ndarray],
    fitted: FittedReasonCodeRgfn,
    indices: np.ndarray,
) -> TensorPartition:
    targets, _, _ = _reason_code_targets(examples)
    values = fitted.scaler.transform(examples)
    return _make_partition(values, targets, np.asarray(indices, dtype=np.int64), include_labels=False)


def evaluate_reason_code_rgfn_models(
    fitted: Mapping[str, FittedReasonCodeRgfn],
    examples: dict[str, np.ndarray],
    splits: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, dict[str, dict[str, object]]], list[dict[str, object]]]:
    targets, label_names, mechanism_count = _reason_code_targets(examples)
    evaluations: dict[str, dict[str, dict[str, object]]] = {}
    rows: list[dict[str, object]] = []
    for scheme in ("random", "spaced"):
        if scheme not in fitted:
            continue
        item = fitted[scheme]
        selection_by_label = {
            str(record["label"]): record for record in item.selection_trace
        }
        test_indices = np.asarray(splits[scheme]["test"], dtype=np.int64)
        partition = _reason_code_rgfn_test_partition(examples, item, test_indices)
        probabilities = predict_reason_code_rgfn(item.model, partition)
        if probabilities.shape != (len(test_indices), len(label_names)):
            raise RuntimeError("reason-code RGFN probabilities do not match the test target shape")
        evaluations[scheme] = {}
        for axis, start, stop in (
            ("mechanism", 0, mechanism_count),
            ("component", mechanism_count, len(label_names)),
        ):
            names = label_names[start:stop]
            summary = reason_code_axis_summary(
                targets[test_indices, start:stop],
                probabilities[:, start:stop],
                item.thresholds[start:stop],
                names,
            )
            for local_index, metric in enumerate(summary["per_label"]):
                index = start + local_index
                selection = selection_by_label[str(label_names[index])]
                rows.append(
                    {
                        "method": "rgfn",
                        "split_scheme": scheme,
                        "axis": axis,
                        "label": str(label_names[index]),
                        "threshold": float(item.thresholds[index]),
                        "threshold_source": str(selection["selection_source"]),
                        "threshold_policy": str(selection["threshold_policy"]),
                        "cv_effective_folds": int(selection["cv_effective_folds"]),
                        "train_support": int(item.train_support[index]),
                        "validation_support": int(item.validation_support[index]),
                        **metric,
                    }
                )
            evaluations[scheme][axis] = {
                "test_indices": test_indices,
                "thresholds": item.thresholds[start:stop].copy(),
                "probabilities": probabilities[:, start:stop],
                "truth": targets[test_indices, start:stop].astype(np.int64),
                **summary,
            }
        item.training["test_evaluated"] = True
    return evaluations, rows


def save_reason_code_rgfn_model(
    fitted: FittedReasonCodeRgfn,
    path: Path,
    manifest_path: Path,
    arm2_source_configuration: Mapping[str, object],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mode": "reason_codes",
            "method": "rgfn",
            "split_scheme": fitted.split_scheme,
            "encoder": str(fitted.config.encoder),
            "state_dict": _copy_state(fitted.model),
            "scaler": fitted.scaler.payload(),
            "configuration": asdict(fitted.config),
            "arm2_source_configuration": dict(arm2_source_configuration),
            "historical_binary_fault_class_weight_not_used": arm2_source_configuration.get("fault_class_weight"),
            "mask_mode": str(fitted.config.mask_mode),
            "window_hours": int(fitted.model.window_hours),
            "n_continuous": int(fitted.model.n_continuous),
            "n_static": int(fitted.model.n_static),
            "n_rule_evidence": int(fitted.model.n_rule_evidence),
            "mechanism_count": int(fitted.mechanism_count),
            "label_names": fitted.label_names.tolist(),
            "selected_thresholds": fitted.thresholds.tolist(),
            "positive_class_weights": fitted.positive_class_weights.tolist(),
            "selection_trace": fitted.selection_trace,
            "training": fitted.training,
            "manifest_path": str(manifest_path),
            "parameter_count": int(sum(parameter.numel() for parameter in fitted.model.parameters())),
        },
        destination,
    )
    return destination


def prepare_hourly_rgfn_split(
    examples: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
) -> PreparedHourlyRgfnSplit:
    required = {"train", "validation", "test"}
    if set(splits) != required:
        raise ValueError("hourly RGFN requires train, validation, and test partitions")
    x_cont = np.asarray(examples["X_cont"])
    if x_cont.ndim != 3 or x_cont.shape[1] < 1:
        raise ValueError("hourly RGFN requires a positive-length hourly input tensor")
    labels = np.asarray(examples["y_binary"], dtype=int)
    scaler = HourlyRgfnScaler.fit(examples, np.asarray(splits["train"], dtype=np.int64))
    values = scaler.transform(examples)
    return PreparedHourlyRgfnSplit(
        train=_make_partition(values, labels, splits["train"], include_labels=True),
        validation=_make_partition(values, labels, splits["validation"], include_labels=True),
        test=_make_partition(values, labels, splits["test"], include_labels=False),
        scaler=scaler,
    )


def _batch_ranges(count: int, batch_size: int):
    for start in range(0, int(count), int(batch_size)):
        yield start, min(start + int(batch_size), int(count))


def _validation_loss(
    model: nn.Module,
    partition: TensorPartition,
    criterion: nn.Module,
    batch_size: int,
) -> float:
    if partition.labels is None:
        raise ValueError("validation labels are required")
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for start, stop in _batch_ranges(partition.count, batch_size):
            output = model(*[value[start:stop] for value in partition.features])
            loss = criterion(output["final_fault_logit"], partition.labels[start:stop])
            total += float(loss.detach().cpu()) * (stop - start)
    return float(total / max(partition.count, 1))


def predict_hourly_rgfn(model: nn.Module, partition: TensorPartition, batch_size: int = 512) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start, stop in _batch_ranges(partition.count, batch_size):
            output = model(*[value[start:stop] for value in partition.features])
            values.append(output["binary_prob"].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(values, axis=0) if values else np.empty(0, dtype=np.float32)


def _copy_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@dataclass
class _CudaGraphTrainingStep:
    graph: torch.cuda.CUDAGraph
    static_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    static_labels: torch.Tensor
    batch_size: int

    def stage(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        if int(indices.numel()) != self.batch_size:
            raise ValueError("CUDA graph staging requires the configured batch size")
        for target, source in zip(self.static_features, features):
            torch.index_select(source, 0, indices, out=target)
        torch.index_select(labels, 0, indices, out=self.static_labels)

    def replay(self) -> None:
        self.graph.replay()


def _make_adam(
    model: nn.Module,
    config: HourlyRgfnTrainingConfig,
    capturable: bool,
) -> torch.optim.Adam:
    kwargs: dict[str, object] = {
        "lr": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
    }
    if capturable:
        kwargs["capturable"] = True
    return torch.optim.Adam(model.parameters(), **kwargs)


def _eager_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    set_to_none: bool,
) -> None:
    optimizer.zero_grad(set_to_none=set_to_none)
    output = model(*features)
    loss = criterion(output["final_fault_logit"], labels)
    loss.backward()
    optimizer.step()


def _model_device_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _restore_model_device_state(model: nn.Module, values: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    with torch.no_grad():
        for name, value in current.items():
            value.copy_(values[name])


def _zero_adam_state(optimizer: torch.optim.Optimizer) -> None:
    with torch.no_grad():
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    value.zero_()


def _prepare_cuda_graph_training_step(
    model: nn.Module,
    optimizer: torch.optim.Adam,
    criterion: nn.Module,
    partition: TensorPartition,
    batch_size: int,
) -> _CudaGraphTrainingStep | None:
    if DEVICE.type != "cuda" or not torch.cuda.is_available() or partition.labels is None:
        return None
    if partition.count < int(batch_size):
        return None
    static_features = tuple(
        torch.zeros(
            (int(batch_size), *value.shape[1:]),
            dtype=value.dtype,
            device=value.device,
        )
        for value in partition.features
    )
    static_labels = torch.zeros(int(batch_size), dtype=partition.labels.dtype, device=partition.labels.device)
    initial_model = _model_device_state(model)
    initial_rng = torch.cuda.get_rng_state(device=DEVICE)
    warmup_stream = torch.cuda.Stream(device=DEVICE)
    graph = torch.cuda.CUDAGraph()
    try:
        warmup_stream.wait_stream(torch.cuda.current_stream(device=DEVICE))
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                _eager_training_step(
                    model,
                    optimizer,
                    criterion,
                    static_features,
                    static_labels,
                    set_to_none=False,
                )
        torch.cuda.current_stream(device=DEVICE).wait_stream(warmup_stream)
        with torch.cuda.graph(graph):
            _eager_training_step(
                model,
                optimizer,
                criterion,
                static_features,
                static_labels,
                set_to_none=False,
            )
        _restore_model_device_state(model, initial_model)
        _zero_adam_state(optimizer)
        optimizer.zero_grad(set_to_none=False)
        torch.cuda.set_rng_state(initial_rng, device=DEVICE)
        return _CudaGraphTrainingStep(
            graph=graph,
            static_features=static_features,
            static_labels=static_labels,
            batch_size=int(batch_size),
        )
    except RuntimeError:
        _restore_model_device_state(model, initial_model)
        torch.cuda.set_rng_state(initial_rng, device=DEVICE)
        return None


def _train_candidate(
    encoder: str,
    prepared: PreparedHourlyRgfnSplit,
    config: HourlyRgfnTrainingConfig,
) -> tuple[nn.Module, dict[str, object]]:
    if prepared.train.labels is None or prepared.validation.labels is None:
        raise ValueError("hourly RGFN training requires train and validation labels")
    if config.batch_size < 1 or config.max_epochs < 1 or config.patience < 1 or config.min_delta < 0.0:
        raise ValueError("hourly RGFN training values must be positive")
    set_hourly_rgfn_seed(config.seed)
    model = build_hourly_rgfn(
        encoder,
        n_continuous=int(prepared.train.features[0].shape[-1]),
        n_static=int(prepared.train.features[3].shape[-1]),
        n_rule_evidence=int(prepared.train.features[4].shape[-1]),
        window_hours=int(prepared.train.features[0].shape[1]),
        mask_mode=str(config.mask_mode),
    ).to(DEVICE)
    optimizer = _make_adam(model, config, capturable=DEVICE.type == "cuda")
    positive_weight = torch.tensor([float(config.fault_class_weight)], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    graph_step = _prepare_cuda_graph_training_step(
        model,
        optimizer,
        criterion,
        prepared.train,
        int(config.batch_size),
    )
    if graph_step is None and DEVICE.type == "cuda":
        optimizer = _make_adam(model, config, capturable=False)
    generator = torch.Generator(device=DEVICE.type)
    generator.manual_seed(int(config.seed))
    best_state = _copy_state(model)
    best_epoch = 0
    best_loss = float("inf")
    stale = 0
    completed = 0
    for epoch in range(1, int(config.max_epochs) + 1):
        model.train()
        permutation = torch.randperm(prepared.train.count, generator=generator, device=DEVICE)
        for start, stop in _batch_ranges(prepared.train.count, config.batch_size):
            batch_indices = permutation[start:stop]
            if graph_step is not None and int(stop - start) == graph_step.batch_size:
                graph_step.stage(prepared.train.features, prepared.train.labels, batch_indices)
                graph_step.replay()
            else:
                batch_features = tuple(
                    value.index_select(0, batch_indices) for value in prepared.train.features
                )
                batch_labels = prepared.train.labels.index_select(0, batch_indices)
                _eager_training_step(
                    model,
                    optimizer,
                    criterion,
                    batch_features,
                    batch_labels,
                    set_to_none=graph_step is None,
                )
        validation_loss = _validation_loss(model, prepared.validation, criterion, max(int(config.batch_size), 512))
        completed = epoch
        if validation_loss < best_loss - float(config.min_delta):
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _copy_state(model)
            stale = 0
        else:
            stale += 1
        if stale >= int(config.patience):
            break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": int(best_epoch),
        "epochs_completed": int(completed),
        "best_validation_loss": float(best_loss),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "cuda_graph_used": bool(graph_step is not None),
        "cuda_graph_batch_size": int(config.batch_size) if graph_step is not None else None,
    }


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def metric_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("metric summary requires rows")
    result: dict[str, object] = {"runs": int(len(rows))}
    for name in METRIC_NAMES:
        values = [float(row[name]) for row in rows]
        result[name] = float(np.mean(values))
        result[f"{name}_std"] = _sample_std(values)
    return result


def _aggregate_validation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[float, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (float(row["fault_class_weight"]), float(row["threshold"]))
        grouped.setdefault(key, []).append(row)
    result = []
    for (weight, threshold), values in sorted(grouped.items()):
        metrics = [value["validation"] for value in values]
        summary = metric_summary(metrics)
        validation = {name: float(summary[name]) for name in ("precision", "recall", "f1", "accuracy")}
        validation_std = {name: float(summary[f"{name}_std"]) for name in ("precision", "recall", "f1", "accuracy")}
        result.append(
            {
                "fault_class_weight": float(weight),
                "threshold": float(threshold),
                "validation": validation,
                "validation_std": validation_std,
                "validation_minimum_metric": float(min(validation[name] for name in ("precision", "recall", "f1"))),
                "seed_count": int(len(values)),
            }
        )
    return result


def save_hourly_rgfn_model(
    path: Path,
    model: nn.Module,
    scaler: HourlyRgfnScaler,
    encoder: str,
    config: HourlyRgfnTrainingConfig,
    selection: dict[str, object],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": str(encoder),
            "state_dict": _copy_state(model),
            "scaler": scaler.payload(),
            "training_config": asdict(config),
            "selection": selection,
            "window_hours": int(model.window_hours),
            "n_continuous": int(model.n_continuous),
            "n_static": int(model.n_static),
            "n_rule_evidence": int(model.n_rule_evidence),
            "mask_mode": str(model.mask_mode),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        },
        destination,
    )
    return destination


def train_hourly_rgfn_variant(
    examples: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    encoder: str,
    split_name: str,
    model_dir: Path,
    seeds: tuple[int, ...] = RGFN_SEEDS,
    weights: tuple[float, ...] = RGFN_WEIGHTS,
    thresholds: tuple[float, ...] = RGFN_THRESHOLDS,
    base_config: HourlyRgfnTrainingConfig | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    if not seeds or not weights or not thresholds:
        raise ValueError("hourly RGFN training requires seeds, weights, and thresholds")
    prepared = prepare_hourly_rgfn_split(examples, splits)
    base = HourlyRgfnTrainingConfig() if base_config is None else base_config
    models: dict[tuple[int, float], nn.Module] = {}
    candidate_info: dict[tuple[int, float], dict[str, object]] = {}
    validation_rows: list[dict[str, object]] = []
    for seed in seeds:
        for weight in weights:
            config = replace(base, seed=int(seed), fault_class_weight=float(weight))
            model, info = _train_candidate(encoder, prepared, config)
            probabilities = predict_hourly_rgfn(model, prepared.validation)
            models[(int(seed), float(weight))] = model
            candidate_info[(int(seed), float(weight))] = {**info, "config": asdict(config)}
            if progress is not None:
                progress(
                    {
                        "encoder": str(encoder),
                        "split": str(split_name),
                        "seed": int(seed),
                        "fault_class_weight": float(weight),
                        **info,
                    }
                )
            validation_labels = prepared.validation.labels.detach().cpu().numpy().astype(int)
            for threshold in thresholds:
                validation_rows.append(
                    {
                        "seed": int(seed),
                        "fault_class_weight": float(weight),
                        "threshold": float(threshold),
                        "validation": binary_metrics(validation_labels, probabilities, float(threshold)),
                        "best_epoch": int(info["best_epoch"]),
                        "best_validation_loss": float(info["best_validation_loss"]),
                    }
                )
    validation_grid = _aggregate_validation_rows(validation_rows)
    best_balanced = select_operating_point(validation_grid, "balanced")
    selected_weight = float(best_balanced["fault_class_weight"])
    selected_threshold = float(best_balanced["threshold"])
    test_labels = np.asarray(examples["y_binary"], dtype=int)[np.asarray(splits["test"], dtype=np.int64)]
    selected_validation_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    model_paths: list[str] = []
    selected_epochs: list[int] = []
    for seed in seeds:
        model = models[(int(seed), selected_weight)]
        validation_match = next(
            row
            for row in validation_rows
            if int(row["seed"]) == int(seed)
            and float(row["fault_class_weight"]) == selected_weight
            and float(row["threshold"]) == selected_threshold
        )
        selected_validation_rows.append(
            {
                "seed": int(seed),
                **validation_match["validation"],
                "best_epoch": int(validation_match["best_epoch"]),
                "best_validation_loss": float(validation_match["best_validation_loss"]),
            }
        )
        probabilities = predict_hourly_rgfn(model, prepared.test)
        test_metric = binary_metrics(test_labels, probabilities, selected_threshold)
        test_rows.append({"seed": int(seed), **test_metric})
        selected_epochs.append(int(candidate_info[(int(seed), selected_weight)]["best_epoch"]))
        model_path = Path(model_dir) / f"hourly_rgfn_{encoder}_{split_name}_seed_{int(seed)}.pt"
        saved = save_hourly_rgfn_model(
            model_path,
            model,
            prepared.scaler,
            encoder,
            replace(base, seed=int(seed), fault_class_weight=selected_weight),
            {
                "split": str(split_name),
                "fault_class_weight": selected_weight,
                "threshold": selected_threshold,
                "validation": validation_match["validation"],
                "test": test_metric,
            },
        )
        model_paths.append(str(saved))
    parameter_count = int(candidate_info[(int(seeds[0]), selected_weight)]["parameter_count"])
    result = {
        "encoder": str(encoder),
        "split": str(split_name),
        "parameter_count": parameter_count,
        "weights": [float(value) for value in weights],
        "thresholds": [float(value) for value in thresholds],
        "seed_count": int(len(seeds)),
        "validation_seed_rows": validation_rows,
        "validation_grid": validation_grid,
        "best_balanced": best_balanced,
        "selected_validation_seed_rows": selected_validation_rows,
        "selected_validation_summary": metric_summary(
            [{name: row[name] for name in METRIC_NAMES} for row in selected_validation_rows]
        ),
        "test_seed_rows": test_rows,
        "test_summary": metric_summary(test_rows),
        "final_test_target_check": target_check(metric_summary(test_rows)),
        "selected_best_epoch_mean": float(np.mean(selected_epochs)),
        "selected_best_epoch_std": _sample_std([float(value) for value in selected_epochs]),
        "test_evaluation_count": int(len(seeds)),
        "test_evaluation_count_per_seed": 1,
        "selection_source": "validation",
        "model_paths": model_paths,
    }
    del models
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


def load_calibrated_baseline(path: Path) -> dict[str, dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results", {})
    output: dict[str, dict[str, object]] = {}
    for split in ("random", "spaced"):
        if split not in results:
            raise KeyError(f"calibrated baseline metrics lack {split}")
        item = results[split]
        output[split] = {
            **{name: float(item["final_test"][name]) for name in METRIC_NAMES},
            "fault_class_weight": float(item["best_balanced"]["fault_class_weight"]),
            "threshold": float(item["best_balanced"]["threshold"]),
            "target_check": item["final_test_target_check"],
            "test_evaluation_count": int(item["test_evaluation_count"]),
        }
    return output


def _table_metric(payload: dict[str, object]) -> dict[str, float]:
    source = payload.get("test_summary", payload)
    result = {name: float(source[name]) for name in ("precision", "recall", "f1")}
    for name in ("precision", "recall", "f1"):
        result[f"{name}_std"] = float(source.get(f"{name}_std", np.nan))
    return result


def master_comparison_frame(
    baseline: dict[str, dict[str, object]],
    gru: dict[str, dict[str, object]],
    conv: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for model_name, source in (("baseline", baseline), ("RGFN-GRU", gru), ("RGFN-CONV", conv)):
        for split in ("random", "spaced"):
            if split not in source:
                raise KeyError(f"comparison source lacks {model_name} {split}")
            metric = _table_metric(source[split])
            rows.append({"model": model_name, "split": split, **metric})
    return pd.DataFrame(rows).sort_values(["split", "model"]).reset_index(drop=True)


def spaced_gap_frame(
    baseline: dict[str, dict[str, object]],
    gru: dict[str, dict[str, object]],
    conv: dict[str, dict[str, object]],
) -> pd.DataFrame:
    baseline_metric = _table_metric(baseline["spaced"])
    rows = []
    for model_name, source in (("RGFN-GRU", gru), ("RGFN-CONV", conv)):
        metric = _table_metric(source["spaced"])
        baseline_std = baseline_metric["f1_std"]
        candidate_std = metric["f1_std"]
        overlap = np.nan
        if np.isfinite(baseline_std) and np.isfinite(candidate_std):
            overlap = bool(
                max(baseline_metric["f1"] - baseline_std, metric["f1"] - candidate_std)
                <= min(baseline_metric["f1"] + baseline_std, metric["f1"] + candidate_std)
            )
        rows.append(
            {
                "model": model_name,
                "baseline_spaced_f1": baseline_metric["f1"],
                "rgfn_spaced_f1": metric["f1"],
                "difference": metric["f1"] - baseline_metric["f1"],
                "rgfn_f1_std": candidate_std,
                "baseline_f1_std": baseline_std,
                "beats_baseline": bool(metric["f1"] > baseline_metric["f1"]),
                "std_overlap": overlap,
            }
        )
    return pd.DataFrame(rows)


def target_frame(
    baseline: dict[str, dict[str, object]],
    gru: dict[str, dict[str, object]],
    conv: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for model_name, source in (("baseline", baseline), ("RGFN-GRU", gru), ("RGFN-CONV", conv)):
        for split in ("random", "spaced"):
            if model_name == "baseline":
                metric = source[split]
                check = target_check(metric)
                any_seed = bool(check["achieved"])
            else:
                metric = source[split]["test_summary"]
                check = target_check(metric)
                any_seed = bool(
                    any(target_check(row)["achieved"] for row in source[split]["test_seed_rows"])
                )
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "mean_precision": float(metric["precision"]),
                    "mean_recall": float(metric["recall"]),
                    "mean_f1": float(metric["f1"]),
                    "mean_achieved_90_90_90": bool(check["achieved"]),
                    "any_seed_achieved_90_90_90": any_seed,
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "model"]).reset_index(drop=True)


def _variant_detail_frame(result: dict[str, object]) -> pd.DataFrame:
    validation = result["selected_validation_summary"]
    test = result["test_summary"]
    point = result["best_balanced"]
    return pd.DataFrame(
        [
            {
                "split": result["split"],
                "fault_class_weight": point["fault_class_weight"],
                "threshold": point["threshold"],
                "validation_precision": validation["precision"],
                "validation_precision_std": validation["precision_std"],
                "validation_recall": validation["recall"],
                "validation_recall_std": validation["recall_std"],
                "validation_f1": validation["f1"],
                "validation_f1_std": validation["f1_std"],
                "test_precision": test["precision"],
                "test_precision_std": test["precision_std"],
                "test_recall": test["recall"],
                "test_recall_std": test["recall_std"],
                "test_f1": test["f1"],
                "test_f1_std": test["f1_std"],
                "tp": test["tp"],
                "tp_std": test["tp_std"],
                "fp": test["fp"],
                "fp_std": test["fp_std"],
                "fn": test["fn"],
                "fn_std": test["fn_std"],
                "tn": test["tn"],
                "tn_std": test["tn_std"],
                "parameter_count": result["parameter_count"],
                "best_epoch_mean": result["selected_best_epoch_mean"],
                "best_epoch_std": result["selected_best_epoch_std"],
            }
        ]
    )


def comparison_report(
    baseline: dict[str, dict[str, object]],
    gru: dict[str, dict[str, object]],
    conv: dict[str, dict[str, object]],
) -> str:
    master = master_comparison_frame(baseline, gru, conv)
    gap = spaced_gap_frame(baseline, gru, conv)
    targets = target_frame(baseline, gru, conv)
    parts = [
        "HOUR-LEVEL RGFN COMPARISON",
        "",
        f"device={DEVICE}",
        "SELECTION",
        "RGFN weights and thresholds are selected from aggregate validation metrics across five seeds.",
        "No candidate test metric is used for weight or threshold selection.",
        "Each selected seed model is evaluated once on its test partition.",
        "",
        "MASTER TEST COMPARISON",
        master.to_string(index=False),
        "",
        "RGFN-GRU SELECTED POINTS AND TEST CONFUSION",
        pd.concat([_variant_detail_frame(gru[name]) for name in ("random", "spaced")], ignore_index=True).to_string(index=False),
        "",
        "RGFN-CONV SELECTED POINTS AND TEST CONFUSION",
        pd.concat([_variant_detail_frame(conv[name]) for name in ("random", "spaced")], ignore_index=True).to_string(index=False),
        "",
        "SPACED F1 COMPARISON",
        gap.to_string(index=False),
        "std_overlap is unavailable because the carried baseline has one saved calibration evaluation and no seed standard deviation.",
        "",
        "TEST TARGET CHECK",
        targets.to_string(index=False),
        "",
        "PARAMETER COUNTS",
        pd.DataFrame(
            [
                {"model": "RGFN-GRU", "parameter_count": gru["random"]["parameter_count"]},
                {"model": "RGFN-CONV", "parameter_count": conv["random"]["parameter_count"]},
            ]
        ).to_string(index=False),
    ]
    return "\n".join(parts)
