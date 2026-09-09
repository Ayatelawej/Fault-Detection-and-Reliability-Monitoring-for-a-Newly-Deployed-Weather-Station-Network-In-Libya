from __future__ import annotations

import numpy as np

from src.model.hourly_baseline import EvidenceFusedHgbClassifier, HourlyBaselineConfig
from src.model.hourly_calibration import (
    CALIBRATION_THRESHOLDS,
    CALIBRATION_WEIGHTS,
    calibrate_split,
    select_operating_point,
)
from src.workflows.train_hourly_rgfn import (
    evidence_fusion_feature_views,
    select_evidence_fusion_by_validation,
    select_one_hour_model_by_validation,
)


class _Factory:
    def __init__(self) -> None:
        self.weights: list[float] = []
        self.validation_calls = 0
        self.test_calls = 0

    def __call__(self, config: HourlyBaselineConfig):
        self.weights.append(float(config.fault_class_weight))
        return _Classifier(self)


class _Classifier:
    def __init__(self, factory: _Factory) -> None:
        self.factory = factory

    def fit(self, values: np.ndarray, labels: np.ndarray):
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        marker = float(values[0, 1])
        if marker == 1.0:
            self.factory.validation_calls += 1
        if marker == 2.0:
            self.factory.test_calls += 1
        probability = values[:, 0].astype(float)
        return np.column_stack([1.0 - probability, probability])


class _ProbabilityEstimator:
    def __init__(self, column: int) -> None:
        self.column = int(column)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        probability = np.asarray(values[:, self.column], dtype=float)
        return np.column_stack([1.0 - probability, probability])


def _inputs() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    values = np.asarray(
        [
            [0.1, 0.0],
            [0.9, 0.0],
            [0.2, 0.0],
            [0.8, 0.0],
            [0.1, 1.0],
            [0.8, 1.0],
            [0.6, 1.0],
            [0.9, 1.0],
            [0.2, 2.0],
            [0.9, 2.0],
            [0.7, 2.0],
            [0.1, 2.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0], dtype=np.int64)
    splits = {
        "train": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "validation": np.asarray([4, 5, 6, 7], dtype=np.int64),
        "test": np.asarray([8, 9, 10, 11], dtype=np.int64),
    }
    return values, labels, splits


def test_calibration_selects_from_validation_and_evaluates_test_once() -> None:
    values, labels, splits = _inputs()
    factory = _Factory()
    result = calibrate_split(
        values,
        labels,
        splits,
        HourlyBaselineConfig(fault_class_weight=1.0),
        weights=(1.0, 4.0),
        thresholds=(0.30, 0.50, 0.70),
        model_factory=factory,
    )

    expected = select_operating_point(result["validation_grid"], "balanced")
    assert result["best_balanced"] == expected
    assert result["selection_source"] == "validation"
    assert result["test_evaluation_count"] == 1
    assert factory.validation_calls == 2
    assert factory.test_calls == 1
    assert factory.weights == [1.0, 4.0]
    assert len(result["validation_grid"]) == 6


def test_balanced_selection_maximizes_the_minimum_validation_metric() -> None:
    rows = [
        {
            "fault_class_weight": 1.0,
            "threshold": 0.50,
            "validation": {"precision": 0.95, "recall": 0.80, "f1": 0.87},
        },
        {
            "fault_class_weight": 2.0,
            "threshold": 0.60,
            "validation": {"precision": 0.88, "recall": 0.88, "f1": 0.88},
        },
    ]
    for row in rows:
        row["validation_minimum_metric"] = min(row["validation"].values())

    selected = select_operating_point(rows, "balanced")

    assert selected["fault_class_weight"] == 2.0
    assert selected["threshold"] == 0.60


def test_default_calibration_grid_has_every_requested_operating_point() -> None:
    values, labels, splits = _inputs()
    factory = _Factory()
    result = calibrate_split(
        values,
        labels,
        splits,
        HourlyBaselineConfig(fault_class_weight=1.0),
        model_factory=factory,
    )

    assert result["weight_count"] == len(CALIBRATION_WEIGHTS)
    assert result["threshold_count"] == len(CALIBRATION_THRESHOLDS)
    assert len(result["validation_grid"]) == len(CALIBRATION_WEIGHTS) * len(CALIBRATION_THRESHOLDS)
    assert factory.validation_calls == len(CALIBRATION_WEIGHTS)
    assert factory.test_calls == 1


def test_one_hour_model_selection_uses_validation_and_ignores_test() -> None:
    rows = [
        {
            "model": "HGB",
            "validation_minimum_metric": 0.91,
            "validation_f1": 0.92,
            "validation_precision": 0.91,
            "validation_recall": 0.93,
            "test_f1": 0.10,
        },
        {
            "model": "RGFN",
            "validation_minimum_metric": 0.89,
            "validation_f1": 0.90,
            "validation_precision": 0.89,
            "validation_recall": 0.92,
            "test_f1": 0.99,
        },
    ]

    assert select_one_hour_model_by_validation(rows)["model"] == "HGB"
    rows[0]["test_f1"] = 0.0
    rows[1]["test_f1"] = 1.0
    assert select_one_hour_model_by_validation(rows)["model"] == "HGB"


def test_evidence_fusion_combines_three_probability_views() -> None:
    values = np.asarray([[0.2, 0.4, 0.8], [0.6, 0.3, 0.1]], dtype=float)
    model = EvidenceFusedHgbClassifier(
        full_estimator=_ProbabilityEstimator(0),
        context_estimator=_ProbabilityEstimator(0),
        rule_estimator=_ProbabilityEstimator(0),
        context_indices=np.asarray([1]),
        rule_indices=np.asarray([2]),
        full_weight=0.5,
        context_weight=0.2,
        rule_weight=0.3,
    )

    probability = model.predict_proba(values)[:, 1]

    assert np.allclose(probability, [0.42, 0.39])


def test_evidence_fusion_views_separate_context_and_rules() -> None:
    groups = {
        "continuous:temperature": np.asarray([0]),
        "mask:all_lags": np.asarray([1]),
        "time_since_last:all_lags": np.asarray([2]),
        "rule:physical": np.asarray([3]),
        "static:elevation": np.asarray([4]),
    }

    context, rules = evidence_fusion_feature_views(groups)

    assert context.tolist() == [0, 1, 2, 4]
    assert rules.tolist() == [3, 4]


def test_evidence_fusion_selection_ignores_test_metrics() -> None:
    rows = [
        {
            "full_weight": 1.0,
            "validation_minimum_metric": 0.90,
            "validation_f1": 0.91,
            "validation_precision": 0.90,
            "validation_recall": 0.92,
            "test_f1": 0.99,
        },
        {
            "full_weight": 0.8,
            "validation_minimum_metric": 0.92,
            "validation_f1": 0.92,
            "validation_precision": 0.92,
            "validation_recall": 0.92,
            "test_f1": 0.10,
        },
    ]

    assert select_evidence_fusion_by_validation(rows)["full_weight"] == 0.8
    rows[0]["test_f1"] = 0.0
    rows[1]["test_f1"] = 1.0
    assert select_evidence_fusion_by_validation(rows)["full_weight"] == 0.8
