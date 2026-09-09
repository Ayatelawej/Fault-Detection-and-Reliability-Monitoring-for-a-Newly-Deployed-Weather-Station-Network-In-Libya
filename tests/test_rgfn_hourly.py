from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.model.feature_spec import CONTINUOUS_FEATURES, RULE_EVIDENCE_FLAGS, STATIC_FEATURES, rule_evidence_feature_names
from src.model.hourly_detection import MASK_MODE_PER_FEATURE, MASK_MODE_PER_HOUR, build_hourly_examples, build_hourly_labels
from src.model.hourly_rgfn import ENCODER_CONV, ENCODER_GRU, ENCODER_MLP, HourlyRgfnConfig, build_hourly_rgfn
from src.model.hourly_rgfn_training import HourlyRgfnTrainingConfig, master_comparison_frame, train_hourly_rgfn_variant


def _hourly_frame() -> pd.DataFrame:
    hours = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    rows = []
    for hour_index, hour in enumerate(hours):
        row: dict[str, object] = {
            "station_id": "S1",
            "hour": hour,
            "data_present": 1,
        }
        for feature_index, name in enumerate(CONTINUOUS_FEATURES):
            row[name] = float(hour_index + feature_index / 100.0)
        for feature_index, name in enumerate(STATIC_FEATURES):
            row[name] = float(feature_index + 1)
        for name in RULE_EVIDENCE_FLAGS:
            row[name] = 0.0
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.loc[3, "r_pressure"] = 99999.0
    return frame


def _episodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "episode_id": "fault_1",
                "station_id": "S1",
                "start_hour": "2026-01-01 01:00:00+00:00",
                "end_hour": "2026-01-01 01:00:00+00:00",
                "binary_fault": 1,
                "label_state": "fault",
                "mechanisms": "spike_impossible",
                "components": "barometer",
            }
        ]
    )


def test_hourly_rgfn_variants_return_valid_gated_probabilities() -> None:
    torch.manual_seed(7)
    count = 4
    x_cont = torch.randn(count, 7, len(CONTINUOUS_FEATURES))
    x_cont[0, 0, 0] = torch.nan
    mask = torch.ones(count, 7, 1)
    time_since_last = torch.zeros(count, 7, 1)
    static = torch.randn(count, len(STATIC_FEATURES))
    rule_evidence = torch.randn(count, len(rule_evidence_feature_names()))
    parameter_counts = []

    for encoder in (ENCODER_GRU, ENCODER_CONV):
        model = build_hourly_rgfn(encoder)
        assert model.mask_mode == MASK_MODE_PER_HOUR
        assert model.input_width == len(CONTINUOUS_FEATURES) + 2
        model.eval()
        with torch.no_grad():
            output = model(x_cont, mask, time_since_last, static, rule_evidence)
        expected_logit = output["alpha"] * output["temporal_logit"] + (1.0 - output["alpha"]) * output["rule_logit"]
        assert output["binary_prob"].shape == (count,)
        assert torch.all((output["alpha"] >= 0.0) & (output["alpha"] <= 1.0))
        assert torch.allclose(output["final_fault_logit"], expected_logit)
        assert torch.allclose(output["fault_logit"], expected_logit)
        assert torch.allclose(output["binary_prob"], torch.sigmoid(expected_logit))
        parameter_counts.append(model.parameter_count())

    assert all(value > 0 for value in parameter_counts)
    assert parameter_counts[0] != parameter_counts[1]


def test_hourly_rgfn_reason_code_outputs_keep_independent_mechanism_and_component_heads() -> None:
    torch.manual_seed(11)
    count = 3
    x_cont = torch.randn(count, 7, len(CONTINUOUS_FEATURES))
    mask = torch.ones(count, 7, 1)
    time_since_last = torch.zeros(count, 7, 1)
    static = torch.randn(count, len(STATIC_FEATURES))
    rule_evidence = torch.randn(count, len(rule_evidence_feature_names()))
    binary = build_hourly_rgfn(ENCODER_GRU)
    model = build_hourly_rgfn(
        ENCODER_GRU,
        output_dim=10,
        mechanism_count=4,
    )

    model.eval()
    with torch.no_grad():
        output = model(x_cont, mask, time_since_last, static, rule_evidence)

    expected = output["alpha"] * output["temporal_logits"] + (1.0 - output["alpha"]) * output["rule_logits"]
    assert output["reason_code_logits"].shape == (count, 10)
    assert output["reason_code_probabilities"].shape == (count, 10)
    assert output["mechanism_logits"].shape == (count, 4)
    assert output["component_logits"].shape == (count, 6)
    assert output["mechanism_probabilities"].shape == (count, 4)
    assert output["component_probabilities"].shape == (count, 6)
    assert torch.all((output["alpha"] >= 0.0) & (output["alpha"] <= 1.0))
    assert torch.allclose(output["reason_code_logits"], expected)
    assert torch.allclose(output["reason_code_probabilities"], torch.sigmoid(expected))
    assert model.parameter_count() > binary.parameter_count()


def test_hourly_rgfn_mlp_uses_one_current_hour_and_retains_the_gate() -> None:
    count = 4
    config = HourlyRgfnConfig(window_hours=1)
    x_cont = torch.randn(count, 1, len(CONTINUOUS_FEATURES))
    mask = torch.ones(count, 1, 1)
    time_since_last = torch.zeros(count, 1, 1)
    static = torch.randn(count, len(STATIC_FEATURES))
    rule_evidence = torch.randn(count, len(rule_evidence_feature_names()))
    model = build_hourly_rgfn(ENCODER_MLP, config=config)

    output = model(x_cont, mask, time_since_last, static, rule_evidence)
    expected = output["alpha"] * output["sensor_logit"] + (1.0 - output["alpha"]) * output["evidence_logit"]

    assert output["binary_prob"].shape == (count,)
    assert torch.allclose(output["fault_logit"], expected)
    assert torch.all((output["alpha"] >= 0.0) & (output["alpha"] <= 1.0))
    with pytest.raises(ValueError, match="one-hour"):
        build_hourly_rgfn(ENCODER_MLP)


@pytest.mark.parametrize(
    ("encoder", "mask_mode", "input_width", "parameter_count"),
    [
        (ENCODER_GRU, MASK_MODE_PER_HOUR, 26, 14995),
        (ENCODER_CONV, MASK_MODE_PER_HOUR, 26, 15907),
        (ENCODER_GRU, MASK_MODE_PER_FEATURE, 49, 18307),
        (ENCODER_CONV, MASK_MODE_PER_FEATURE, 49, 18115),
    ],
)
def test_hourly_rgfn_accepts_only_the_configured_mask_layout(
    encoder: str,
    mask_mode: str,
    input_width: int,
    parameter_count: int,
) -> None:
    count = 4
    x_cont = torch.zeros((count, 7, len(CONTINUOUS_FEATURES)))
    correct_width = 1 if mask_mode == MASK_MODE_PER_HOUR else len(CONTINUOUS_FEATURES)
    mask = torch.ones((count, 7, correct_width))
    time_since_last = torch.zeros((count, 7, 1))
    static = torch.zeros((count, len(STATIC_FEATURES)))
    rule_evidence = torch.zeros((count, len(rule_evidence_feature_names())))
    model = build_hourly_rgfn(encoder, config=HourlyRgfnConfig(mask_mode=mask_mode))

    output = model(x_cont, mask, time_since_last, static, rule_evidence)

    assert output["binary_prob"].shape == (count,)
    assert model.input_width == input_width
    assert model.parameter_count() == parameter_count
    wrong_width = len(CONTINUOUS_FEATURES) if correct_width == 1 else 1
    wrong_mask = torch.ones((count, 7, wrong_width))
    with pytest.raises(ValueError, match="mask mode"):
        model(x_cont, wrong_mask, time_since_last, static, rule_evidence)


def test_hourly_rgfn_training_supports_per_feature_mask(tmp_path) -> None:
    count = 30
    generator = np.random.default_rng(5)
    labels = np.asarray([index % 2 for index in range(count)], dtype=np.int64)
    examples = {
        "X_cont": generator.normal(size=(count, 7, len(CONTINUOUS_FEATURES))).astype(np.float32),
        "mask": np.ones((count, 7, len(CONTINUOUS_FEATURES)), dtype=np.float32),
        "time_since_last": np.zeros((count, 7, 1), dtype=np.float32),
        "static": generator.normal(size=(count, 2)).astype(np.float32),
        "rule_evidence": generator.normal(size=(count, 2)).astype(np.float32),
        "y_binary": labels,
    }
    splits = {
        "train": np.arange(0, 18, dtype=np.int64),
        "validation": np.arange(18, 24, dtype=np.int64),
        "test": np.arange(24, 30, dtype=np.int64),
    }
    result = train_hourly_rgfn_variant(
        examples,
        splits,
        ENCODER_GRU,
        "synthetic",
        tmp_path,
        seeds=(0,),
        weights=(1.0,),
        thresholds=(0.5,),
        base_config=HourlyRgfnTrainingConfig(
            batch_size=64,
            max_epochs=1,
            patience=1,
            mask_mode=MASK_MODE_PER_FEATURE,
        ),
    )

    assert result["parameter_count"] == 17123
    assert len(result["model_paths"]) == 1


def test_short_examples_use_only_current_and_earlier_hours() -> None:
    hourly = _hourly_frame()
    labels = build_hourly_labels(hourly, _episodes())
    examples = build_hourly_examples(hourly, labels, window_hours=7)
    target_hour = "2026-01-01 02:00:00+00:00"
    target_index = examples["hour"].tolist().index(target_hour)
    pressure_index = CONTINUOUS_FEATURES.index("r_pressure")

    assert examples["X_cont"].shape == (5, 7, len(CONTINUOUS_FEATURES))
    assert examples["X_cont"][target_index, -1, pressure_index] == 2.0
    assert not np.any(examples["X_cont"][target_index, :, pressure_index] == 99999.0)
    assert examples["mask"][target_index, :4, 0].sum() == 0.0
    assert examples["mask"][target_index, 4:, 0].sum() == 3.0


def _baseline_metrics(value: float) -> dict[str, float]:
    return {"precision": value, "recall": value - 0.01, "f1": value - 0.02}


def _rgfn_metrics(value: float) -> dict[str, object]:
    return {
        "test_summary": {
            "precision": value,
            "recall": value - 0.01,
            "f1": value - 0.02,
            "precision_std": 0.01,
            "recall_std": 0.01,
            "f1_std": 0.01,
        }
    }


def test_master_comparison_has_one_row_per_model_and_split() -> None:
    baseline = {
        "random": _baseline_metrics(0.91),
        "spaced": _baseline_metrics(0.89),
    }
    gru = {
        "random": _rgfn_metrics(0.92),
        "spaced": _rgfn_metrics(0.90),
    }
    conv = {
        "random": _rgfn_metrics(0.93),
        "spaced": _rgfn_metrics(0.91),
    }

    comparison = master_comparison_frame(baseline, gru, conv)

    assert len(comparison) == 6
    assert set(comparison["model"]) == {"baseline", "RGFN-GRU", "RGFN-CONV"}
    assert set(comparison["split"]) == {"random", "spaced"}
    assert comparison.groupby(["model", "split"]).size().eq(1).all()
