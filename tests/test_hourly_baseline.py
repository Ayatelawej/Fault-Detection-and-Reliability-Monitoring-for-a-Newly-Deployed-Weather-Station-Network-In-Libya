from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model.hourly_baseline import (
    HourlyBaselineConfig,
    HourlyReasonCodeConfig,
    HourlyReasonCodeLogisticConfig,
    HourlyLogisticScaler,
    SPLIT_FRACTIONS_80_20,
    aggregate_reason_code_prediction_rows,
    baseline_report,
    build_reason_code_cv_folds,
    detector_channel_evidence_for_rows,
    evaluate_reason_code_models,
    filter_eligible_examples,
    fit_reason_code_models,
    fit_reason_code_logistic_models,
    flatten_hourly_features,
    load_reason_code_manifest_splits,
    make_split_manifest,
    prepare_reason_code_population,
    random_split_indices,
    reason_code_metrics,
    reason_code_comparison_rows,
    reason_code_aggregated_metric_rows,
    reason_code_connected_group_metadata,
    reason_code_cv_group_ids,
    reason_code_prediction_frame,
    select_reason_code_event_configuration,
    select_reason_code_threshold,
    select_reason_code_threshold_from_training_oof,
    select_reason_code_threshold_mean_fold_f1,
    resolve_fault_class_weight,
    run_baseline_matrix,
    save_model_bundle,
    spaced_split_indices,
    validation_importance_reference,
    write_metrics_json,
)
from src.model.hourly_detection import MASK_MODE_PER_FEATURE, MASK_MODE_PER_HOUR
from src.model.hourly_rgfn_training import (
    HourlyReasonCodeRgfnConfig,
    diagnostic_reason_code_threshold,
    fit_reason_code_rgfn_models,
)
from src.workflows.train_hourly_baseline import (
    _frozen_selection_test_metrics,
    _verify_saved_prediction_artifact,
)


def _examples(window_hours: int) -> dict[str, np.ndarray]:
    count = 90
    hours = np.asarray(
        [f"2026-{1 + index // 30:02d}-{1 + index % 28:02d} 00:00:00+00:00" for index in range(count)],
        dtype=object,
    )
    labels = np.asarray([int(index % 6 == 0) for index in range(count)], dtype=np.int64)
    display = np.where(labels == 1, "fault", "clean").astype(object)
    display[5] = "excluded"
    labels[5] = -1
    continuous = np.zeros((count, window_hours, 2), dtype=np.float32)
    for index in range(count):
        continuous[index, :, 0] = float(index % 7)
        continuous[index, :, 1] = float(labels[index] == 1)
    source_ids = np.full(count, "", dtype=object)
    for index in np.flatnonzero(labels == 1):
        source_ids[index] = f"episode_{index // 2:03d}"
    mechanism_names = np.asarray(
        ["spike_impossible", "stuck_flatline", "statistical_anomaly", "calibration_offset"],
        dtype=object,
    )
    component_names = np.asarray(
        [
            "anemometer",
            "barometer",
            "light_uv",
            "rain_gauge",
            "thermo_hygrometer",
            "wind_vane",
        ],
        dtype=object,
    )
    mechanisms = np.zeros((count, len(mechanism_names)), dtype=np.float32)
    components = np.zeros((count, len(component_names)), dtype=np.float32)
    for ordinal, index in enumerate(np.flatnonzero(labels == 1)):
        mechanisms[index] = [int((ordinal + offset) % 2 == 0) for offset in range(len(mechanism_names))]
        components[index] = [int((ordinal + offset) % 2 == 0) for offset in range(len(component_names))]
    mechanism_text = np.asarray(
        [
            "|".join(name for name, active in zip(mechanism_names, row) if active)
            for row in mechanisms
        ],
        dtype=object,
    )
    component_text = np.asarray(
        [
            "|".join(name for name, active in zip(component_names, row) if active)
            for row in components
        ],
        dtype=object,
    )
    return {
        "X_cont": continuous,
        "mask": np.ones((count, window_hours, 1), dtype=np.float32),
        "time_since_last": np.zeros((count, window_hours, 1), dtype=np.float32),
        "static": np.ones((count, 3), dtype=np.float32),
        "rule_evidence": np.zeros((count, 4), dtype=np.float32),
        "y_binary": labels,
        "y_mechanism": mechanisms,
        "y_component": components,
        "mechanism_target_available": (labels == 1) & mechanisms.astype(bool).any(axis=1),
        "component_target_available": (labels == 1) & components.astype(bool).any(axis=1),
        "mechanism_label_names": mechanism_names,
        "component_label_names": component_names,
        "station_id": np.asarray([f"S{index % 3}" for index in range(count)], dtype=object),
        "hour": hours,
        "display_state": display,
        "mechanisms": mechanism_text,
        "components": component_text,
        "detectors_fired": np.where(labels == 1, "robust_zscore", "").astype(object),
        "source_episode_ids": source_ids,
        "continuous_feature_names": np.asarray(["signal", "fault_signal"], dtype=object),
        "static_feature_names": np.asarray(["ctx_a", "ctx_b", "ctx_c"], dtype=object),
        "rule_evidence_feature_names": np.asarray(["rule_a", "rule_b", "rule_c", "rule_d"], dtype=object),
    }


def test_flattened_feature_contract_includes_every_numeric_input() -> None:
    short, _ = filter_eligible_examples(_examples(7))
    long, _ = filter_eligible_examples(_examples(49))
    short_values, short_names, _ = flatten_hourly_features(short)
    long_values, long_names, _ = flatten_hourly_features(long)

    assert short_values.shape[1] == 7 * (2 + 2) + 4 + 3
    assert long_values.shape[1] == 49 * (2 + 2) + 4 + 3
    assert len(short_names) == short_values.shape[1]
    assert len(long_names) == long_values.shape[1]
    assert "row_present_t0" in short_names
    assert "time_since_last_t0" in short_names
    assert "rule_a" in short_names
    assert "ctx_a" in short_names


def test_flattening_supports_per_feature_masks_without_changing_per_hour_values() -> None:
    examples, _ = filter_eligible_examples(_examples(7))
    default_values, default_names, default_groups = flatten_hourly_features(examples)
    explicit_values, explicit_names, explicit_groups = flatten_hourly_features(
        examples,
        mask_mode=MASK_MODE_PER_HOUR,
    )

    assert np.array_equal(default_values, explicit_values, equal_nan=True)
    assert default_names == explicit_names
    assert set(default_groups) == set(explicit_groups)
    for name in default_groups:
        assert np.array_equal(default_groups[name], explicit_groups[name])

    feature_examples = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in examples.items()}
    feature_examples["X_cont"][0, -1, 1] = np.nan
    feature_examples["mask"] = (~np.isnan(feature_examples["X_cont"])).astype(np.float32)
    feature_values, feature_names, feature_groups = flatten_hourly_features(
        feature_examples,
        mask_mode=MASK_MODE_PER_FEATURE,
    )

    assert feature_values.shape[1] == 7 * (2 * 2 + 1) + 4 + 3
    assert "signal_present_t0" in feature_names
    assert "fault_signal_present_t0" in feature_names
    assert "feature_mask:signal" in feature_groups
    assert "feature_mask:fault_signal" in feature_groups
    missing_index = feature_names.index("fault_signal_present_t0")
    present_index = feature_names.index("signal_present_t0")
    assert feature_values[0, missing_index] == 0.0
    assert feature_values[0, present_index] == 1.0


def test_four_runs_apply_weight_and_exclude_ineligible_rows() -> None:
    examples, excluded = filter_eligible_examples(_examples(7))
    values, _, _ = flatten_hourly_features(examples)
    config = HourlyBaselineConfig(
        seed=17,
        fault_class_weight=4.0,
        max_iter=3,
        max_leaf_nodes=4,
        min_samples_leaf=2,
    )
    rows, models, split_maps, _ = run_baseline_matrix(
        values,
        examples["y_binary"],
        examples["station_id"],
        examples["hour"],
        examples["display_state"],
        examples["source_episode_ids"],
        "short",
        config,
    )

    assert excluded == 1
    assert len(rows) == 2
    assert set(models) == {"short-random", "short-spaced"}
    assert all(model.get_params()["class_weight"] == {0: 1.0, 1: 4.0} for model in models.values())
    for row in rows:
        assert set(row["validation"]) >= {"precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn"}
        assert set(row["test"]) >= {"precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn"}
        assert sum(row["test"][key] for key in ("tp", "fp", "fn", "tn")) == len(split_maps[row["split_scheme"]]["test"])
    assert not np.equal(examples["display_state"], "excluded").any()


def test_two_windows_produce_four_metric_sets_and_model_artifacts(tmp_path) -> None:
    config = HourlyBaselineConfig(
        seed=19,
        fault_class_weight=3.0,
        max_iter=3,
        max_leaf_nodes=4,
        min_samples_leaf=2,
    )
    all_rows = []
    paths = []
    for window_name, window_hours in (("short", 7), ("long", 49)):
        examples, _ = filter_eligible_examples(_examples(window_hours))
        values, names, _ = flatten_hourly_features(examples)
        rows, models, _, _ = run_baseline_matrix(
            values,
            examples["y_binary"],
            examples["station_id"],
            examples["hour"],
            examples["display_state"],
            examples["source_episode_ids"],
            window_name,
            config,
        )
        all_rows.extend(rows)
        for row in rows:
            path = save_model_bundle(
                models[row["run"]],
                tmp_path / f"{row['run']}.joblib",
                names,
                window_hours,
                config,
                row["split_scheme"],
            )
            paths.append(path)
    metrics_path = write_metrics_json({"runs": all_rows}, tmp_path / "metrics.json")

    assert {row["run"] for row in all_rows} == {
        "short-random",
        "short-spaced",
        "long-random",
        "long-spaced",
    }
    assert all(path.exists() for path in paths)
    assert metrics_path.exists()


def test_split_membership_is_deterministic_complete_and_spaced_by_fault_groups() -> None:
    examples, _ = filter_eligible_examples(_examples(7))
    labels = examples["y_binary"]
    random_one = random_split_indices(labels, seed=11)
    random_two = random_split_indices(labels, seed=11)
    spaced_one, detail_one = spaced_split_indices(
        labels,
        examples["station_id"],
        examples["hour"],
        examples["display_state"],
        examples["source_episode_ids"],
    )
    spaced_two, detail_two = spaced_split_indices(
        labels,
        examples["station_id"],
        examples["hour"],
        examples["display_state"],
        examples["source_episode_ids"],
    )

    assert all(np.array_equal(random_one[name], random_two[name]) for name in random_one)
    assert all(np.array_equal(spaced_one[name], spaced_two[name]) for name in spaced_one)
    assert detail_one["fault_group_count"] == detail_two["fault_group_count"]
    partitions = np.full(len(labels), "", dtype=object)
    for name, indices in spaced_one.items():
        partitions[indices] = name
    for episode_id in np.unique(examples["source_episode_ids"]):
        if not episode_id:
            continue
        member = np.equal(examples["source_episode_ids"], episode_id)
        assert len(set(partitions[member])) == 1


def test_fault_weight_resolves_to_the_not_fault_ratio() -> None:
    labels = np.asarray([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int64)

    assert resolve_fault_class_weight(labels) == 3.0
    assert resolve_fault_class_weight(labels, 5.0) == 5.0


def _reason_code_full_splits(examples: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    labels = np.asarray(examples["y_binary"], dtype=int)
    faults = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    assignments = np.full(len(labels), "", dtype=object)
    assignments[faults[:9]] = "train"
    assignments[faults[9:13]] = "validation"
    assignments[faults[13:]] = "test"
    for partition, indices in zip(("train", "validation", "test"), np.array_split(negatives, 3)):
        assignments[indices] = partition
    return {
        name: np.flatnonzero(assignments == name).astype(np.int64)
        for name in ("train", "validation", "test")
    }


def test_reason_code_population_reuses_the_full_binary_manifest_exactly(tmp_path) -> None:
    examples, _ = filter_eligible_examples(_examples(7))
    full_splits = _reason_code_full_splits(examples)
    manifest = make_split_manifest(
        {"random": full_splits, "spaced": full_splits},
        examples["y_binary"],
        examples["station_id"],
        examples["hour"],
        examples["display_state"],
        examples["source_episode_ids"],
    )
    manifest_path = tmp_path / "binary_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    loaded = load_reason_code_manifest_splits(examples, manifest_path)
    fault_examples, fault_splits, trace = prepare_reason_code_population(examples, loaded)

    assert len(fault_examples["y_binary"]) == int(np.equal(examples["y_binary"], 1).sum())
    assert np.equal(fault_examples["y_binary"], 1).all()
    assert np.equal(fault_examples["display_state"], "fault").all()
    assert fault_examples["y_mechanism"].shape == (15, 4)
    assert fault_examples["y_component"].shape == (15, 6)
    for scheme in ("random", "spaced"):
        combined = np.concatenate([fault_splits[scheme][name] for name in ("train", "validation", "test")])
        assert len(np.unique(combined)) == len(fault_examples["y_binary"])
        assert set(combined) == set(range(len(fault_examples["y_binary"])))
        for partition in ("train", "validation", "test"):
            expected = loaded[scheme][partition]
            expected = expected[np.equal(examples["y_binary"][expected], 1)]
            expected_keys = {
                (str(examples["station_id"][index]), str(examples["hour"][index]))
                for index in expected
            }
            actual_keys = {
                (str(fault_examples["station_id"][index]), str(fault_examples["hour"][index]))
                for index in fault_splits[scheme][partition]
            }
            assert actual_keys == expected_keys
    assert len(trace) == 6
    assert all(row["selection_source"] == "existing_binary_split_manifest" for row in trace)


class _ReasonCodeRecordingClassifier:
    instances: list["_ReasonCodeRecordingClassifier"] = []

    def __init__(self, class_weight: float) -> None:
        self.class_weight = class_weight
        self.predict_sizes: list[int] = []
        self.fit_labels: np.ndarray | None = None
        self.__class__.instances.append(self)

    def fit(self, values: np.ndarray, labels: np.ndarray) -> "_ReasonCodeRecordingClassifier":
        self.fit_labels = np.asarray(labels, dtype=int)
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        self.predict_sizes.append(len(values))
        scores = np.linspace(0.10, 0.90, len(values), dtype=float)
        return np.column_stack([1.0 - scores, scores])


def _recording_reason_code_factory(
    _config: HourlyReasonCodeConfig,
    class_weight: float,
) -> _ReasonCodeRecordingClassifier:
    return _ReasonCodeRecordingClassifier(class_weight)


def test_reason_code_models_use_training_oof_cv_and_emit_localisation_rows(tmp_path) -> None:
    _ReasonCodeRecordingClassifier.instances = []
    examples, _ = filter_eligible_examples(_examples(7))
    full_splits = _reason_code_full_splits(examples)
    fault_examples, fault_splits, _ = prepare_reason_code_population(
        examples,
        {"random": full_splits, "spaced": full_splits},
    )
    values, _, _ = flatten_hourly_features(fault_examples)
    fitted, selection_trace = fit_reason_code_models(
        values,
        fault_examples,
        fault_splits,
        HourlyReasonCodeConfig(max_iter=2, max_leaf_nodes=4, min_samples_leaf=2),
        model_factory=_recording_reason_code_factory,
    )

    assert len(fitted) == 20
    assert len(selection_trace) == 20
    assert all(instance.fit_labels is not None for instance in _ReasonCodeRecordingClassifier.instances)
    final_instances = [
        instance
        for instance in _ReasonCodeRecordingClassifier.instances
        if len(instance.fit_labels) == len(fault_splits["random"]["train"])
    ]
    cv_instances = [
        instance for instance in _ReasonCodeRecordingClassifier.instances if instance not in final_instances
    ]
    assert len(final_instances) == 20
    assert len(cv_instances) == sum(int(row["cv_effective_folds"]) for row in selection_trace)
    assert all(instance.predict_sizes == [4] for instance in final_instances)
    assert all(len(instance.predict_sizes) == 1 for instance in cv_instances)
    assert any(row["selection_source"] == "training_oof_cross_validation" for row in selection_trace)
    assert {
        row["selection_source"] for row in selection_trace
    }.issubset({"training_oof_cross_validation", "predeclared_fixed_0_5"})
    assert all(row["cv_grouping"] == "connected_source_episode" for row in selection_trace)
    assert all(
        2 <= int(row["cv_effective_folds"]) <= 4
        for row in selection_trace
        if row["selection_source"] == "training_oof_cross_validation"
    )
    assert all(
        row["oof_coverage_complete"] is True
        for row in selection_trace
        if row["selection_source"] == "training_oof_cross_validation"
    )
    assert all(row["outer_validation_used_for_threshold"] is False for row in selection_trace)
    assert all(row["outer_test_used_for_threshold"] is False for row in selection_trace)
    assert all(row["test_metrics_read_during_selection"] is False for row in selection_trace)
    assert all(row["positive_class_weight"] > 0.0 for row in selection_trace)

    evaluations, metrics = evaluate_reason_code_models(fitted, values, fault_examples, fault_splits)
    assert all(instance.predict_sizes == [4, 2] for instance in final_instances)
    assert all(len(instance.predict_sizes) == 1 for instance in cv_instances)
    assert len(metrics) == 20
    assert {row["threshold_source"] for row in metrics}.issubset(
        {"training_oof_cross_validation", "predeclared_fixed_0_5"}
    )
    assert {row["axis"] for row in metrics} == {"mechanism", "component"}

    test_indices = np.concatenate(
        [fault_splits[scheme]["test"] for scheme in ("random", "spaced")]
    )
    feature_frame = pd.DataFrame(
        {
            "station_id": fault_examples["station_id"][test_indices],
            "time_utc": fault_examples["hour"][test_indices],
            "stat_flag_zscore_windspeed_avg_kmh": 1.0,
            "stat_flag_physical_suspect_uv_high": 1.0,
        }
    ).drop_duplicates(["station_id", "time_utc"])
    predictions = reason_code_prediction_frame(fault_examples, evaluations, feature_frame)
    assert len(predictions) == 4
    assert set(predictions["split_scheme"]) == {"random", "spaced"}
    assert set(predictions["split"]) == {"test"}
    assert predictions["detector_groups_fired"].eq("robust_zscore").all()
    assert predictions["tensor_detector_groups"].eq(predictions["detector_groups_fired"]).all()
    assert predictions["detector_channels_fired"].eq("windspeed_avg_kmh").all()
    assert predictions["detector_component_evidence"].eq("anemometer").all()
    assert "mechanism_probability_spike_impossible" in predictions
    assert "component_probability_anemometer" in predictions
    assert predictions["true_mechanisms"].str.len().gt(0).all()
    assert predictions["true_components"].str.len().gt(0).all()


def test_reason_code_logistic_models_use_fold_train_scaling_and_training_oof_thresholds() -> None:
    _ReasonCodeRecordingClassifier.instances = []
    examples, _ = filter_eligible_examples(_examples(7))
    full_splits = _reason_code_full_splits(examples)
    fault_examples, fault_splits, _ = prepare_reason_code_population(
        examples,
        {"random": full_splits, "spaced": full_splits},
    )
    values, _, _ = flatten_hourly_features(fault_examples)
    train_indices = fault_splits["random"]["train"]
    scaler = HourlyLogisticScaler.fit(values, train_indices)
    shifted = values.copy()
    non_train = np.setdiff1d(np.arange(len(values), dtype=np.int64), train_indices)
    shifted[non_train] += 10_000.0
    shifted_scaler = HourlyLogisticScaler.fit(shifted, train_indices)

    assert np.array_equal(scaler.median, shifted_scaler.median)
    assert np.array_equal(scaler.iqr, shifted_scaler.iqr)

    fitted, selection_trace = fit_reason_code_logistic_models(
        values,
        fault_examples,
        fault_splits,
        HourlyReasonCodeLogisticConfig(max_iter=2),
        model_factory=lambda _config, class_weight: _ReasonCodeRecordingClassifier(class_weight),
    )

    assert len(fitted) == 20
    assert len(selection_trace) == 20
    assert all(item.method == "logistic" for item in fitted)
    assert all(item.scaler is not None for item in fitted)
    final_instances = [
        instance
        for instance in _ReasonCodeRecordingClassifier.instances
        if len(instance.fit_labels) == len(fault_splits["random"]["train"])
    ]
    cv_instances = [
        instance for instance in _ReasonCodeRecordingClassifier.instances if instance not in final_instances
    ]
    assert len(final_instances) == 20
    assert len(cv_instances) == sum(int(row["cv_effective_folds"]) for row in selection_trace)
    assert all(instance.predict_sizes == [4] for instance in final_instances)
    assert all(len(instance.predict_sizes) == 1 for instance in cv_instances)
    assert any(row["selection_source"] == "training_oof_cross_validation" for row in selection_trace)
    assert all(
        detail["scaler_fit_examples"] == detail["fit_examples"]
        for row in selection_trace
        if row["selection_source"] == "training_oof_cross_validation"
        for detail in row["cv_fold_details"]
    )
    assert all(row["test_metrics_read_during_selection"] is False for row in selection_trace)
    evaluations, metric_rows = evaluate_reason_code_models(fitted, values, fault_examples, fault_splits)
    assert all(instance.predict_sizes == [4, 2] for instance in final_instances)
    assert all(len(instance.predict_sizes) == 1 for instance in cv_instances)
    assert len(metric_rows) == 20
    assert set(evaluations) == {"random", "spaced"}
    assert all(row["method"] == "logistic" for row in metric_rows)


def test_reason_code_oof_threshold_maximises_mean_fold_f1_not_pooled_f1() -> None:
    labels = np.asarray([0, 1, 0, 0, 0, 0, 0, 1], dtype=int)
    probabilities = np.asarray([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.4], dtype=float)
    fold_ids = np.asarray([1, 1, 2, 2, 2, 2, 2, 2], dtype=int)

    cv_threshold, cv_metrics, candidate_count = select_reason_code_threshold_mean_fold_f1(
        labels,
        probabilities,
        fold_ids,
    )
    pooled_threshold, _, _ = select_reason_code_threshold(labels, probabilities)

    assert candidate_count == 2
    assert cv_threshold == 0.1
    assert pooled_threshold == 0.4
    assert cv_metrics["mean_oof_fold_f1"] > 0.45
    assert len(cv_metrics["fold_metrics"]) == 2


def test_reason_code_grouped_cv_reduces_fold_count_and_preserves_connected_episodes() -> None:
    target = np.asarray([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=int)
    source_episode_ids = np.asarray(
        ["a|b", "b", "c", "n1", "n2", "n3", "n4", "n5", "n6"],
        dtype=object,
    )
    groups = reason_code_cv_group_ids(source_episode_ids)
    indices = np.arange(len(target), dtype=np.int64)
    folds, trace = build_reason_code_cv_folds(
        target,
        indices,
        seed=17,
        requested_folds=5,
        groups=groups,
    )
    repeated_folds, repeated_trace = build_reason_code_cv_folds(
        target,
        indices,
        seed=17,
        requested_folds=5,
        groups=groups,
    )

    assert trace["cv_grouping"] == "connected_source_episode"
    assert trace["cv_effective_folds"] == 2
    assert trace["cv_status"] == "reduced_to_group_support"
    assert groups[0] == groups[1]
    assert len(folds) == len(repeated_folds) == 2
    assert trace["cv_fold_details"] == repeated_trace["cv_fold_details"]
    for (fit_indices, oof_indices), (repeat_fit, repeat_oof) in zip(folds, repeated_folds):
        assert np.array_equal(fit_indices, repeat_fit)
        assert np.array_equal(oof_indices, repeat_oof)
        assert not set(groups[fit_indices]).intersection(set(groups[oof_indices]))
        assert np.equal(target[fit_indices], 1).any()
        assert np.equal(target[oof_indices], 1).any()


def test_reason_code_cv_fallback_never_reads_outer_validation_or_test() -> None:
    target = np.asarray([1, 0, 0, 0, 0], dtype=int)
    train_indices = np.arange(len(target), dtype=np.int64)
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fail_if_called(
        fit_indices: np.ndarray,
        oof_indices: np.ndarray,
        _fold: int,
    ) -> np.ndarray:
        calls.append((fit_indices, oof_indices))
        raise AssertionError("infeasible CV must not use a validation fallback")

    threshold, trace = select_reason_code_threshold_from_training_oof(
        target,
        train_indices,
        seed=2026,
        fit_predict_fold=fail_if_called,
    )

    assert threshold == 0.5
    assert calls == []
    assert trace["selection_source"] == "predeclared_fixed_0_5"
    assert trace["cv_status"] == "not_possible_insufficient_minority_support"
    assert trace["cv_effective_folds"] == 0


def test_reason_code_comparison_table_has_all_fixed_rows_without_test_ranking() -> None:
    summary = {
        "macro": {
            "precision": 0.6,
            "recall": 0.5,
            "f1": 0.55,
            "accuracy": 0.8,
            "labels_with_test_support": 1,
            "unsupported_labels": [],
        },
        "micro": {
            "precision": 0.61,
            "recall": 0.51,
            "f1": 0.56,
            "accuracy": 0.81,
            "support": 4,
        },
    }
    evaluations = {
        method: {
            scheme: {axis: summary for axis in ("mechanism", "component")}
            for scheme in ("random", "spaced")
        }
        for method in ("logistic", "gradient_boosted", "rgfn")
    }
    rows = reason_code_comparison_rows(
        evaluations,
        {
            method: {"random": "completed", "spaced": "completed"}
            for method in evaluations
        },
    )

    assert len(rows) == 12
    assert [
        (row["method"], row["split_scheme"], row["axis"])
        for row in rows
    ] == [
        (method, scheme, axis)
        for method in ("logistic", "gradient_boosted", "rgfn")
        for scheme in ("random", "spaced")
        for axis in ("mechanism", "component")
    ]
    assert all("best" not in row for row in rows)
    assert all(row["status"] == "completed" for row in rows)


def test_reason_code_rgfn_timebox_leaves_incomplete_splits_unevaluated() -> None:
    examples, _ = filter_eligible_examples(_examples(7))
    full_splits = _reason_code_full_splits(examples)
    fault_examples, fault_splits, _ = prepare_reason_code_population(
        examples,
        {"random": full_splits, "spaced": full_splits},
    )
    config = HourlyReasonCodeRgfnConfig(max_epochs=1, patience=1)
    fitted, selection_trace, status = fit_reason_code_rgfn_models(
        fault_examples,
        fault_splits,
        {"random": config, "spaced": config},
        timebox_seconds=1e-12,
    )

    assert fitted == {}
    assert selection_trace == []
    assert {detail["status"] for detail in status.values()} == {"timebox_exceeded"}
    assert all(detail["test_evaluated"] is False for detail in status.values())


def test_reason_code_metrics_marks_zero_positive_test_support_not_estimable() -> None:
    metrics = reason_code_metrics(
        np.asarray([0, 0, 0], dtype=int),
        np.asarray([0.1, 0.9, 0.2], dtype=float),
        0.5,
    )

    assert metrics["support"] == 0
    assert metrics["estimated"] is False
    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["f1"] is None
    assert metrics["accuracy"] == 2.0 / 3.0


def test_rgfn_validation_diagnostic_allows_single_class_reason_label() -> None:
    threshold, metrics, candidate_count, status = diagnostic_reason_code_threshold(
        np.asarray([0, 0, 0], dtype=int),
        np.asarray([0.1, 0.3, 0.2], dtype=float),
    )

    assert threshold is None
    assert metrics is None
    assert candidate_count == 0
    assert status == "not_estimable_single_class_validation"


def test_true_80_20_runs_without_a_validation_partition() -> None:
    examples, _ = filter_eligible_examples(_examples(7))
    values, _, _ = flatten_hourly_features(examples)
    config = HourlyBaselineConfig(
        seed=23,
        fault_class_weight=4.0,
        max_iter=3,
        max_leaf_nodes=4,
        min_samples_leaf=2,
    )
    rows, _, split_maps, _ = run_baseline_matrix(
        values,
        examples["y_binary"],
        examples["station_id"],
        examples["hour"],
        examples["display_state"],
        examples["source_episode_ids"],
        "short",
        config,
        SPLIT_FRACTIONS_80_20,
        "80_20",
    )

    assert set(split_maps["random"]) == {"train", "test"}
    assert set(split_maps["spaced"]) == {"train", "test"}
    assert {row["run"] for row in rows} == {"short-random-80_20", "short-spaced-80_20"}
    assert all(row["validation"] is None for row in rows)
    assert all(row["split_configuration"] == "80_20" for row in rows)
    assert all(len(split_maps[name]["train"]) > len(split_maps[name]["test"]) for name in split_maps)


def _metrics(f1: float) -> dict[str, float | int]:
    return {
        "precision": f1 - 0.01,
        "recall": f1 - 0.02,
        "f1": f1,
        "accuracy": f1 + 0.01,
        "tp": 3,
        "fp": 1,
        "fn": 1,
        "tn": 10,
    }


def test_importance_reference_uses_validation_not_test_metrics() -> None:
    rows = [
        {
            "run": "validation_reference",
            "window": "short",
            "split_scheme": "spaced",
            "feature_dimension": 8,
            "model_iterations": 3,
            "validation": _metrics(0.90),
            "test": _metrics(0.50),
        },
        {
            "run": "test_favorite",
            "window": "long",
            "split_scheme": "random",
            "feature_dimension": 8,
            "model_iterations": 3,
            "validation": _metrics(0.80),
            "test": _metrics(0.99),
        },
    ]

    reference = validation_importance_reference(rows)
    report = baseline_report(
        rows,
        [{"feature_group": "rule_evidence", "importance_f1_drop": 0.10}],
        HourlyBaselineConfig(),
        {"fault_group_count": 2},
        reference,
    )

    assert reference["run"] == "validation_reference"
    assert "importance_reference_run=validation_reference" in report
    assert "test_favorite" in report
    assert "best_run" not in report
    assert "BEST-RUN" not in report
    assert "Held-out test metrics are descriptive comparisons only" in report


def _postprocess_group_fixture() -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    source_ids = np.asarray(["episode_a", "episode_a", "episode_b", "episode_b", "episode_c|episode_d", "episode_d"], dtype=object)
    partitions = {
        scheme: {
            "train": np.asarray([0], dtype=np.int64),
            "validation": np.asarray([1], dtype=np.int64),
            "test": np.asarray([2, 3, 4, 5], dtype=np.int64),
        }
        for scheme in ("random", "spaced")
    }
    return source_ids, partitions


def test_reason_code_episode_aggregation_uses_connected_groups_and_all_rules() -> None:
    source_ids, partitions = _postprocess_group_fixture()
    group_ids, diagnostics = reason_code_connected_group_metadata(source_ids, partitions)

    assert group_ids[4] == group_ids[5]
    rows = pd.DataFrame(
        {
            "method": "gradient_boosted",
            "split_scheme": "spaced",
            "axis": "mechanism",
            "label": "spike_impossible",
            "connected_episode_group_id": group_ids[[2, 3, 4, 5]],
            "truth": [1, 1, 0, 0],
            "probability": [0.80, 0.20, 0.80, 0.20],
            "threshold": [0.50, 0.50, 0.50, 0.50],
            "hourly_prediction": [1, 0, 1, 0],
        }
    )

    aggregated = aggregate_reason_code_prediction_rows(rows, diagnostics)
    assert aggregated["connected_episode_group_id"].nunique() == 2
    by_group_rule = aggregated.set_index(["connected_episode_group_id", "aggregation_rule"])["prediction"]
    for group_id in np.unique(group_ids[[2, 3]]):
        assert by_group_rule[(group_id, "any")] == 1
        assert by_group_rule[(group_id, "majority")] == 0
        assert by_group_rule[(group_id, "mean_probability")] == 1
    assert set(aggregated["aggregation_rule"]) == {"any", "majority", "mean_probability"}


def test_reason_code_episode_metrics_mark_zero_support_not_estimable() -> None:
    aggregated = pd.DataFrame(
        {
            "method": ["logistic", "logistic", "logistic", "logistic"],
            "split_scheme": ["spaced"] * 4,
            "aggregation_rule": ["any"] * 4,
            "axis": ["component"] * 4,
            "label": ["supported", "supported", "unsupported", "unsupported"],
            "connected_episode_group_id": ["a", "b", "a", "b"],
            "truth": [1, 0, 0, 0],
                "prediction": [1, 0, 1, 0],
                "evaluation_partition": ["test"] * 4,
                "evaluation_unit": ["complete_connected_event_group"] * 4,
                "is_complete_within_evaluation_partition": [True] * 4,
        }
    )

    metrics = pd.DataFrame(reason_code_aggregated_metric_rows(aggregated))
    unsupported = metrics.loc[
        metrics["average"].eq("per_label") & metrics["label"].eq("unsupported")
    ].iloc[0]
    macro = metrics.loc[metrics["average"].eq("macro")].iloc[0]

    assert unsupported["support"] == 0
    assert unsupported["estimated"] is False
    assert pd.isna(unsupported["f1"])
    assert macro["labels_with_evaluation_support"] == 1
    assert macro["unsupported_labels"] == "unsupported"


def _validation_event_selection_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scores = {
        ("logistic", "any"): (0.74, 0.81, 0.77, 0.90),
        ("logistic", "majority"): (0.76, 0.79, 0.77, 0.91),
        ("logistic", "mean_probability"): (0.78, 0.78, 0.78, 0.91),
        ("gradient_boosted", "any"): (0.79, 0.83, 0.81, 0.92),
        ("gradient_boosted", "majority"): (0.82, 0.82, 0.82, 0.93),
        ("gradient_boosted", "mean_probability"): (0.86, 0.88, 0.87, 0.94),
        ("rgfn", "any"): (0.62, 0.73, 0.67, 0.83),
        ("rgfn", "majority"): (0.64, 0.70, 0.66, 0.84),
        ("rgfn", "mean_probability"): (0.63, 0.71, 0.66, 0.84),
    }
    common = {
        "split_scheme": "spaced",
        "axis": "mechanism",
        "evaluation_partition": "validation",
        "evaluation_unit": "complete_connected_event_group",
        "episode_groups": 7,
        "complete_evaluation_groups": 7,
        "fragmented_evaluation_groups": 0,
        "labels_total": 4,
        "labels_with_evaluation_support": 3,
        "unsupported_labels": "calibration_offset",
    }
    for (method, rule), (precision, recall, f1, accuracy) in scores.items():
        rows.append(
            {
                **common,
                "method": method,
                "aggregation_rule": rule,
                "average": "macro",
                "label": "",
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
            }
        )
        for label in (
            "calibration_offset",
            "spike_impossible",
            "stuck_flatline",
            "statistical_anomaly",
        ):
            rows.append(
                {
                    **common,
                    "method": method,
                    "aggregation_rule": rule,
                    "average": "per_label",
                    "label": label,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "accuracy": accuracy,
                    "estimated": True,
                    "support": 1 if label == "calibration_offset" else 7,
                }
            )
    return pd.DataFrame(rows)


def test_reason_code_event_configuration_uses_only_spaced_validation_metrics() -> None:
    metrics = _validation_event_selection_metrics()
    random = metrics.loc[
        metrics["average"].eq("macro") & metrics["method"].eq("logistic")
    ].copy()
    random["split_scheme"] = "random"
    random[["precision", "recall", "f1", "accuracy"]] = 0.999
    trace, candidates = select_reason_code_event_configuration(
        pd.concat([metrics, random], ignore_index=True)
    )

    assert trace["selected_method"] == "gradient_boosted"
    assert trace["selected_aggregation_rule"] == "mean_probability"
    assert trace["selection_partition"] == "validation"
    assert trace["selection_split_scheme"] == "spaced"
    assert trace["axis"] == "mechanism"
    assert trace["selection_labels"] == [
        "spike_impossible",
        "statistical_anomaly",
        "stuck_flatline",
    ]
    assert trace["excluded_low_support_labels"] == ["calibration_offset"]
    assert trace["test_metrics_read_during_configuration_selection"] is False
    assert len(candidates) == 9
    assert candidates.loc[candidates["selected"], "method"].tolist() == ["gradient_boosted"]


def test_reason_code_event_configuration_has_fixed_validation_tie_breaks() -> None:
    metrics = _validation_event_selection_metrics()
    logistic = metrics["method"].eq("logistic") & metrics["aggregation_rule"].eq("any")
    boosted = metrics["method"].eq("gradient_boosted") & metrics["aggregation_rule"].eq("mean_probability")
    metrics.loc[logistic, ["f1", "precision", "accuracy", "recall"]] = [0.91, 0.95, 0.94, 0.90]
    metrics.loc[boosted, ["f1", "precision", "accuracy", "recall"]] = [0.91, 0.94, 0.95, 0.99]
    trace, _ = select_reason_code_event_configuration(metrics)
    assert trace["selected_method"] == "logistic"
    assert trace["selected_aggregation_rule"] == "any"

    metrics.loc[boosted, ["f1", "precision", "accuracy", "recall"]] = [0.91, 0.95, 0.94, 0.90]
    first, _ = select_reason_code_event_configuration(metrics)
    second, _ = select_reason_code_event_configuration(metrics.sample(frac=1.0, random_state=7))
    assert first == second
    assert first["selected_method"] == "gradient_boosted"
    assert first["selected_aggregation_rule"] == "mean_probability"


def test_reason_code_validation_aggregation_rejects_non_validation_rows() -> None:
    source_ids, partitions = _postprocess_group_fixture()
    group_ids, diagnostics = reason_code_connected_group_metadata(source_ids, partitions)
    rows = pd.DataFrame(
        {
            "method": ["gradient_boosted"],
            "split_scheme": ["spaced"],
            "split": ["test"],
            "axis": ["mechanism"],
            "label": ["stuck_flatline"],
            "connected_episode_group_id": [group_ids[1]],
            "truth": [1],
            "probability": [0.9],
            "threshold": [0.5],
            "hourly_prediction": [1],
        }
    )
    with pytest.raises(ValueError, match="requested partition"):
        aggregate_reason_code_prediction_rows(
            rows,
            diagnostics,
            evaluation_partition="validation",
        )


def test_reason_code_event_configuration_rejects_fragmented_validation_groups() -> None:
    metrics = _validation_event_selection_metrics()
    selection_rows = metrics["average"].eq("macro")
    metrics.loc[selection_rows, "evaluation_unit"] = "mixed_complete_and_fragmented_validation_event_groups"
    metrics.loc[selection_rows, "fragmented_evaluation_groups"] = 1
    with pytest.raises(ValueError, match="complete validation"):
        select_reason_code_event_configuration(metrics)


def test_reason_code_event_configuration_rejects_test_metric_rows() -> None:
    metrics = _validation_event_selection_metrics()
    metrics["evaluation_partition"] = "test"
    with pytest.raises(ValueError, match="validation metrics only"):
        select_reason_code_event_configuration(metrics)


def test_reason_code_event_configuration_enforces_spaced_mechanism_scope() -> None:
    metrics = _validation_event_selection_metrics()
    with pytest.raises(ValueError, match="spaced split"):
        select_reason_code_event_configuration(metrics, split_scheme="random")
    with pytest.raises(ValueError, match="mechanism labels"):
        select_reason_code_event_configuration(metrics, axis="component")


def test_reason_code_event_configuration_names_zero_support_labels() -> None:
    metrics = _validation_event_selection_metrics()
    calibration = metrics["average"].eq("per_label") & metrics["label"].eq("calibration_offset")
    metrics.loc[calibration, "support"] = 0
    metrics.loc[calibration, "estimated"] = False
    trace, _ = select_reason_code_event_configuration(metrics)

    assert trace["labels_total"] == 4
    assert trace["labels_with_positive_validation_support"] == 3
    assert trace["labels_support_qualified"] == 3
    assert trace["zero_support_validation_labels"] == ["calibration_offset"]
    assert trace["excluded_low_support_labels"] == ["calibration_offset"]


def test_frozen_selection_test_metrics_keeps_the_validation_label_set_explicit() -> None:
    selected = pd.DataFrame(
        {
            "method": ["gradient_boosted"] * 4,
            "split_scheme": ["spaced"] * 4,
            "aggregation_rule": ["any"] * 4,
            "axis": ["mechanism"] * 4,
            "label": ["supported", "supported", "zero_support", "zero_support"],
            "connected_episode_group_id": ["a", "b", "a", "b"],
            "truth": [1, 0, 0, 0],
            "prediction": [1, 0, 1, 0],
            "evaluation_partition": ["test"] * 4,
            "evaluation_unit": ["complete_connected_event_group"] * 4,
            "is_complete_within_evaluation_partition": [True] * 4,
        }
    )
    trace = {
        "selected_method": "gradient_boosted",
        "selected_aggregation_rule": "any",
        "selection_split_scheme": "spaced",
        "axis": "mechanism",
        "selection_labels": ["supported", "zero_support"],
    }
    metrics = _frozen_selection_test_metrics(selected, trace)
    macro = metrics.loc[metrics["average"].eq("macro")].iloc[0]
    zero = metrics.loc[
        metrics["average"].eq("per_label") & metrics["label"].eq("zero_support")
    ].iloc[0]

    assert macro["frozen_selection_labels"] == "supported|zero_support"
    assert macro["frozen_selection_label_count"] == 2
    assert macro["frozen_selection_labels_with_test_support"] == 1
    assert macro["frozen_selection_labels_without_test_support"] == "zero_support"
    assert zero["estimated"] is False
    assert bool(macro["test_metrics_used_for_selection"]) is False


def test_saved_reason_code_prediction_artifact_requires_exact_model_match() -> None:
    rows = pd.DataFrame(
        {
            "method": ["logistic"],
            "split_scheme": ["spaced"],
            "axis": ["mechanism"],
            "label": ["stuck_flatline"],
            "station_id": ["S1"],
            "hour": ["2026-01-01 00:00:00+00:00"],
            "source_episode_ids": ["episode_1"],
            "truth": [1],
            "probability": [0.9],
            "threshold": [0.5],
            "hourly_prediction": [1],
        }
    )

    details = _verify_saved_prediction_artifact(rows, rows.copy())
    assert details["logistic"]["rows"] == 1
    assert details["logistic"]["max_abs_probability_delta"] == 0.0

    duplicated = pd.concat([rows, rows], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        _verify_saved_prediction_artifact(duplicated, rows)
