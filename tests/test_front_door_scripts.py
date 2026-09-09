from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import (
    build_hourly_dataset,
    build_reliability_foundations,
    build_station_health,
    generate_report_assets,
    run_dashboard,
)
from scripts import evaluate_outage_risk
from scripts.evaluate_outage_risk import parse_args as parse_outage_args
from scripts.generate_report_assets import parse_args as parse_report_args
from scripts.train_hourly_detection import parse_args as parse_training_args
from scripts.tune_hourly_detection import parse_args as parse_tuning_args
from src.workflows import train_hourly_baseline, tune_hourly_detection


def test_reliability_foundations_start_from_frozen_input_without_rewriting_it(
    tmp_path,
    monkeypatch,
) -> None:
    hourly_states = pd.DataFrame({"station_id": ["station"]})
    full_events = pd.DataFrame({"event_id": ["full"]})
    network_windows = pd.DataFrame({"window_id": ["network"]})
    classification = pd.DataFrame({"availability_class": ["online"]})
    partial_events = pd.DataFrame({"event_id": ["partial"]})
    structural_gaps = pd.DataFrame({"gap_id": []})
    calls: list[str] = []

    monkeypatch.setattr(
        build_reliability_foundations,
        "MERGED_DATASET_PATH",
        tmp_path / "station_hourly_merged.csv",
    )
    monkeypatch.setattr(
        build_reliability_foundations,
        "HOURLY_ROW_STATES_PATH",
        tmp_path / "hourly_row_states.parquet",
    )
    monkeypatch.setattr(
        build_reliability_foundations,
        "AVAILABILITY_EVENTS_PATH",
        tmp_path / "availability_events.parquet",
    )
    monkeypatch.setattr(
        build_reliability_foundations,
        "NETWORK_OUTAGE_WINDOWS_PATH",
        tmp_path / "network_outage_windows.csv",
    )
    monkeypatch.setattr(
        build_reliability_foundations,
        "require_files",
        lambda *args, **kwargs: calls.append("preflight"),
    )
    monkeypatch.setattr(
        build_reliability_foundations,
        "run_data_audit",
        lambda: calls.append("audit"),
    )
    monkeypatch.setattr(
        build_reliability_foundations.pd,
        "read_parquet",
        lambda path: hourly_states,
    )
    monkeypatch.setattr(
        build_reliability_foundations,
        "build_availability_events",
        lambda frame: full_events,
    )
    monkeypatch.setattr(
        build_reliability_foundations,
        "detect_network_outage_windows",
        lambda *args, **kwargs: network_windows,
    )
    monkeypatch.setattr(
        build_reliability_foundations,
        "assign_outage_class",
        lambda events, windows: events,
    )
    monkeypatch.setattr(
        build_reliability_foundations,
        "write_operational_availability_outputs",
        lambda frame: (classification, partial_events, structural_gaps),
    )
    monkeypatch.setattr(
        build_reliability_foundations,
        "write_station_reliability_summary",
        lambda **kwargs: calls.append("summary"),
    )
    monkeypatch.setattr(
        build_reliability_foundations,
        "write_availability_report",
        lambda *args: calls.append("report"),
    )

    before = b"frozen input"
    build_reliability_foundations.MERGED_DATASET_PATH.write_bytes(before)
    result = build_reliability_foundations.build_reliability_foundations()

    assert result == {
        "hourly_rows": 1,
        "full_outage_events": 1,
        "network_outage_windows": 1,
        "partial_outage_events": 1,
    }
    assert build_reliability_foundations.MERGED_DATASET_PATH.read_bytes() == before
    assert calls == ["preflight", "audit", "summary", "report"]


def test_training_front_door_routes_modes_without_consuming_runner_options() -> None:
    mode, remaining = parse_training_args(["baseline", "--seed", "7"])

    assert mode == "baseline"
    assert remaining == ["--seed", "7"]

    mode, remaining = parse_training_args(["reason-codes", "--seed", "11"])

    assert mode == "reason-codes"
    assert remaining == ["--seed", "11"]

    mode, remaining = parse_training_args(
        ["one-hour-comparison", "--tensor", "one_hour.npz"]
    )

    assert mode == "one-hour-comparison"
    assert remaining == ["--tensor", "one_hour.npz"]

    mode, remaining = parse_training_args(
        ["one-hour-july", "--tensor", "july.npz"]
    )

    assert mode == "one-hour-july"
    assert remaining == ["--tensor", "july.npz"]

    mode, remaining = parse_training_args(
        ["evidence-fusion", "--tensor", "one_hour.npz"]
    )

    assert mode == "evidence-fusion"
    assert remaining == ["--tensor", "one_hour.npz"]


def test_hourly_dataset_custom_window_requires_explicit_output() -> None:
    with pytest.raises(ValueError, match="provided together"):
        build_hourly_dataset.main(["--window-hours", "1"])


def test_reason_code_postprocess_is_described_as_retrospective_not_dashboard_output() -> None:
    help_text = train_hourly_baseline.parse_reason_code_args().format_help().lower()
    source = Path(train_hourly_baseline.__file__).read_text(encoding="utf-8").lower()

    assert "retrospective event-level analysis" in help_text
    assert "operationalise" not in help_text
    assert "reason_code_operational_output.csv" not in source
    assert "dashboard_visible" not in source


def test_tuning_front_door_routes_resume_mode_without_consuming_runner_options() -> None:
    mode, remaining = parse_tuning_args(["resume", "--phase", "report"])

    assert mode == "resume"
    assert remaining == ["--phase", "report"]


def test_report_front_door_selects_one_figure_set() -> None:
    assert parse_report_args(["--set", "results"]) == "results"
    assert parse_report_args(["--set", "july-evaluation"]) == "july-evaluation"
    assert parse_report_args([]) == "methodology"


def test_station_health_front_door_exposes_separate_long_horizon_mode() -> None:
    args = build_station_health.parse_args(["--long-horizon-forecast"])

    assert args.long_horizon_forecast
    assert not args.forecast
    assert not args.scorecard


def test_dashboard_source_contains_no_evaluation_metrics_or_ground_truth() -> None:
    source = Path(run_dashboard.__file__).read_text(encoding="utf-8").lower()

    assert "confusion_matrix" not in source
    assert "roc_auc" not in source
    assert "average_precision" not in source
    assert "truth_fault" not in source
    assert "episode_labels" not in source
    assert 'key="selected_station_id"' in source
    assert "bundle.registry.itertuples" in source


def test_report_all_preflights_result_inputs_before_writing_methodology_figures(monkeypatch) -> None:
    built: list[str] = []

    monkeypatch.setattr(generate_report_assets, "require_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        generate_report_assets,
        "require_result_inputs",
        lambda: (_ for _ in ()).throw(FileNotFoundError("missing result evidence")),
    )
    monkeypatch.setattr(generate_report_assets, "build_methodology", lambda: built.append("methodology"))
    monkeypatch.setattr(generate_report_assets, "build_results", lambda: built.append("results"))

    with pytest.raises(FileNotFoundError, match="missing result evidence"):
        generate_report_assets.main(["--set", "all"])

    assert built == []


def test_hourly_dataset_checks_all_required_inputs_before_creating_outputs(tmp_path) -> None:
    output = tmp_path / "hourly_labels.csv"

    with pytest.raises(FileNotFoundError) as error:
        build_hourly_dataset.main(
            [
                "--source",
                str(tmp_path / "missing_source.csv"),
                "--features",
                str(tmp_path / "missing_features.parquet"),
                "--labels",
                str(tmp_path / "missing_labels.csv"),
                "--labels-output",
                str(output),
            ],
        )

    message = str(error.value)
    assert "canonical merged dataset" in message
    assert "feature matrix" in message
    assert "live episode labels" in message
    assert not output.exists()


def test_station_health_preflights_inputs_before_writing_outputs(tmp_path) -> None:
    output = tmp_path / "station_health_scores.parquet"

    with pytest.raises(FileNotFoundError) as error:
        build_station_health.main(
            [
                "--observations",
                str(tmp_path / "missing_station_hourly.csv"),
                "--reference-dir",
                str(tmp_path / "missing_reference"),
                "--scores-output",
                str(output),
            ]
        )

    assert "canonical merged dataset" in str(error.value)
    assert not output.exists()


def test_station_health_scorecard_mode_is_parameterised_and_exclusive() -> None:
    args = build_station_health.parse_args(
        ["--scorecard", "--reference-hour", "2026-06-30T23:00:00+02:00"]
    )

    assert args.scorecard
    assert not args.forecast
    assert args.reference_hour == "2026-06-30T23:00:00+02:00"
    with pytest.raises(SystemExit):
        build_station_health.parse_args(["--scorecard", "--forecast"])


def test_station_health_scorecard_preflights_every_input_before_writing(tmp_path) -> None:
    table = tmp_path / "scorecard.csv"
    report = tmp_path / "scorecard.txt"
    invariants = tmp_path / "scorecard.json"
    causality = tmp_path / "scorecard_causality.csv"

    with pytest.raises(FileNotFoundError) as error:
        build_station_health.main(
            [
                "--scorecard",
                "--scores-output",
                str(tmp_path / "missing_health.parquet"),
                "--station-registry",
                str(tmp_path / "missing_registry.csv"),
                "--availability-classification",
                str(tmp_path / "missing_availability.parquet"),
                "--forecast-model-dir",
                str(tmp_path / "missing_models"),
                "--scorecard-output",
                str(table),
                "--scorecard-report-output",
                str(report),
                "--scorecard-invariants-output",
                str(invariants),
                "--scorecard-causality-output",
                str(causality),
            ]
        )

    message = str(error.value)
    assert "causal station-health scores" in message
    assert "station registry" in message
    assert "availability classification" in message
    assert "health forecast 1h model" in message
    assert "health forecast 24h model" in message
    assert not any(path.exists() for path in (table, report, invariants, causality))


def test_baseline_training_reports_missing_tensors_before_creating_output_directory(tmp_path) -> None:
    output_dir = tmp_path / "baseline_output"

    with pytest.raises(FileNotFoundError) as error:
        train_hourly_baseline.main(
            [
                "--short-tensor",
                str(tmp_path / "missing_short.npz"),
                "--long-tensor",
                str(tmp_path / "missing_long.npz"),
                "--output-dir",
                str(output_dir),
            ],
        )

    assert "short hourly tensor" in str(error.value)
    assert "long hourly tensor" in str(error.value)
    assert not output_dir.exists()


def test_tuning_reports_all_core_prerequisites_before_creating_output_directory(tmp_path) -> None:
    output_dir = tmp_path / "tuning_output"

    with pytest.raises(FileNotFoundError) as error:
        tune_hourly_detection.main(
            [
                "--tensor",
                str(tmp_path / "missing_short.npz"),
                "--manifest",
                str(tmp_path / "missing_manifest.csv"),
                "--boosted-metrics",
                str(tmp_path / "missing_calibration.json"),
                "--prior-rgfn-metrics",
                str(tmp_path / "missing_rgfn.json"),
                "--output-dir",
                str(output_dir),
            ],
        )

    message = str(error.value)
    assert "short hourly tensor" in message
    assert "baseline split manifest" in message
    assert "calibrated baseline metrics" in message
    assert "prior RGFN metrics" in message
    assert not output_dir.exists()


def test_outage_front_door_accepts_an_output_directory() -> None:
    args = parse_outage_args(["--output-dir", "temporary-evaluation"])

    assert str(args.output_dir) == "temporary-evaluation"


def test_risk_front_door_accepts_fault_label_target() -> None:
    args = parse_outage_args(["--target", "fault"])

    assert args.target == "fault"
    assert not args.run_models


def test_risk_front_door_accepts_the_explicit_causal_training_route() -> None:
    args = parse_outage_args(["--target", "all", "--train-risk-models"])

    assert args.target == "all"
    assert args.train_risk_models
    assert not args.run_models


def test_risk_front_door_accepts_the_discrete_hazard_comparison_route() -> None:
    args = parse_outage_args(["--target", "all", "--train-discrete-hazard"])

    assert args.target == "all"
    assert args.train_discrete_hazard
    assert not args.train_risk_models
    assert not args.run_models


def test_risk_front_door_accepts_the_threshold_reselection_route() -> None:
    args = parse_outage_args(["--target", "all", "--reselect-forecast-thresholds"])

    assert args.target == "all"
    assert args.reselect_forecast_thresholds
    assert not args.train_risk_models
    assert not args.train_discrete_hazard


def test_causal_risk_training_requires_all_targets_before_preflight() -> None:
    with pytest.raises(ValueError, match="requires --target all"):
        evaluate_outage_risk.main(["--target", "outage", "--train-risk-models"])


def test_causal_risk_training_front_door_routes_to_the_new_path(tmp_path, monkeypatch) -> None:
    report_path = tmp_path / "forecast_risk_report.txt"
    report_path.write_text("forecast complete\n", encoding="utf-8")
    calls: dict[str, object] = {}

    monkeypatch.setattr(evaluate_outage_risk, "require_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluate_outage_risk,
        "train_forecast_risk_models",
        lambda output_dir, **kwargs: calls.update(
            {"train_output_dir": output_dir, "train_kwargs": kwargs}
        )
        or {"result": "causal"},
    )
    monkeypatch.setattr(
        evaluate_outage_risk,
        "write_forecast_risk_outputs",
        lambda result, output_dir: calls.update(
            {"write_result": result, "write_output_dir": output_dir}
        )
        or {"report": report_path},
    )
    monkeypatch.setattr(
        evaluate_outage_risk,
        "evaluate_all_with_predictions",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("the legacy prototype path must not run")
        ),
    )

    evaluate_outage_risk.main(
        ["--target", "all", "--train-risk-models", "--output-dir", str(tmp_path)]
    )

    assert calls["train_output_dir"] == tmp_path
    assert calls["write_output_dir"] == tmp_path
    assert calls["write_result"] == {"result": "causal"}


def test_threshold_reselection_front_door_routes_without_training(tmp_path, monkeypatch) -> None:
    report_path = tmp_path / "forecast_risk_threshold_reselection_report.txt"
    report_path.write_text("threshold reselection complete\n", encoding="utf-8")
    calls: dict[str, object] = {}

    monkeypatch.setattr(evaluate_outage_risk, "require_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluate_outage_risk,
        "reselect_forecast_risk_thresholds",
        lambda source_dir: calls.update({"source_dir": source_dir})
        or {"result": "threshold-reselection"},
    )
    monkeypatch.setattr(
        evaluate_outage_risk,
        "write_forecast_threshold_reselection_outputs",
        lambda result, output_dir: calls.update(
            {"write_result": result, "write_output_dir": output_dir}
        )
        or {"report": report_path},
    )
    monkeypatch.setattr(
        evaluate_outage_risk,
        "train_forecast_risk_models",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("threshold reselection must not train a model")
        ),
    )

    evaluate_outage_risk.main(
        [
            "--target",
            "all",
            "--reselect-forecast-thresholds",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert calls["source_dir"] == evaluate_outage_risk.FORECAST_OUTPUT_DIR
    assert calls["write_result"] == {"result": "threshold-reselection"}
    assert calls["write_output_dir"] == tmp_path


def test_discrete_hazard_front_door_routes_to_the_hazard_path(tmp_path, monkeypatch) -> None:
    report_path = tmp_path / "discrete_hazard_report.txt"
    report_path.write_text("hazard complete\n", encoding="utf-8")
    calls: dict[str, object] = {}

    monkeypatch.setattr(evaluate_outage_risk, "require_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluate_outage_risk,
        "train_discrete_hazard_models",
        lambda output_dir, **kwargs: calls.update(
            {"train_output_dir": output_dir, "train_kwargs": kwargs}
        )
        or {"result": "hazard"},
    )
    monkeypatch.setattr(
        evaluate_outage_risk,
        "write_discrete_hazard_outputs",
        lambda result, output_dir: calls.update(
            {"write_result": result, "write_output_dir": output_dir}
        )
        or {"report": report_path},
    )

    evaluate_outage_risk.main(
        ["--target", "all", "--train-discrete-hazard", "--output-dir", str(tmp_path)]
    )

    assert calls["train_output_dir"] == tmp_path
    assert calls["write_output_dir"] == tmp_path
    assert calls["write_result"] == {"result": "hazard"}


def test_discrete_hazard_report_remains_writable_when_timeboxed_before_testing() -> None:
    empty_frame_names = [
        "metrics",
        "selection_trace",
        "calibration",
        "feature_counts",
        "feature_audit",
        "feature_future_validation",
        "hazard_support",
        "independent_onset_support",
        "onset_construction",
        "onset_eligibility",
        "network_event_policy",
        "direct_onset_comparison",
        "manifest_validation",
        "deployment_scope",
    ]
    result = {name: pd.DataFrame() for name in empty_frame_names}
    result["run_status"] = pd.DataFrame(
        [{"run_status": "timeboxed_partial", "test_configurations_completed": 0}]
    )

    report = evaluate_outage_risk._discrete_hazard_report(result)

    assert "RUN STATUS" in report
    assert "timeboxed_partial" in report
    assert "Configurations meeting precision, recall, and F1 >= 0.80:" in report


def test_fault_risk_front_door_refuses_model_fitting() -> None:
    with pytest.raises(ValueError, match="label-only run"):
        evaluate_outage_risk.main(["--target", "fault", "--run-models"])


def test_outage_front_door_does_not_fit_models_without_explicit_flag(
    tmp_path,
    monkeypatch,
) -> None:
    report_path = tmp_path / "outage_risk_label_split_report.txt"
    report_path.write_text("labels only\n", encoding="utf-8")
    output_paths = {
        "partition_summary": tmp_path / "summary.csv",
        "purge_summary": tmp_path / "purge.csv",
        "label_changes": tmp_path / "changes.csv",
        "report": report_path,
    }
    monkeypatch.setattr(evaluate_outage_risk, "require_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluate_outage_risk,
        "build_label_split_report",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        evaluate_outage_risk,
        "write_label_split_outputs",
        lambda *args, **kwargs: output_paths,
    )
    monkeypatch.setattr(
        evaluate_outage_risk,
        "evaluate_all_with_predictions",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("model fitting must require --run-models")
        ),
    )

    evaluate_outage_risk.main(["--output-dir", str(tmp_path)])


def test_outage_evaluation_checks_inputs_before_creating_output_directory(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "evaluation_output"
    monkeypatch.setattr(
        evaluate_outage_risk,
        "AVAILABILITY_CLASSIFICATION_PATH",
        tmp_path / "missing_classification.parquet",
    )

    with pytest.raises(FileNotFoundError) as error:
        evaluate_outage_risk.main(
            [
                "--events",
                str(tmp_path / "missing_events.parquet"),
                "--output-dir",
                str(output_dir),
            ],
        )

    assert "hourly availability classification" in str(error.value)
    assert "availability events" in str(error.value)
    assert not output_dir.exists()
