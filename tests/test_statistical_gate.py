from __future__ import annotations

import numpy as np
import pandas as pd

from src.rules.statistical_gate import (
    build_benign_review,
    build_statistical_evidence,
    build_statistical_review,
    detector_thresholds,
)


def _raw(channel: str, values: list[float]) -> pd.DataFrame:
    times = pd.date_range("2026-07-01 12:00", periods=len(values), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "station_id": ["STA"] * len(values),
            "hour_utc": times,
            channel: values,
        },
    )


def _scores(
    raw: pd.DataFrame,
    channel: str,
    zscore: bool = True,
    iforest: bool = True,
) -> pd.DataFrame:
    count = len(raw)
    result = pd.DataFrame(
        {
            "station_id": raw["station_id"],
            "hour_utc": raw["hour_utc"],
            "channel": [channel] * count,
            "zscore": [0.1] * (count - 1) + [5.0],
            "iforest_score": [0.1] * (count - 1) + [5.0],
            "flag_zscore": [False] * (count - 1) + [zscore],
            "flag_iforest": [False] * (count - 1) + [iforest],
            "flag_stuck": [False] * count,
            "flag_physical": [False] * count,
        },
    )
    result["flag"] = result["flag_zscore"] | result["flag_iforest"]
    result["rolling_variance"] = 1.0
    return result


def _external(short: str, time: pd.Timestamp, zscore: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": ["STA"],
            "time_utc": [time],
            f"station_{short}": [10.0],
            f"ref_{short}": [5.0],
            f"r_{short}": [5.0],
            f"base_{short}": [0.0],
            f"bmad_{short}": [1.0],
            f"z_{short}": [zscore],
        },
    )


def test_context_extreme_with_era5_agreement_passes_full_gate() -> None:
    raw = _raw("solar_radiation_high_wm2", [1200.0] * 16 + [1500.0])
    scores = _scores(raw, "solar_radiation_high_wm2")
    external = _external("solar", raw["hour_utc"].iloc[-1], 4.0)
    external["ref_solar"] = 500.0

    result = build_statistical_evidence(raw, scores, external)

    assert len(result) == 1
    assert result.loc[0, "context_baseline_n"] == 16
    assert result.loc[0, "contextual_outlier"]
    assert result.loc[0, "era5_available"]
    assert result.loc[0, "era5_agrees"]
    assert result.loc[0, "evidence_path"] == "A"
    assert result.loc[0, "full_gate_passed"]


def test_normal_high_solar_stays_benign_despite_two_detector_flags() -> None:
    raw = _raw("solar_radiation_high_wm2", [1300.0] * 17)
    scores = _scores(raw, "solar_radiation_high_wm2", zscore=True, iforest=False)
    external = _external("solar", raw["hour_utc"].iloc[-1], 4.0)
    external["ref_solar"] = 500.0

    result = build_statistical_evidence(raw, scores, external)

    assert not result.loc[0, "contextual_outlier"]
    assert result.loc[0, "exactly_one_detector"]
    assert result.loc[0, "evidence_path"] == ""
    assert not result.loc[0, "full_gate_passed"]
    assert "context_not_outlier" in result.loc[0, "gate_failure_reasons"]


def test_opposite_sign_era5_rejects_an_otherwise_contextual_outlier() -> None:
    raw = _raw("temp_avg_c", [20.0] * 16 + [25.0])
    scores = _scores(raw, "temp_avg_c")
    external = _external("temp", raw["hour_utc"].iloc[-1], -4.0)

    result = build_statistical_evidence(raw, scores, external)

    assert result.loc[0, "era5_available"]
    assert not result.loc[0, "era5_sign_matches_context"]
    assert not result.loc[0, "full_gate_passed"]
    assert "era5_not_agree" in result.loc[0, "gate_failure_reasons"]


def test_single_detector_with_strong_era5_residual_passes_path_b() -> None:
    raw = _raw("temp_avg_c", [20.0] * 16 + [25.0])
    scores = _scores(raw, "temp_avg_c", zscore=True, iforest=False)
    external = _external("temp", raw["hour_utc"].iloc[-1], -4.0)

    result = build_statistical_evidence(raw, scores, external)

    assert result.loc[0, "exactly_one_detector"]
    assert result.loc[0, "path_b_era5_strong"]
    assert result.loc[0, "evidence_path"] == "B"
    assert result.loc[0, "full_gate_passed"]
    assert result.loc[0, "gate_failure_reasons"] == ""


def test_single_detector_with_weak_era5_residual_stays_benign() -> None:
    raw = _raw("temp_avg_c", [20.0] * 16 + [25.0])
    scores = _scores(raw, "temp_avg_c", zscore=True, iforest=False)
    external = _external("temp", raw["hour_utc"].iloc[-1], 2.9)

    result = build_statistical_evidence(raw, scores, external)

    assert result.loc[0, "exactly_one_detector"]
    assert not result.loc[0, "path_b_era5_strong"]
    assert result.loc[0, "evidence_path"] == ""
    assert not result.loc[0, "full_gate_passed"]
    assert "path_b_era5_not_strong" in result.loc[0, "gate_failure_reasons"]


def test_single_detector_solar_context_outlier_stays_benign() -> None:
    raw = _raw("solar_radiation_high_wm2", [1200.0] * 16 + [1500.0])
    scores = _scores(raw, "solar_radiation_high_wm2", zscore=True, iforest=False)
    external = _external("solar", raw["hour_utc"].iloc[-1], 4.0)
    external["ref_solar"] = 500.0

    result = build_statistical_evidence(raw, scores, external)

    assert result.loc[0, "contextual_outlier"]
    assert not result.loc[0, "path_b_external_comparable"]
    assert result.loc[0, "evidence_path"] == ""
    assert not result.loc[0, "full_gate_passed"]
    assert "path_b_external_not_comparable" in result.loc[0, "gate_failure_reasons"]


def test_unmapped_channel_can_pass_with_strong_local_context() -> None:
    raw = _raw("humidity_avg_pct", [50.0] * 16 + [80.0])
    scores = _scores(raw, "humidity_avg_pct")

    result = build_statistical_evidence(raw, scores)

    assert result.loc[0, "era5_metric"] == ""
    assert not result.loc[0, "era5_available"]
    assert result.loc[0, "era5_agrees"]
    assert result.loc[0, "full_gate_passed"]


def test_context_needs_fifteen_leave_one_out_reference_values() -> None:
    raw = _raw("humidity_avg_pct", [50.0] * 14 + [80.0])
    scores = _scores(raw, "humidity_avg_pct")

    result = build_statistical_evidence(raw, scores)

    assert result.loc[0, "context_baseline_n"] == 14
    assert not result.loc[0, "context_available"]
    assert not result.loc[0, "full_gate_passed"]


def test_contextual_unavailable_from_forces_context_unavailable_despite_enough_samples() -> None:
    raw = _raw("humidity_avg_pct", [50.0] * 16 + [80.0])
    scores = _scores(raw, "humidity_avg_pct")
    target_hour = raw["hour_utc"].iloc[-1]

    blocked = build_statistical_evidence(
        raw, scores, contextual_unavailable_from=target_hour,
    )
    unblocked = build_statistical_evidence(raw, scores)

    assert unblocked.loc[0, "context_available"]
    assert not blocked.loc[0, "context_available"]
    assert pd.isna(blocked.loc[0, "contextual_zscore"])
    assert not blocked.loc[0, "contextual_outlier"]
    assert blocked.loc[0, "context_baseline_n"] == unblocked.loc[0, "context_baseline_n"]


def test_contextual_unavailable_from_leaves_earlier_rows_untouched() -> None:
    raw = _raw("humidity_avg_pct", [50.0] * 16 + [80.0])
    scores = _scores(raw, "humidity_avg_pct")
    far_future = raw["hour_utc"].iloc[-1] + pd.Timedelta(days=1)

    result = build_statistical_evidence(
        raw, scores, contextual_unavailable_from=far_future,
    )

    assert result.loc[0, "context_available"]


def test_rain_context_uses_logarithmic_detector_scale() -> None:
    raw = _raw("precip_rate_mmh", [0.0] * 16 + [99.0])
    scores = _scores(raw, "precip_rate_mmh")

    result = build_statistical_evidence(raw, scores)

    assert result.loc[0, "context_value"] == np.log1p(99.0)
    assert result.loc[0, "full_gate_passed"]


def test_direction_requires_both_projections_to_pass() -> None:
    raw = _raw("winddir_avg_deg", [90.0] * 16 + [180.0])
    sin = _scores(raw, "winddir_sin")
    cos = _scores(raw, "winddir_cos")
    scores = pd.concat([sin, cos], ignore_index=True)

    result = build_statistical_evidence(raw, scores)

    assert result["direction_pair_passed"].all()
    assert result["full_gate_passed"].all()


def test_benign_review_includes_every_final_benign_episode() -> None:
    raw = _raw("humidity_avg_pct", [50.0] * 16 + [80.0])
    scores = _scores(raw, "humidity_avg_pct", zscore=True, iforest=False)
    labels = pd.DataFrame(
        {
            "episode_id": ["v2_000001"],
            "station_id": ["STA"],
            "start_hour": [raw["hour_utc"].iloc[-1]],
            "end_hour": [raw["hour_utc"].iloc[-1]],
            "duration_hours": [1],
            "binary_fault": [0],
            "mechanisms": [""],
            "period": ["june"],
        },
    )

    result = build_benign_review(labels, raw, scores)

    assert len(result) == 1
    assert result.loc[0, "review_rank"] == 1
    assert result.loc[0, "single_detector_score"] > 0


def test_statistical_review_keeps_a_shared_witness_for_every_matching_episode() -> None:
    time = pd.Timestamp("2026-07-17 12:00", tz="UTC")
    evidence = pd.DataFrame(
        {
            "station_id": ["STA"],
            "hour_utc": [time],
            "channel": ["temp_avg_c"],
            "raw_channel": ["temp_avg_c"],
            "observed_value": [25.0],
            "zscore": [5.0],
            "iforest_score": [5.0],
            "zscore_threshold": [3.0],
            "iforest_threshold": [3.0],
            "flag_zscore": [True],
            "flag_iforest": [True],
            "both_detectors_same_channel_hour": [True],
            "exactly_one_detector": [False],
            "flag_physical": [False],
            "flag_stuck": [False],
            "physically_normal": [True],
            "context_month": [7],
            "context_hour": [12],
            "context_baseline_n": [16],
            "context_median": [20.0],
            "context_mad": [0.0],
            "context_floor": [0.1],
            "contextual_zscore": [50.0],
            "context_available": [True],
            "contextual_outlier": [True],
            "era5_metric": [""],
            "era5_zscore": [float("nan")],
            "era5_available": [False],
            "era5_sign_matches_context": [False],
            "era5_agrees": [True],
            "path_b_era5_strong": [False],
            "direction_pair_passed": [True],
            "evidence_path": ["A"],
            "full_gate_passed": [True],
            "gate_failure_reasons": [""],
        },
    )
    labels = pd.DataFrame(
        {
            "episode_id": ["v2_000001", "v2_000002"],
            "station_id": ["STA", "STA"],
            "start_hour": [time - pd.Timedelta(hours=1), time],
            "end_hour": [time, time + pd.Timedelta(hours=1)],
            "duration_hours": [2, 2],
            "period": ["june", "june"],
            "mechanisms": ["statistical_anomaly", "statistical_anomaly"],
        },
    )

    result = build_statistical_review(labels, evidence)

    assert set(result["episode_id"]) == {"v2_000001", "v2_000002"}
    assert set(result["evidence_path"]) == {"A"}


def test_detector_thresholds_uses_frozen_values_instead_of_recomputing() -> None:
    raw = _raw("temp_avg_c", [10.0, 11.0, 12.0, 200.0])
    scores = _scores(raw, "temp_avg_c")
    frozen = {("temp_avg_c", "zscore"): 42.0, ("temp_avg_c", "iforest"): 43.0}

    default_result = detector_thresholds(scores)
    frozen_result = detector_thresholds(scores, frozen=frozen)

    assert default_result.loc[0, "zscore_threshold"] != 42.0
    assert frozen_result.loc[0, "zscore_threshold"] == 42.0
    assert frozen_result.loc[0, "iforest_threshold"] == 43.0


def test_detector_thresholds_falls_back_to_recomputing_missing_channels() -> None:
    raw = _raw("temp_avg_c", [10.0, 11.0, 12.0, 200.0])
    scores = _scores(raw, "temp_avg_c")

    result = detector_thresholds(scores, frozen={})

    assert not result.empty
    assert pd.notna(result.loc[0, "zscore_threshold"])
