from __future__ import annotations

import numpy as np
import pandas as pd

from src.rules.calibration_corroboration import (
    BORDERLINE_EVIDENCE_COLUMNS,
    CALIBRATION_CORROBORATION_COLUMNS,
    build_calibration_corroboration,
    resolve_borderline_labels,
    tag_borderline_review,
)


def _external_frame(target_pressure: float, solar_zscore: float = 0.0) -> pd.DataFrame:
    hours = pd.date_range("2026-01-01", periods=1001, freq="h", tz="UTC")
    station_ids = ["TARGET", "REF01", "REF02", "REF03", "REF04", "REF05", "REF06", "REF07"]
    rows = []
    for station_id in station_ids:
        is_target = station_id == "TARGET"
        pressure = target_pressure if is_target else 0.0
        for hour in hours:
            rows.append(
                {
                    "station_id": station_id,
                    "time_utc": hour,
                    "r_pressure": pressure,
                    "base_pressure": pressure,
                    "bmad_pressure": 0.2,
                    "z_pressure": 4.0 if is_target and pressure else 0.0,
                    "r_temp": 0.0,
                    "base_temp": 0.0,
                    "bmad_temp": 0.2,
                    "z_temp": 0.0,
                    "r_dewpoint": 0.0,
                    "base_dewpoint": 0.0,
                    "bmad_dewpoint": 0.2,
                    "z_dewpoint": 0.0,
                    "r_solar": 100.0 if is_target else 0.0,
                    "z_solar": solar_zscore if is_target else 0.0,
                },
            )
    return pd.DataFrame(rows)


def _labels() -> pd.DataFrame:
    start = pd.Timestamp("2026-01-10 00:00", tz="UTC")
    return pd.DataFrame(
        {
            "episode_id": ["v2_000001", "v2_000002"],
            "station_id": ["TARGET", "TARGET"],
            "start_hour": [start, start],
            "end_hour": [start + pd.Timedelta(hours=1), start + pd.Timedelta(hours=1)],
            "duration_hours": [2, 2],
            "binary_fault": [0, 0],
            "label_state": ["borderline_review", "borderline_review"],
            "mechanisms": ["", ""],
            "components": ["", ""],
            "fired_channels": ["", ""],
            "period": ["june", "june"],
        },
    )


def _calibration_corroboration_run() -> pd.DataFrame:
    start = pd.Timestamp("2026-01-10 00:00", tz="UTC")
    row = {column: np.nan for column in CALIBRATION_CORROBORATION_COLUMNS}
    row.update(
        {
            "corroboration_run_id": "calibration_pressure_TARGET_01",
            "station_id": "TARGET",
            "channel": "pressure",
            "verdict": "confirmed",
            "sustained_offset": True,
            "tier": "HIGH",
            "start_hour": start,
            "end_hour": start + pd.Timedelta(hours=1),
            "duration_hours": 2.0,
            "confirmed_offset_value": -10.0,
            "resolved_episode_count": 0,
            "resolved_episode_ids": "",
        },
    )
    return pd.DataFrame([row], columns=CALIBRATION_CORROBORATION_COLUMNS)


def test_constant_pressure_offset_is_confirmed() -> None:
    result = build_calibration_corroboration(_external_frame(-10.0))
    target = result.loc[
        result["station_id"].eq("TARGET") & result["channel"].eq("pressure"),
    ]

    assert len(target) == 1
    assert target.iloc[0]["verdict"] == "confirmed"
    assert bool(target.iloc[0]["sustained_offset"])
    assert target.iloc[0]["confirmed_offset_value"] == -10.0


def test_frozen_metrics_are_used_instead_of_recomputing_full_period_stats() -> None:
    external = _external_frame(-10.0)
    frozen_metrics = {
        (station_id, channel): {
            "full_period_covered_hours": 0,
            "full_period_median_residual": -999.0,
            "full_period_residual_mad": 0.0,
            "full_period_fraction_abs_residual_z_ge_3": 0.0,
        }
        for station_id in external["station_id"].unique()
        for channel in ["pressure", "temp", "dewpoint"]
    }

    result = build_calibration_corroboration(external, frozen_metrics=frozen_metrics)
    target = result.loc[
        result["station_id"].eq("TARGET") & result["channel"].eq("pressure"),
    ]

    assert target.iloc[0]["full_period_median_residual"] == -999.0


def test_noise_only_station_has_no_sustained_offset() -> None:
    result = build_calibration_corroboration(_external_frame(0.0))
    target = result.loc[
        result["station_id"].eq("TARGET") & result["channel"].eq("pressure"),
    ]

    assert len(target) == 1
    assert target.iloc[0]["verdict"] == "clean"
    assert not bool(target.iloc[0]["sustained_offset"])


def test_solar_is_not_a_calibration_corroboration_channel() -> None:
    result = build_calibration_corroboration(_external_frame(0.0, solar_zscore=20.0))

    assert "solar" not in set(result["channel"])


def test_matching_borderline_flips_to_fault_and_unmatched_stays_held_out() -> None:
    labels = _labels()
    evidence = pd.DataFrame(
        {
            "episode_id": ["v2_000001"],
            "station_id": ["TARGET"],
            "hour_utc": [pd.Timestamp("2026-01-10 00:00", tz="UTC")],
            "raw_channel": ["pressure_max_hpa"],
            "calibration_channel": ["pressure"],
            "evidence_kind": ["external_residual"],
            "era5_zscore": [4.0],
        },
        columns=BORDERLINE_EVIDENCE_COLUMNS,
    )

    result, matches = resolve_borderline_labels(
        labels,
        evidence,
        _calibration_corroboration_run(),
    )
    resolved = result.set_index("episode_id")

    assert len(matches) == 1
    assert resolved.loc["v2_000001", "label_state"] == "fault"
    assert resolved.loc["v2_000001", "binary_fault"] == 1
    assert resolved.loc["v2_000001", "mechanisms"] == "calibration_offset"
    assert resolved.loc["v2_000002", "label_state"] == "borderline_review"
    assert resolved.loc["v2_000002", "binary_fault"] == 0


def test_broad_review_state_marks_external_or_near_flatline_evidence() -> None:
    labels = _labels()
    review = pd.DataFrame(
        {
            "episode_id": ["v2_000001", "v2_000002"],
            "station_id": ["TARGET", "TARGET"],
            "period": ["june", "june"],
            "max_abs_era5_zscore": [3.0, 0.0],
            "near_flatline_hours": [0, 1],
        },
    )

    result, tagged = tag_borderline_review(labels, review)

    assert len(tagged) == 2
    assert set(result["label_state"]) == {"borderline_review"}
