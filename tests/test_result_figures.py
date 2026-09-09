from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.rules.result_figures import (
    localized_pressure_series,
    plot_localized_spatial,
    plot_pressure_offsets,
    plot_solar_underread,
    plot_spatial_vs_external,
    plot_systemic_adjudication,
    pressure_offset_summary,
    solar_monthly_fleet_ratio,
    solar_station_clear_day_ratio,
    spatial_external_pairs,
    systemic_tier_counts,
)
from src.workflows.build_july_evaluation_figures import (
    BinaryEvaluation,
    build_class_distribution_figure,
    build_confusion_matrix_figure,
    build_roc_pr_figure,
)


def _assert_png(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 1000


def test_pressure_offset_summary_median_iqr_and_confirmed_flag() -> None:
    external = pd.DataFrame(
        {
            "station_id": ["A", "A", "A", "B", "B"],
            "time_utc": pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC"),
            "r_pressure": [-3.0, -5.0, -7.0, 1.0, 3.0],
        }
    )
    reviewed = pd.DataFrame(
        {
            "station_id": ["A"],
            "label": ["calibration_offset"],
            "reasons": ["external_offset|channel=pressure|level_p50=-5.0"],
            "start_hour": [pd.Timestamp("2026-01-01", tz="UTC")],
            "end_hour": [pd.Timestamp("2026-01-02", tz="UTC")],
        }
    )

    result = pressure_offset_summary(external, reviewed)
    row = result.loc[result["station_id"].eq("A")].iloc[0]

    assert row["median"] == pytest.approx(-5.0)
    assert row["p25"] == pytest.approx(-6.0)
    assert row["p75"] == pytest.approx(-4.0)
    assert bool(row["confirmed_offset"])


def test_solar_monthly_and_station_clear_day_ratios() -> None:
    external = pd.DataFrame(
        {
            "station_id": ["A", "A", "B", "B"],
            "time_utc": pd.to_datetime(
                ["2026-01-01 10:00", "2026-01-02 10:00", "2026-01-01 10:00", "2026-02-01 10:00"],
                utc=True,
            ),
            "clear_sky_ratio": [0.5, 0.9, 0.7, 0.8],
        }
    )

    monthly = solar_monthly_fleet_ratio(external)
    station = solar_station_clear_day_ratio(external, top_fraction=0.5)

    assert monthly.loc[monthly["month"].eq("2026-01"), "fleet_median"].iloc[0] == pytest.approx(0.7)
    assert station.loc[station["station_id"].eq("A"), "clear_day_ratio"].iloc[0] == pytest.approx(0.9)


def test_systemic_tier_counts_order_and_sum() -> None:
    evidence = pd.DataFrame({"support_tier": ["supported", "benign", "benign", "single_channel"]})

    result = systemic_tier_counts(evidence)

    assert result["support_tier"].tolist() == ["benign", "single_channel", "inconclusive", "supported"]
    assert int(result["episodes"].sum()) == 4


def test_external_spatial_pairing_with_isolated_and_confirmed() -> None:
    external = pd.DataFrame(
        {
            "station_id": ["A", "A", "B"],
            "time_utc": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "r_pressure": [-2.0, -4.0, 1.0],
        }
    )
    spatial = pd.DataFrame(
        {
            "station_id": ["A", "A", "B"],
            "time_utc": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "spatial_offset_level_pressure": [0.0, 2.0, None],
            "spatial_isolated": [False, False, True],
        }
    )
    reviewed = pd.DataFrame(
        {
            "station_id": ["A"],
            "label": ["calibration_offset"],
            "reasons": ["external_offset|channel=pressure|level_p50=-3.0"],
            "start_hour": [pd.Timestamp("2026-01-01", tz="UTC")],
            "end_hour": [pd.Timestamp("2026-01-02", tz="UTC")],
        }
    )

    result = spatial_external_pairs(external, spatial, reviewed)
    row = result.loc[result["station_id"].eq("A")].iloc[0]
    isolated = result.loc[result["station_id"].eq("B")].iloc[0]

    assert row["external_median"] == pytest.approx(-3.0)
    assert row["spatial_median"] == pytest.approx(1.0)
    assert bool(row["confirmed_offset"])
    assert bool(isolated["spatial_isolated"])


def test_localized_three_line_reconstruction() -> None:
    time = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    merged = pd.DataFrame({"station_id": ["A", "A"], "hour_utc": time, "pressure_max_hpa": [1000.0, 1002.0]})
    external = pd.DataFrame({"station_id": ["A", "A"], "time_utc": time, "r_pressure": [5.0, 6.0]})
    spatial = pd.DataFrame({"station_id": ["A", "A"], "time_utc": time, "r_spatial_pressure": [2.0, 4.0]})
    episode = pd.Series({"station_id": "A", "start_hour": time[0], "end_hour": time[-1]})

    result = localized_pressure_series(merged, external, spatial, episode)

    assert result.loc[0, "station_pressure"] == pytest.approx(1001.0)
    assert result.loc[0, "healthy_neighbor_pressure"] == pytest.approx(998.0)
    assert result.loc[0, "era5_pressure"] == pytest.approx(995.5)


def test_plotting_functions_write_pngs(tmp_path: Path) -> None:
    pressure = pd.DataFrame(
        {
            "station_id": ["A", "B"],
            "median": [-5.0, -1.0],
            "p25": [-6.0, -2.0],
            "p75": [-4.0, 0.0],
            "n": [10, 10],
            "confirmed_offset": [True, False],
            "confirmed_level_p50": [-5.0, None],
        }
    )
    monthly = pd.DataFrame({"month": ["2026-01"], "fleet_median": [0.7], "p25": [0.6], "p75": [0.8], "n": [10]})
    station = pd.DataFrame({"station_id": ["A", "B"], "clear_day_ratio": [0.6, 0.9], "n_days": [1, 1]})
    counts = pd.DataFrame({"support_tier": ["benign", "single_channel", "inconclusive", "supported"], "episodes": [1, 2, 3, 4]})
    pairs = pd.DataFrame({"station_id": ["A", "B"], "external_median": [-5.0, -1.0], "spatial_median": [-4.0, None], "confirmed_offset": [True, False], "spatial_isolated": [False, True]})
    episodes = pd.DataFrame(
        {
            "station_id": ["A"],
            "start_hour": [pd.Timestamp("2026-01-01", tz="UTC")],
            "end_hour": [pd.Timestamp("2026-01-01 01:00", tz="UTC")],
        }
    )
    time = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    merged = pd.DataFrame({"station_id": ["A", "A"], "hour_utc": time, "pressure_max_hpa": [1000.0, 1001.0]})
    external = pd.DataFrame({"station_id": ["A", "A"], "time_utc": time, "r_pressure": [1.0, 2.0]})
    spatial = pd.DataFrame({"station_id": ["A", "A"], "time_utc": time, "r_spatial_pressure": [3.0, 4.0]})
    paths = [
        plot_pressure_offsets(pressure, tmp_path / "pressure.png"),
        plot_solar_underread(monthly, station, tmp_path / "solar.png"),
        plot_systemic_adjudication(counts, tmp_path / "systemic.png"),
        plot_spatial_vs_external(pairs, tmp_path / "pairs.png"),
        plot_localized_spatial(episodes, merged, external, spatial, tmp_path / "localized.png"),
    ]

    for path in paths:
        _assert_png(path)


def test_selected_detector_evaluation_figures_write_pngs(tmp_path: Path) -> None:
    evaluations = (
        BinaryEvaluation(
            "Development held-out test",
            np.array([0, 0, 1, 1]),
            np.array([0.05, 0.20, 0.75, 0.90]),
            np.array([0, 0, 1, 1]),
        ),
        BinaryEvaluation(
            "Independent July test",
            np.array([0, 0, 1, 1]),
            np.array([0.10, 0.60, 0.55, 0.80]),
            np.array([0, 1, 1, 1]),
        ),
    )
    paths = [
        tmp_path / "distribution.png",
        tmp_path / "confusion.png",
        tmp_path / "curves.png",
    ]

    build_class_distribution_figure(evaluations, paths[0])
    build_confusion_matrix_figure(evaluations, paths[1])
    build_roc_pr_figure(evaluations, paths[2])

    for path in paths:
        _assert_png(path)
