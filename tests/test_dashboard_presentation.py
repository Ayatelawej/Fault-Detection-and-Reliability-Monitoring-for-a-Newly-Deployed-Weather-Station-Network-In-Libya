from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.dashboard.presentation import (
    HORIZONS, event_neighbor_comparison, event_weather_comparison, fault_status,
    forecast_table, health_breakdown, history_chart, network_table, score_text,
)
from src.dashboard.replay import ReplayBundle, station_history


def _row():
    return pd.Series({
        "station_id": "A", "city": "Tripoli", "status": "Fault alert",
        "availability_class": "online", "fault_detected": 1, "health_total": 80.0,
        "health_band": "Healthy", "weighted_health_availability": 25.,
        "weighted_health_sensor_completeness": 15., "weighted_health_fault_burden": 20.,
        "weighted_health_reference_consistency": 12., "weighted_health_stability": 8.,
        **{f"forecast_{h}h": 80. - h for h in HORIZONS},
    })


def _bundle():
    hours = pd.date_range("2026-07-01", periods=3, freq="h", tz="UTC")
    health = pd.DataFrame([
        {"station_id": "A", "hour_utc": hours[0], "is_transmitting": True,
         "health_total": 80., "temp_avg_c": 30., "temperature_2m": 29.},
        {"station_id": "A", "hour_utc": hours[1], "is_transmitting": True,
         "health_total": 70., "temp_avg_c": np.nan, "temperature_2m": 999.},
        {"station_id": "A", "hour_utc": hours[2], "is_transmitting": True,
         "health_total": 60., "temp_avg_c": 900., "temperature_2m": 0.},
        {"station_id": "B", "hour_utc": hours[0], "is_transmitting": True,
         "health_total": 90., "temp_avg_c": 28.},
        {"station_id": "B", "hour_utc": hours[1], "is_transmitting": False,
         "health_total": 89., "temp_avg_c": 1000.},
        {"station_id": "B", "hour_utc": hours[2], "is_transmitting": True,
         "health_total": 88., "temp_avg_c": 0.},
    ])
    bundle = ReplayBundle(health, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        pd.DataFrame([{"station_id": "A", "neighbor_id": "B", "distance_km": 12.}]),
        pd.DataFrame([{"station_id": "A", "city": "Alpha"}, {"station_id": "B", "city": "Beta"}]))
    event = pd.Series({"station_id": "A", "start_hour": hours[0], "end_hour": hours[1], "duration_hours": 2})
    return bundle, event


def test_network_has_all_five_forecast_horizons_without_mutation():
    snapshot = pd.DataFrame([_row()])
    before = snapshot.copy(deep=True)
    table = network_table(snapshot)
    assert list(table.columns[-5:]) == [f"+{h} h" for h in HORIZONS]
    assert table.iloc[0]["+24 h"] == 56.
    assert "Band" not in table
    pd.testing.assert_frame_equal(snapshot, before)


def test_health_breakdown_uses_actual_maxima_and_sums_to_total():
    row = _row()
    table = health_breakdown(row)
    assert table.Maximum.tolist() == [30., 20., 25., 15., 10.]
    assert table.Maximum.sum() == 100.
    assert table.Points.sum() == row.health_total
    assert score_text(table.Points[0], table.Maximum[0]) == "25.0 / 30"


@pytest.mark.parametrize("value", [np.nan, pd.NA, None])
def test_missing_scores_are_not_displayed_as_zero(value):
    row = _row()
    row["forecast_3h"] = value
    assert pd.isna(forecast_table(row).loc[1, "Health /100"])
    assert score_text(value) == "Not available"
    assert "Source" not in forecast_table(row)


def test_fault_status_distinguishes_alert_no_alert_and_unavailable():
    row = _row()
    assert fault_status(row) == "Likely fault"
    row["fault_detected"] = 0
    assert fault_status(row) == "No alert"
    row["fault_detected"] = pd.NA
    assert fault_status(row) == "Not available"
    row["availability_class"] = "full_outage"
    assert fault_status(row) == "Not assessed"


def test_weather_comparison_is_paired_and_stops_at_event_end():
    bundle, event = _bundle()
    before = bundle.health.copy(deep=True)
    result = event_weather_comparison(bundle, event).set_index("Variable")
    assert result.loc["Temperature", "Paired hours"] == 1
    assert result.loc["Temperature", "Station mean"] == 30.
    assert result.loc["Temperature", "Reference mean"] == 29.
    assert result.loc["Temperature", "Mean difference"] == 1.
    assert result.loc["Dew point", "Paired hours"] == 0
    assert pd.isna(result.loc["Dew point", "Mean difference"])
    pd.testing.assert_frame_equal(bundle.health, before)


def test_neighbor_comparison_reports_identity_distance_and_matched_support():
    bundle, event = _bundle()
    result = event_neighbor_comparison(bundle, event).iloc[0]
    assert result.Neighbor == "B"
    assert result.Location == "Beta"
    assert result["Distance (km)"] == 12.
    assert result["Reporting hours"] == 1
    assert result["Paired temperature hours"] == 1
    assert result["Station − neighbor (°C)"] == 2.
    empty = event_neighbor_comparison(replace(bundle, neighbors=bundle.neighbors.iloc[:0]), event)
    assert empty.empty and "Neighbor" in empty


def test_reporting_neighbor_with_no_paired_values_is_not_claimed_to_agree():
    bundle, event = _bundle()
    bundle.health.loc[bundle.health.station_id.eq("B"), "temp_avg_c"] = np.nan
    row = event_neighbor_comparison(bundle, event).iloc[0]
    assert row["Reporting hours"] == 1
    assert row["Paired temperature hours"] == 0
    assert pd.isna(row["Station − neighbor (°C)"])


def test_full_history_remains_past_only_and_chart_supports_horizontal_navigation():
    bundle, event = _bundle()
    history = station_history(bundle, "A", event.end_hour, lookback_hours=None)
    assert history.hour_utc.max() == event.end_hour
    fig = history_chart(history, "A")
    assert fig.layout.dragmode == "pan"
    assert fig.layout.xaxis.rangeslider.visible
    assert fig.layout.yaxis.fixedrange
    assert list(fig.layout.yaxis.range) == [0, 100]
    assert not fig.data[0].connectgaps
    assert max(fig.data[0].x) == event.end_hour


def test_history_chart_preserves_missing_clock_hours():
    hours = pd.to_datetime(["2026-07-01T00:00Z", "2026-07-01T02:00Z"])
    fig = history_chart(pd.DataFrame({"hour_utc": hours, "health_total": [80., 90.]}), "A")
    assert len(fig.data[0].x) == 3
    assert pd.isna(fig.data[0].y[1])


def test_single_health_reading_has_no_empty_slider_or_millisecond_axis():
    hour = pd.Timestamp("2026-07-01T00:00:00Z")
    fig = history_chart(pd.DataFrame({"hour_utc": [hour], "health_total": [80.]}), "A")
    assert not fig.layout.xaxis.rangeslider.visible
    assert not fig.layout.xaxis.rangeselector.visible
    assert fig.layout.xaxis.dtick == 3_600_000
    assert fig.layout.xaxis.range == (hour - pd.Timedelta(hours=1), hour + pd.Timedelta(hours=1))
    assert len(fig.data[0].x) == 1  # padding the axis must not invent observations


def test_timeline_reserves_room_for_axis_title_and_uses_matching_background():
    bundle, event = _bundle()
    fig = history_chart(station_history(bundle, "A", event.end_hour), "A")
    assert fig.layout.margin.b >= 100
    assert fig.layout.xaxis.automargin
    assert fig.layout.xaxis.title.standoff >= 12
    assert fig.layout.xaxis.rangeslider.bgcolor == "#eef2f6"
