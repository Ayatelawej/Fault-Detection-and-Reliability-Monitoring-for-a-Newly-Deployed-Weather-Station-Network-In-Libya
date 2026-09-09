from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest

from src.availability.build_availability_events import (
    AVAILABILITY_CLASS_EXCLUDED,
    AVAILABILITY_CLASS_FULL_OUTAGE,
    AVAILABILITY_CLASS_ONLINE,
    AVAILABILITY_CLASS_PARTIAL_OUTAGE,
    SENSOR_GROUP_CHANNELS,
    build_hourly_availability_classification,
    build_partial_outage_events,
    build_sensor_group_availability,
    sensor_group_channels,
)
from src.availability.build_network_outage_windows import (
    NETWORK_OUTAGE_MIN_STATIONS,
)
from src.availability.health_score import (
    HEALTH_COMPONENT_COLUMNS,
    _causal_outage_base_score_cap,
    build_hard_zero_health_baseline,
    build_station_health_scores,
    build_health_version_comparison,
    is_hard_zero_health_baseline,
    outage_duration_multiplier,
    validate_delete_future_health_scores,
)
from src.availability.health_forecast import (
    FittedHealthForecastModel,
    HEALTH_FORECAST_ALPHA_GRID,
    HEALTH_FORECAST_CORE_FEATURES,
    HEALTH_FORECAST_HORIZONS,
    HEALTH_FORECAST_LONG_FEATURES,
    HEALTH_FORECAST_LONG_HORIZONS,
    _band_labels,
    _binary_classification_metrics,
    _delta_predictions_from_levels,
    _health_from_residual,
    _feature_columns_for_set,
    _multiclass_metrics,
    _select_residual_alpha,
    _trajectory_labels,
    _validate_transmitting_population_counts,
    build_health_forecast_dataset,
    build_health_forecast_features,
    health_forecast_inference_frame,
    health_forecast_horizon_frame,
    roll_forward_health_no_new_incident,
    validate_delete_future_health_forecast_features,
)
from src.availability.operational_scorecard import (
    SCORECARD_HORIZONS,
    build_operational_scorecard,
    validate_delete_future_operational_scorecard,
)
from src.config.paths import (
    AVAILABILITY_CLASSIFICATION_PATH,
    AVAILABILITY_EVENTS_PATH,
    AVAILABILITY_REPORT_PATH,
    HOURLY_ROW_STATES_PATH,
    MEASUREMENT_COLUMNS,
    NETWORK_OUTAGE_WINDOWS_PATH,
    PARTIAL_OUTAGE_EVENTS_PATH,
    STATION_RELIABILITY_SUMMARY_PATH,
    STRUCTURAL_AVAILABILITY_GAPS_PATH,
)
from src.dashboard.replay import (
    ReplayBundle,
    build_replay_snapshot,
    replay_hours,
    segment_predicted_fault_events,
    station_history,
)
from src.features.row_state import (
    ROW_STATE_COMPLETE,
    ROW_STATE_PARTIAL,
    ROW_STATE_TERMINAL_PADDED,
    ROW_STATE_TRUE_OUTAGE,
    ROW_STATE_WARMUP,
)

EXPECTED_AVAILABILITY_EVENT_COUNT = 2_398
EXPECTED_NETWORK_OUTAGE_WINDOW_COUNT = 47
EXPECTED_STRUCTURAL_GAP_COUNT = 4
EXPECTED_STRUCTURAL_GAP_HOURS = 1_454
EXPECTED_OUTAGE_CLASSES = {
    "local",
    "network_midnight",
    "network_other",
    "unknown",
}
EXPECTED_FINAL_OUTAGE_CLASSES = {
    "local",
    "network_midnight",
    "network_other",
}
EXPECTED_NETWORK_OUTAGE_CLASSES = {
    "network_midnight",
    "network_other",
}
EXPECTED_OUTAGE_CLASS_COUNTS = {
    "local": 1_968,
    "network_midnight": 267,
    "network_other": 163,
}


@pytest.fixture(scope="module")
def availability_events_df() -> pd.DataFrame:
    return pd.read_parquet(AVAILABILITY_EVENTS_PATH)


@pytest.fixture(scope="module")
def hourly_row_states_df() -> pd.DataFrame:
    return pd.read_parquet(HOURLY_ROW_STATES_PATH)


@pytest.fixture(scope="module")
def network_outage_windows_df() -> pd.DataFrame:
    return pd.read_csv(NETWORK_OUTAGE_WINDOWS_PATH)


@pytest.fixture(scope="module")
def availability_classification_df() -> pd.DataFrame:
    return pd.read_parquet(AVAILABILITY_CLASSIFICATION_PATH)


@pytest.fixture(scope="module")
def partial_outage_events_df() -> pd.DataFrame:
    return pd.read_parquet(PARTIAL_OUTAGE_EVENTS_PATH)


@pytest.fixture(scope="module")
def structural_gaps_df() -> pd.DataFrame:
    return pd.read_csv(STRUCTURAL_AVAILABILITY_GAPS_PATH)


@pytest.fixture(scope="module")
def station_reliability_summary_df() -> pd.DataFrame:
    return pd.read_csv(STATION_RELIABILITY_SUMMARY_PATH)


def _synthetic_hourly_rows(hours: list[pd.Timestamp]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "station_id": "SYNTHETIC",
            "hour_utc": pd.to_datetime(hours, utc=True),
            "data_present": 1,
            "row_state": ROW_STATE_COMPLETE,
        }
    )
    for column in MEASUREMENT_COLUMNS:
        frame[column] = 1.0
    return frame


def _synthetic_health_inputs(
    periods: int = 220,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hours = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    variation = pd.Series(range(periods), dtype=float).map(
        lambda value: value % 17 / 100.0
    )
    observations = pd.DataFrame(
        {
            "station_id": "SYNTHETIC",
            "hour_utc": hours,
            "data_present": 1,
            "n_raw_records": 1,
        }
    )
    for index, column in enumerate(MEASUREMENT_COLUMNS):
        if "pressure" in column:
            base = 1010.0
        elif "humidity" in column:
            base = 50.0
        elif column == "winddir_avg_deg":
            base = 180.0
        elif "wind" in column:
            base = 10.0
        elif "solar" in column:
            base = 120.0
        elif "uv" in column:
            base = 2.0
        elif "precip" in column:
            base = 0.1
        elif "dewpoint" in column:
            base = 10.0
        else:
            base = 20.0
        observations[column] = base + variation + index / 10_000.0
    reference = pd.DataFrame(
        {
            "station_id": "SYNTHETIC",
            "hour_utc": hours,
            "pressure_msl": 1010.0 + variation,
            "temperature_2m": 20.0 + variation,
            "dew_point_2m": 10.0 + variation,
            "wind_speed_10m": (10.0 + variation) / 3.6,
            "shortwave_radiation": 120.0 + variation,
        }
    )
    return observations, reference


def _synthetic_operational_scorecard_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    observations, reference = _synthetic_health_inputs(periods=240)
    base = build_station_health_scores(observations, reference)
    station_frames = []
    for station_id in ("IDERNA7", "IALWAH18", "FULL"):
        station = base.copy(deep=True)
        station["station_id"] = station_id
        station_frames.append(station)
    scores = pd.concat(station_frames, ignore_index=True)
    reference_hour = scores["hour_utc"].max()
    scores["causal_fault_evidence"] = False
    online = scores["station_id"].eq("IDERNA7")
    partial = scores["station_id"].eq("IALWAH18")
    full = scores["station_id"].eq("FULL")
    online_fault_hours = pd.date_range(
        reference_hour - pd.Timedelta(hours=1), reference_hour, freq="h"
    )
    scores.loc[
        online & scores["hour_utc"].isin(online_fault_hours),
        "causal_fault_evidence",
    ] = True
    partial_hours = pd.date_range(
        reference_hour - pd.Timedelta(hours=2), reference_hour, freq="h"
    )
    scores.loc[partial & scores["hour_utc"].isin(partial_hours), "availability_class"] = (
        AVAILABILITY_CLASS_PARTIAL_OUTAGE
    )
    scores.loc[partial & scores["hour_utc"].isin(partial_hours), "is_transmitting"] = True
    scores.loc[partial & scores["hour_utc"].isin(partial_hours), "absent_sensor_groups"] = (
        "light_uv|rain_gauge"
    )
    scores.loc[partial & scores["hour_utc"].isin(partial_hours), "partial_outage_run_hours"] = [
        1.0,
        2.0,
        3.0,
    ]
    full_hours = pd.date_range(
        reference_hour - pd.Timedelta(hours=4), reference_hour, freq="h"
    )
    scores.loc[full & scores["hour_utc"].isin(full_hours), "availability_class"] = (
        AVAILABILITY_CLASS_FULL_OUTAGE
    )
    scores.loc[full & scores["hour_utc"].isin(full_hours), "is_transmitting"] = False
    scores.loc[full & scores["hour_utc"].isin(full_hours), "full_outage_run_hours"] = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]
    targets = {"IDERNA7": (85.0, "Healthy"), "IALWAH18": (55.0, "Degraded"), "FULL": (20.0, "Critical")}
    weights = {
        "weighted_health_availability": 30.0,
        "weighted_health_sensor_completeness": 20.0,
        "weighted_health_fault_burden": 25.0,
        "weighted_health_reference_consistency": 15.0,
        "weighted_health_stability": 10.0,
    }
    for station_id, (total, band) in targets.items():
        current = scores["station_id"].eq(station_id) & scores["hour_utc"].eq(reference_hour)
        scores.loc[current, "health_total"] = total
        scores.loc[current, "health_band"] = band
        for column, weight in weights.items():
            scores.loc[current, column] = total * weight / 100.0
    registry = pd.DataFrame(
        {
            "station_id": ["IDERNA7", "IALWAH18", "FULL"],
            "station_name": ["Derna", "Awjila", "Full Station"],
            "city": ["Derna", "Awjila", "Test"],
            "country": ["Libya", "Libya", "Libya"],
            "latitude": [32.7, 29.1, 30.0],
            "longitude": [22.6, 21.2, 20.0],
            "elevation": [3.0, 16.0, 10.0],
        }
    )
    return scores, registry


def _persistence_forecast_models() -> dict[int, FittedHealthForecastModel]:
    return {
        horizon: FittedHealthForecastModel(
            family="deterministic",
            feature_columns=(),
            estimator=None,
            final_policy="persistence",
            horizon_h=horizon,
            regime="transmitting_origin",
        )
        for horizon in SCORECARD_HORIZONS
    }


def _event_start_in_window_mask(
    events: pd.DataFrame,
    windows: pd.DataFrame,
    outage_classes: set[str] | None = None,
) -> pd.Series:
    if outage_classes is not None:
        windows = windows.loc[windows["outage_class"].isin(outage_classes)]

    event_starts = pd.to_datetime(
        events["start_utc"],
        utc=True,
        errors="coerce",
    )
    backfill_starts = pd.to_datetime(
        windows["backfill_start_utc"],
        utc=True,
        errors="coerce",
    )
    backfill_ends = pd.to_datetime(
        windows["backfill_end_utc"],
        utc=True,
        errors="coerce",
    )

    in_window = pd.Series(False, index=events.index)
    for backfill_start, backfill_end in zip(backfill_starts, backfill_ends):
        if pd.isna(backfill_start) or pd.isna(backfill_end):
            continue
        in_window = in_window | (
            event_starts.ge(backfill_start)
            & event_starts.le(backfill_end)
        )
    return in_window.fillna(False)


def test_availability_events_output_exists() -> None:
    assert AVAILABILITY_EVENTS_PATH.exists(), (
        f"Missing availability output: {AVAILABILITY_EVENTS_PATH}"
    )


def test_availability_event_count_expected(
    availability_events_df: pd.DataFrame,
) -> None:
    assert len(availability_events_df) == EXPECTED_AVAILABILITY_EVENT_COUNT


def test_availability_event_durations_are_positive(
    availability_events_df: pd.DataFrame,
) -> None:
    duration_hours = pd.to_numeric(
        availability_events_df["duration_hours"],
        errors="coerce",
    )
    assert duration_hours.gt(0).all()


def test_availability_event_start_before_end(
    availability_events_df: pd.DataFrame,
) -> None:
    start_utc = pd.to_datetime(
        availability_events_df["start_utc"],
        utc=True,
        errors="coerce",
    )
    end_utc = pd.to_datetime(
        availability_events_df["end_utc"],
        utc=True,
        errors="coerce",
    )
    assert start_utc.notna().all()
    assert end_utc.notna().all()
    assert start_utc.le(end_utc).all()


def test_availability_event_duration_matches_bounds(
    availability_events_df: pd.DataFrame,
) -> None:
    start_utc = pd.to_datetime(
        availability_events_df["start_utc"],
        utc=True,
        errors="coerce",
    )
    end_utc = pd.to_datetime(
        availability_events_df["end_utc"],
        utc=True,
        errors="coerce",
    )
    duration_hours = pd.to_numeric(
        availability_events_df["duration_hours"],
        errors="coerce",
    )
    computed_duration = ((end_utc - start_utc) / pd.Timedelta(hours=1)) + 1
    assert computed_duration.eq(duration_hours).all()


def test_availability_event_outage_classes_are_final(
    availability_events_df: pd.DataFrame,
) -> None:
    assert set(availability_events_df["outage_class"].unique()) <= (
        EXPECTED_FINAL_OUTAGE_CLASSES
    )
    assert not availability_events_df["outage_class"].eq("unknown").any()


def test_availability_event_hours_match_true_outage_rows(
    availability_events_df: pd.DataFrame,
    hourly_row_states_df: pd.DataFrame,
) -> None:
    duration_hours = pd.to_numeric(
        availability_events_df["duration_hours"],
        errors="coerce",
    )
    true_outage_rows = (
        hourly_row_states_df["row_state"]
        .astype("string")
        .eq(ROW_STATE_TRUE_OUTAGE)
        .fillna(False)
        .sum()
    )
    assert int(duration_hours.sum()) == int(true_outage_rows)


def test_network_outage_windows_output_exists() -> None:
    assert NETWORK_OUTAGE_WINDOWS_PATH.exists(), (
        f"Missing network outage output: {NETWORK_OUTAGE_WINDOWS_PATH}"
    )


def test_network_outage_windows_have_expected_classes(
    network_outage_windows_df: pd.DataFrame,
) -> None:
    assert "outage_class" in network_outage_windows_df.columns
    assert set(network_outage_windows_df["outage_class"].unique()) <= (
        EXPECTED_NETWORK_OUTAGE_CLASSES
    )


def test_network_outage_window_class_matches_start_hour(
    network_outage_windows_df: pd.DataFrame,
) -> None:
    window_start_utc = pd.to_datetime(
        network_outage_windows_df["window_start_utc"],
        utc=True,
        errors="coerce",
    )
    expected_classes = window_start_utc.dt.hour.apply(
        lambda hour: "network_midnight"
        if hour in {22, 23}
        else "network_other"
    )
    assert network_outage_windows_df["outage_class"].eq(expected_classes).all()


def test_network_outage_windows_station_count_threshold(
    network_outage_windows_df: pd.DataFrame,
) -> None:
    station_count = pd.to_numeric(
        network_outage_windows_df["station_count"],
        errors="coerce",
    )
    assert station_count.ge(NETWORK_OUTAGE_MIN_STATIONS).all()


def test_network_outage_window_bounds_are_ordered(
    network_outage_windows_df: pd.DataFrame,
) -> None:
    window_start_utc = pd.to_datetime(
        network_outage_windows_df["window_start_utc"],
        utc=True,
        errors="coerce",
    )
    window_end_utc = pd.to_datetime(
        network_outage_windows_df["window_end_utc"],
        utc=True,
        errors="coerce",
    )
    backfill_start_utc = pd.to_datetime(
        network_outage_windows_df["backfill_start_utc"],
        utc=True,
        errors="coerce",
    )
    backfill_end_utc = pd.to_datetime(
        network_outage_windows_df["backfill_end_utc"],
        utc=True,
        errors="coerce",
    )

    assert window_start_utc.notna().all()
    assert window_end_utc.notna().all()
    assert backfill_start_utc.notna().all()
    assert backfill_end_utc.notna().all()
    assert window_start_utc.le(window_end_utc).all()
    assert backfill_start_utc.le(backfill_end_utc).all()


def test_availability_event_class_count_still_matches_expected(
    availability_events_df: pd.DataFrame,
) -> None:
    class_counts = availability_events_df["outage_class"].value_counts()
    expected_counts = pd.Series(EXPECTED_OUTAGE_CLASS_COUNTS)
    actual_counts = class_counts.reindex(expected_counts.index, fill_value=0)
    assert actual_counts.eq(expected_counts).all()
    assert int(actual_counts.sum()) == EXPECTED_AVAILABILITY_EVENT_COUNT


def test_network_events_fall_in_window_ranges(
    availability_events_df: pd.DataFrame,
    network_outage_windows_df: pd.DataFrame,
) -> None:
    in_window = _event_start_in_window_mask(
        availability_events_df,
        network_outage_windows_df,
    )
    network_event = availability_events_df["outage_class"].isin(
        EXPECTED_NETWORK_OUTAGE_CLASSES,
    )
    assert in_window.loc[network_event].all()


@pytest.mark.parametrize(
    "outage_class",
    ["network_midnight", "network_other"],
)
def test_network_events_fall_in_matching_window_ranges(
    availability_events_df: pd.DataFrame,
    network_outage_windows_df: pd.DataFrame,
    outage_class: str,
) -> None:
    in_matching_window = _event_start_in_window_mask(
        availability_events_df,
        network_outage_windows_df,
        {outage_class},
    )
    matching_events = availability_events_df["outage_class"].eq(outage_class)
    assert in_matching_window.loc[matching_events].all()


def test_local_events_do_not_fall_in_window_ranges(
    availability_events_df: pd.DataFrame,
    network_outage_windows_df: pd.DataFrame,
) -> None:
    in_window = _event_start_in_window_mask(
        availability_events_df,
        network_outage_windows_df,
    )
    local = availability_events_df["outage_class"].eq("local")
    assert (~in_window.loc[local]).all()


def test_network_outage_window_count_is_frozen(
    network_outage_windows_df: pd.DataFrame,
) -> None:
    assert len(network_outage_windows_df) == EXPECTED_NETWORK_OUTAGE_WINDOW_COUNT


def test_sensor_group_mapping_matches_taxonomy() -> None:
    assert sensor_group_channels() == {
        "anemometer": (
            "windspeed_avg_kmh",
            "windspeed_high_kmh",
            "windspeed_low_kmh",
            "windgust_avg_kmh",
            "windgust_high_kmh",
            "windgust_low_kmh",
        ),
        "barometer": (
            "pressure_max_hpa",
            "pressure_min_hpa",
            "pressure_trend_hpa",
        ),
        "light_uv": ("solar_radiation_high_wm2", "uv_high"),
        "rain_gauge": ("precip_rate_mmh", "precip_total_mm"),
        "thermo_hygrometer": (
            "humidity_avg_pct",
            "humidity_high_pct",
            "humidity_low_pct",
            "temp_avg_c",
            "temp_high_c",
            "temp_low_c",
        ),
        "wind_vane": ("winddir_avg_deg",),
    }


def test_group_level_partial_outage_requires_an_entire_group_to_be_absent() -> None:
    start = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    frame = _synthetic_hourly_rows([start + pd.Timedelta(hours=index) for index in range(4)])
    frame.loc[0, list(SENSOR_GROUP_CHANNELS["anemometer"])] = pd.NA
    frame.loc[1, "windspeed_avg_kmh"] = pd.NA
    frame.loc[1, "row_state"] = ROW_STATE_PARTIAL
    frame.loc[2, MEASUREMENT_COLUMNS] = pd.NA
    frame.loc[2, "data_present"] = 0
    frame.loc[2, "row_state"] = ROW_STATE_TRUE_OUTAGE
    frame.loc[3, list(SENSOR_GROUP_CHANNELS["light_uv"])] = pd.NA
    frame.loc[3, "row_state"] = ROW_STATE_WARMUP

    classification, structural_gaps = build_hourly_availability_classification(
        frame,
        include_structural_gaps=False,
    )

    assert structural_gaps.empty
    assert classification["availability_class"].tolist() == [
        AVAILABILITY_CLASS_PARTIAL_OUTAGE,
        AVAILABILITY_CLASS_ONLINE,
        AVAILABILITY_CLASS_FULL_OUTAGE,
        AVAILABILITY_CLASS_PARTIAL_OUTAGE,
    ]
    assert classification["absent_sensor_groups"].tolist() == [
        "anemometer",
        "",
        "",
        "light_uv",
    ]
    assert frame.loc[1, "row_state"] == ROW_STATE_PARTIAL


def test_partial_outage_events_merge_contiguous_hours_and_union_groups() -> None:
    start = pd.Timestamp("2026-01-02 00:00:00", tz="UTC")
    classification = pd.DataFrame(
        {
            "station_id": ["SYNTHETIC"] * 4,
            "hour_utc": [start + pd.Timedelta(hours=index) for index in range(4)],
            "availability_class": [
                AVAILABILITY_CLASS_PARTIAL_OUTAGE,
                AVAILABILITY_CLASS_PARTIAL_OUTAGE,
                AVAILABILITY_CLASS_FULL_OUTAGE,
                AVAILABILITY_CLASS_PARTIAL_OUTAGE,
            ],
            "absent_sensor_groups": [
                "anemometer",
                "barometer",
                "",
                "light_uv",
            ],
        }
    )

    events = build_partial_outage_events(classification)

    assert len(events) == 2
    assert events["duration_hours"].tolist() == [2, 1]
    assert events["absent_sensor_groups"].tolist() == [
        "anemometer|barometer",
        "light_uv",
    ]


def test_structural_gap_hours_are_materialized_as_full_outages() -> None:
    start = pd.Timestamp("2026-01-03 00:00:00", tz="UTC")
    frame = _synthetic_hourly_rows([start, start + pd.Timedelta(hours=481)])
    frame.loc[0, MEASUREMENT_COLUMNS] = pd.NA
    frame.loc[0, "data_present"] = 0
    frame.loc[0, "row_state"] = ROW_STATE_TRUE_OUTAGE

    classification, structural_gaps = build_hourly_availability_classification(frame)
    materialized = classification.loc[
        classification["source_kind"].eq("materialized_structural_gap")
    ]

    assert len(structural_gaps) == 1
    assert int(structural_gaps.iloc[0]["gap_duration_hours"]) == 481
    assert int(structural_gaps.iloc[0]["omitted_hour_count"]) == 480
    assert len(materialized) == 480
    assert materialized["availability_class"].eq(
        AVAILABILITY_CLASS_FULL_OUTAGE
    ).all()
    assert materialized["hour_utc"].min() == start + pd.Timedelta(hours=1)
    assert materialized["hour_utc"].max() == start + pd.Timedelta(hours=480)


def test_sensor_group_availability_uses_only_transmitting_hours() -> None:
    start = pd.Timestamp("2026-01-04 00:00:00", tz="UTC")
    frame = _synthetic_hourly_rows([start, start + pd.Timedelta(hours=1)])
    frame.loc[1, list(SENSOR_GROUP_CHANNELS["rain_gauge"])] = pd.NA
    classification, _ = build_hourly_availability_classification(
        frame,
        include_structural_gaps=False,
    )

    availability = build_sensor_group_availability(classification).set_index(
        "sensor_group"
    )

    assert int(availability.loc["rain_gauge", "transmitting_hours"]) == 2
    assert int(availability.loc["rain_gauge", "absent_hours"]) == 1
    assert float(availability.loc["rain_gauge", "availability_pct"]) == 50.0


def test_operational_availability_outputs_exist() -> None:
    assert AVAILABILITY_CLASSIFICATION_PATH.exists()
    assert PARTIAL_OUTAGE_EVENTS_PATH.exists()
    assert STRUCTURAL_AVAILABILITY_GAPS_PATH.exists()
    assert AVAILABILITY_REPORT_PATH.exists()
    assert STATION_RELIABILITY_SUMMARY_PATH.exists()


def test_operational_availability_classification_accounts_for_all_rows(
    hourly_row_states_df: pd.DataFrame,
    availability_classification_df: pd.DataFrame,
    structural_gaps_df: pd.DataFrame,
) -> None:
    observed = availability_classification_df["source_kind"].eq("observed_row")
    materialized = availability_classification_df["source_kind"].eq(
        "materialized_structural_gap"
    )
    assert int(observed.sum()) == len(hourly_row_states_df)
    assert int(materialized.sum()) == int(
        pd.to_numeric(
            structural_gaps_df["omitted_hour_count"],
            errors="coerce",
        ).sum()
    )
    assert len(availability_classification_df) == int(observed.sum() + materialized.sum())

    in_scope = availability_classification_df["availability_class"].isin(
        {
            AVAILABILITY_CLASS_FULL_OUTAGE,
            AVAILABILITY_CLASS_PARTIAL_OUTAGE,
            AVAILABILITY_CLASS_ONLINE,
        }
    )
    excluded = availability_classification_df["availability_class"].eq(
        AVAILABILITY_CLASS_EXCLUDED
    )
    assert int(in_scope.sum() + excluded.sum()) == len(availability_classification_df)
    assert availability_classification_df.loc[in_scope, "availability_scope"].ne(
        "terminal_padded"
    ).all()


def test_frozen_full_outage_rows_remain_unchanged_in_operational_layer(
    availability_events_df: pd.DataFrame,
    availability_classification_df: pd.DataFrame,
) -> None:
    observed_full = availability_classification_df.loc[
        availability_classification_df["source_kind"].eq("observed_row")
        & availability_classification_df["availability_class"].eq(
            AVAILABILITY_CLASS_FULL_OUTAGE
        )
    ]
    event_hours = pd.to_numeric(
        availability_events_df["duration_hours"],
        errors="coerce",
    ).sum()
    assert len(observed_full) == int(event_hours)


def test_current_structural_gaps_are_explicitly_accounted_for(
    availability_classification_df: pd.DataFrame,
    structural_gaps_df: pd.DataFrame,
) -> None:
    assert len(structural_gaps_df) == EXPECTED_STRUCTURAL_GAP_COUNT
    omitted_hours = pd.to_numeric(
        structural_gaps_df["omitted_hour_count"],
        errors="coerce",
    )
    assert int(omitted_hours.sum()) == EXPECTED_STRUCTURAL_GAP_HOURS
    materialized = availability_classification_df.loc[
        availability_classification_df["source_kind"].eq(
            "materialized_structural_gap"
        )
    ]
    assert len(materialized) == EXPECTED_STRUCTURAL_GAP_HOURS
    assert materialized["availability_class"].eq(
        AVAILABILITY_CLASS_FULL_OUTAGE
    ).all()


def test_partial_events_cover_exactly_the_partial_classification_hours(
    availability_classification_df: pd.DataFrame,
    partial_outage_events_df: pd.DataFrame,
) -> None:
    partial_hours = availability_classification_df["availability_class"].eq(
        AVAILABILITY_CLASS_PARTIAL_OUTAGE
    ).sum()
    event_hours = pd.to_numeric(
        partial_outage_events_df["duration_hours"],
        errors="coerce",
    ).sum()
    assert int(event_hours) == int(partial_hours)
    assert partial_outage_events_df["availability_class"].eq(
        AVAILABILITY_CLASS_PARTIAL_OUTAGE
    ).all()


def test_station_summary_uses_current_data_end_and_group_availability(
    hourly_row_states_df: pd.DataFrame,
    station_reliability_summary_df: pd.DataFrame,
) -> None:
    expected_data_end = pd.to_datetime(
        hourly_row_states_df["hour_utc"],
        utc=True,
        errors="coerce",
    ).max()
    summary_data_end = pd.to_datetime(
        station_reliability_summary_df["dataset_end_utc"],
        utc=True,
        errors="coerce",
    )
    assert summary_data_end.eq(expected_data_end).all()
    for sensor_group in SENSOR_GROUP_CHANNELS:
        availability = pd.to_numeric(
            station_reliability_summary_df[
                f"{sensor_group}_availability_pct"
            ],
            errors="coerce",
        )
        assert availability.between(0, 100).all()


def test_station_health_score_marks_warmup_and_preserves_weighted_sum() -> None:
    observations, reference = _synthetic_health_inputs()
    scores = build_station_health_scores(observations, reference)

    assert scores.loc[:166, "health_status"].eq("insufficient_history").all()
    scored = scores.loc[scores["health_total"].notna()].copy()
    weighted_sum = scored.loc[:, [
        f"weighted_{component}" for component in HEALTH_COMPONENT_COLUMNS
    ]].sum(axis=1)
    assert (scored["health_total"] - weighted_sum).abs().lt(1e-10).all()
    assert scored.loc[:, list(HEALTH_COMPONENT_COLUMNS)].ge(0.0).all().all()
    assert scored.loc[:, list(HEALTH_COMPONENT_COLUMNS)].le(1.0).all().all()
    assert not scores.duplicated(["station_id", "hour_utc"]).any()


def test_station_health_score_handles_partial_and_full_outages() -> None:
    observations, reference = _synthetic_health_inputs(periods=240)
    partial_hour = observations.loc[170, "hour_utc"]
    observations.loc[170:175, ["solar_radiation_high_wm2", "uv_high"]] = float("nan")
    full_hour = observations.loc[180, "hour_utc"]
    observations = observations.drop(index=list(range(180, 187))).reset_index(drop=True)
    scores = build_station_health_scores(observations, reference)

    partial = scores.loc[
        scores["hour_utc"].between(partial_hour, partial_hour + pd.Timedelta(hours=5))
    ].copy()
    full = scores.loc[
        scores["hour_utc"].between(full_hour, full_hour + pd.Timedelta(hours=6))
    ].copy()
    preceding = scores.loc[scores["hour_utc"].eq(full_hour - pd.Timedelta(hours=1))].iloc[0]
    recovered = scores.loc[scores["hour_utc"].eq(full_hour + pd.Timedelta(hours=7))].iloc[0]

    assert partial["availability_class"].eq(AVAILABILITY_CLASS_PARTIAL_OUTAGE).all()
    assert partial["partial_outage_run_hours"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert partial["outage_duration_multiplier"].is_monotonic_decreasing
    assert partial["health_availability"].lt(1.0).all()
    assert partial["health_sensor_completeness"].lt(1.0).all()

    assert full["health_status"].eq("full_outage").all()
    assert full["full_outage_run_hours"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert full["outage_duration_multiplier"].is_monotonic_decreasing
    assert full["health_total"].gt(0.0).all()
    assert full["health_total"].is_monotonic_decreasing
    assert full.iloc[0]["health_total"] < preceding["health_total"]
    assert recovered["health_total"] > full.iloc[-1]["health_total"]


def test_outage_duration_multiplier_is_monotonic_and_less_severe_for_partial() -> None:
    duration = pd.Series([1.0, 6.0, 24.0, 168.0])
    full = outage_duration_multiplier(duration, pd.Series([0.0] * len(duration)))
    partial = outage_duration_multiplier(pd.Series([0.0] * len(duration)), duration)

    assert full.between(0.0, 1.0).all()
    assert partial.between(0.0, 1.0).all()
    assert full.is_monotonic_decreasing
    assert partial.is_monotonic_decreasing
    assert partial.gt(full).all()
    assert full.iloc[-1] < 0.01


def test_outage_score_cap_prevents_missing_telemetry_from_improving_health() -> None:
    base_total = pd.Series([60.0, 74.0, 71.0, 68.0, 72.0, 80.0])
    availability_class = pd.Series(
        [
            AVAILABILITY_CLASS_ONLINE,
            AVAILABILITY_CLASS_FULL_OUTAGE,
            AVAILABILITY_CLASS_FULL_OUTAGE,
            AVAILABILITY_CLASS_FULL_OUTAGE,
            AVAILABILITY_CLASS_PARTIAL_OUTAGE,
            AVAILABILITY_CLASS_PARTIAL_OUTAGE,
        ]
    )
    capped = _causal_outage_base_score_cap(base_total, availability_class)

    assert capped.tolist() == [60.0, 60.0, 60.0, 60.0, 68.0, 68.0]


def test_health_progressive_comparison_reconstructs_the_previous_hard_zero_rule() -> None:
    observations, reference = _synthetic_health_inputs(periods=240)
    observations = observations.drop(index=list(range(180, 187))).reset_index(drop=True)
    progressive = build_station_health_scores(observations, reference)
    hard_zero = build_hard_zero_health_baseline(progressive)
    comparison = build_health_version_comparison(hard_zero, progressive)

    full = hard_zero["availability_class"].eq(AVAILABILITY_CLASS_FULL_OUTAGE)
    assert is_hard_zero_health_baseline(hard_zero)
    assert hard_zero.loc[full, "health_total"].dropna().eq(0.0).all()
    distribution = comparison["metrics"].loc[
        comparison["metrics"]["section"].eq("distribution")
    ].set_index("metric")
    assert distribution.loc["exact_zero_rows", "hard_zero_baseline"] > 0
    assert distribution.loc["exact_zero_rows", "progressive_duration"] == pytest.approx(0.0)
    assert set(comparison["ranking"]["station_id"]) == {"SYNTHETIC"}


def test_station_health_score_isolated_from_future_source_changes() -> None:
    observations, reference = _synthetic_health_inputs(periods=420)
    cutoff = observations.loc[190, "hour_utc"]
    observations.loc[
        observations["hour_utc"].gt(cutoff), "pressure_max_hpa"
    ] = 9_999.0
    reference.loc[reference["hour_utc"].gt(cutoff), "pressure_msl"] = 9_999.0
    full_scores = build_station_health_scores(observations, reference)
    audit = validate_delete_future_health_scores(
        observations,
        reference,
        full_scores=full_scores,
        sample_keys=pd.DataFrame(
            {"station_id": ["SYNTHETIC"], "hour_utc": [cutoff]}
        ),
    )

    assert audit["passed"].all()


def test_health_forecast_uses_five_horizons_and_exact_future_clock_hours() -> None:
    observations, reference = _synthetic_health_inputs(periods=240)
    scores = build_station_health_scores(observations, reference)
    assert HEALTH_FORECAST_HORIZONS == (1, 3, 6, 12, 24)
    bundle = build_health_forecast_dataset(scores)
    origin_hour = pd.Timestamp("2026-01-08 06:00:00", tz="UTC")
    for horizon in HEALTH_FORECAST_HORIZONS:
        frame = health_forecast_horizon_frame(bundle, horizon)
        origin = frame.loc[frame["hour_utc"].eq(origin_hour)]
        assert len(origin) == 1
        row = origin.iloc[0]
        expected_hour = origin_hour + pd.Timedelta(hours=horizon)
        expected = scores.loc[
            scores["hour_utc"].eq(expected_hour),
            "health_total",
        ].iloc[0]
        assert row["label_end_utc"] == expected_hour
        assert row["target_health_total"] == pytest.approx(expected)
        assert row["target_delta_health"] == pytest.approx(
            expected - row["health_total"]
        )


def test_health_forecast_supports_exact_daily_horizons_through_one_week() -> None:
    observations, reference = _synthetic_health_inputs(periods=480)
    scores = build_station_health_scores(observations, reference)
    assert HEALTH_FORECAST_LONG_HORIZONS == (48, 72, 96, 120, 144, 168)
    bundle = build_health_forecast_dataset(scores, horizons=(168,))
    frame = health_forecast_horizon_frame(bundle, 168)
    origin_hour = pd.Timestamp("2026-01-10 00:00:00", tz="UTC")
    row = frame.loc[frame["hour_utc"].eq(origin_hour)].iloc[0]
    expected_hour = origin_hour + pd.Timedelta(hours=168)
    expected = scores.loc[
        scores["hour_utc"].eq(expected_hour), "health_total"
    ].iloc[0]

    assert row["label_end_utc"] == expected_hour
    assert row["target_health_total"] == pytest.approx(expected)
    projected = roll_forward_health_no_new_incident(scores, 168)
    assert projected.loc[scores["health_total"].notna()].between(0.0, 100.0).all()


def test_long_horizon_features_are_causal_and_available() -> None:
    observations, reference = _synthetic_health_inputs(periods=900)
    scores = build_station_health_scores(observations, reference)
    cutoff = pd.Timestamp("2026-01-25 00:00:00", tz="UTC")
    original = build_health_forecast_features(scores)
    assert set(HEALTH_FORECAST_LONG_FEATURES).difference(
        {
            "feature_target_hour_sin",
            "feature_target_hour_cos",
            "feature_target_day_of_year_sin",
            "feature_target_day_of_year_cos",
        }
    ).issubset(original.frame.columns)
    altered = scores.copy(deep=True)
    altered.loc[altered["hour_utc"].gt(cutoff), "health_total"] = 0.0
    rebuilt = build_health_forecast_features(
        altered,
        station_ids=original.station_ids,
    )
    columns = [
        column
        for column in HEALTH_FORECAST_LONG_FEATURES
        if column in original.frame.columns
    ]
    before = original.frame.loc[original.frame["hour_utc"].eq(cutoff), columns]
    after = rebuilt.frame.loc[rebuilt.frame["hour_utc"].eq(cutoff), columns]

    assert np.isclose(
        before.to_numpy(dtype=float),
        after.to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    ).all()


def test_long_horizon_features_do_not_change_existing_feature_sets() -> None:
    observations, reference = _synthetic_health_inputs(periods=900)
    scores = build_station_health_scores(observations, reference)
    bundle = build_health_forecast_features(scores)
    core = _feature_columns_for_set(bundle, "core")
    full = _feature_columns_for_set(bundle, "full_engineered")
    long_horizon = _feature_columns_for_set(bundle, "long_horizon")
    new_features = set(HEALTH_FORECAST_LONG_FEATURES).difference(
        HEALTH_FORECAST_CORE_FEATURES
    )

    assert core == HEALTH_FORECAST_CORE_FEATURES
    assert not set(full).intersection(new_features)
    assert new_features.issubset(long_horizon)


def test_no_new_incident_roll_forward_is_identity_at_zero_and_progressive_in_outage() -> None:
    observations, reference = _synthetic_health_inputs(periods=240)
    observations = observations.drop(index=list(range(180, 188))).reset_index(drop=True)
    scores = build_station_health_scores(observations, reference)
    zero = roll_forward_health_no_new_incident(scores, 0)
    assert np.isclose(
        zero.to_numpy(dtype=float),
        pd.to_numeric(scores["health_total"], errors="coerce").to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    ).all()
    projected = roll_forward_health_no_new_incident(scores, 6)
    outage_hour = pd.Timestamp("2026-01-08 12:00:00", tz="UTC")
    position = scores.index[scores["hour_utc"].eq(outage_hour)][0]
    assert scores.loc[position, "availability_class"] == AVAILABILITY_CLASS_FULL_OUTAGE
    assert projected.iloc[position] < scores.loc[position, "health_total"]


def test_health_forecast_features_ignore_future_values() -> None:
    observations, reference = _synthetic_health_inputs(periods=240)
    scores = build_station_health_scores(observations, reference)
    cutoff = pd.Timestamp("2026-01-08 18:00:00", tz="UTC")
    original = build_health_forecast_features(scores)
    altered = scores.copy(deep=True)
    future = altered["hour_utc"].gt(cutoff)
    altered.loc[future, "health_total"] = 0.0
    altered.loc[future, "pressure_max_hpa"] = 9_999.0
    altered_features = build_health_forecast_features(
        altered,
        station_ids=original.station_ids,
    )
    before = original.frame.loc[
        original.frame["hour_utc"].eq(cutoff), list(original.feature_columns)
    ].iloc[0].to_numpy(dtype=float)
    after = altered_features.frame.loc[
        altered_features.frame["hour_utc"].eq(cutoff), list(original.feature_columns)
    ].iloc[0].to_numpy(dtype=float)
    assert np.isclose(before, after, rtol=1e-10, atol=1e-12, equal_nan=True).all()
    audit = validate_delete_future_health_forecast_features(
        scores,
        full_bundle=original,
        sample_size=1,
    )
    assert audit["passed"].all()
    assert "station_id_categorical" in set(audit["feature"])


def test_health_forecast_network_features_use_same_hour_peers() -> None:
    observations, reference = _synthetic_health_inputs(periods=240)
    base_scores = build_station_health_scores(observations, reference)
    station_scores = []
    for station_id in ("STATION_A", "STATION_B", "STATION_C"):
        station = base_scores.copy(deep=True)
        station["station_id"] = station_id
        station_scores.append(station)
    scores = pd.concat(station_scores, ignore_index=True)
    target_hour = pd.Timestamp("2026-01-09 08:00:00", tz="UTC")
    prior_hour = target_hour - pd.Timedelta(hours=24)
    starting_health = {"STATION_A": 50.0, "STATION_B": 60.0, "STATION_C": 70.0}
    transmitting = {"STATION_A": True, "STATION_B": False, "STATION_C": True}
    trailing_hours = pd.date_range(prior_hour, target_hour, freq="h")
    for station_id, initial_health in starting_health.items():
        for elapsed, hour in enumerate(trailing_hours):
            scores.loc[
                scores["station_id"].eq(station_id) & scores["hour_utc"].eq(hour),
                "health_total",
            ] = initial_health + float(elapsed)
        scores.loc[
            scores["station_id"].eq(station_id) & scores["hour_utc"].eq(target_hour),
            "is_transmitting",
        ] = transmitting[station_id]

    bundle = build_health_forecast_features(scores)
    rows = bundle.frame.loc[bundle.frame["hour_utc"].eq(target_hour)].set_index(
        "station_id"
    )

    assert rows["feature_fraction_other_stations_transmitting"].to_dict() == {
        "STATION_A": pytest.approx(0.5),
        "STATION_B": pytest.approx(1.0),
        "STATION_C": pytest.approx(0.5),
    }
    assert rows["feature_network_median_health_slope_24h"].eq(1.0).all()


def test_residual_alpha_can_fall_back_to_roll_forward_and_bounds_health() -> None:
    validation = pd.DataFrame(
        {
            "baseline_no_new_incident_level": [10.0, 50.0, 90.0],
            "health_total": [80.0, 50.0, 20.0],
            "target_health_total": [10.0, 50.0, 90.0],
        }
    )
    residual_prediction = np.array([80.0, -40.0, 80.0])

    alpha, metrics, trace = _select_residual_alpha(
        validation, residual_prediction
    )

    assert alpha == 0.0
    assert metrics["mae"] == pytest.approx(0.0)
    assert tuple(row["alpha"] for row in trace) == HEALTH_FORECAST_ALPHA_GRID
    bounded = _health_from_residual(
        validation,
        np.array([-1_000.0, 0.0, 1_000.0]),
        alpha=1.0,
    )
    assert bounded.tolist() == [0.0, 50.0, 100.0]
    bounded_delta = _delta_predictions_from_levels(
        validation, {"selected": bounded}
    )["selected"]
    current = validation["health_total"].to_numpy(dtype=float)
    assert (bounded_delta >= -current).all()
    assert (bounded_delta <= 100.0 - current).all()


def test_saved_deterministic_forecast_wrapper_reproduces_policy_and_delta() -> None:
    frame = pd.DataFrame(
        {
            "health_total": [40.0, 70.0],
            "baseline_persistence_level": [40.0, 70.0],
            "baseline_trend_level": [35.0, 75.0],
            "baseline_no_new_incident_level": [30.0, 65.0],
        }
    )
    model = FittedHealthForecastModel(
        family="deterministic",
        feature_columns=(),
        estimator=None,
        final_policy="no_new_incident_roll_forward",
        horizon_h=24,
        regime="full_outage_origin",
    )

    assert model.predict_health(frame).tolist() == [30.0, 65.0]
    assert model.predict_delta(frame).tolist() == [-10.0, -5.0]


def test_direct_classification_helpers_use_required_boundaries_and_metrics() -> None:
    delta = np.array([-10.0, -5.0, -4.999, 0.0, 4.999, 5.0, 10.0])
    assert _trajectory_labels(delta).tolist() == [
        "Deteriorating",
        "Deteriorating",
        "Stable",
        "Stable",
        "Stable",
        "Improving",
        "Improving",
    ]
    assert _band_labels(
        np.array([0.0, 39.999, 40.0, 59.999, 60.0, 79.999, 80.0, 100.0])
    ).tolist() == [
        "Critical",
        "Critical",
        "Degraded",
        "Degraded",
        "Watch",
        "Watch",
        "Healthy",
        "Healthy",
    ]

    binary = _binary_classification_metrics(
        np.array([False, False, True, True]),
        np.array([False, True, True, True]),
        np.array([0.1, 0.7, 0.8, 0.9]),
    )
    assert binary["support_positive"] == 2
    assert binary["accuracy"] == pytest.approx(0.75)
    assert binary["balanced_accuracy"] == pytest.approx(0.75)
    assert binary["precision"] == pytest.approx(2.0 / 3.0)
    assert binary["recall"] == pytest.approx(1.0)
    assert binary["f1"] == pytest.approx(0.8)
    assert binary["pr_auc"] == pytest.approx(1.0)


def test_health_forecast_transmitting_population_contract_matches_all_primary_outputs() -> None:
    horizon = 3
    population = 4
    common = pd.DataFrame(
        {
            "horizon_h": [horizon],
            "scope": ["transmitting_origin"],
            "method": ["selected_residual_forecast"],
            "n": [population],
        }
    )
    run = SimpleNamespace(
        metrics=common.assign(target="level"),
        trajectory_metrics=common.copy(),
        band_metrics=common.copy(),
        deterioration_metrics=common.copy(),
        calibration=pd.DataFrame(
            {
                "horizon_h": [horizon, horizon],
                "scope": ["transmitting_origin", "transmitting_origin"],
                "n": [2, 2],
            }
        ),
    )

    counts = _validate_transmitting_population_counts(run)

    assert counts.to_dict(orient="records") == [
        {
            "horizon_h": horizon,
            "regression_n": population,
            "trajectory_n": population,
            "band_n": population,
            "deterioration_n": population,
            "calibration_n": population,
        }
    ]

    run.band_metrics = common.assign(n=population - 1)
    with pytest.raises(RuntimeError, match="band population"):
        _validate_transmitting_population_counts(run)

    multiclass = _multiclass_metrics(
        np.array(
            ["Deteriorating", "Deteriorating", "Stable", "Stable", "Improving"]
        ),
        np.array(["Deteriorating", "Stable", "Stable", "Stable", "Improving"]),
        ("Deteriorating", "Stable", "Improving"),
    )
    assert multiclass["accuracy"] == pytest.approx(0.8)
    assert multiclass["balanced_accuracy"] == pytest.approx(5.0 / 6.0)
    assert multiclass["recall_deteriorating"] == pytest.approx(0.5)
    assert multiclass["recall_stable"] == pytest.approx(1.0)
    assert multiclass["recall_improving"] == pytest.approx(1.0)


def test_health_forecast_inference_frame_keeps_latest_rows_without_future_targets() -> None:
    observations, reference = _synthetic_health_inputs(periods=220)
    scores = build_station_health_scores(observations, reference)
    bundle = build_health_forecast_features(scores)
    latest = scores["hour_utc"].max()

    inference = health_forecast_inference_frame(bundle, 24)

    row = inference.loc[inference["hour_utc"].eq(latest)]
    assert len(row) == 1
    assert row["baseline_persistence_level"].notna().all()
    assert row["baseline_no_new_incident_level"].notna().all()
    assert "target_health_total" not in row.columns


def test_operational_scorecard_preserves_roster_scope_and_history_contracts() -> None:
    scores, registry = _synthetic_operational_scorecard_inputs()
    reference_hour = scores["hour_utc"].max()

    run = build_operational_scorecard(
        scores,
        registry,
        forecast_models=_persistence_forecast_models(),
        reference_hour=reference_hour,
        expected_station_count=3,
    )
    table = run.table.set_index("station_id")

    assert list(run.table["station_id"]) == ["FULL", "IALWAH18", "IDERNA7"]
    assert table.loc["FULL", "transmission_status"] == "full_outage"
    assert table.loc["FULL", "current_outage_run_hours"] == pytest.approx(5.0)
    assert table.loc["IALWAH18", "transmission_status"] == "partial_outage"
    assert table.loc["IALWAH18", "current_outage_run_hours"] == pytest.approx(3.0)
    assert table.loc["IALWAH18", "absent_sensor_groups"] == "light_uv|rain_gauge"
    assert table.loc["IDERNA7", "transmission_status"] == "transmitting"
    assert table.loc["IDERNA7", "fault_detected"]
    assert table.loc["IDERNA7", "current_fault_run_hours"] == 2
    assert table.loc["IDERNA7", "fault_hours_trailing_24h"] == 2
    assert table.loc["IDERNA7", "reason_code_status"] == "not_available_causal_deployment"
    assert table.loc["IDERNA7", "trained_binary_detector_status"] == (
        "not_available_causal_deployment"
    )
    assert table.loc["FULL", "forecast_scope"] == "not_applicable_active_full_outage"
    assert pd.isna(table.loc["FULL", "forecast_health_1h"])
    assert table.loc["IALWAH18", "forecast_scope"] == (
        "applicable_transmitting_partial_outage"
    )
    for horizon in SCORECARD_HORIZONS:
        assert table.loc["IALWAH18", f"forecast_status_{horizon}h"] == "available"
        assert table.loc["IALWAH18", f"forecast_health_{horizon}h"] == pytest.approx(55.0)
        assert table.loc["IALWAH18", f"forecast_change_{horizon}h"] == pytest.approx(0.0)
    assert table.loc["IALWAH18", "uptime_7d_pct"] == pytest.approx(100.0)
    assert table.loc["FULL", "full_outage_events_trailing_30d"] == 1
    assert table.loc["FULL", "hours_since_last_outage_ended_status"] == (
        "active_full_outage"
    )
    assert table.loc["IDERNA7", "spatial_context_status"] == (
        "no_spatial_neighbour_not_penalised"
    )
    assert table.loc["IALWAH18", "spatial_context_status"] == (
        "no_spatial_neighbour_not_penalised"
    )
    assert int(run.metadata["unexplained_nulls"]) == 0
    assert run.inconsistencies["code"].eq("healthy_band_with_active_fault").any()


def test_operational_scorecard_default_skips_terminal_padding_and_is_delete_future_safe() -> None:
    scores, registry = _synthetic_operational_scorecard_inputs()
    latest = scores["hour_utc"].max()
    padded = scores["hour_utc"].gt(latest - pd.Timedelta(hours=2))
    scores.loc[padded, "is_transmitting"] = False
    scores.loc[padded, "availability_class"] = AVAILABILITY_CLASS_FULL_OUTAGE
    expected = latest - pd.Timedelta(hours=2)

    run = build_operational_scorecard(
        scores,
        registry,
        forecast_models=_persistence_forecast_models(),
        expected_station_count=3,
    )

    assert run.metadata["reference_hour_utc"] == expected.isoformat()
    assert run.metadata["terminal_hours_skipped_by_default"] == 2
    audit = validate_delete_future_operational_scorecard(
        scores,
        registry,
        availability=None,
        forecast_models=_persistence_forecast_models(),
        calibration_corroboration=None,
        reference_hour=expected,
        expected_station_count=3,
    )
    assert not audit.empty
    assert audit["passed"].all()

    with pytest.raises(RuntimeError, match="expected 4"):
        build_operational_scorecard(
            scores,
            registry,
            forecast_models=_persistence_forecast_models(),
            expected_station_count=4,
        )


def _synthetic_replay_bundle() -> ReplayBundle:
    hours = pd.to_datetime(["2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z"])
    health = pd.DataFrame(
        [
            {
                "station_id": station_id,
                "hour_utc": hour,
                "availability_class": availability,
                "absent_sensor_groups": "" if availability == "online" else "anemometer",
                "full_outage_run_hours": 0.0 if availability == "online" else float(index + 1),
                "partial_outage_run_hours": 0.0,
                "is_transmitting": availability != "full_outage",
                "health_total": health_total,
                "health_band": "Healthy" if health_total >= 80 else "Critical",
                "weighted_health_availability": health_total * 0.35,
                "weighted_health_sensor_completeness": health_total * 0.25,
                "weighted_health_fault_burden": health_total * 0.16,
                "weighted_health_reference_consistency": health_total * 0.14,
                "weighted_health_stability": health_total * 0.10,
                **{
                    f"sensor_group_present_{group}": availability != "full_outage"
                    for group in (
                        "anemometer",
                        "barometer",
                        "light_uv",
                        "rain_gauge",
                        "thermo_hygrometer",
                        "wind_vane",
                    )
                },
                **{
                    f"outage_projection_{horizon}h": max(0.0, health_total - horizon)
                    for horizon in (1, 3, 6, 12, 24)
                },
            }
            for index, hour in enumerate(hours)
            for station_id, availability, health_total in [
                ("A", "online", 90.0 - index),
                ("B", "full_outage", 20.0 - index),
            ]
        ]
    )
    forecasts = pd.DataFrame(
        [
            {
                "station_id": "A",
                "hour_utc": hours[1],
                "horizon_h": horizon,
                "predicted_frozen_selected_policy": 88.0 - horizon,
            }
            for horizon in (1, 3, 6, 12, 24)
        ]
        + [
            {
                "station_id": "B",
                "hour_utc": hours[1],
                "horizon_h": horizon,
                "predicted_frozen_selected_policy": 25.0,
            }
            for horizon in (1, 3, 6, 12, 24)
        ]
    )
    detections = pd.DataFrame(
        [
            {
                "station_id": "A",
                "hour_utc": hours[0],
                "random_probability": 0.20,
                "random_prediction": 0,
            },
            {
                "station_id": "A",
                "hour_utc": hours[1],
                "random_probability": 0.81,
                "random_prediction": 1,
            },
            {
                "station_id": "B",
                "hour_utc": hours[0],
                "random_probability": float("nan"),
                "random_prediction": pd.NA,
            },
            {
                "station_id": "B",
                "hour_utc": hours[1],
                "random_probability": float("nan"),
                "random_prediction": pd.NA,
            },
        ]
    )
    statistical_scores = pd.DataFrame(
        [
            {
                "station_id": "A",
                "hour_utc": hours[1],
                "channel": "temp_avg_c",
                "zscore": 5.0,
                "rolling_variance": 1.0,
                "iforest_score": 0.2,
                "flag_zscore": True,
                "flag_stuck": False,
                "flag_iforest": False,
                "flag_physical": False,
                "flag_physical_suspect": False,
            }
        ]
    )
    spatial_neighbors = pd.DataFrame(
        [{"station_id": "A", "neighbor_id": "B", "distance_km": 10.0}]
    )
    registry = pd.DataFrame(
        [
            {"station_id": "A", "station_name": "Alpha", "city": "A", "latitude": 32.0, "longitude": 13.0},
            {"station_id": "B", "station_name": "Beta", "city": "B", "latitude": 31.0, "longitude": 14.0},
        ]
    )
    return ReplayBundle(
        health,
        forecasts,
        detections,
        statistical_scores,
        spatial_neighbors,
        registry,
    )


def test_july_replay_snapshot_preserves_stations_and_uses_selected_detector_output() -> None:
    bundle = _synthetic_replay_bundle()
    snapshot = build_replay_snapshot(bundle, "2026-07-01T01:00:00Z")

    assert snapshot["station_id"].tolist() == ["B", "A"]
    alpha = snapshot.loc[snapshot["station_id"].eq("A")].iloc[0]
    beta = snapshot.loc[snapshot["station_id"].eq("B")].iloc[0]
    assert alpha["status"] == "Fault alert"
    assert alpha["fault_probability"] == pytest.approx(0.81)
    assert alpha["forecast_24h"] == pytest.approx(64.0)
    assert alpha["category"] == "Active faults"
    assert beta["status"] == "Full outage"
    assert beta["full_outage_run_hours"] == pytest.approx(2.0)
    assert beta["forecast_1h"] == pytest.approx(18.0)
    assert beta["forecast_source_1h"] == "Continued-outage projection"
    assert beta["category"] == "In outage"


def test_predicted_fault_events_break_on_negative_hours_and_time_gaps() -> None:
    hours = pd.to_datetime(
        [
            "2026-07-01T00:00:00Z",
            "2026-07-01T01:00:00Z",
            "2026-07-01T02:00:00Z",
            "2026-07-01T04:00:00Z",
            "2026-07-01T05:00:00Z",
        ]
    )
    detections = pd.DataFrame(
        {
            "station_id": ["A"] * 5,
            "hour_utc": hours,
            "random_probability": [0.8, 0.9, 0.2, 0.7, 0.8],
            "random_prediction": [1, 1, 0, 1, 1],
        }
    )

    events = segment_predicted_fault_events(detections)

    assert events["duration_hours"].tolist() == [2, 2]
    assert events["start_hour"].tolist() == [hours[0], hours[3]]
    assert events["status"].tolist() == ["closed", "active"]

    later = segment_predicted_fault_events(detections, "2026-07-01T06:00:00Z")

    assert later["status"].tolist() == ["closed", "closed"]


def test_predicted_event_replay_does_not_reveal_future_duration() -> None:
    detections = pd.DataFrame(
        {
            "station_id": ["A", "A", "A"],
            "hour_utc": pd.to_datetime(
                ["2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z", "2026-07-01T02:00:00Z"]
            ),
            "random_probability": [0.8, 0.9, 0.95],
            "random_prediction": [1, 1, 1],
        }
    )

    visible = segment_predicted_fault_events(detections, "2026-07-01T01:00:00Z")

    assert visible.iloc[0]["duration_hours"] == 2
    assert visible.iloc[0]["end_hour"] == pd.Timestamp("2026-07-01T01:00:00Z")


def test_july_replay_history_is_past_only() -> None:
    bundle = _synthetic_replay_bundle()
    history = station_history(bundle, "A", "2026-07-01T00:00:00Z")

    assert history["hour_utc"].tolist() == [pd.Timestamp("2026-07-01T00:00:00Z")]
    assert replay_hours(bundle) == [
        pd.Timestamp("2026-07-01T00:00:00Z"),
        pd.Timestamp("2026-07-01T01:00:00Z"),
    ]
