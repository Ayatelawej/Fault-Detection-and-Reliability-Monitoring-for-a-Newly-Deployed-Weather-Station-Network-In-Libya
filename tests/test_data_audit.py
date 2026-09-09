from __future__ import annotations

import pandas as pd
import pytest

from src.config.paths import (
    CANONICAL_COLUMN_ORDER,
    DATA_AUDIT_SUMMARY_PATH,
    EXPECTED_FROZEN_N_COLS,
    EXPECTED_FROZEN_N_ROWS,
    EXPECTED_STATION_COUNT,
    FIGURES_DIR,
    HOURLY_ROW_STATES_PATH,
    MERGED_DATASET_PATH,
    MISSINGNESS_HEATMAP_PATH,
    STATION_COVERAGE_FIGURE_PATH,
    STATION_REGISTRY_PATH,
)
from src.features.row_state import (
    ROW_STATE_BEFORE_FIRST,
    ROW_STATE_COMPLETE,
    ROW_STATE_INVALID,
    ROW_STATE_NEVER_ACTIVE,
    ROW_STATE_PARTIAL,
    ROW_STATE_TERMINAL_PADDED,
    ROW_STATE_TRUE_OUTAGE,
    ROW_STATE_WARMUP,
    classify_row_states,
)

MISSINGNESS_BY_VARIABLE_PATH = DATA_AUDIT_SUMMARY_PATH.parent / "missingness_by_variable.csv"

EXPECTED_ROW_STATES = {
    ROW_STATE_COMPLETE,
    ROW_STATE_PARTIAL,
    ROW_STATE_TRUE_OUTAGE,
    ROW_STATE_TERMINAL_PADDED,
    ROW_STATE_WARMUP,
    ROW_STATE_BEFORE_FIRST,
    ROW_STATE_NEVER_ACTIVE,
    ROW_STATE_INVALID,
}

EXPECTED_STATUS_CLASSES = {
    "outage_dominated",
    "weak",
    "usable",
    "reliable_but_terminal_gap",
    "reliable_reference_candidate",
}


@pytest.fixture(scope="module")
def merged_df() -> pd.DataFrame:
    return pd.read_csv(MERGED_DATASET_PATH, low_memory=False)


@pytest.fixture(scope="module")
def registry_df() -> pd.DataFrame:
    return pd.read_csv(STATION_REGISTRY_PATH)


@pytest.fixture(scope="module")
def classified_df(
    merged_df: pd.DataFrame,
    registry_df: pd.DataFrame,
) -> pd.DataFrame:
    return classify_row_states(merged_df, registry_df)


def test_row_state_module_imports() -> None:
    from src.features.row_state import WARMUP_DAYS, classify_row_states

    assert callable(classify_row_states)
    assert WARMUP_DAYS == 7


def test_frozen_shape_and_schema(merged_df: pd.DataFrame) -> None:
    assert merged_df.shape == (
        EXPECTED_FROZEN_N_ROWS,
        EXPECTED_FROZEN_N_COLS,
    )
    assert list(merged_df.columns) == CANONICAL_COLUMN_ORDER


def test_station_hour_key_is_unique(merged_df: pd.DataFrame) -> None:
    assert not merged_df.duplicated(
        subset=["station_id", "hour_utc"],
    ).any()
    assert merged_df["station_id"].nunique() == EXPECTED_STATION_COUNT


def test_hour_utc_is_parseable_and_utc(merged_df: pd.DataFrame) -> None:
    parsed = pd.to_datetime(
        merged_df["hour_utc"],
        utc=True,
        errors="raise",
    )
    assert parsed.notna().all()
    assert merged_df["hour_utc"].str.endswith("+00:00").all()


def test_utc_columns_are_consistent(merged_df: pd.DataFrame) -> None:
    assert (merged_df["hour_utc"] == merged_df["timestamp_utc_dt"]).all()

    hour_utc_without_offset = (
        pd.to_datetime(merged_df["hour_utc"], utc=True)
        .dt.tz_convert("UTC")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    assert (hour_utc_without_offset == merged_df["timestamp_utc"]).all()


def test_data_present_is_binary(merged_df: pd.DataFrame) -> None:
    assert set(merged_df["data_present"].dropna().unique()) <= {0, 1}


def test_timestamp_local_exists_for_live_rows(
    merged_df: pd.DataFrame,
) -> None:
    live_rows = merged_df["data_present"] == 1
    assert merged_df.loc[live_rows, "timestamp_local"].notna().all()


def test_row_state_classification_on_full_dataset(
    classified_df: pd.DataFrame,
) -> None:
    assert len(classified_df) == EXPECTED_FROZEN_N_ROWS
    assert classified_df["row_state"].notna().all()
    assert set(classified_df["row_state"].unique()) <= EXPECTED_ROW_STATES

    present_mask = pd.to_numeric(
        classified_df["data_present"],
        errors="coerce",
    ).eq(1)
    row_state_counts = classified_df["row_state"].value_counts()

    present_row_states = [
        ROW_STATE_COMPLETE,
        ROW_STATE_PARTIAL,
        ROW_STATE_WARMUP,
    ]
    absent_row_states = [
        ROW_STATE_TRUE_OUTAGE,
        ROW_STATE_TERMINAL_PADDED,
        ROW_STATE_BEFORE_FIRST,
        ROW_STATE_NEVER_ACTIVE,
        ROW_STATE_INVALID,
    ]

    present_state_count = int(row_state_counts.reindex(
        present_row_states,
        fill_value=0,
    ).sum())
    absent_state_count = int(row_state_counts.reindex(
        absent_row_states,
        fill_value=0,
    ).sum())

    assert int(present_mask.sum()) == present_state_count
    assert int((~present_mask).sum()) == absent_state_count


def test_audit_outputs_exist() -> None:
    expected_paths = [
        HOURLY_ROW_STATES_PATH,
        DATA_AUDIT_SUMMARY_PATH,
        MISSINGNESS_BY_VARIABLE_PATH,
        STATION_COVERAGE_FIGURE_PATH,
        MISSINGNESS_HEATMAP_PATH,
    ]
    for path in expected_paths:
        assert path.exists(), f"Missing audit output: {path}"

    audit_summary = pd.read_csv(DATA_AUDIT_SUMMARY_PATH)
    assert len(audit_summary) == EXPECTED_STATION_COUNT
    assert "status_class" in audit_summary.columns

    hourly_row_states = pd.read_parquet(HOURLY_ROW_STATES_PATH)
    assert len(hourly_row_states) == EXPECTED_FROZEN_N_ROWS


def test_status_class_consistency(registry_df: pd.DataFrame) -> None:
    assert registry_df["status_class"].notna().all()
    assert set(registry_df["status_class"].unique()) <= EXPECTED_STATUS_CLASSES
