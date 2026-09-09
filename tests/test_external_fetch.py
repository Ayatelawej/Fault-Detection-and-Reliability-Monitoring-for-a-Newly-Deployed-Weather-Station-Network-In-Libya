from __future__ import annotations

import pandas as pd
import pytest

import scripts.fetch_reference_data as reference_fetch
from scripts.fetch_reference_data import (
    ExternalReferenceError,
    active_reference_registry,
    build_request_params,
    expected_reference_index,
    fetch_station,
    parse_response,
    validate_reference_index,
)
from src.rules.config import (
    EXTERNAL_CELL_SELECTION,
    EXTERNAL_END_DATE,
    EXTERNAL_EXPECTED_ROWS,
    EXTERNAL_HOURLY_VARS,
    EXTERNAL_MODELS,
    EXTERNAL_START_DATE,
)


def _station_row() -> pd.Series:
    return pd.Series(
        {
            "station_id": "STA",
            "latitude": 32.1,
            "longitude": 13.2,
            "elevation": 80.0,
        }
    )


def _payload(periods: int = 3) -> dict[str, object]:
    times = pd.date_range("2025-06-15", periods=periods, freq="h").strftime(
        "%Y-%m-%dT%H:%M"
    ).tolist()
    hourly = {"time": times}
    for index, variable in enumerate(EXTERNAL_HOURLY_VARS):
        hourly[variable] = [float(index + offset) for offset in range(periods)]
    return {
        "latitude": 32.0,
        "longitude": 13.0,
        "elevation": 100.0,
        "hourly": hourly,
    }


def test_build_request_params_uses_external_config() -> None:
    result = build_request_params(_station_row())

    assert result["latitude"] == 32.1
    assert result["longitude"] == 13.2
    assert result["elevation"] == 80.0
    assert result["start_date"] == EXTERNAL_START_DATE
    assert result["end_date"] == EXTERNAL_END_DATE
    assert result["hourly"] == EXTERNAL_HOURLY_VARS
    assert result["models"] == EXTERNAL_MODELS
    assert result["cell_selection"] == EXTERNAL_CELL_SELECTION
    assert result["timezone"] == "UTC"
    assert result["wind_speed_unit"] == "ms"


def test_parse_response_returns_tz_aware_utc_index_and_variables() -> None:
    frame, metadata = parse_response(_payload())

    assert frame.index.name == "time_utc"
    assert str(frame.index.tz) == "UTC"
    assert list(frame.columns) == EXTERNAL_HOURLY_VARS
    assert frame.shape == (3, len(EXTERNAL_HOURLY_VARS))
    assert metadata == {
        "response_latitude": 32.0,
        "response_longitude": 13.0,
        "model_elevation": 100.0,
    }


def test_validate_row_count_rejects_truncated_fixture() -> None:
    frame, _ = parse_response(_payload(EXTERNAL_EXPECTED_ROWS - 1))

    with pytest.raises(Exception, match="expected"):
        validate_reference_index(frame)


def test_reference_index_contract_is_dynamic_and_july_inclusive() -> None:
    expected = expected_reference_index()

    assert EXTERNAL_EXPECTED_ROWS == 9_888
    assert len(expected) == EXTERNAL_EXPECTED_ROWS
    assert expected[0] == pd.Timestamp("2025-06-15 00:00:00+00:00")
    assert expected[-1] == pd.Timestamp("2026-07-31 23:00:00+00:00")


def test_validate_reference_index_rejects_a_same_length_shifted_range() -> None:
    frame, _ = parse_response(_payload(EXTERNAL_EXPECTED_ROWS))
    frame.index = frame.index + pd.Timedelta(hours=1)

    with pytest.raises(ExternalReferenceError, match="timestamps"):
        validate_reference_index(frame)


def test_fetch_station_reuses_a_valid_cache_without_a_network_request(tmp_path, monkeypatch) -> None:
    frame, _ = parse_response(_payload(EXTERNAL_EXPECTED_ROWS))
    monkeypatch.setattr(reference_fetch, "REFERENCE_DIR", tmp_path)
    frame.to_parquet(reference_fetch.station_cache_path("STA"))

    def unexpected_fetch(_: dict[str, object]) -> dict[str, object]:
        pytest.fail("a valid cache must not issue a network request")

    monkeypatch.setattr(reference_fetch, "fetch_payload", unexpected_fetch)
    row = fetch_station(_station_row(), force=False)

    assert row["status"] == "skipped"
    assert row["rows"] == EXTERNAL_EXPECTED_ROWS


def test_active_reference_registry_excludes_the_retired_station() -> None:
    registry = pd.DataFrame([_station_row().to_dict(), {**_station_row().to_dict(), "station_id": "IJANZO4"}])

    active = active_reference_registry(registry)

    assert active["station_id"].tolist() == ["STA"]
