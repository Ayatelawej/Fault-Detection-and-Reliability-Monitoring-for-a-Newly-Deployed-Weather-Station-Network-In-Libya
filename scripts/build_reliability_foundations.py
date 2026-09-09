from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.availability.build_availability_events import (
    build_availability_events,
    write_availability_report,
    write_operational_availability_outputs,
)
from src.availability.build_network_outage_windows import (
    NETWORK_OUTAGE_MIN_STATIONS,
    NETWORK_OUTAGE_TIME_WINDOW_HOURS,
    assign_outage_class,
    detect_network_outage_windows,
)
from src.availability.build_station_reliability_summary import (
    write_station_reliability_summary,
)
from src.config.paths import (
    AVAILABILITY_EVENTS_PATH,
    HOURLY_ROW_STATES_PATH,
    MERGED_DATASET_PATH,
    NETWORK_OUTAGE_WINDOWS_PATH,
)
from src.features.run_data_audit import run_data_audit
from src.workflows.prerequisites import require_files


def build_reliability_foundations() -> dict[str, int]:
    require_files(
        "Reliability foundations",
        {"frozen hourly dataset": MERGED_DATASET_PATH},
    )
    run_data_audit()
    hourly_states = pd.read_parquet(HOURLY_ROW_STATES_PATH)
    full_events = build_availability_events(hourly_states)
    network_windows = detect_network_outage_windows(
        full_events,
        min_stations=NETWORK_OUTAGE_MIN_STATIONS,
        time_window_hours=NETWORK_OUTAGE_TIME_WINDOW_HOURS,
    )
    full_events = assign_outage_class(full_events, network_windows)
    full_events.to_parquet(AVAILABILITY_EVENTS_PATH, index=False)
    network_windows.to_csv(NETWORK_OUTAGE_WINDOWS_PATH, index=False)
    classification, partial_events, structural_gaps = (
        write_operational_availability_outputs(hourly_states)
    )
    write_station_reliability_summary(
        availability_classification=classification,
        full_events=full_events,
        partial_events=partial_events,
    )
    write_availability_report(
        full_events,
        classification,
        partial_events,
        structural_gaps,
    )
    return {
        "hourly_rows": int(len(hourly_states)),
        "full_outage_events": int(len(full_events)),
        "network_outage_windows": int(len(network_windows)),
        "partial_outage_events": int(len(partial_events)),
    }


def main() -> None:
    counts = build_reliability_foundations()
    print("RELIABILITY FOUNDATIONS BUILT")
    for name, value in counts.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
