from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from src.dashboard.replay import (
    FROZEN_STATISTICS_END,
    FROZEN_STATISTICS_ROWS,
    HEALTH_COMPONENT_COLUMNS,
    SELECTED_DETECTOR_THRESHOLD,
    ReplayBundle,
    build_replay_snapshot,
    event_detector_evidence,
    event_layer_status,
    load_replay_bundle,
    replay_hours,
    segment_predicted_fault_events,
    station_history,
    station_sensor_states,
)


DASHBOARD_SCHEMA_VERSION = "2026-08-13-simple-v1"


@st.cache_data(show_spinner="Loading July replay...")
def _load(schema_version: str) -> ReplayBundle:
    _ = schema_version
    return load_replay_bundle()


def _sidebar(hours: list[pd.Timestamp]) -> tuple[pd.Timestamp, float]:
    if "replay_index" not in st.session_state:
        st.session_state.replay_index = 0
    if "replay_running" not in st.session_state:
        st.session_state.replay_running = False
    selected = st.sidebar.select_slider(
        "Simulated July hour (UTC)",
        options=list(range(len(hours))),
        value=min(st.session_state.replay_index, len(hours) - 1),
        format_func=lambda index: hours[index].strftime("%d Jul %Y, %H:%M"),
    )
    st.session_state.replay_index = int(selected)
    left, right = st.sidebar.columns(2)
    if left.button("Play", width="stretch"):
        st.session_state.replay_running = True
    if right.button("Pause", width="stretch"):
        st.session_state.replay_running = False
    speed = st.sidebar.select_slider("Seconds per hour", [0.25, 0.5, 1.0, 2.0], value=1.0)
    st.sidebar.caption("Historical replay · no live station feed")
    return hours[st.session_state.replay_index], float(speed)


def _network_page(snapshot: pd.DataFrame) -> None:
    counts = snapshot["category"].value_counts()
    columns = st.columns(4)
    for column, category in zip(columns, ["Healthy", "Needs attention", "In outage", "Active faults"], strict=True):
        column.metric(category, int(counts.get(category, 0)))
    st.caption("The four categories are mutually exclusive and total 26 stations.")
    st.dataframe(
        snapshot[["station_id", "city", "health_total", "health_band", "status", "finding", "health_24h"]],
        hide_index=True,
        width="stretch",
        column_config={
            "station_id": "Station",
            "city": "Location",
            "health_total": st.column_config.NumberColumn("Health", format="%.1f"),
            "health_band": "Band",
            "status": "Status",
            "finding": "Current finding",
            "health_24h": "Health +24 h",
        },
    )


def _station_page(bundle: ReplayBundle, snapshot: pd.DataFrame, hour: pd.Timestamp) -> None:
    station_labels = {
        str(row.station_id): f"{row.station_id} · {row.city}"
        for row in bundle.registry.itertuples(index=False)
    }
    station_ids = sorted(station_labels)
    station_id = st.selectbox(
        "Station",
        station_ids,
        format_func=station_labels.get,
        key="selected_station_id",
    )
    row = snapshot.loc[snapshot["station_id"].eq(station_id)].iloc[0]
    metrics = st.columns(4)
    metrics[0].metric("Status", row["status"])
    metrics[1].metric("Health", f"{row['health_total']:.1f}")
    metrics[2].metric("Band", row["health_band"])
    metrics[3].metric("Fault probability", "N/A" if pd.isna(row["fault_probability"]) else f"{row['fault_probability']:.1%}")
    st.markdown("#### Health history")
    history = station_history(bundle, station_id, hour).set_index("hour_utc")
    st.line_chart(history, y="health_total", height=250)
    left, right = st.columns(2)
    left.markdown("#### Health components")
    left.dataframe(
        pd.DataFrame(
            {
                "Component": list(HEALTH_COMPONENT_COLUMNS),
                "Points": [row[column] for column in HEALTH_COMPONENT_COLUMNS.values()],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    right.markdown("#### Sensor groups")
    right.dataframe(station_sensor_states(bundle, station_id, hour), hide_index=True, width="stretch")
    forecast = pd.DataFrame(
        {
            "Horizon": [1, 3, 6, 12, 24],
            "Health": [row[f"forecast_{horizon}h"] for horizon in (1, 3, 6, 12, 24)],
            "Source": [row[f"forecast_source_{horizon}h"] for horizon in (1, 3, 6, 12, 24)],
        }
    )
    forecast["Health"] = forecast["Health"].where(forecast["Health"].abs().ge(0.05), 0.0).round(1)
    st.markdown("#### Health forecast")
    st.line_chart(forecast.set_index("Horizon"), y="Health", height=250)
    st.dataframe(
        forecast,
        hide_index=True,
        width="stretch",
        column_config={
            "Horizon": st.column_config.NumberColumn("Horizon (hours)", format="%d"),
            "Health": st.column_config.NumberColumn("Health", format="%.1f"),
        },
    )
    if row["status"] == "Full outage":
        st.info("This is a continued-outage projection. It assumes the outage continues and does not predict recovery.")


def _event_page(bundle: ReplayBundle, hour: pd.Timestamp) -> None:
    events = segment_predicted_fault_events(bundle.detections, hour).sort_values("start_hour", ascending=False)
    if events.empty:
        st.info("No predicted fault event exists up to this simulated hour.")
        return
    event_id = st.selectbox(
        "Predicted event",
        events["event_id"].tolist(),
        format_func=lambda value: _event_name(events.loc[events["event_id"].eq(value)].iloc[0]),
    )
    event = events.loc[events["event_id"].eq(event_id)].iloc[0]
    columns = st.columns(4)
    columns[0].metric("Station", event["station_id"])
    columns[1].metric("Status", event["status"].title())
    columns[2].metric("Duration", f"{int(event['duration_hours'])} h")
    columns[3].metric("Peak probability", f"{event['peak_probability']:.1%}")
    st.caption(f"Selected EF-HGB threshold: {SELECTED_DETECTOR_THRESHOLD:.0%}")
    evidence = event_detector_evidence(bundle, event)
    if evidence.empty:
        st.warning("No individual saved detector flag fired inside this EF-HGB-positive event.")
    else:
        st.dataframe(
            evidence,
            hide_index=True,
            width="stretch",
            column_config={
                "Score": st.column_config.NumberColumn(format="%.4f"),
                "Threshold": st.column_config.NumberColumn(format="%.4f"),
                "Margin": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption("Components are detector-evidence groups, not model-predicted components.")
    external, spatial = event_layer_status(bundle, event)
    st.info(external)
    st.info(spatial)


def _event_name(event: pd.Series) -> str:
    return f"{event['station_id']} · {event['start_hour']:%d Jul %H:%M} · {int(event['duration_hours'])} h · {event['status']}"


def main() -> None:
    st.set_page_config(page_title="Station Reliability Monitor", page_icon="🌦️", layout="wide")
    st.title("Weather Station Reliability Monitor")
    st.caption("Historical July 2026 operational replay")
    try:
        bundle = _load(DASHBOARD_SCHEMA_VERSION)
    except (FileNotFoundError, KeyError, ValueError) as error:
        st.error(str(error))
        st.stop()
    hours = replay_hours(bundle)
    hour, speed = _sidebar(hours)
    snapshot = build_replay_snapshot(bundle, hour)
    st.header(hour.strftime("%A, %d July 2026 · %H:%M UTC"))
    network, station, evidence = st.tabs(["Network", "Station", "Evidence"])
    with network:
        _network_page(snapshot)
    with station:
        _station_page(bundle, snapshot, hour)
    with evidence:
        _event_page(bundle, hour)
    st.caption(
        f"Selected EF-HGB and health-forecast policies · rule statistics frozen through "
        f"{FROZEN_STATISTICS_END:%d %B %Y} on {FROZEN_STATISTICS_ROWS:,} rows"
    )
    if st.session_state.replay_running:
        time.sleep(speed)
        st.session_state.replay_index = (st.session_state.replay_index + 1) % len(hours)
        st.rerun()


if __name__ == "__main__":
    main()
