from __future__ import annotations

from html import escape
from pathlib import Path
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.presentation import (
    ACCENT, HEALTH_HELP, HORIZONS, event_neighbor_comparison,
    event_weather_comparison, fault_status, forecast_table, health_breakdown,
    history_chart, network_table, score_text,
)
from src.dashboard.replay import (
    SELECTED_DETECTOR_THRESHOLD, ReplayBundle, build_replay_snapshot,
    event_detector_evidence, event_reason_history, load_replay_bundle,
    replay_hours, segment_predicted_fault_events, station_history, station_sensor_states,
)

DASHBOARD_SCHEMA_VERSION = "2026-09-24-validation-selected-v6"
STYLE_PATH = Path(__file__).resolve().parents[1] / "src/dashboard/styles.css"
CHART_CONFIG = {"displaylogo": False, "scrollZoom": False,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"]}


@st.cache_resource(show_spinner="Loading station observations…")
def _load(schema_version: str) -> ReplayBundle:
    # The bundle is read-only. Avoid copying the large saved tables on every tick.
    _ = schema_version
    return load_replay_bundle()


def _select_hour() -> None:
    st.session_state.replay_index = int(st.session_state.timeline_hour)
    st.session_state.replay_running = False


def _step_hour(delta: int, total: int) -> None:
    st.session_state.replay_index = max(0, min(total - 1, st.session_state.replay_index + delta))
    st.session_state.replay_running = False


def _toggle_playback() -> None:
    st.session_state.replay_running = not st.session_state.replay_running


def _sidebar(hours: list[pd.Timestamp]) -> tuple[pd.Timestamp, float]:
    st.session_state.setdefault("replay_index", 0)
    st.session_state.setdefault("replay_running", False)
    st.session_state.replay_index = min(st.session_state.replay_index, len(hours) - 1)
    st.session_state.timeline_hour = st.session_state.replay_index
    st.sidebar.markdown("### Time controls")
    st.sidebar.select_slider(
        "Viewing time (UTC)", options=list(range(len(hours))), key="timeline_hour",
        format_func=lambda index: hours[index].strftime("%d %b, %H:%M"), on_change=_select_hour,
        help="Browse the saved July observations. This control does not connect to a live station feed.",
    )
    previous, play, following = st.sidebar.columns([1, 1.4, 1])
    previous.button("←", help="Previous hour", on_click=_step_hour, args=(-1, len(hours)), width="stretch")
    play.button("Pause" if st.session_state.replay_running else "Play", on_click=_toggle_playback, width="stretch")
    following.button("→", help="Next hour", on_click=_step_hour, args=(1, len(hours)), width="stretch")
    speed = st.sidebar.select_slider("Playback speed", [0.5, 1.0, 2.0, 4.0], value=2.0,
                                     format_func=lambda value: f"{value:g} s / hour")
    st.sidebar.caption(f"{hours[0]:%d %b} – {hours[-1]:%d %b %Y}")
    return hours[st.session_state.replay_index], float(speed)


def _header(hour: pd.Timestamp) -> None:
    state = "Playing" if st.session_state.replay_running else "Paused"
    st.html(
        '<div class="monitor-header"><div><div class="monitor-eyebrow">Libya · Weather stations</div>'
        '<h1>Station Monitor</h1></div><div class="monitor-clock" aria-live="polite">'
        f'<span class="state">VIEWING TIME · {state.upper()}</span>'
        f'<span class="time">{hour:%H:%M} <small>UTC</small></span>'
        f'<span class="date">{hour:%A, %d %B %Y}</span></div></div>'
    )


def _network_page(snapshot: pd.DataFrame) -> None:
    counts = snapshot["category"].value_counts()
    for col, category, label in zip(st.columns(4),
            ["Healthy", "Needs attention", "In outage", "Active faults"],
            ["Healthy", "Needs attention", "Full outages", "Fault alerts"], strict=True):
        col.metric(label, int(counts.get(category, 0)))
    st.subheader("All stations")
    st.caption("Current health and forecasts are out of 100. Select a column heading to sort.")
    numeric = {"Health /100": st.column_config.ProgressColumn("Health /100", min_value=0,
                max_value=100, format="%.1f"), "Condition": st.column_config.TextColumn(help=HEALTH_HELP)}
    numeric.update({f"+{h} h": st.column_config.NumberColumn(format="%.1f", help=f"Health out of 100, {h} hours ahead. Offline stations use a continued-outage projection; blank means unavailable.")
                    for h in HORIZONS})
    st.dataframe(network_table(snapshot), hide_index=True, width="stretch", height="content",
                 column_config=numeric, key="network_stations")
    alerts = snapshot.loc[snapshot.fault_detected.eq(1) & ~snapshot.availability_class.eq("full_outage")]
    if not alerts.empty:
        with st.expander(f"Likely fault explanations · {len(alerts)} stations"):
            details = alerts[["station_id", "likely_mechanisms", "likely_components", "weather_note"]].rename(
                columns={"station_id": "Station", "likely_mechanisms": "Likely mechanism",
                         "likely_components": "Likely component", "weather_note": "Weather context"})
            st.dataframe(details, hide_index=True, height="content", width="stretch")


def _health_components(row: pd.Series) -> None:
    st.markdown("#### How this health score adds up")
    parts = []
    for component in health_breakdown(row).itertuples(index=False):
        width = 0 if pd.isna(component.Points) else max(0, min(100, 100 * component.Points / component.Maximum))
        parts.append(
            f'<div class="health-part"><span class="label">{escape(component.Component)}</span>'
            f'<span class="points">{score_text(component.Points, component.Maximum)}</span>'
            f'<div class="track"><div class="fill" style="width:{width:.2f}%"></div></div>'
            f'<span class="meaning">{escape(component.Meaning)}</span></div>'
        )
    st.html("".join(parts))
    st.caption(f"Total: {score_text(row.health_total)}. Small differences can arise from rounding.")
    if row.availability_class in ("full_outage", "partial_outage"):
        st.caption("These points already include the reduction for outage duration; it is not deducted a second time.")


def _station_page(bundle: ReplayBundle, snapshot: pd.DataFrame, hour: pd.Timestamp) -> None:
    labels = {str(row.station_id): f"{row.station_id} · {row.city}" for row in bundle.registry.itertuples(index=False)}
    if "selected_station_id" not in st.session_state:
        st.session_state.selected_station_id = st.session_state.get("station_memory", sorted(labels)[0])
    station_id = st.selectbox("Choose station", sorted(labels), format_func=labels.get, key="selected_station_id")
    # Widget keys are cleaned up when another view is selected; retain this separately.
    st.session_state.station_memory = station_id
    row = snapshot.loc[snapshot.station_id.eq(station_id)].iloc[0]
    metrics = st.columns(4)
    connection = row["status"] if row["status"] in ("Full outage", "Partial outage") else "Transmitting"
    metrics[0].metric("Connection", connection)
    metrics[1].metric("Health", score_text(row.health_total))
    metrics[2].metric("Condition", str(row.health_band).replace("insufficient_history", "Not enough history"), help=HEALTH_HELP)
    metrics[3].metric("Fault status", fault_status(row), help="The detector's decision for this hour, not a confirmed hardware diagnosis.")
    st.markdown("#### What needs attention")
    if row.availability_class == "full_outage":
        st.warning(f"No observations received for {int(row.full_outage_run_hours)} hours. Sensor faults cannot be assessed during full outage.")
    else:
        if row.availability_class == "partial_outage":
            st.warning("Missing sensor groups: " + str(row.absent_sensor_groups).replace("_", " "))
        if pd.isna(row.fault_detected):
            st.info("No detector result is available at this hour.")
        elif row.fault_detected == 1:
            st.write("**Likely mechanism:** " + row.likely_mechanisms)
            st.write("**Likely component:** " + row.likely_components)
            st.caption("Suggested explanations for this hour; inspection is needed to confirm the cause.")
        else:
            st.success("No sensor-fault alert at this hour.")
    if row.weather_note:
        st.info(row.weather_note)
    with st.expander("About the fault assessment"):
        if row.availability_class == "full_outage" or pd.isna(row.fault_probability):
            st.write("No model assessment is available for this hour.")
        else:
            st.write(f"Model score: **{row.fault_probability:.3f} / 1.000** · alert threshold: **{SELECTED_DETECTOR_THRESHOLD:.2f}**.")
        st.write("The model gives each hour a score from 0 to 1. A score of 0.30 or above raises a likely-fault alert. For example, 0.80 is a model score—not a verified 80% chance that hardware is broken. Health /100 summarizes the station's overall condition separately.")
    st.markdown("#### Health forecast")
    for col, (_, forecast) in zip(st.columns(5), forecast_table(row).iterrows(), strict=True):
        col.metric(f"+{int(forecast['Hours ahead'])} hours", score_text(forecast['Health /100']))
    if row.availability_class == "full_outage":
        st.caption("Continued-outage projection: assumes the station stays offline; it does not predict recovery.")
    st.markdown("#### Health history")
    history = station_history(bundle, station_id, hour, lookback_hours=None)
    if history.health_total.notna().any():
        st.plotly_chart(history_chart(history, station_id), width="stretch", theme=None,
                        config=CHART_CONFIG, key=f"history_{station_id}")
        if history.health_total.notna().sum() > 1:
            st.caption("Drag left/right or move the shaded timeline underneath. Times are UTC.")
        else:
            st.caption("Only one health reading is available at this viewing time. Move forward to see the history and timeline.")
    else:
        st.info("Not enough history to show a health score yet.")
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        _health_components(row)
    with right:
        st.markdown("#### Sensor availability")
        st.dataframe(station_sensor_states(bundle, station_id, hour), hide_index=True, height="content", width="stretch")
        st.caption("Present means observations were received; it does not prove the sensor is accurate.")


def _event_context(bundle: ReplayBundle, event: pd.Series) -> None:
    st.markdown("#### Weather reference comparison")
    st.caption("Station and ERA5/Open-Meteo values at the same hours. Difference = station minus reference.")
    weather = event_weather_comparison(bundle, event)
    if not weather["Paired hours"].any():
        st.info("No paired station/reference measurements are available for this alert period.")
    else:
        st.dataframe(weather.round(2), hide_index=True, height="content", width="stretch")
        rows = bundle.health.loc[bundle.health.station_id.eq(event.station_id)
                                 & bundle.health.hour_utc.between(event.start_hour, event.end_hour)].sort_values("hour_utc")
        if {"temp_avg_c", "temperature_2m"}.issubset(rows.columns):
            fig = go.Figure()
            for column, label, color in [("temp_avg_c", "Station", ACCENT), ("temperature_2m", "Weather reference", "#8b689f")]:
                fig.add_trace(go.Scatter(x=rows.hour_utc, y=rows[column], name=label,
                    mode="lines+markers", line={"color": color}, connectgaps=False,
                    hovertemplate="%{x|%d %b, %H:%M} UTC<br>%{y:.1f} °C<extra>%{fullData.name}</extra>"))
            fig.update_layout(template="plotly_white", height=250, dragmode="pan",
                              margin={"l": 5, "r": 5, "t": 10, "b": 5}, yaxis_title="Temperature (°C)",
                              xaxis_title="Time (UTC)", legend={"orientation": "h", "y": 1.15})
            st.plotly_chart(fig, width="stretch", config=CHART_CONFIG, key=f"reference_{event.event_id}")
        st.caption("Context, not a new fault test. Pressure compares the station's hourly maximum with reference sea-level pressure. Solar is omitted because hourly maxima and reference radiation are not directly comparable.")
    st.markdown("#### Nearby stations")
    neighbors = event_neighbor_comparison(bundle, event)
    if neighbors.empty:
        st.info("This station has no configured nearby stations.")
    else:
        st.dataframe(neighbors.round(2), hide_index=True, height="content", width="stretch")
        st.caption(f"Counts cover the selected {int(event.duration_hours)}-hour alert period. Temperature differences use only matched hours; a reporting neighbor is not automatically an agreeing neighbor. Blank differences mean no paired values.")


def _event_page(bundle: ReplayBundle, hour: pd.Timestamp) -> None:
    events = segment_predicted_fault_events(bundle.detections, hour).sort_values("start_hour", ascending=False)
    if events.empty:
        st.info("No fault alerts up to the viewing time. Move the time control forward to inspect later alerts.")
        return
    choices = ["All stations"] + sorted(events.station_id.unique().tolist())
    station_filter = st.selectbox("Filter by station", choices, key="evidence_station")
    if station_filter != "All stations":
        events = events.loc[events.station_id.eq(station_filter)]
    event_id = st.selectbox("Choose alert period", events.event_id.tolist(),
                            format_func=lambda value: _event_name(events.loc[events.event_id.eq(value)].iloc[0]))
    event = events.loc[events.event_id.eq(event_id)].iloc[0]
    metrics = st.columns(4)
    metrics[0].metric("Station", event.station_id)
    metrics[1].metric("Alert", "Active" if event.status == "active" else "Ended")
    metrics[2].metric("Duration", f"{int(event.duration_hours)} hours")
    metrics[3].metric("Peak model score", f"{event.peak_probability:.3f} / 1", help="Not a calibrated probability of hardware failure.")
    st.caption(f"{event.start_hour:%d %b %Y, %H:%M} – {event.end_hour:%d %b %Y, %H:%M} UTC")
    st.markdown("#### Likely causes by hour")
    reasons = event_reason_history(bundle, event)
    if reasons.empty:
        st.info("No reason-code output is available for this alert period.")
    else:
        readable = reasons[["hour_utc", "likely_mechanisms", "likely_components"]].copy()
        for axis in ("mechanism", "component"):
            column = f"likely_{axis}s"
            readable[column] = readable[column].fillna("").str.replace("_", " ", regex=False)
            readable.loc[readable[column].eq(""), column] = reasons[f"{axis}_status"].str.replace("_", " ", regex=False)
        st.dataframe(readable.rename(columns={"hour_utc": "Time (UTC)", "likely_mechanisms": "Likely mechanism",
                                              "likely_components": "Likely component"}),
                     hide_index=True, height="content", width="stretch")
    _event_context(bundle, event)
    with st.expander("Detector checks and thresholds"):
        evidence = event_detector_evidence(bundle, event)
        if evidence.empty:
            st.info("The model raised an alert, but no individual saved detector check is listed for this period.")
        else:
            st.dataframe(evidence, hide_index=True, height="content", width="stretch", column_config={
                key: st.column_config.NumberColumn(format="%.4f") for key in ("Score", "Threshold", "Margin")})
        st.caption("These are supporting checks, not confirmed causes. Positive margins show how far a check crossed its threshold; stuck variance uses the opposite direction.")


def _event_name(event: pd.Series) -> str:
    return f"{event.station_id} · {event.start_hour:%d %b, %H:%M} · {int(event.duration_hours)} h · {event.status}"


def main() -> None:
    st.set_page_config(page_title="Station Monitor", page_icon="🌤️", layout="wide")
    st.html(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>")
    try:
        bundle = _load(DASHBOARD_SCHEMA_VERSION)
    except (FileNotFoundError, KeyError, ValueError) as error:
        st.error(str(error))
        st.stop()
    hours = replay_hours(bundle)
    if not hours:
        st.info("No station observations are available.")
        st.stop()
    hour, speed = _sidebar(hours)
    _header(hour)
    snapshot = build_replay_snapshot(bundle, hour)
    view = st.segmented_control("View", ["Network", "Station", "Evidence"], default="Network",
                                required=True, key="monitor_view", label_visibility="collapsed")
    # Only build the selected view: charts/evidence do not run in hidden tabs.
    if view == "Station":
        _station_page(bundle, snapshot, hour)
    elif view == "Evidence":
        _event_page(bundle, hour)
    else:
        _network_page(snapshot)
    if st.session_state.replay_running:
        if st.session_state.replay_index >= len(hours) - 1:
            st.session_state.replay_running = False
            st.rerun()
        time.sleep(speed)
        st.session_state.replay_index += 1
        st.rerun()


if __name__ == "__main__":
    main()
