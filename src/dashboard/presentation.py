"""Display-only dashboard helpers. Never change model outputs or health scores."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.availability.health_score import HEALTH_WEIGHTS
from src.dashboard.replay import HEALTH_COMPONENT_COLUMNS, ReplayBundle

HORIZONS = (1, 3, 6, 12, 24)
ACCENT = "#167d8d"
HEALTH_HELP = "Healthy: 80–100 · Watch: 60–<80 · Degraded: 40–<60 · Critical: below 40."
COMPONENT_HELP = {
    "Availability": "How consistently the station has sent observations.",
    "Sensor completeness": "How consistently its sensor groups have been present.",
    "Fault burden": "Fewer recent fault indicators earn more points.",
    "Reference consistency": "Consistency with external weather and reference context.",
    "Stability": "Fewer interruptions and longer recovery earn more points.",
}


def score_text(value: object, maximum: float = 100) -> str:
    if pd.isna(value):
        return "Not available"
    return f"{float(value):.1f} / {maximum:g}"


def fault_status(row: pd.Series) -> str:
    if row.get("availability_class") == "full_outage":
        return "Not assessed"
    if pd.isna(row.get("fault_detected", np.nan)):
        return "Not available"
    return "Likely fault" if row["fault_detected"] == 1 else "No alert"


def network_table(snapshot: pd.DataFrame) -> pd.DataFrame:
    table = snapshot[["station_id", "city", "status", "health_total", "health_band"]].rename(
        columns={"station_id": "Station", "city": "Location", "status": "Status",
                 "health_total": "Health /100", "health_band": "Condition"}
    ).copy()
    for horizon in HORIZONS:
        table[f"+{horizon} h"] = snapshot[f"forecast_{horizon}h"].round(1)
    table["Health /100"] = table["Health /100"].round(1)
    return table


def health_breakdown(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([
        {"Component": label, "Points": row[column],
         "Maximum": HEALTH_WEIGHTS[column.removeprefix("weighted_")],
         "Meaning": COMPONENT_HELP[label]}
        for label, column in HEALTH_COMPONENT_COLUMNS.items()
    ])


def forecast_table(row: pd.Series) -> pd.DataFrame:

    return pd.DataFrame({
        "Hours ahead": HORIZONS,
        "Health /100": [row[f"forecast_{h}h"] for h in HORIZONS],
    }).round({"Health /100": 1})


def history_chart(history: pd.DataFrame, station_id: str) -> go.Figure:
    points = history.sort_values("hour_utc").set_index("hour_utc")
    if not points.empty:
        points = points.reindex(pd.date_range(points.index.min(), points.index.max(), freq="h"))
    valid = points["health_total"].dropna()
    has_timeline = len(valid) > 1
    fig = go.Figure(go.Scatter(
        x=points.index, y=points["health_total"], mode="lines", connectgaps=False,
        line={"color": ACCENT, "width": 2.5}, name="Health",
        hovertemplate="%{x|%d %b, %H:%M} UTC<br>Health: %{y:.1f} / 100<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white", height=430 if has_timeline else 310,
        dragmode="pan", uirevision=f"history-{station_id}-{has_timeline}",
        margin={"l": 55, "r": 20, "t": 45, "b": 100 if has_timeline else 65, "autoexpand": True},
        font={"color": "#24364b"},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title": {"text": "Time (UTC)", "standoff": 12}, "automargin": True,
               "type": "date", "tickformat": "%d %b<br>%H:%M", "nticks": 6,
               "rangeslider": {"visible": has_timeline, "thickness": .12,
                               "bgcolor": "#eef2f6", "bordercolor": "#dbe3ec", "borderwidth": 1},
               "rangeselector": {"visible": has_timeline, "buttons": [
                   {"count": 1, "label": "1 day", "step": "day", "stepmode": "backward"},
                   {"count": 3, "label": "3 days", "step": "day", "stepmode": "backward"},
                   {"count": 7, "label": "7 days", "step": "day", "stepmode": "backward"},
                   {"label": "All", "step": "all"},
               ]}},
        yaxis={"title": "Health /100", "range": [0, 100], "fixedrange": True, "automargin": True},
    )
    if has_timeline:
        end = points.index.max()
        fig.update_xaxes(range=[max(points.index.min(), end - pd.Timedelta(hours=72)), end])
    else:
        fig.update_traces(mode="lines+markers")
        if len(valid):

            hour = valid.index[0]
            fig.update_xaxes(range=[hour - pd.Timedelta(hours=1), hour + pd.Timedelta(hours=1)],
                             dtick=3_600_000)
    return fig


def event_weather_comparison(bundle: ReplayBundle, event: pd.Series) -> pd.DataFrame:
    """Paired raw values at the same event hours; not a new fault decision."""
    rows = bundle.health.loc[
        bundle.health.station_id.eq(event.station_id)
        & bundle.health.hour_utc.between(event.start_hour, event.end_hour)
    ]
    specs = [
        ("Temperature", "°C", "temp_avg_c", "temperature_2m"),
        ("Dew point", "°C", "dewpoint_avg_c", "dew_point_2m"),
        ("Wind speed", "km/h", "windspeed_avg_kmh", "wind_speed_10m"),
        ("Pressure (station hourly max)", "hPa", "pressure_max_hpa", "pressure_msl"),
    ]
    output = []
    for label, unit, station_col, reference_col in specs:
        station = pd.to_numeric(rows.get(station_col, pd.Series(np.nan, index=rows.index)), errors="coerce")
        reference = pd.to_numeric(rows.get(reference_col, pd.Series(np.nan, index=rows.index)), errors="coerce")
        paired = np.isfinite(station) & np.isfinite(reference)
        output.append({"Variable": label, "Unit": unit, "Paired hours": int(paired.sum()),
                       "Station mean": station[paired].mean(), "Reference mean": reference[paired].mean(),
                       "Mean difference": (station[paired] - reference[paired]).mean()})
    return pd.DataFrame(output)


def event_neighbor_comparison(bundle: ReplayBundle, event: pd.Series) -> pd.DataFrame:
    columns = ["Neighbor", "Location", "Distance (km)", "Reporting hours", "Paired temperature hours",
               "Station − neighbor (°C)"]
    neighbors = bundle.neighbors.loc[bundle.neighbors.station_id.eq(event.station_id)].drop_duplicates("neighbor_id")
    event_rows = bundle.health.loc[bundle.health.hour_utc.between(event.start_hour, event.end_hour)]
    station = event_rows.loc[event_rows.station_id.eq(event.station_id)].set_index("hour_utc")
    station_temp = station.get("temp_avg_c", pd.Series(np.nan, index=station.index))
    cities = bundle.registry.set_index("station_id")["city"].to_dict()
    output = []
    for neighbor in neighbors.itertuples(index=False):
        observations = event_rows.loc[
            event_rows.station_id.eq(neighbor.neighbor_id) & event_rows.is_transmitting.fillna(False)
        ].set_index("hour_utc")
        temp = observations.get("temp_avg_c", pd.Series(np.nan, index=observations.index))
        matched = pd.concat([station_temp.rename("station"), temp.rename("neighbor")], axis=1)
        paired = matched.replace([np.inf, -np.inf], np.nan).dropna()
        output.append({"Neighbor": neighbor.neighbor_id, "Location": cities.get(neighbor.neighbor_id, "—"),
                       "Distance (km)": getattr(neighbor, "distance_km", np.nan),
                       "Reporting hours": observations.index.nunique(), "Paired temperature hours": len(paired),
                       "Station − neighbor (°C)": (paired.station - paired.neighbor).mean()})
    return pd.DataFrame(output, columns=columns)
