from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "outputs" / "figures"
REGISTRY_PATH = ROOT / "data" / "merged" / "station_registry.csv"
ROW_STATES_PATH = ROOT / "data" / "processed" / "hourly_row_states.parquet"


COLORS = {
    "ink": "#27323a",
    "muted": "#66737c",
    "blue": "#2f6f9f",
    "green": "#43835f",
    "gold": "#c48a2c",
    "red": "#b75b5b",
    "teal": "#3f8f8f",
    "line": "#d7dde2",
    "panel": "#f6f8fa",
}


STATUS_COLORS = {
    "reliable_reference_candidate": "#397367",
    "reliable_but_terminal_gap": "#5f8fc2",
    "usable": "#c58b32",
    "weak": "#a46aa9",
    "outage_dominated": "#b85858",
}


def _save(fig: plt.Figure, filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = COLORS["ink"],
    fontsize: int = 10,
) -> None:
    rect = Rectangle(
        xy,
        width,
        height,
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        fill(text, 24),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        zorder=3,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = COLORS["muted"],
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color=color,
            shrinkA=4,
            shrinkB=4,
            zorder=1,
        )
    )


def build_station_network_map() -> Path:
    registry = pd.read_csv(REGISTRY_PATH)
    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    for status, group in registry.groupby("status_class"):
        label = status.replace("_", " ")
        ax.scatter(
            group["longitude"],
            group["latitude"],
            s=70,
            label=label,
            color=STATUS_COLORS.get(status, COLORS["muted"]),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

    for _, row in registry.iterrows():
        if row["station_id"] in {"IALWAH18", "IDERNA7", "INALUT3", "ITRIPO33"}:
            ax.annotate(
                row["station_id"],
                (row["longitude"], row["latitude"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=COLORS["ink"],
            )

    ax.set_title("Weather-station network coverage across Libya", fontsize=14, pad=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(10.5, 23.3)
    ax.set_ylim(28.5, 33.4)
    ax.grid(True, color=COLORS["line"], linewidth=0.8)
    ax.legend(loc="lower left", fontsize=8, frameon=True, title="Status class")
    ax.text(
        0.99,
        0.02,
        "26 personal weather stations; marker color shows audit status class",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=COLORS["muted"],
    )
    return _save(fig, "report_station_network_map.png")


def build_architecture_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(12.0, 6.8))
    ax.set_axis_off()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.6)

    boxes = {
        "raw": ((0.5, 4.25), "WU hourly observations"),
        "registry": ((0.5, 2.75), "Station registry and install windows"),
        "row": ((3.0, 3.5), "Row-state classifier"),
        "events": ((5.4, 3.5), "Availability events and network windows"),
        "channels": ((3.0, 1.35), "Channel transforms: wind sin/cos and precip log1p"),
        "base": ((5.4, 1.35), "Per-station robust baselines"),
        "detectors": ((7.9, 2.45), "Statistical detectors: robust z-score, stuck runs, Isolation Forest"),
        "review": ((7.9, 0.55), "Review queue, labels, and future ML layers"),
    }

    for key, (xy, text) in boxes.items():
        face = {
            "raw": "#eaf2f8",
            "registry": "#eef6ef",
            "row": "#fff4df",
            "events": "#f9ecec",
            "channels": "#e8f5f4",
            "base": "#eef6ef",
            "detectors": "#f3f0fa",
            "review": "#f6f8fa",
        }[key]
        width = 2.15 if key in {"detectors", "review"} else 1.95
        _box(ax, xy, width, 0.82, text, face, fontsize=9)

    _arrow(ax, (2.45, 4.66), (3.0, 3.91))
    _arrow(ax, (2.45, 3.16), (3.0, 3.91))
    _arrow(ax, (4.95, 3.91), (5.4, 3.91))
    _arrow(ax, (2.45, 4.43), (3.0, 1.76))
    _arrow(ax, (4.95, 1.76), (5.4, 1.76))
    _arrow(ax, (7.35, 1.76), (7.9, 2.86))
    _arrow(ax, (7.35, 3.91), (7.9, 2.86))
    _arrow(ax, (8.98, 2.45), (8.98, 1.37))

    ax.text(
        5.5,
        6.05,
        "Hybrid fault-detection architecture",
        ha="center",
        va="center",
        fontsize=15,
        color=COLORS["ink"],
        weight="bold",
    )
    ax.text(
        5.5,
        5.72,
        "Completed reliability layer feeds statistical scoring and later review/ML work",
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["muted"],
    )
    return _save(fig, "report_hybrid_fault_detection_architecture.png")


def build_row_state_flow() -> Path:
    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    ax.set_axis_off()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)

    _box(ax, (0.45, 4.7), 1.95, 0.7, "Hourly station row", "#eaf2f8", fontsize=9)
    _box(ax, (2.95, 4.7), 1.95, 0.7, "Valid station and timestamp?", "#fff4df", fontsize=9)
    _box(ax, (5.45, 4.7), 1.95, 0.7, "Inside active station window?", "#fff4df", fontsize=9)
    _box(ax, (7.95, 4.7), 1.95, 0.7, "data_present = 1?", "#fff4df", fontsize=9)

    outputs = [
        ((2.95, 3.25), "invalid"),
        ((5.3, 3.25), "before_first /\nterminal_padded /\nnever_active"),
        ((7.85, 3.25), "true_outage\ncandidate"),
        ((1.1, 1.55), "warmup"),
        ((3.55, 1.55), "online_partial\nmissing"),
        ((6.0, 1.55), "online_complete"),
    ]
    for xy, text in outputs:
        _box(ax, xy, 2.05, 0.75, text, "#f6f8fa", fontsize=9)

    _box(ax, (8.35, 1.55), 1.75, 0.75, "row_state", "#eef6ef", fontsize=9)

    _arrow(ax, (2.4, 5.05), (2.95, 5.05))
    _arrow(ax, (4.9, 5.05), (5.45, 5.05))
    _arrow(ax, (7.4, 5.05), (7.95, 5.05))
    _arrow(ax, (3.92, 4.7), (3.92, 4.0))
    _arrow(ax, (6.42, 4.7), (6.42, 4.0))
    _arrow(ax, (8.92, 4.7), (8.92, 4.0))
    _arrow(ax, (8.88, 3.25), (9.22, 2.3))
    _arrow(ax, (2.12, 1.92), (8.35, 1.92))
    _arrow(ax, (5.6, 1.92), (8.35, 1.92))
    _arrow(ax, (8.05, 1.92), (8.35, 1.92))

    ax.text(
        5.0,
        5.7,
        "Row-state classification flow",
        ha="center",
        fontsize=15,
        color=COLORS["ink"],
        weight="bold",
    )
    ax.text(
        5.0,
        0.75,
        "The classifier separates lifecycle padding, warmup, complete observations, partial missingness, and outage candidates before event extraction.",
        ha="center",
        fontsize=9,
        color=COLORS["muted"],
    )
    return _save(fig, "report_row_state_classification_flow.png")


def build_outage_event_flow() -> Path:
    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)

    states = [
        "ok",
        "ok",
        "out",
        "out",
        "out",
        "ok",
        "out",
        "out",
        "ok",
        "ok",
    ]
    x0 = 0.8
    for i, state in enumerate(states):
        color = "#b85858" if state == "out" else "#397367"
        ax.add_patch(
            Rectangle((x0 + i * 0.55, 3.55), 0.42, 0.42, facecolor=color, edgecolor="white")
        )
    ax.text(0.8, 4.2, "Hourly row states", fontsize=10, color=COLORS["ink"])
    ax.text(0.8, 3.25, "green = online, red = true_outage_candidate", fontsize=8, color=COLORS["muted"])

    _box(ax, (0.8, 1.95), 2.0, 0.75, "Run-length encode outage hours", "#fff4df", fontsize=9)
    _box(ax, (3.5, 1.95), 2.0, 0.75, "Create event start, end, duration", "#eaf2f8", fontsize=9)
    _box(ax, (6.2, 1.95), 2.0, 0.75, "Detect coordinated network windows", "#f9ecec", fontsize=9)
    _box(ax, (8.9, 1.95), 1.8, 0.75, "Assign\noutage_class", "#eef6ef", fontsize=9)

    _arrow(ax, (3.2, 3.55), (1.8, 2.7))
    _arrow(ax, (2.8, 2.32), (3.5, 2.32))
    _arrow(ax, (5.5, 2.32), (6.2, 2.32))
    _arrow(ax, (8.2, 2.32), (8.9, 2.32))

    ax.text(
        5.0,
        4.62,
        "Outage event construction from hourly row states",
        ha="center",
        fontsize=15,
        color=COLORS["ink"],
        weight="bold",
    )
    ax.text(
        5.0,
        0.95,
        "Candidate outage hours become duration-bearing events, then local station failures are separated from coordinated network windows.",
        ha="center",
        fontsize=9,
        color=COLORS["muted"],
    )
    return _save(fig, "report_outage_event_construction_flow.png")


def _longest_zero_wind_run(station: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    mask = (
        station["data_present"].eq(1)
        & station["windspeed_avg_kmh"].eq(0)
        & station["windgust_avg_kmh"].eq(0)
    )
    run_id = mask.ne(mask.shift(fill_value=False)).cumsum()
    best_start = best_end = None
    best_len = 0
    for _, group in station.loc[mask].groupby(run_id[mask]):
        if len(group) > best_len:
            best_len = len(group)
            best_start = group["hour_utc"].min()
            best_end = group["hour_utc"].max()
    if best_start is None or best_end is None:
        raise ValueError("No zero wind run found for ITRIPO33")
    return best_start, best_end, best_len


def build_itripo33_stuck_wind_panel() -> Path:
    cols = [
        "station_id",
        "hour_utc",
        "windspeed_avg_kmh",
        "windgust_avg_kmh",
        "data_present",
    ]
    df = pd.read_parquet(ROW_STATES_PATH, columns=cols)
    station = df.loc[df["station_id"].eq("ITRIPO33")].copy()
    station["hour_utc"] = pd.to_datetime(station["hour_utc"], utc=True)
    station = station.sort_values("hour_utc")
    start, end, hours = _longest_zero_wind_run(station)
    panel_start = start - pd.Timedelta(hours=24)
    panel_end = end + pd.Timedelta(hours=24)
    panel = station.loc[
        station["hour_utc"].between(panel_start, panel_end)
    ].copy()

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(
        panel["hour_utc"],
        panel["windspeed_avg_kmh"],
        color=COLORS["blue"],
        linewidth=1.8,
        marker="o",
        markersize=2.5,
        label="Average wind speed",
    )
    ax.plot(
        panel["hour_utc"],
        panel["windgust_avg_kmh"],
        color=COLORS["gold"],
        linewidth=1.8,
        marker="o",
        markersize=2.5,
        label="Average wind gust",
    )
    ymax = max(8.0, float(panel[["windspeed_avg_kmh", "windgust_avg_kmh"]].max().max()) + 1.0)
    ax.axvspan(start, end, color=COLORS["red"], alpha=0.18, label="Zero-wind flatline")
    ax.text(
        start + (end - start) / 2,
        ymax * 0.86,
        f"{hours} consecutive hours at 0.0 km/h",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["red"],
        weight="bold",
    )
    ax.set_title("ITRIPO33 stuck-wind example: wind speed and gust flatline", fontsize=14, pad=12)
    ax.set_ylabel("km/h")
    ax.set_xlabel("UTC time")
    ax.set_ylim(-0.4, ymax)
    ax.grid(True, color=COLORS["line"], linewidth=0.8)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    fig.autofmt_xdate(rotation=0)
    return _save(fig, "report_itripo33_stuck_wind_panel.png")


def main() -> None:
    builders = [
        build_station_network_map,
        build_architecture_diagram,
        build_row_state_flow,
        build_outage_event_flow,
        build_itripo33_stuck_wind_panel,
    ]
    for builder in builders:
        print(builder())


if __name__ == "__main__":
    main()
