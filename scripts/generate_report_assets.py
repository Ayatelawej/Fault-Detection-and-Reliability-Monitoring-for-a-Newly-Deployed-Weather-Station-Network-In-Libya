from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.workflows.build_methodology_figures import main as build_methodology
from src.workflows.build_result_figures import (
    main as build_results,
    require_result_inputs,
)
from src.config.paths import HOURLY_ROW_STATES_PATH, STATION_REGISTRY_PATH
from src.workflows.prerequisites import require_files


def parse_args(argv: list[str] | None = None) -> str:
    parser = ArgumentParser(description="Generate report figure assets.")
    parser.add_argument(
        "--set",
        choices=("current", "methodology", "results", "july-evaluation", "all"),
        default="methodology",
        dest="figure_set",
    )
    return str(parser.parse_args(argv).figure_set)


def main(argv: list[str] | None = None) -> None:
    figure_set = parse_args(argv)
    if figure_set == "july-evaluation":

        from src.workflows.build_current_figures import binary, OUT
        OUT.mkdir(parents=True, exist_ok=True)
        binary()
        return
    if figure_set == "current":
        from src.workflows.build_current_figures import main as build_current
        build_current()
        return
    if figure_set in {"methodology", "all"}:
        require_files(
            "Methodology figure generation",
            {
                "station registry": STATION_REGISTRY_PATH,
                "hourly availability states": HOURLY_ROW_STATES_PATH,
            },
        )
    if figure_set in {"results", "all"}:
        require_result_inputs()
    if figure_set in {"methodology", "all"}:
        build_methodology()
    if figure_set in {"results", "all"}:
        build_results()
    if figure_set == "all":
        from src.workflows.build_current_figures import main as build_current
        build_current()


if __name__ == "__main__":
    main()
