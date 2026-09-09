from __future__ import annotations

import pytest

from src.features.rebuild import (
    DEFAULT_OUTPUTS,
    isolated_output_paths,
    rebuild_detection_features,
)


def test_rebuild_requires_reference_inputs_before_writing_outputs(tmp_path) -> None:
    merged_path = tmp_path / "station_hourly_merged.csv"
    registry_path = tmp_path / "station_registry.csv"
    output_path = tmp_path / "outputs" / "scores.parquet"
    merged_path.write_text("station_id,hour_utc\n", encoding="utf-8")
    registry_path.write_text("station_id\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="public reference parquet"):
        rebuild_detection_features(
            merged_path=merged_path,
            registry_path=registry_path,
            reference_dir=tmp_path / "missing_reference",
            five_min_dir=tmp_path / "missing_five_minute",
            output_dir=output_path.parent,
        )

    assert not output_path.exists()


def test_noncanonical_rebuild_requires_output_directory_before_input_checks(tmp_path) -> None:
    merged_path = tmp_path / "combined.csv"

    with pytest.raises(ValueError, match="requires --output-dir"):
        rebuild_detection_features(merged_path=merged_path)

    assert not list(tmp_path.rglob("*"))


def test_isolated_output_map_redirects_every_canonical_output(tmp_path) -> None:
    output_dir = tmp_path / "july_features"
    paths = isolated_output_paths(output_dir)

    assert set(paths) == set(DEFAULT_OUTPUTS)
    assert all(path.parent == output_dir.resolve() for path in paths.values())
    assert {path.name for path in paths.values()} == {
        path.name for path in DEFAULT_OUTPUTS.values()
    }
    assert not {
        path.resolve() for path in paths.values()
    }.intersection(path.resolve() for path in DEFAULT_OUTPUTS.values())
