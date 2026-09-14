from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard.replay import (
    JULY_DETECTION_PATH,
    JULY_FORECAST_PATH,
    JULY_HEALTH_PATH,
    JULY_NEIGHBORS_PATH,
    JULY_REASON_PATH,
    JULY_SCORES_PATH,
)


def _require_local_replay_inputs(paths):
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        pytest.skip(
            "Optional frozen July replay artifacts are missing; restore them as described "
            "in docs/RELEASE_CHECKLIST.md: " + ", ".join(missing)
        )


def test_missing_local_replay_input_skips_explicitly(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="Optional frozen July replay artifacts"):
        _require_local_replay_inputs([tmp_path / "missing.parquet"])


def test_present_local_replay_input_does_not_skip(tmp_path):
    artifact = tmp_path / "present.parquet"
    artifact.touch()
    _require_local_replay_inputs([artifact])


def test_dashboard_starts_and_renders_mid_july_hour_400():
    # Only ignored release inputs are optional; missing tracked files, bad schemas,
    # gate mismatches, and application errors must still fail when inputs exist.
    _require_local_replay_inputs(
        [
            JULY_HEALTH_PATH,
            JULY_FORECAST_PATH,
            JULY_DETECTION_PATH,
            JULY_SCORES_PATH,
            JULY_NEIGHBORS_PATH,
            JULY_REASON_PATH,
        ]
    )
    app = AppTest.from_file(str(Path(__file__).parents[1] / "scripts" / "run_dashboard.py"))
    app.run(timeout=30)
    assert not app.exception
    app.sidebar.select_slider[0].set_value(400).run(timeout=30)
    assert not app.exception
    assert app.session_state["replay_index"] == 400
