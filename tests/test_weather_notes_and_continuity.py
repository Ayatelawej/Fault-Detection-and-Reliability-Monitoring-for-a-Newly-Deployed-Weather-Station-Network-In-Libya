from dataclasses import replace
import numpy as np
import pandas as pd
from src.dashboard.replay import build_replay_snapshot
from test_availability import _synthetic_replay_bundle


def test_weather_note_preserves_snapshot_decisions_and_hides_outage_notes():
    base = _synthetic_replay_bundle()
    notes = base.detections[['station_id', 'hour_utc']].copy()
    notes['weather_note'] = 'Weather-consistent; alert retained.'
    original = build_replay_snapshot(base, '2026-07-01T01:00:00Z')
    annotated = build_replay_snapshot(replace(base, weather_notes=notes), '2026-07-01T01:00:00Z')
    pd.testing.assert_frame_equal(original.drop(columns='weather_note'), annotated.drop(columns='weather_note'))
    assert annotated.set_index('station_id').loc['A', 'weather_note']
    assert not annotated.set_index('station_id').loc['B', 'weather_note']
