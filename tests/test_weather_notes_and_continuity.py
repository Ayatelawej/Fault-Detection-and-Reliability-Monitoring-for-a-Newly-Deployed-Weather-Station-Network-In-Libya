from dataclasses import replace
import numpy as np
import pandas as pd
from src.dashboard.replay import build_replay_snapshot
from test_availability import _synthetic_replay_bundle
from scripts.test_temperature_refinement import add_continuity


def test_weather_note_preserves_snapshot_decisions_and_hides_outage_notes():
    base = _synthetic_replay_bundle()
    notes = base.detections[['station_id', 'hour_utc']].copy()
    notes['weather_note'] = 'Weather-consistent; alert retained.'
    original = build_replay_snapshot(base, '2026-07-01T01:00:00Z')
    annotated = build_replay_snapshot(replace(base, weather_notes=notes), '2026-07-01T01:00:00Z')
    pd.testing.assert_frame_equal(original.drop(columns='weather_note'), annotated.drop(columns='weather_note'))
    assert annotated.set_index('station_id').loc['A', 'weather_note']
    assert not annotated.set_index('station_id').loc['B', 'weather_note']


def test_continuity_uses_prior_evidence_not_current_or_future():
    hours = pd.date_range('2026-06-01', periods=30, freq='h', tz='UTC')
    context = pd.DataFrame(dict(station_id='s', hour_utc=hours, temp_avg_c=np.arange(30.), r_temp=0.,
        temp_avg_c_past_hour_z=0., r_temp_past_z=0., r_spatial_temp_past_z=0., physical_or_confirmed_stuck=False))
    context.loc[10, ['temp_avg_c_past_hour_z', 'r_temp_past_z']] = 5.
    result = add_continuity(context)
    assert result.loc[10, 'prior_supported_temperature_6h'] == 0
    assert result.loc[11, 'prior_supported_temperature_6h'] == 1
    pd.testing.assert_frame_equal(result.iloc[:20], add_continuity(context.iloc[:20]))
