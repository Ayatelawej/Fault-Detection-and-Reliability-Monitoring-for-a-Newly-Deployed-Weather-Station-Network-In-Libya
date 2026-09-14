import numpy as np
import pandas as pd

from scripts.investigate_july_temperature import guard_masks, thermal_context


def test_missing_or_conflicting_evidence_keeps_alert():
    rows = pd.DataFrame({
        'temperature_stat': [True] * 7, 'other_stat': [False] * 7,
        'physical_or_confirmed_stuck': [False] * 7,
        'temp_low_c_past_hour_z': [1.] * 7, 'temp_avg_c_past_hour_z': [1.] * 7,
        'temp_high_c_past_hour_z': [1.] * 7, 'r_temp_past_z': [0.] * 7,
        'r_spatial_temp_past_z': [0.] * 7,
    })
    rows.loc[1, 'temp_high_c_past_hour_z'] = 4.
    rows.loc[2, 'r_temp_past_z'] = np.nan
    rows.loc[3, 'r_spatial_temp_past_z'] = np.nan
    rows.loc[4, 'physical_or_confirmed_stuck'] = True
    rows.loc[5, 'other_stat'] = True
    rows.loc[6, 'temp_avg_c_past_hour_z'] = np.nan
    masks = guard_masks(rows)
    assert not masks['unchanged'].any()
    assert masks['temperature_context_and_reference'].tolist() == [True, False, False, True, False, False, False]
    assert masks['temperature_context_reference_and_peers'].tolist() == [True, False, False, False, False, False, False]


def test_guard_context_does_not_use_future_observations():
    hours = pd.date_range('2026-06-01', periods=24 * 36, freq='h', tz='UTC')
    values = 25 + np.sin(np.arange(len(hours)) / 5) + np.cos(np.arange(len(hours)) / 72)
    raw = pd.DataFrame(dict(station_id='s', hour_utc=hours,
                            temp_low_c=values - 1, temp_avg_c=values, temp_high_c=values + 1))
    external = pd.DataFrame(dict(station_id='s', hour_utc=hours, r_temp=np.sin(values), ref_temp=values - np.sin(values)))
    spatial = pd.DataFrame(dict(station_id='s', hour_utc=hours, r_spatial_temp=np.cos(values)))
    full = thermal_context(raw, external, spatial)
    n = 24 * 32
    prefix = thermal_context(raw.iloc[:n], external.iloc[:n], spatial.iloc[:n])
    pd.testing.assert_frame_equal(full.iloc[:n].reset_index(drop=True), prefix)
