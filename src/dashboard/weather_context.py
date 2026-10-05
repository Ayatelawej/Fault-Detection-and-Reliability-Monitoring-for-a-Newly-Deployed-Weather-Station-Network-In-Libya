"""Score-neutral weather context. These checks never change detector outputs."""
import numpy as np
import pandas as pd
from src.model.reason_code_utils import features_for_station

def mad(values):
    return np.nanmedian(np.abs(values - np.nanmedian(values)))


def thermal_context(raw, external, spatial):
    """New safeguard features use current/past data, not retrospective labels."""
    joined = raw.merge(external[['station_id', 'hour_utc', 'r_temp', 'ref_temp']], on=['station_id', 'hour_utc'], how='left', validate='one_to_one')
    joined = joined.merge(spatial[['station_id', 'hour_utc', 'r_spatial_temp']], on=['station_id', 'hour_utc'], how='left', validate='one_to_one')
    frames = []
    for station, rows in joined.groupby('station_id', sort=True):
        r = rows.set_index('hour_utc').sort_index()
        r = r.reindex(pd.date_range(r.index.min(), r.index.max(), freq='h'))
        result = pd.DataFrame(index=r.index)
        for channel in ['temp_low_c', 'temp_avg_c', 'temp_high_c']:
            value = pd.to_numeric(r[channel], errors='coerce')
            center = value.groupby(value.index.hour).transform(lambda s: s.shift(1).rolling(30, min_periods=15).median())
            spread = value.groupby(value.index.hour).transform(lambda s: s.shift(1).rolling(30, min_periods=15).apply(mad, raw=True))
            result[channel + '_past_hour_z'] = (value - center) / spread.clip(lower=.1)
        for field, floor in [('r_temp', .3), ('r_spatial_temp', .5)]:
            value = pd.to_numeric(r[field], errors='coerce')
            history = value.shift(1).rolling(720, min_periods=240)
            result[field + '_past_z'] = (value - history.mean()) / history.std().clip(lower=floor)


        _, ev = features_for_station(r)
        protected = pd.Series(False, index=r.index)
        for channels in ev.values():
            protected |= channels['spike_impossible'] | channels['stuck_flatline']
        result['physical_or_confirmed_stuck'] = protected
        result['station_id'] = station
        result['hour_utc'] = result.index
        result['temp_avg_c'] = r.temp_avg_c
        result['ref_temp'] = r.ref_temp
        result['r_temp'] = r.r_temp
        frames.append(result.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def guard_masks(rows):
    thermal = ['temp_low_c_past_hour_z', 'temp_avg_c_past_hour_z', 'temp_high_c_past_hour_z']
    same_hour_normal = rows[thermal].notna().all(axis=1) & rows[thermal].abs().le(3.5).all(axis=1)
    eligible = (rows.temperature_stat.eq(True) & rows.other_stat.eq(False)
                & rows.physical_or_confirmed_stuck.eq(False)).fillna(False)
    ref_agrees = rows.r_temp_past_z.notna() & rows.r_temp_past_z.abs().lt(3)
    peers_agree = rows.r_spatial_temp_past_z.notna() & rows.r_spatial_temp_past_z.abs().lt(3)
    return {'unchanged': pd.Series(False, index=rows.index),
            'temperature_context_and_reference': eligible & same_hour_normal & ref_agrees,
            'temperature_context_reference_and_peers': eligible & same_hour_normal & ref_agrees & peers_agree}
