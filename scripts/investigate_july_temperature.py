"""Targeted fixed-policy shadow checks; no fitting or deployment changes."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.model.final_reason_codes import ROOT, JULY_RAW, JULY_REFS, JULY_GATE, JULY_OUTPUT, MODEL_DIR
from src.model.reason_code_rebuild import features_for_station, minimum_one, MECH, COMP, sha
from src.model.hourly_baseline import load_hourly_tensor, flatten_hourly_features, load_reason_code_manifest_splits, binary_metrics
from sklearn.metrics import precision_recall_fscore_support


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
        # Cached legacy stuck flags backfill an episode. Use the already-tested
        # current-hour causal primitives for protection instead of those flags.
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


def run(output):
    if output.exists():
        raise FileExistsError(output)
    spatial_path = ROOT / 'data/eval/july_2026_features/spatial_residuals.parquet'
    scores_path = ROOT / 'data/eval/july_2026_features/statistical_anomaly_scores.parquet'
    model_path = ROOT / 'data/hourly_detection/one_hour_final/models/evidence_fusion/selected_ef_hgb_random_01h.joblib'
    paths = [JULY_RAW, JULY_REFS, JULY_GATE, JULY_OUTPUT / 'reason_code_predictions.parquet', spatial_path, scores_path, model_path, MODEL_DIR / 'reason_heads.joblib']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    raw = pd.read_csv(JULY_RAW)
    raw.hour_utc = pd.to_datetime(raw.hour_utc, utc=True)
    external = pd.read_parquet(JULY_REFS).rename(columns={'time_utc': 'hour_utc'})
    spatial = pd.read_parquet(spatial_path).rename(columns={'time_utc': 'hour_utc'})
    for frame in (external, spatial):
        frame.hour_utc = pd.to_datetime(frame.hour_utc, utc=True)
    context = thermal_context(raw, external, spatial)
    cutoff = pd.Timestamp('2026-07-15T12:00Z')
    short = thermal_context(raw[raw.hour_utc.le(cutoff)], external[external.hour_utc.le(cutoff)], spatial[spatial.hour_utc.le(cutoff)])
    pd.testing.assert_frame_equal(context[context.hour_utc.le(cutoff)].reset_index(drop=True), short.reset_index(drop=True))
    print('Temperature/reference guard features pass all-station delete-future check', flush=True)
    scores = pd.read_parquet(scores_path, columns=['station_id', 'hour_utc', 'channel', 'flag_zscore', 'flag_iforest'])
    scores['temperature_stat'] = scores.channel.str.startswith('temp_') & (scores.flag_zscore | scores.flag_iforest)
    scores['other_stat'] = ~scores.channel.str.startswith('temp_') & (scores.flag_zscore | scores.flag_iforest)
    flags = scores.groupby(['station_id', 'hour_utc'])[['temperature_stat', 'other_stat']].max().reset_index()
    context = context.merge(flags, on=['station_id', 'hour_utc'], how='left', validate='one_to_one')
    z = load_hourly_tensor(ROOT / 'data/hourly_detection/one_hour_final/hourly_detection_01h.npz')
    x, _, _ = flatten_hourly_features(z)
    splits = load_reason_code_manifest_splits(z, ROOT / 'data/hourly_detection/hourly_baseline_split_manifest.csv')['random']
    model = joblib.load(model_path)['estimator']
    evaluations = {}
    for part in ('validation', 'test'):
        ii = splits[part]
        frame = pd.DataFrame(dict(station_id=z['station_id'][ii], hour_utc=pd.to_datetime(z['hour'][ii], utc=True), truth_fault=z['y_binary'][ii], random_probability=model.predict_proba(x[ii])[:, 1]))
        frame['random_prediction'] = frame.random_probability.ge(.3).astype(int)
        evaluations['development_random_' + part] = frame
    evaluations['july'] = pd.read_parquet(JULY_GATE)
    rows, ledgers = [], []
    for scope, frame in evaluations.items():
        joined = frame.merge(context, on=['station_id', 'hour_utc'], how='left', validate='one_to_one')
        for policy, suppress in guard_masks(joined).items():
            alert = joined.random_prediction.eq(1)
            predicted = alert & ~suppress
            removed = alert & suppress
            m = binary_metrics(joined.truth_fault.to_numpy(), predicted.to_numpy(float), .5)
            rows.append(dict(scope=scope, policy=policy, **m,
                false_alerts_removed=int((removed & joined.truth_fault.eq(0)).sum()),
                labelled_faults_lost=int((removed & joined.truth_fault.eq(1)).sum())))
            joined[policy + '_prediction'] = predicted
        if scope == 'july':
            ledgers.append(joined)
    july = ledgers[0]
    saved_reasons = pd.read_parquet(JULY_OUTPUT / 'reason_code_predictions.parquet')
    targets = pd.read_parquet(ROOT / 'data/eval/july_2026_reason_code_evaluation/aligned_reference.parquet')
    assert july[['station_id', 'hour_utc']].equals(targets[['station_id', 'hour_utc']])
    assert saved_reasons[['station_id', 'hour_utc']].equals(targets[['station_id', 'hour_utc']])
    bundle = joblib.load(MODEL_DIR / 'reason_heads.joblib')
    resolved = targets[MECH + COMP].any(axis=1)
    eligible = ~targets.truth_fault | resolved
    reason_rows = []
    for axis, labels in [('mechanism', MECH), ('component', COMP)]:
        probs = saved_reasons[[f'{axis}_score__{label}' for label in labels]].to_numpy(float)
        thresholds = np.array([bundle['heads'][f'{axis}:{label}']['threshold'] for label in labels])
        base = minimum_one(probs >= thresholds, probs, np.ones(len(labels), bool))
        for policy in guard_masks(july):
            prediction = base & july[policy + '_prediction'].to_numpy()[:, None]
            p, r, f, _ = precision_recall_fscore_support(targets.loc[eligible, labels], prediction[eligible], average='micro', zero_division=0)
            reason_rows.append(dict(axis=axis, policy=policy, precision=p, recall=r, f1=f,
                resolved_faults_lost=int((july.random_prediction.eq(1) & ~july[policy + '_prediction'] & resolved).sum())))
    # Equal-weight matched station/hour cells, not a claim about climate normals.
    summer = context[context.hour_utc.between(pd.Timestamp('2026-06-01', tz='UTC'), pd.Timestamp('2026-07-31T23:00', tz='UTC'))].copy()
    summer['month'], summer['utc_hour'] = summer.hour_utc.dt.month, summer.hour_utc.dt.hour
    paired = summer.groupby(['station_id', 'utc_hour', 'month'])[['temp_avg_c', 'ref_temp']].agg(['mean', 'count']).unstack('month')
    matched = paired[(paired[('temp_avg_c', 'count', 6)] >= 5) & (paired[('temp_avg_c', 'count', 7)] >= 5) &
                     (paired[('ref_temp', 'count', 6)] >= 5) & (paired[('ref_temp', 'count', 7)] >= 5)]
    deltas = pd.DataFrame({'station_delta_c': matched[('temp_avg_c', 'mean', 7)] - matched[('temp_avg_c', 'mean', 6)],
                          'reference_delta_c': matched[('ref_temp', 'mean', 7)] - matched[('ref_temp', 'mean', 6)]}).reset_index()
    audit = dict(input_hashes=hashes, production_unchanged=True, no_refitting=True, no_threshold_sweep=True,
        guard_delete_future_passed=True, guard_cutoff=str(cutoff),
        paired_station_hour_cells=len(deltas), paired_station_mean_july_minus_june=float(deltas.station_delta_c.mean()),
        paired_reference_mean_july_minus_june=float(deltas.reference_delta_c.mean()),
        method='Two fixed shadow vetoes: temperature-only statistical alert, no current physical/confirmed-stuck evidence, all three temperature channels within 3.5 MAD of previous 30 same-hour days (>=15); current external residual within 3 standard deviations of previous 720h residual history (>=240). Strict variant also requires analogous peer residual agreement. Missing required evidence preserves alert.',
        limitations='Exploratory: July already inspected. Development controls use existing random split, not newly trained chronological models. Archived reference/peer arrival-time availability unverified. Base binary predictions inherit legacy preprocessing and retrospective stuck flags; new guard causal protection does not establish causality of that base model. July-June warmth is not a long-term climate anomaly test.')
    assert hashes == {str(p.relative_to(ROOT)): sha(p) for p in paths}
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output / 'binary_comparison.csv', index=False)
    pd.DataFrame(reason_rows).to_csv(output / 'reason_comparison.csv', index=False)
    july.to_parquet(output / 'july_shadow_predictions.parquet', index=False)
    deltas.to_csv(output / 'paired_temperature_changes.csv', index=False)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(pd.DataFrame(reason_rows).to_string(index=False), flush=True)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/eval/july_temperature_investigation')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)
