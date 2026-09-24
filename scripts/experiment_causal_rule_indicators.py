"""Isolated HGB ablation of retrospective stuck flags and rule families."""
from pathlib import Path
import sys
import os
import json
import hashlib
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.model.hourly_baseline import (load_hourly_tensor, filter_eligible_examples,
    flatten_hourly_features, _fault_groups, HourlyBaselineConfig)
from src.model.hourly_calibration import calibrate_split, validation_grid_frame
from src.model.feature_spec import RULE_EVIDENCE_FLAGS
from src.rules.score import _build_scored_wide_frame
from src.rules.channel_handlers import sensor_group_for_channel
from src.rules.config import STUCK_IGNORE_ZERO_CHANNELS, STUCK_SKIP_CHANNELS
from scripts.experiment_blocked_fault_detection import blocked_split, validate_split


def causal_stuck(series, channel):
    if channel in STUCK_SKIP_CHANNELS:
        return pd.Series(False, index=series.index)
    result = series.rolling(24, min_periods=24).var().lt(1e-6).fillna(False)
    if channel in STUCK_IGNORE_ZERO_CHANNELS:
        result &= series.abs().rolling(24, min_periods=24).mean().gt(1e-9).fillna(False)
    return result


def run(prepare_only=False):
    out = ROOT / 'data/eval/causal_rule_indicators_blocked_20260923'
    if not prepare_only:
        out.mkdir(parents=True, exist_ok=False)
    # Deleting future observations must not change earlier causal flags.
    synthetic = pd.Series(np.r_[np.arange(12), np.repeat(7., 50), np.arange(20)])
    for cut in (12, 25, 36, 50, 70):
        pd.testing.assert_series_equal(causal_stuck(synthetic, 'windspeed_avg_kmh').iloc[:cut],
                                      causal_stuck(synthetic.iloc[:cut], 'windspeed_avg_kmh'))
    examples, _ = filter_eligible_examples(load_hourly_tensor(ROOT / 'data/hourly_detection/one_hour_final/hourly_detection_01h.npz'))
    x, names, fg = flatten_hourly_features(examples)
    y = np.asarray(examples['y_binary'], int)
    hours = pd.to_datetime(examples['hour'], utc=True)
    groups = np.array([f'normal:{s}:{str(h)[:10]}' for s,h in zip(examples['station_id'], hours)], dtype=object)
    for key, indices in _fault_groups(y, examples['source_episode_ids']).items():
        groups[indices] = 'event:' + key
    splits = blocked_split(hours, groups)
    validate_split(hours, groups, splits)
    retained = np.concatenate([splits[k] for k in ('train','validation','test')])
    digests = {k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in splits.items()}
    old_design = json.loads((ROOT / 'data/eval/no_rule_indicators_blocked_20260923/design.json').read_text())
    assert digests == old_design['partition_index_sha256']
    remap = np.full(len(y), -1, int)
    remap[retained] = np.arange(len(retained))
    splits = {k: remap[v] for k,v in splits.items()}
    index = pd.MultiIndex.from_arrays([examples['station_id'], hours], names=['station_id','hour_utc'])
    scores = pd.read_parquet(ROOT / 'data/processed/statistical_anomaly_scores.parquet',
        columns=['station_id','hour_utc','channel','flag_stuck','flag_physical','flag_physical_suspect','flag_zscore','flag_iforest'])
    scores.hour_utc = pd.to_datetime(scores.hour_utc, utc=True)
    raw = pd.read_csv(ROOT / 'data/merged/station_hourly_merged.csv')
    raw.hour_utc = pd.to_datetime(raw.hour_utc, utc=True)
    channels = list(scores.channel.unique())
    raw_channels = [c for c in channels if c not in ('winddir_sin','winddir_cos')] + ['winddir_avg_deg']
    wide = _build_scored_wide_frame(raw, raw_channels)
    frames = []
    for station, part in wide.groupby('station_id', sort=False):
        part = part.set_index('hour_utc').sort_index()
        part = part.reindex(pd.date_range(part.index.min(), part.index.max(), freq='h', name='hour_utc'))
        flags = pd.DataFrame({c: causal_stuck(part[c], c) for c in channels}, index=part.index)
        # Actual-data prefix check for every channel at the middle of the station record.
        cut = len(part)//2
        for c in channels:
            pd.testing.assert_series_equal(flags[c].iloc[:cut], causal_stuck(part[c].iloc[:cut], c), check_names=False)
        flags['station_id'] = station
        frames.append(flags.reset_index().melt(id_vars=['station_id','hour_utc'], var_name='channel', value_name='causal_stuck'))
    scores = scores.merge(pd.concat(frames, ignore_index=True), on=['station_id','hour_utc','channel'], validate='one_to_one')
    scores['group'] = scores.channel.map(sensor_group_for_channel)
    scores['statistical'] = scores.flag_zscore | scores.flag_iforest
    scores['old_any'] = scores.statistical | scores.flag_stuck | scores.flag_physical
    scores['causal_any'] = scores.statistical | scores.causal_stuck | scores.flag_physical
    scores['causal_suspect'] = scores.flag_physical_suspect & scores.causal_any

    def rule_frame(mode):
        output = pd.DataFrame(index=index)
        group_column = 'old_any' if mode == 'original' else ('causal_any' if mode == 'causal_all' else 'statistical')
        grouped = scores.groupby(['station_id','hour_utc','group'])[group_column].max().unstack('group')
        for flag in RULE_EVIDENCE_FLAGS:
            if flag.startswith('stat_sensor_group_flag_'):
                group = flag.removeprefix('stat_sensor_group_flag_')
                series = grouped[group] if group in grouped else pd.Series(False, index=grouped.index)
            else:
                prefix = next(p for p in ('stat_flag_physical_suspect_', 'stat_flag_physical_', 'stat_flag_stuck_') if flag.startswith(p))
                channel = flag.removeprefix(prefix)
                column = prefix.removeprefix('stat_').removesuffix('_')
                if mode != 'original':
                    column = {'flag_stuck':'causal_stuck','flag_physical_suspect':'causal_suspect'}.get(column,column)
                series = scores.loc[scores.channel.eq(channel)].set_index(['station_id','hour_utc'])[column]
            output[flag] = series.reindex(index).astype('boolean').fillna(False).to_numpy(dtype=float)
        return output

    original_rules = rule_frame('original')
    for flag in RULE_EVIDENCE_FLAGS:
        for suffix in ('any','rate'):
            np.testing.assert_array_equal(x[:, names.index(flag+'_'+suffix)], original_rules[flag].to_numpy())
    print('Original rule inputs reproduced; causal stuck flags passed prefix checks.', flush=True)
    base = sorted(i for key,indices in fg.items() if not key.startswith('rule:') for i in indices)
    if prepare_only:
        return dict(x=x, names=names, base=base, rules=rule_frame('causal_all'),
                    y=y, index=index, splits=splits, retained=retained, raw=raw, digests=digests)
    baseline = pd.read_csv(ROOT / 'data/eval/no_rule_indicators_blocked_20260923/comparison.csv')
    rows = baseline.to_dict(orient='records')
    variants = [('causal_all','Causal stuck + existing hard/suspect + rebuilt groups'),
        ('causal_stat_stuck','Causal stuck + statistical-only groups'),
        ('stat_only','Statistical-only groups')]
    changes = {}
    for mode, label in variants:
        rules = rule_frame(mode)
        selected = RULE_EVIDENCE_FLAGS if mode == 'causal_all' else [f for f in RULE_EVIDENCE_FLAGS
            if f.startswith('stat_sensor_group_flag_') or (mode == 'causal_stat_stuck' and f.startswith('stat_flag_stuck_'))]
        values = x.copy()
        keep = list(base)
        for flag in selected:
            for suffix in ('any','rate'):
                i = names.index(flag+'_'+suffix)
                values[:,i] = rules[flag].to_numpy()
                keep.append(i)
        keep.sort()
        changes[mode] = {flag: int((rules[flag] != original_rules[flag]).sum()) for flag in selected}
        values = values[retained][:,keep]
        result = calibrate_split(values, y[retained], splits, HourlyBaselineConfig())
        choice = result['best_balanced']
        row = {'model': label, 'input_columns': len(keep), 'threshold': choice['threshold'], 'class_weight': choice['fault_class_weight']}
        for partition, metrics in [('validation',choice['validation']),('test',result['final_test'])]:
            row.update({f'{partition}_{k}':metrics[k] for k in ('precision','recall','f1','accuracy')})
        rows.append(row)
        validation_grid_frame(result).to_csv(out / f'{mode}_validation_grid.csv', index=False)
        (out / f'{mode}_features.json').write_text(json.dumps([names[i] for i in keep], indent=2), encoding='utf-8')
        pd.DataFrame(rows).to_csv(out / 'comparison.csv', index=False)
        print(json.dumps(row), flush=True)
    (out/'design.json').write_text(json.dumps({'partition_index_sha256':digests,
        'partition_counts':{k:len(v) for k,v in splits.items()}, 'changed_indicators':changes,
        'original_input_parity_verified':True, 'causal_stuck_prefix_checks_passed':True,
        'stuck_definition':'Trailing 24 hourly slots, variance below 1e-6; original zero/skip exceptions; no backfill; missing clock slots explicitly reindexed',
        'statistical_groups':'OR of cached robust-z and Isolation Forest flags only; no hard or stuck flags',
        'selection':'Original HGB grid and validation maximin rule; test not used for selection',
        'limitations':'Labels, continuous features and cached statistical detector fitting unchanged. Not a full causal preprocessing audit or independent hardware validation. Previously examined test period.',
        'deployment_changed':False}, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
