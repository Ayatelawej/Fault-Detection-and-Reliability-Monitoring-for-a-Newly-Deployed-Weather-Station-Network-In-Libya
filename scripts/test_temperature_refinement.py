"""One isolated temperature specialist plus score-neutral annotation preview."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from scripts.investigate_july_temperature import thermal_context, guard_masks
from src.model.final_reason_codes import ROOT, JULY_RAW, JULY_REFS, JULY_GATE, JULY_OUTPUT, MODEL_DIR
from src.model.reason_code_rebuild import fit_estimator, probability, event_weights, sha, minimum_one, MECH, COMP
from src.model.hourly_baseline import load_hourly_tensor, _fault_groups, binary_metrics


def add_continuity(context):
    """Continuity from prior observed evidence, never true labels or episode IDs."""
    frames = []
    for _, g in context.groupby('station_id', sort=True):
        g = g.sort_values('hour_utc').copy()
        g['temperature_change_1h'] = g.temp_avg_c.diff()
        g['temperature_change_24h'] = g.temp_avg_c.diff(24)
        g['residual_change_1h'] = g.r_temp.diff()
        strong = g.temp_avg_c_past_hour_z.abs().gt(3.5) & (g.r_temp_past_z.abs().ge(3) | g.r_spatial_temp_past_z.abs().ge(3))
        for window in (6, 24):
            g[f'prior_supported_temperature_{window}h'] = strong.shift(1).rolling(window, min_periods=1).sum()
            g[f'prior_physical_stuck_{window}h'] = g.physical_or_confirmed_stuck.astype(float).shift(1).rolling(window, min_periods=1).sum()
        g['prior_temperature_coverage_24h'] = g.temp_avg_c.notna().astype(float).shift(1).rolling(24, min_periods=1).mean()
        for period, values in [(24, g.hour_utc.dt.hour), (366, g.hour_utc.dt.dayofyear)]:
            g[f'calendar_sin_{period}'] = np.sin(2 * np.pi * values / period)
            g[f'calendar_cos_{period}'] = np.cos(2 * np.pi * values / period)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def eligibility(frame):
    return (frame.temperature_stat.eq(True) & frame.other_stat.eq(False) & frame.physical_or_confirmed_stuck.eq(False)).fillna(False)


def run(out):
    if out.exists():
        raise FileExistsError(out)
    scores_path = ROOT / 'data/eval/july_2026_features/statistical_anomaly_scores.parquet'
    spatial_path = ROOT / 'data/eval/july_2026_features/spatial_residuals.parquet'
    tensor_path = ROOT / 'data/hourly_detection/one_hour_final/hourly_detection_01h.npz'
    protected = [JULY_RAW, JULY_REFS, JULY_GATE, JULY_OUTPUT / 'reason_code_predictions.parquet', scores_path, spatial_path, tensor_path, MODEL_DIR / 'reason_heads.joblib']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    raw = pd.read_csv(JULY_RAW)
    raw.hour_utc = pd.to_datetime(raw.hour_utc, utc=True)
    ext = pd.read_parquet(JULY_REFS).rename(columns={'time_utc': 'hour_utc'})
    spatial = pd.read_parquet(spatial_path).rename(columns={'time_utc': 'hour_utc'})
    for frame in (ext, spatial):
        frame.hour_utc = pd.to_datetime(frame.hour_utc, utc=True)
    context = thermal_context(raw, ext, spatial)
    full = add_continuity(context)
    cutoff = pd.Timestamp('2026-07-15T12:00Z')
    short = add_continuity(context[context.hour_utc.le(cutoff)])
    pd.testing.assert_frame_equal(full[full.hour_utc.le(cutoff)].reset_index(drop=True), short.reset_index(drop=True))
    names = [c for c in full if c not in ('station_id', 'hour_utc', 'physical_or_confirmed_stuck')]
    scores = pd.read_parquet(scores_path, columns=['station_id', 'hour_utc', 'channel', 'flag_zscore', 'flag_iforest'])
    stat = scores.flag_zscore | scores.flag_iforest
    scores['temperature_stat'] = scores.channel.str.startswith('temp_') & stat
    scores['other_stat'] = ~scores.channel.str.startswith('temp_') & stat
    flags = scores.groupby(['station_id', 'hour_utc'])[['temperature_stat', 'other_stat']].max().reset_index()
    full = full.merge(flags, on=['station_id', 'hour_utc'], how='left', validate='one_to_one')
    print('Causal specialist features and continuity assembled', flush=True)
    z = load_hourly_tensor(tensor_path)
    index = pd.MultiIndex.from_arrays([z['station_id'], pd.to_datetime(z['hour'], utc=True)], names=['station_id', 'hour_utc'])
    june = full.set_index(['station_id', 'hour_utc']).reindex(index)
    assert index.get_level_values('hour_utc').max() < pd.Timestamp('2026-07-01', tz='UTC')
    y = np.asarray(z['y_binary'], int)
    groups = np.array([f'normal:{s}:{h:%Y-%m-%d}' for s, h in index], dtype=object)
    for key, ii in _fault_groups(y, z['source_episode_ids']).items():
        groups[ii] = 'event:' + key
    ii = np.flatnonzero(eligibility(june))
    x, target, group = june.iloc[ii][names].to_numpy('float32'), y[ii], groups[ii]
    assert len(np.unique(target)) == 2
    oof, folds = np.zeros(len(ii)), np.zeros(len(ii), int)
    for fold, (train, test) in enumerate(GroupKFold(3).split(x, target, group)):
        assert not set(group[train]) & set(group[test])
        model = fit_estimator(x[train], target[train], group[train])
        oof[test] = probability(model, x[test])
        folds[test] = fold
    weights = event_weights(group)
    choices = []
    for threshold in [0.] + list(np.arange(.05, 1., .05)):
        fold_scores = []
        for fold in range(3):
            mask = folds == fold
            pp, yy, ww = oof[mask] >= threshold, target[mask], weights[mask]
            tp, fp, fn = ww[(yy == 1) & pp].sum(), ww[(yy == 0) & pp].sum(), ww[(yy == 1) & ~pp].sum()
            fold_scores.append(2 * tp / max(2 * tp + fp + fn, 1e-12))
        choices.append(dict(threshold=float(threshold), mean_grouped_oof_event_f1=float(np.mean(fold_scores))))
    selected = max(choices, key=lambda r: (r['mean_grouped_oof_event_f1'], -r['threshold']))
    specialist = fit_estimator(x, target, group)
    print('Specialist frozen using June only: ' + json.dumps(selected), flush=True)

    # July labels enter only here for scoring, after model/policy selection.
    gate = pd.read_parquet(JULY_GATE)
    july = gate[['station_id', 'hour_utc']].merge(full, on=['station_id', 'hour_utc'], how='left', validate='one_to_one')
    active = gate.random_prediction.eq(1).to_numpy()
    applies = eligibility(july).to_numpy() & active
    specialist_score = np.full(len(july), np.nan)
    specialist_score[applies] = probability(specialist, july.loc[applies, names].to_numpy('float32'))
    suppressed = applies & (specialist_score < selected['threshold'])
    refined = active & ~suppressed
    annotations = guard_masks(july)['temperature_context_reference_and_peers'].to_numpy() & active
    annotated = gate[['station_id', 'hour_utc', 'random_probability', 'random_prediction']].copy()
    annotated['weather_consistent_temperature'] = annotations
    annotated['weather_note'] = np.where(annotations, 'Temperature pattern consistent with recent history, reference weather and neighbours. Alert retained; sensor fault not ruled out.', '')
    assert annotated.random_prediction.equals(gate.random_prediction)
    assert annotated.random_probability.equals(gate.random_probability)
    rows = []
    for policy, pp in [('unchanged', active), ('annotation_only', active), ('temperature_specialist', refined)]:
        rows.append(dict(policy=policy, **binary_metrics(gate.truth_fault.to_numpy(), pp.astype(float), .5)))
    reasons = pd.read_parquet(JULY_OUTPUT / 'reason_code_predictions.parquet')
    refs = pd.read_parquet(ROOT / 'data/eval/july_2026_reason_code_evaluation/aligned_reference.parquet')
    assert gate[['station_id', 'hour_utc']].equals(refs[['station_id', 'hour_utc']])
    assert gate[['station_id', 'hour_utc']].equals(reasons[['station_id', 'hour_utc']])
    bundle = joblib.load(MODEL_DIR / 'reason_heads.joblib')
    valid = ~refs.truth_fault | refs[MECH + COMP].any(axis=1)
    from sklearn.metrics import precision_recall_fscore_support
    reason_rows = []
    for axis, labels in [('mechanism', MECH), ('component', COMP)]:
        p = reasons[[f'{axis}_score__{label}' for label in labels]].to_numpy(float)
        thresholds = np.array([bundle['heads'][f'{axis}:{label}']['threshold'] for label in labels])
        codes = minimum_one(p >= thresholds, p, np.ones(len(labels), bool))
        for policy, pp in [('unchanged_minimum_one', active), ('specialist_minimum_one', refined)]:
            precision, recall, f1, _ = precision_recall_fscore_support(refs.loc[valid, labels], (codes & pp[:, None])[valid], average='micro', zero_division=0)
            reason_rows.append(dict(axis=axis, policy=policy, precision=precision, recall=recall, f1=f1))
    report = dict(input_hashes=hashes, selected=selected, june_candidate_hours=len(ii), june_fault_candidates=int(target.sum()),
        feature_names=names, baseline_oof_event_f1=choices[0]['mean_grouped_oof_event_f1'],
        july_eligible_alerts=int(applies.sum()), false_alerts_removed=int((suppressed & gate.truth_fault.eq(0)).sum()),
        labelled_faults_lost=int((suppressed & gate.truth_fault.eq(1)).sum()), annotation_hours=int(annotations.sum()),
        annotation_does_not_change_scores=True, continuity_prefix_invariant=True, production_unchanged=True,
        limitations='Exploratory specialist only; fixed June-frozen upstream statistical flags remain inherited, so not a fully chronological preprocessing validation. No July tuning. OOF threshold objective is temperature-candidate event F1, not independently held-out full-system F1. Archived weather/peer arrival times unverified. No model-based continuity or true event history is an input.')
    assert hashes == {str(p.relative_to(ROOT)): sha(p) for p in protected}
    out.mkdir(parents=True, exist_ok=False)
    annotated.to_parquet(out / 'weather_annotations.parquet', index=False)
    diag = annotated[['station_id', 'hour_utc']].copy()
    diag['specialist_score'], diag['specialist_applies'], diag['specialist_prediction'] = specialist_score, applies, refined
    diag.to_parquet(out / 'specialist_predictions.parquet', index=False)
    pd.DataFrame(rows).to_csv(out / 'binary_results.csv', index=False)
    pd.DataFrame(reason_rows).to_csv(out / 'reason_results.csv', index=False)
    pd.DataFrame(choices).to_csv(out / 'june_threshold_selection.csv', index=False)
    joblib.dump(dict(model=specialist, feature_names=names, **selected), out / 'specialist.joblib')
    (out / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(pd.DataFrame(reason_rows).to_string(index=False), flush=True)
    print(json.dumps({k: v for k, v in report.items() if k not in ('input_hashes', 'feature_names')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/eval/temperature_specialist_check')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)
