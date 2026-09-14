"""Small pre-July-selected reason-only tests; never change the binary gate."""
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
from scripts.investigate_july_temperature import thermal_context
from scripts.test_temperature_refinement import add_continuity
from src.model.final_reason_codes import (ROOT, FREEZE, MODEL_DIR, JULY_RAW,
    JULY_REFS, JULY_GATE, JULY_OUTPUT, load_observations, build_features)
from src.model.reason_code_rebuild import MECH, COMP, probability, fit_estimator, sha
from src.model.hourly_baseline import (load_hourly_tensor, flatten_hourly_features,
    load_reason_code_manifest_splits)


def fallback(scores, thresholds, gate, floor=0., margin=0., enabled=True):
    """Keep threshold codes; finite top score may fill an empty alerted axis."""
    scores = np.asarray(scores, float)
    result = (scores >= thresholds) & np.asarray(gate, bool)[:, None]
    if enabled:
        safe = np.where(np.isfinite(scores), scores, -np.inf)
        top = safe.argmax(axis=1)
        ranked = np.sort(safe, axis=1)
        best, runner = ranked[:, -1], ranked[:, -2]
        gap = np.zeros(len(scores))
        finite = np.isfinite(best)
        gap[finite] = best[finite] - runner[finite]
        fill = gate & ~result.any(axis=1) & finite & (best >= floor) & (gap >= margin)
        result[np.flatnonzero(fill), top[fill]] = True
    return result


def counts(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp, fp, fn = int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum())
    return dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
        f1=2 * tp / max(2 * tp + fp + fn, 1), tp=tp, fp=fp, fn=fn)


def weather_codes(scores, thresholds, gate, j, weather_score, weight, threshold):
    updated = scores.copy()
    if weight > 0:
        updated[:, j] = (1 - weight) * scores[:, j] + weight * weather_score
    limits = thresholds.copy()
    limits[j] = threshold if weight > 0 else thresholds[j]
    return fallback(updated, limits, gate, enabled=False), updated, limits


def context_frame(raw_path, refs_path, spatial_path, cutoff):
    raw = pd.read_csv(raw_path)
    raw.hour_utc = pd.to_datetime(raw.hour_utc, utc=True)
    refs = pd.read_parquet(refs_path).rename(columns={'time_utc': 'hour_utc'})
    spatial = pd.read_parquet(spatial_path).rename(columns={'time_utc': 'hour_utc'})
    for frame in (refs, spatial):
        frame.hour_utc = pd.to_datetime(frame.hour_utc, utc=True)
    return add_continuity(thermal_context(raw[raw.hour_utc.le(cutoff)],
        refs[refs.hour_utc.le(cutoff)], spatial[spatial.hour_utc.le(cutoff)]))


def run(out):
    if out.exists():
        raise FileExistsError(out)
    dev = ROOT / 'data/eval/reason_code_rebuild_20260912_v2'
    raw = ROOT / 'data/merged/station_hourly_merged.csv'
    refs = ROOT / 'data/features/external_residuals.parquet'
    spatial = ROOT / 'data/features/spatial_residuals.parquet'
    j_spatial = ROOT / 'data/eval/july_2026_features/spatial_residuals.parquet'
    binary_model = ROOT / 'data/hourly_detection/one_hour_final/models/evidence_fusion/selected_ef_hgb_random_01h.joblib'
    tensor = ROOT / 'data/hourly_detection/one_hour_final/hourly_detection_01h.npz'
    protected = [raw, refs, spatial, j_spatial, JULY_RAW, JULY_REFS, JULY_GATE,
        JULY_OUTPUT / 'reason_code_predictions.parquet', MODEL_DIR / 'reason_heads.joblib',
        binary_model, tensor, dev / 'aligned_reference.parquet',
        ROOT / 'src/dashboard/replay.py', ROOT / 'scripts/run_dashboard.py']
    protected += sorted((dev / 'models').glob('random_*.joblib'))
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    z = load_hourly_tensor(tensor)
    index = pd.MultiIndex.from_arrays([z['station_id'], pd.to_datetime(z['hour'], utc=True)], names=['station_id', 'hour'])
    assert index.get_level_values('hour').max() <= FREEZE
    splits = load_reason_code_manifest_splits(z, ROOT / 'data/hourly_detection/hourly_baseline_split_manifest.csv')['random']
    ii = splits['validation']
    target = pd.read_parquet(dev / 'aligned_reference.parquet')
    target.hour = pd.to_datetime(target.hour, utc=True)
    target = target.set_index(['station_id', 'hour']).reindex(index).iloc[ii]
    assert not target[MECH + COMP + ['original_fault']].isna().any().any()
    valid = (~target.original_fault.astype(bool) | target[MECH + COMP].any(axis=1)).to_numpy()
    membership = pd.read_csv(dev / 'split_membership.csv')
    membership = membership[(membership.scheme == 'random') & (membership.partition == 'validation')].set_index('row_index')
    groups = membership.reindex(ii).group.to_numpy(str)
    bx, names, _ = flatten_hourly_features(z)
    detector = joblib.load(binary_model)
    assert names == detector['feature_names']
    gate = detector['estimator'].predict_proba(bx[ii])[:, 1] >= detector['config']['threshold']
    del bx
    features, _ = build_features(load_observations(raw, refs, FREEZE))
    x = features.reindex(index[ii]).to_numpy('float32')
    feature_names = features.columns.tolist()
    del features
    probabilities, thresholds = {}, {}
    for axis, labels in [('mechanism', MECH), ('component', COMP)]:
        pp, tt = [], []
        for label in labels:
            head = joblib.load(dev / 'models' / f'random_{axis}_{label}.joblib')
            assert head['feature_names'] == feature_names
            pp.append(np.column_stack([probability(m, x[:, cols]) for m, cols in zip(head['models'], head['views'], strict=True)]) @ head['weights'])
            tt.append(head['threshold'])
        probabilities[axis], thresholds[axis] = np.column_stack(pp), np.array(tt)
    del x
    print('Held-out development-validation reason scores reproduced; no base-model fitting', flush=True)
    selected, choices = {}, []
    for axis, labels in [('mechanism', MECH), ('component', COMP)]:
        y, p, t = target[labels].to_numpy(bool), probabilities[axis], thresholds[axis]
        candidates = [dict(enabled=False, floor=0., margin=0.)]
        candidates += [dict(enabled=True, floor=f, margin=m) for f in (0., .3, .5, .7) for m in (0., .1, .2, .3)]
        rows = []
        for setting in candidates:
            metric = counts(y[valid], fallback(p, t, gate, **setting)[valid])
            rows.append(dict(axis=axis, method='selective_fallback', **setting, **metric))
        best = max(rows, key=lambda r: (r['f1'], not r['enabled'], r['floor'], r['margin']))
        selected[axis] = dict(fallback={k: best[k] for k in ('enabled', 'floor', 'margin')}, fallback_selection_f1=best['f1'])
        choices.extend(rows)
    print('Fallback settings frozen from pre-July validation: ' + json.dumps(selected), flush=True)

    context = context_frame(raw, refs, spatial, FREEZE)
    weather_names = [c for c in context if c not in ('station_id', 'hour_utc', 'physical_or_confirmed_stuck')]
    wx = context.set_index(['station_id', 'hour_utc']).reindex(index[ii])[weather_names].to_numpy('float32')
    trainrows = np.flatnonzero(valid & gate)
    assert len(np.unique(groups[trainrows])) >= 3
    models = {}
    for axis, labels, label in [('mechanism', MECH, 'statistical_anomaly'), ('component', COMP, 'thermo_hygrometer')]:
        p, t = probabilities[axis], thresholds[axis]
        j = labels.index(label)
        # Same-axis base scores plus weather inputs; no station IDs, labels or episode IDs.
        xx = np.column_stack([p, wx]).astype('float32')
        y = target[labels].to_numpy(bool)
        oof = np.full(len(ii), np.nan)
        for train, hold in GroupKFold(3).split(trainrows, groups=groups[trainrows]):
            fi, oi = trainrows[train], trainrows[hold]
            assert not set(groups[fi]) & set(groups[oi])
            model = fit_estimator(xx[fi], y[fi, j].astype(int), groups[fi])
            oof[oi] = probability(model, xx[oi])
        rows = []
        for weight in (0., .25, .5, 1.):
            for threshold in ([float(t[j])] if weight == 0 else sorted(set([float(t[j]), .3, .5, .7, .9]))):
                prediction, _, _ = weather_codes(p, t, gate, j, oof, weight, threshold)
                rows.append(dict(axis=axis, method='weather_refinement', weight=weight, threshold=threshold,
                    **counts(y[valid], prediction[valid])))
        best = max(rows, key=lambda r: (r['f1'], -r['weight'], -abs(r['threshold'] - t[j])))
        selected[axis]['weather'] = {k: best[k] for k in ('weight', 'threshold')}
        selected[axis]['weather_selection_f1'] = best['f1']
        selected[axis]['weather_label'] = label
        choices.extend(rows)
        models[axis] = fit_estimator(xx[trainrows], y[trainrows, j].astype(int), groups[trainrows])
        print(f'{axis} weather head frozen: {best}', flush=True)
    # All choices are now fixed. Only now load July targets and scores.
    july = pd.read_parquet(ROOT / 'data/eval/july_2026_reason_code_evaluation/aligned_reference.parquet')
    deployed = pd.read_parquet(JULY_OUTPUT / 'reason_code_predictions.parquet')
    binary = pd.read_parquet(JULY_GATE)
    for frame in (july, binary):
        assert frame[['station_id', 'hour_utc']].equals(deployed[['station_id', 'hour_utc']])
    assert np.array_equal(deployed.random_prediction, binary.random_prediction)
    assert np.array_equal(deployed.random_probability, binary.random_probability)
    jgate = deployed.random_prediction.eq(1).to_numpy()
    jvalid = (~july.truth_fault.astype(bool) | july[MECH + COMP].any(axis=1)).to_numpy()
    jc = context_frame(JULY_RAW, JULY_REFS, j_spatial, deployed.hour_utc.max())
    # Rebuild weather features on a truncated input history for every station.
    cutoff = pd.Timestamp('2026-07-15T12:00Z')
    # Recompute full context prefix from original inputs, not from precomputed rolling values.
    prefix = context_frame(JULY_RAW, JULY_REFS, j_spatial, cutoff)
    pd.testing.assert_frame_equal(jc[jc.hour_utc.le(cutoff)].reset_index(drop=True), prefix.reset_index(drop=True))
    ji = pd.MultiIndex.from_frame(deployed[['station_id', 'hour_utc']])
    jwx = jc.set_index(['station_id', 'hour_utc']).reindex(ji)[weather_names].to_numpy('float32')
    bundle = joblib.load(MODEL_DIR / 'reason_heads.joblib')
    summary, ledgers = [], deployed[['station_id', 'hour_utc', 'random_probability', 'random_prediction']].copy()
    for axis, labels in [('mechanism', MECH), ('component', COMP)]:
        p = deployed[[f'{axis}_score__{label}' for label in labels]].to_numpy(float)
        t = np.array([bundle['heads'][f'{axis}:{label}']['threshold'] for label in labels])
        wscore = np.full(len(july), np.nan)
        wscore[jgate] = probability(models[axis], np.column_stack([p, jwx])[jgate].astype('float32'))
        j = labels.index(selected[axis]['weather_label'])
        weather, wp, wt = weather_codes(p, t, jgate, j, wscore, **selected[axis]['weather'])
        predictions = dict(deployed_threshold=fallback(p, t, jgate, enabled=False),
            minimum_one=fallback(p, t, jgate),
            selective_fallback=fallback(p, t, jgate, **selected[axis]['fallback']),
            weather_refinement=weather,
            weather_plus_selective_fallback=fallback(wp, wt, jgate, **selected[axis]['fallback']))
        expected = [' | '.join(np.asarray(labels)[r]) for r in predictions['deployed_threshold']]
        assert expected == deployed[f'likely_{axis}s'].fillna('').tolist()
        for policy, pred in predictions.items():
            assert not pred[~jgate].any()
            summary.append(dict(axis=axis, policy=policy, population=int(jvalid.sum()),
                assigned_alerts=int(pred.any(axis=1).sum()), **counts(july.loc[jvalid, labels], pred[jvalid])))
            ledgers[f'{policy}_{axis}'] = [' | '.join(np.asarray(labels)[r]) for r in pred]
        ledgers[f'{axis}_weather_score'] = wscore
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in protected}
    audit = dict(selected=selected, input_hashes=before, production_unchanged=True,
        binary_decisions_unchanged=True, prefix_invariant=True, validation_hours=len(ii),
        weather_training_alerts=len(trainrows), unknown_july_hours_excluded=int((~jvalid).sum()),
        weather_features=weather_names, threads=2,
        protocol='Fallback selected on original random validation. Weather heads use 3-fold event-disjoint OOF within resolved/normal validation alerts; blend and threshold selected on validation cascade micro-F1, then refit those alerts. No July fitting or parameter selection.',
        limitations='Exploratory reused development data. Original random base-model split shares events across train/validation; grouped meta folds do not remove that. Policies learned from split-specific reason heads transfer to the full-June refit, whose score distributions/thresholds differ. Validation selection scores are not independent test accuracy. Archived weather arrival times unverified; evidence-derived labels, 843 unknown July fault hours excluded. Combined variant is a fixed composition, not separately tuned.')
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(choices).to_csv(out / 'development_selection.csv', index=False)
    ledgers.to_parquet(out / 'predictions.parquet', index=False)
    joblib.dump(dict(models=models, selected=selected, weather_features=weather_names), out / 'weather_reason_refiners.joblib')
    (out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/eval/july_reason_only_refinements')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)
