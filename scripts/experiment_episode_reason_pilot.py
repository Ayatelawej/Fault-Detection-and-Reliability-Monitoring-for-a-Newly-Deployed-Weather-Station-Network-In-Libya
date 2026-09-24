"""Isolated matched-label pilot; February selection, March-April test."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import sys
import json
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.model.hourly_baseline import load_hourly_tensor, _fault_groups
from src.model.final_reason_codes import load_observations, build_features, FREEZE, apply_output_policy
from src.model.reason_code_rebuild import (MECH, COMP, feature_views, fit_estimator,
    probability, select_policy, event_weights, multilabel_rows, sha)
from scripts.experiment_blocked_reason_codes import training_rows


def run():
    started = time.perf_counter()
    out = ROOT / 'data/eval/episode_reason_pilot_20260924'
    out.mkdir(exist_ok=False)
    protected = [ROOT / 'data/model/reason_codes/final/reason_heads.joblib',
        ROOT / 'data/eval/july_2026_reason_codes_mixed_v2/reason_code_predictions.parquet']
    before = [sha(p) for p in protected]
    z = load_hourly_tensor(ROOT / 'data/hourly_detection/one_hour_final/hourly_detection_01h.npz')
    index = pd.MultiIndex.from_arrays([z['station_id'], pd.to_datetime(z['hour'], utc=True)],
                                    names=['station_id', 'hour'])
    fault = np.asarray(z['y_binary'], int)
    groups = np.array([f'normal:{s}:{str(t)[:10]}' for s,t in index], dtype=object)
    for k, ii in _fault_groups(fault, z['source_episode_ids']).items():
        groups[ii] = 'event:' + k
    membership = pd.read_csv(ROOT / 'data/eval/blocked_ef_hgb_20260914/split_membership.csv')
    splits = {p: membership.loc[membership.partition.eq(p), 'row_index'].to_numpy(int)
              for p in ('train', 'validation', 'test')}
    train, val, test = (splits[p] for p in ('train', 'validation', 'test'))
    assert not set(groups[train]) & set(groups[test])
    assert not set(groups[train]) & set(groups[val])
    ref = pd.read_parquet(ROOT / 'data/eval/reason_code_rebuild_20260912_v2/aligned_reference.parquet')
    ref.hour = pd.to_datetime(ref.hour, utc=True)
    ref = ref.set_index(['station_id', 'hour']).reindex(index)
    assert not ref[MECH+COMP].isna().any().any()
    features, _ = build_features(load_observations(ROOT / 'data/merged/station_hourly_merged.csv',
        ROOT / 'data/features/external_residuals.parquet', FREEZE))
    assert index.isin(features.index).all()
    names = list(features.columns)
    x = features.reindex(index).to_numpy('float32')
    del features
    results, support, selections, details = [], [], [], []
    for axis, labels in [('mechanism', MECH), ('component', COMP)]:
        original_names = list(z[axis+'_label_names'])
        episode = np.asarray(z['y_'+axis], int)[:, [original_names.index(n) for n in labels]]
        current = ref[labels].to_numpy(int)
        assert not episode[fault == 0].any()
        for part, ii in splits.items():
            support.append(dict(axis=axis, partition=part, fault_hours=int(fault[ii].sum()),
                episode_supported=int(episode[ii].any(axis=1).sum()),
                current_supported=int(current[ii].any(axis=1).sum())))
        for target_name, yy in [('current_hour', current), ('episode', episode)]:
            fit = training_rows(train, fault, yy.any(axis=1))
            known_val = val[(fault[val] == 0) | yy[val].any(axis=1)]
            scores = np.full((len(test),len(labels)), np.nan)
            thresholds = np.full(len(labels), np.inf)
            for j, label in enumerate(labels):
                events = len(np.unique(groups[fit[yy[fit,j] == 1]]))
                if events < 3:
                    continue
                views = feature_views(names,axis,label)
                models = [fit_estimator(x[fit][:, cols], yy[fit,j], groups[fit]) for cols in views]
                vp = np.column_stack([probability(m,x[known_val][:,cols]) for m,cols in zip(models,views)])
                weights, threshold, vf = select_policy(yy[known_val,j],vp,event_weights(groups[known_val]),
                                                        np.zeros(len(known_val),int))
                scores[:,j] = np.column_stack([probability(m,x[test][:,cols]) for m,cols in zip(models,views)]) @ weights
                thresholds[j] = threshold
                selections.append(dict(training_target=target_name,axis=axis,label=label,
                    threshold=threshold,weights=weights.tolist(),validation_event_f1=vf))
                print(f'{target_name}/{axis}/{label} fitted',flush=True)
            for policy in ('minimum_one','mixed'):
                mode = 'minimum_one' if policy == 'minimum_one' or axis == 'mechanism' else 'threshold'
                pred = apply_output_policy(scores,thresholds,np.ones(len(test),bool),mode)
                for truth_name, truth, mask in [
                    ('episode_all_supported',episode,episode[test].any(axis=1)),
                    ('episode_previously_unknown',episode,episode[test].any(axis=1)&~current[test].any(axis=1)),
                    ('current_supported',current,current[test].any(axis=1))]:
                    if not mask.any(): continue
                    rows, summary = multilabel_rows(truth[test][mask],pred[mask],groups[test][mask],labels,
                        dict(training_target=target_name,axis=axis,policy=policy,evaluation=truth_name))
                    results.append(summary); details.extend(rows)
    pd.DataFrame(results).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(details).to_csv(out/'per_label.csv',index=False)
    pd.DataFrame(support).to_csv(out/'population.csv',index=False)
    (out/'selection.json').write_text(json.dumps(selections,indent=2),encoding='utf-8')
    assert before == [sha(p) for p in protected]
    (out/'audit.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-started,
        protected_unchanged=True,features=len(names),threads=2,
        selection='Compact fusion/threshold grid on February, not original three-fold OOF; fixed HGB settings.',
        split='Saved blocked membership: February validation, March-April test; training includes May-June.',
        limitations=['Episode labels may describe future-confirmed reasons, not real-time current-hour truth.',
        'Predictors include physical/stuck evidence used in target construction; agreement is not independent diagnosis.',
        'Conditional known-fault evaluation, not detector-to-reason performance.',
        'Exploratory comparison on previously inspected development data.']),indent=2),encoding='utf-8')
    print(pd.DataFrame(results).to_string(index=False),flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
