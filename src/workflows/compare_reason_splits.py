"""Original episode targets; matched compact EF-HGB protocol across three splits."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import time, json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.model.hourly_baseline import load_hourly_tensor, _fault_groups
from src.model.final_reason_codes import build_features, load_observations, FREEZE, apply_output_policy
from src.model.reason_code_utils import (sha, MECH, COMP, feature_views, fit_estimator,
    probability, select_policy, event_weights, multilabel_rows)
from src.model.reason_sampling import training_rows
ROOT = Path(__file__).resolve().parents[2]


def run_three():
    started = time.perf_counter()
    out = ROOT/'data/eval/episode_reasons_three_splits_20260924_v2'
    out.mkdir(exist_ok=False)
    paths = dict(tensor=ROOT/'data/hourly_detection/one_hour_final/hourly_detection_01h.npz',
        reason_splits=ROOT/'data/eval/reason_code_rebuild_20260912_v2/split_membership.csv',
        blocked_split=ROOT/'data/eval/blocked_ef_hgb_20260914/split_membership.csv',
        deployed_model=ROOT/'data/model/reason_codes/final/reason_heads.joblib',
        july=ROOT/'data/eval/july_2026_reason_codes_mixed_v2/reason_code_predictions.parquet')
    hashes = {k:sha(p) for k,p in paths.items()}
    z = load_hourly_tensor(paths['tensor'])
    index = pd.MultiIndex.from_arrays([z['station_id'],pd.to_datetime(z['hour'],utc=True)],names=['station_id','hour'])
    fault = np.asarray(z['y_binary'],int)
    groups = np.array([f'normal:{s}:{str(t)[:10]}' for s,t in index],dtype=object)
    for k,ii in _fault_groups(fault,z['source_episode_ids']).items(): groups[ii]='event:'+k
    memberships = pd.read_csv(paths['reason_splits'])
    split_frames = {p:memberships.loc[memberships.scheme.eq(p)] for p in ('random','grouped')}
    split_frames['temporal_holdout'] = pd.read_csv(paths['blocked_split'])
    features,_ = build_features(load_observations(ROOT/'data/merged/station_hourly_merged.csv',
        ROOT/'data/features/external_residuals.parquet',FREEZE))
    assert index.isin(features.index).all()
    names = list(features.columns)
    x = features.reindex(index).to_numpy('float32')
    del features
    results, details, selections, populations, ledger, split_audit = [],[],[],[],[],[]
    for scheme, membership in split_frames.items():
        splits = {p:membership.loc[membership.partition.eq(p),'row_index'].to_numpy(int) for p in ('train','validation','test')}
        train,val,test = (splits[p] for p in ('train','validation','test'))
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            assert not np.intersect1d(splits[a],splits[b]).size
            overlap=len(set(groups[splits[a]])&set(groups[splits[b]]))
            split_audit.append(dict(split=scheme,left=a,right=b,overlapping_groups=overlap))
            if scheme!='random': assert overlap==0
        for axis,labels in [('mechanism',MECH),('component',COMP)]:
            original_names=list(z[axis+'_label_names'])
            y=np.asarray(z['y_'+axis],int)[:,[original_names.index(n) for n in labels]]
            np.testing.assert_array_equal(y.any(axis=1),fault.astype(bool))
            fit=training_rows(train,fault,y.any(axis=1))
            for part,ii in splits.items():
                populations.append(dict(split=scheme,axis=axis,partition=part,hours=len(ii),
                    fault_hours=int(fault[ii].sum()),reason_labelled_fault_hours=int(y[ii].any(axis=1).sum())))
            scores=np.full((len(test),len(labels)),np.nan)
            thresholds=np.full(len(labels),np.inf)
            for j,label in enumerate(labels):
                events=len(np.unique(groups[fit[y[fit,j]==1]]))
                if events<3:
                    selections.append(dict(split=scheme,axis=axis,label=label,status='unsupported',training_events=events))
                    continue
                views=feature_views(names,axis,label)
                models=[fit_estimator(x[fit][:,cols],y[fit,j],groups[fit]) for cols in views]
                vp=np.column_stack([probability(m,x[val][:,cols]) for m,cols in zip(models,views)])
                weights,threshold,vf=select_policy(y[val,j],vp,event_weights(groups[val]),np.zeros(len(val),int))
                scores[:,j]=np.column_stack([probability(m,x[test][:,cols]) for m,cols in zip(models,views)])@weights
                thresholds[j]=threshold
                selections.append(dict(split=scheme,axis=axis,label=label,status='fitted',training_events=events,
                    threshold=threshold,weights=weights.tolist(),validation_event_f1=vf))
                print(f'{scheme}/{axis}/{label} fitted',flush=True)
            pred=apply_output_policy(scores,thresholds,np.ones(len(test),bool),'minimum_one')
            mask=fault[test].astype(bool)
            rows,summary=multilabel_rows(y[test][mask],pred[mask],groups[test][mask],labels,
                dict(split=scheme,axis=axis,policy='minimum_one',target='original_episode',scope='all_reference_fault_hours'))
            details.extend(rows);results.append(summary)
            for j,label in enumerate(labels):
                ledger.append(pd.DataFrame(dict(split=scheme,axis=axis,label=label,row_index=test[mask],
                    truth=y[test][mask,j],score=scores[mask,j],threshold=thresholds[j],prediction=pred[mask,j])))
        pd.DataFrame(results).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(details).to_csv(out/'per_label.csv',index=False)
    pd.DataFrame(populations).to_csv(out/'population.csv',index=False)
    pd.DataFrame(split_audit).to_csv(out/'split_audit.csv',index=False)
    pd.concat(ledger).to_parquet(out/'test_predictions.parquet',index=False)
    (out/'selection.json').write_text(json.dumps(selections,indent=2),encoding='utf-8')
    assert hashes=={k:sha(p) for k,p in paths.items()}
    previous=pd.read_csv(ROOT/'data/eval/episode_reason_pilot_20260924/summary.csv')
    for axis in ('mechanism','component'):
        old=previous.loc[previous.training_target.eq('episode')&previous.axis.eq(axis)&previous.policy.eq('minimum_one')&previous.evaluation.eq('episode_all_supported')].iloc[0]
        new=next(r for r in results if r['split']=='temporal_holdout' and r['axis']==axis)
        np.testing.assert_allclose([new['micro_f1'],new['macro_f1']],[old.micro_f1,old.macro_f1],atol=1e-12)
    (out/'audit.json').write_text(json.dumps(dict(input_hashes=hashes,protected_unchanged=True,
        temporal_pilot_reproduced=True,elapsed_seconds=time.perf_counter()-started,features=len(names),threads=2,
        selection='Same per-head fusion/threshold grid on each validation partition; no test selection or OOF retuning.',
        limitations=['Random can share events across partitions; grouped and blocked do not.',
        'Grouped is the saved reason-specific balanced split, not binary spaced membership.',
        'Episode targets may use later evidence; this is event-label prediction, not guaranteed contemporaneous confirmation.',
        'Conditional known-fault metrics; no binary detector gate.',
        'Previously explored development data; not independent prospective validation.']),indent=2),encoding='utf-8')
    print(pd.DataFrame(results).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2): run_three()
