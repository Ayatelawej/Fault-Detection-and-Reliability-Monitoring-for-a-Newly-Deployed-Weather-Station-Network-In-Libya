"""Comparable HGB, EF-HGB and RGFN episode-head validation and test tables."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from src.model.final_reason_codes import FREEZE,build_features,load_observations,apply_output_policy
from src.model.reason_code_utils import MECH,COMP,feature_views,fit_estimator,probability,event_weights,select_policy,multilabel_rows
from src.model.hourly_baseline import load_hourly_tensor,_fault_groups
from src.model.reason_rgfn_adapter import ReasonRgfnEstimator
from src.model.reason_sampling import training_rows

OUT=ROOT/'data/eval/final_reason_comparison_20260924'

def balanced(groups,y):
    w=event_weights(groups);w[y==1]*=w[y==0].sum()/max(w[y==1].sum(),1e-12);return w/w.mean()

def run():
    OUT.mkdir(exist_ok=False);start=time.perf_counter()
    z=load_hourly_tensor(ROOT/'data/hourly_detection/one_hour_final/hourly_detection_01h.npz')
    index=pd.MultiIndex.from_arrays([z['station_id'],pd.to_datetime(z['hour'],utc=True)],names=['station_id','hour'])
    fault=np.asarray(z['y_binary'],int);groups=np.array([f'normal:{s}:{str(t)[:10]}' for s,t in index],object)
    for k,ii in _fault_groups(fault,z['source_episode_ids']).items():groups[ii]='event:'+k
    feature,_=build_features(load_observations(ROOT/'data/merged/station_hourly_merged.csv',ROOT/'data/features/external_residuals.parquet',FREEZE))
    names=list(feature);x=feature.reindex(index).to_numpy('float32');del feature
    mem=pd.read_csv(ROOT/'data/eval/reason_code_rebuild_20260912_v2/split_membership.csv')
    frames={'random':mem[mem.scheme.eq('random')],'temporal':pd.read_csv(ROOT/'data/eval/blocked_ef_hgb_20260914/split_membership.csv')}
    summaries=[];details=[];selections=[];ledgers=[]
    for scheme,frame in frames.items():
        splits={p:frame.loc[frame.partition.eq(p),'row_index'].to_numpy(int) for p in ('train','validation','test')}
        tr,va,te=(splits[p] for p in ('train','validation','test'))
        for axis,labels in [('mechanism',MECH),('component',COMP)]:
            original=list(z[axis+'_label_names']);y=np.asarray(z['y_'+axis],int)[:,[original.index(n) for n in labels]]
            np.testing.assert_array_equal(y.any(axis=1),fault.astype(bool))
            fit=training_rows(tr,fault,y.any(axis=1));wv=event_weights(groups[va])
            for family in ('HGB','EF-HGB','RGFN'):
                scores={p:np.full((len(splits[p]),len(labels)),np.nan) for p in ('validation','test')};limits=np.full(len(labels),np.inf);heads={}
                for j,label in enumerate(labels):
                    if len(np.unique(groups[fit[y[fit,j]==1]]))<3:continue
                    views=feature_views(names,axis,label)
                    if family=='RGFN':
                        full,context,rules=views
                        est=ReasonRgfnEstimator([list(full).index(c) for c in context],[list(full).index(c) for c in rules])
                        est.fit(x[fit][:,full],y[fit,j],balanced(groups[fit],y[fit,j]),
                            (x[va][:,full],y[va,j],balanced(groups[va],y[va,j])))
                        models=[est];views=[full]
                    else:
                        views=views if family=='EF-HGB' else views[:1]
                        models=[fit_estimator(x[fit][:,c],y[fit,j],groups[fit]) for c in views]
                    vp=np.column_stack([probability(m,x[va][:,c]) for m,c in zip(models,views)])
                    if family=='EF-HGB':mix,threshold,vf=select_policy(y[va,j],vp,wv,np.zeros(len(va),int))
                    else:
                        _,threshold,vf=select_policy(y[va,j],np.tile(vp,(1,3)),wv,np.zeros(len(va),int));mix=np.ones(1)
                    limits[j]=threshold;heads[label]=dict(models=models,views=views,weights=mix,threshold=threshold)
                    selections.append(dict(split=scheme,axis=axis,model=family,label=label,threshold=threshold,weights=mix.tolist(),
                        best_epoch=getattr(models[0],'best_epoch',None),validation_event_f1=vf))
                    scores['validation'][:,j]=vp@mix
                    scores['test'][:,j]=np.column_stack([probability(m,x[te][:,c]) for m,c in zip(models,views)])@mix
                    print(f'{scheme}/{axis}/{family}/{label} complete',flush=True)
                joblib.dump(dict(heads=heads,feature_names=names),OUT/f'{scheme}_{axis}_{family}.joblib')
                for part in ('validation','test'):
                    ii=splits[part];mask=fault[ii]==1
                    pred=apply_output_policy(scores[part],limits,np.ones(len(ii),bool),'minimum_one')
                    rr,ss=multilabel_rows(y[ii][mask],pred[mask],groups[ii][mask],labels,dict(split=scheme,partition=part,axis=axis,model=family))
                    summaries.append(ss);details.extend(rr)
                    for j,label in enumerate(labels):ledgers.append(pd.DataFrame(dict(split=scheme,partition=part,axis=axis,model=family,label=label,row_index=ii[mask],truth=y[ii][mask,j],prediction=pred[mask,j])))
                pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(details).to_csv(OUT/'per_label.csv',index=False);pd.concat(ledgers).to_parquet(OUT/'predictions.parquet',index=False)
    (OUT/'selection.json').write_text(json.dumps(selections,indent=2),encoding='utf-8')
    f=pd.DataFrame(summaries);valid=f[f.split.eq('temporal')&f.partition.eq('validation')]
    choices={axis:g.sort_values(['micro_f1','macro_f1','model'],ascending=[False,False,True]).iloc[0].model for axis,g in valid.groupby('axis')}
    (OUT/'chosen.json').write_text(json.dumps(dict(selected=choices,criterion='temporal validation micro-F1 per axis; macro-F1 then model name tie-break',
        elapsed_seconds=time.perf_counter()-start,rgfn='Existing one-hour MLP gated architecture, separate sensor-scoped heads; seed 2026, max80 epochs, patience8, batch1024; validation loss checkpoint.',
        population='All reference fault hours; original episode labels; minimum-one both outputs; no detector gate.',
        limitations='Exploratory previously inspected splits; RGFN single-seed matched-feature adaptation, not old current-hour RGFN results.'),indent=2),encoding='utf-8')
    print('Chosen '+json.dumps(choices),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
