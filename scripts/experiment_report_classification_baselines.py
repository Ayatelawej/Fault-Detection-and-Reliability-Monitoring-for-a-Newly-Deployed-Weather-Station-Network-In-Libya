"""Report baselines on unchanged historical binary and episode-reason tasks."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
from pathlib import Path
import sys,json,time,warnings
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from src.model.hourly_baseline import load_hourly_tensor,flatten_hourly_features,_fault_groups,binary_metrics
from src.model.hourly_calibration import CALIBRATION_WEIGHTS,CALIBRATION_THRESHOLDS
from src.model.final_reason_codes import FREEZE,build_features,load_observations,apply_output_policy
from src.model.reason_code_rebuild import MECH,COMP,feature_views,fit_estimator,probability,event_weights,select_policy,sha
from src.model.reason_code_rebuild import multilabel_rows
from scripts.experiment_blocked_reason_codes import training_rows


def prepare(x,train,valid,test):
    x=np.asarray(x,dtype=float); x[~np.isfinite(x)]=np.nan
    prep=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler())
    return prep,prep.fit_transform(x[train]),prep.transform(x[valid]),prep.transform(x[test])


def logistic(x,y,c,weight=None):
    m=LogisticRegression(C=c,max_iter=2500,solver='lbfgs',random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        m.fit(x,y,sample_weight=weight)
    return m


def run():
    started=time.perf_counter(); out=ROOT/'data/eval/report_classification_baselines_20260924'; out.mkdir(exist_ok=False)
    protected=[ROOT/'data/model/reason_codes/episode_v2/reason_heads.joblib',ROOT/'data/eval/july_2026_reason_codes_episode_v2/reason_code_predictions.parquet',ROOT/'data/hourly_detection/one_hour_final/models/evidence_fusion/selected_ef_hgb_random_01h.joblib']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in protected}
    z=load_hourly_tensor(ROOT/'data/hourly_detection/one_hour_final/hourly_detection_01h.npz')
    xbinary,_,_=flatten_hourly_features(z); fault=np.asarray(z['y_binary'],int)
    idx=pd.MultiIndex.from_arrays([z['station_id'],pd.to_datetime(z['hour'],utc=True)],names=['station_id','hour'])
    groups=np.array([f'normal:{s}:{str(t)[:10]}' for s,t in idx],dtype=object)
    for k,ii in _fault_groups(fault,z['source_episode_ids']).items():groups[ii]='event:'+k
    mem=pd.read_csv(ROOT/'data/eval/reason_code_rebuild_20260912_v2/split_membership.csv')
    frames={'random':mem[mem.scheme.eq('random')],'temporal_holdout':pd.read_csv(ROOT/'data/eval/blocked_ef_hgb_20260914/split_membership.csv')}
    binary_rows=[];reason_rows=[];grids=[];detail=[];selections=[];ledger=[]
    splits={name:{p:f.loc[f.partition.eq(p),'row_index'].to_numpy(int) for p in ('train','validation','test')} for name,f in frames.items()}
    for name,sp in splits.items():
        tr,va,te=(sp[p] for p in ('train','validation','test'))
        prep,xt,xv,xe=prepare(xbinary,tr,va,te)
        best=None
        for c in (.1,1.,10.):
            for weight in CALIBRATION_WEIGHTS:
                model=logistic(xt,fault[tr],c,np.where(fault[tr]==1,weight,1.))
                pv=model.predict_proba(xv)[:,1]
                for threshold in CALIBRATION_THRESHOLDS:
                    m=binary_metrics(fault[va],pv,threshold)
                    row=dict(task='binary',split=name,C=c,class_weight=weight,threshold=threshold,**m)
                    grids.append(row)
                    key=(min(m[k] for k in ('precision','recall','f1')),m['f1'],m['precision'],m['recall'])
                    if best is None or key>best[0]:best=(key,model,row)
        _,model,choice=best; pt=model.predict_proba(xe)[:,1]; tm=binary_metrics(fault[te],pt,choice['threshold'])
        binary_rows.append(dict(split=name,model='Logistic regression',C=choice['C'],class_weight=choice['class_weight'],threshold=choice['threshold'],
            **{f'validation_{k}':choice[k] for k in ('accuracy','precision','recall','f1')},**{f'test_{k}':tm[k] for k in ('accuracy','precision','recall','f1')}))
        joblib.dump(dict(preprocessor=prep,model=model,selection=choice),out/f'binary_logistic_{name}.joblib')
        pd.DataFrame(binary_rows).to_csv(out/'binary_metrics.csv',index=False)
        print('Binary '+name+': '+json.dumps(binary_rows[-1]),flush=True)
    features,_=build_features(load_observations(ROOT/'data/merged/station_hourly_merged.csv',ROOT/'data/features/external_residuals.parquet',FREEZE))
    names=list(features.columns); x=features.reindex(idx).to_numpy('float32'); del features
    old_predictions=pd.read_parquet(ROOT/'data/eval/episode_reasons_three_splits_20260924_v2/test_predictions.parquet')
    for name,sp in splits.items():
        tr,va,te=(sp[p] for p in ('train','validation','test'))
        if name=='random':
            gatebundle=joblib.load(protected[-1]); detected=gatebundle['estimator'].predict_proba(xbinary[te])[:,1]>=.3
        else:
            gate=pd.read_parquet(ROOT/'data/eval/blocked_ef_hgb_20260914/predictions.parquet').set_index('row_index').loc[te]
            detected=gate.prediction_february_selected.to_numpy(bool)
        for axis,labels in [('mechanism',MECH),('component',COMP)]:
            original=list(z[axis+'_label_names']); y=np.asarray(z['y_'+axis],int)[:,[original.index(n) for n in labels]]
            np.testing.assert_array_equal(y.any(axis=1),fault.astype(bool))
            fit=training_rows(tr,fault,y.any(axis=1)); wval=event_weights(groups[va])
            for family in ('Logistic regression','HGB','EF-HGB'):
                scores=np.full((len(te),len(labels)),np.nan); limits=np.full(len(labels),np.inf)
                if family=='EF-HGB':
                    pred=np.zeros((len(te),len(labels)),bool)
                    for j,label in enumerate(labels):
                        p=old_predictions.loc[old_predictions.split.eq(name)&old_predictions.axis.eq(axis)&old_predictions.label.eq(label)].set_index('row_index')
                        ii=np.flatnonzero(fault[te]==1)
                        pred[ii,j]=p.loc[te[ii],'prediction'].to_numpy(bool)
                else:
                    for j,label in enumerate(labels):
                        if len(np.unique(groups[fit[y[fit,j]==1]]))<3:continue
                        cols=feature_views(names,axis,label)[0]
                        yy=y[fit,j]; w=event_weights(groups[fit]);w[yy==1]*=w[yy==0].sum()/max(w[yy==1].sum(),1e-12);w/=w.mean()
                        candidates=[]
                        if family=='Logistic regression':
                            prep,xt,xv,xe=prepare(x[:,cols],fit,va,te)
                            for c in (.1,1.,10.):
                                m=logistic(xt,yy,c,w);pv=m.predict_proba(xv)[:,1]
                                _,threshold,vf=select_policy(y[va,j],np.tile(pv[:,None],(1,3)),wval,np.zeros(len(va),int))
                                candidates.append((vf,-c,m,threshold,c))
                            _,_,m,threshold,c=max(candidates,key=lambda v:v[:2]);pt=m.predict_proba(xe)[:,1]
                        else:
                            m=fit_estimator(x[fit][:,cols],yy,groups[fit]);pv=probability(m,x[va][:,cols])
                            _,threshold,vf=select_policy(y[va,j],np.tile(pv[:,None],(1,3)),wval,np.zeros(len(va),int))
                            pt=probability(m,x[te][:,cols]);c=None
                        scores[:,j]=pt;limits[j]=threshold
                        selections.append(dict(split=name,axis=axis,model=family,label=label,threshold=threshold,C=c))
                    pred=apply_output_policy(scores,limits,np.ones(len(te),bool),'minimum_one')
                for scope,mask in [('all_reference_fault_hours',fault[te]==1),('correctly_detected_fault_hours',(fault[te]==1)&detected)]:
                    rr,ss=multilabel_rows(y[te][mask],pred[mask],groups[te][mask],labels,dict(split=name,axis=axis,model=family,scope=scope))
                    reason_rows.append(ss);detail.extend(rr)
                for j,label in enumerate(labels):
                    mask=fault[te]==1
                    ledger.append(pd.DataFrame(dict(split=name,axis=axis,model=family,label=label,row_index=te[mask],truth=y[te][mask,j],prediction=pred[mask,j],detected=detected[mask])))
                pd.DataFrame(reason_rows).to_csv(out/'reason_metrics.csv',index=False)
                print(f'{name}/{axis}/{family} done',flush=True)
    pd.DataFrame(grids).to_csv(out/'binary_validation_grid.csv',index=False)
    pd.DataFrame(selections).to_csv(out/'reason_selection.csv',index=False)
    pd.DataFrame(detail).to_csv(out/'reason_per_label.csv',index=False)
    pd.concat(ledger).to_parquet(out/'reason_predictions.parquet',index=False)
    assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in protected}
    (out/'audit.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-started,protected_unchanged=True,hashes=hashes,
        selection='Binary: maximin validation precision/recall/F1; logistic C and class weight grid. Reasons: per-head validation event-weighted F1 threshold, logistic C grid; same full feature view and training weights as EF full arm.',
        limitations=['Binary uses original 67-feature pipeline, including previously identified retrospective stuck indicators; historical comparison, not causal validation.',
        'Detected-fault subset excludes false alerts and missed faults; not end-to-end performance.',
        'Detected subset uses saved matching-split EF-HGB gates: random .30, original blocked February-selected .35.',
        'Episode labels and split data have been previously explored. No model promoted.']),indent=2),encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
