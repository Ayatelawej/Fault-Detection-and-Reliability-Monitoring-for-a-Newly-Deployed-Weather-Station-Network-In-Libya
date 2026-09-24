"""Versioned full-development refits and July evaluation; activation is separate."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import sys,json,copy,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from src.model.hourly_baseline import load_hourly_tensor,flatten_hourly_features,EvidenceFusedHgbClassifier,binary_metrics,_fault_groups
from src.model.reason_code_rebuild import MECH,COMP,fit_estimator,sha
from src.model.final_reason_codes import FREEZE,load_observations,build_features,JULY_RAW,JULY_REFS
from src.model.reason_rgfn_adapter import ReasonRgfnEstimator
from scripts.final_reason_model_comparison import balanced
from src.availability import health_forecast as hf
from scripts.final_forecast_model_comparison import metrics as forecast_metrics

OUT=ROOT/'data/eval/final_system_release_20260924'
MODELS=ROOT/'data/model/final_system_20260924'

def write(name,value): (OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')

def binary():
    OUT.mkdir(exist_ok=True);MODELS.mkdir(exist_ok=True)
    if (MODELS/'binary.joblib').exists():raise FileExistsError('Binary already staged')
    table=pd.read_csv(ROOT/'data/eval/blocked_model_comparison_20260916_v2/comparison.csv')
    choice=table.sort_values(['validation_minimum_metric','validation_f1','validation_precision','validation_recall'],ascending=False).iloc[0]
    if choice.model!='EF-HGB':raise ValueError('Expected verified EF-HGB selection; add other release adapters if this changes')
    source=joblib.load(ROOT/'data/eval/blocked_model_comparison_20260916_v2/models/ef_hgb.joblib')
    original=source['estimator'];z=load_hourly_tensor(ROOT/'data/hourly_detection/one_hour_final/hourly_detection_01h.npz')
    assert pd.to_datetime(z['hour'],utc=True).max()<=FREEZE
    x,names,_=flatten_hourly_features(z);y=np.asarray(z['y_binary'],int)
    estimators=[]
    for estimator,cols in zip((original.full_estimator,original.context_estimator,original.rule_estimator),
        (np.arange(x.shape[1]),original.context_indices,original.rule_indices)):
        m=clone(estimator);m.fit(x[:,cols],y);estimators.append(m)
    model=EvidenceFusedHgbClassifier(*estimators,original.context_indices,original.rule_indices,original.full_weight,original.context_weight,original.rule_weight)
    joblib.dump(dict(estimator=model,feature_names=names,threshold=float(choice.threshold),selection=choice.to_dict()),MODELS/'binary.joblib')
    gate=pd.read_parquet(ROOT/'data/eval/one_hour_candidate/july_ef_hgb_binary_predictions.parquet')
    july=load_hourly_tensor(ROOT/'data/eval/one_hour_candidate/tensors/hourly_detection_01h.npz')
    jx,jnames,_=flatten_hourly_features(july);assert names==jnames
    idx=pd.MultiIndex.from_arrays([july['station_id'],pd.to_datetime(july['hour'],utc=True)])
    ii=pd.Series(np.arange(len(idx)),index=idx).reindex(pd.MultiIndex.from_frame(gate[['station_id','hour_utc']])).to_numpy(int)
    gate['random_probability']=model.predict_proba(jx[ii])[:,1];gate['random_prediction']=(gate.random_probability>=float(choice.threshold)).astype(int)
    gate.to_parquet(OUT/'binary_predictions.parquet',index=False)
    write('binary_selection.json',dict(model=choice.model,threshold=float(choice.threshold),validation_minimum=float(choice.validation_minimum_metric),
        training_rows=len(y),training_fault_hours=int(y.sum()),trained_through=str(FREEZE),model_sha256=sha(MODELS/'binary.joblib'),
        limitations='Original historical 67-feature pipeline including retrospective stuck indicators; no claim that causal binary limitations are repaired.'))
    write('july_binary_metrics.json',binary_metrics(gate.truth_fault.to_numpy(int),gate.random_probability.to_numpy(),float(choice.threshold)))
    print('Binary staged and July evaluated',flush=True)

def reasons():
    if (MODELS/'reasons').exists():raise FileExistsError('Reasons already staged')
    choice=json.loads((ROOT/'data/eval/final_reason_comparison_20260924/chosen.json').read_text())['selected']
    z=load_hourly_tensor(ROOT/'data/hourly_detection/one_hour_final/hourly_detection_01h.npz')
    index=pd.MultiIndex.from_arrays([z['station_id'],pd.to_datetime(z['hour'],utc=True)],names=['station_id','hour'])
    fault=np.asarray(z['y_binary'],int);groups=np.array([f'normal:{s}:{str(t)[:10]}' for s,t in index],object)
    for k,ii in _fault_groups(fault,z['source_episode_ids']).items():groups[ii]='event:'+k
    features,_=build_features(load_observations(ROOT/'data/merged/station_hourly_merged.csv',ROOT/'data/features/external_residuals.parquet',FREEZE))
    names=list(features);x=features.reindex(index).to_numpy('float32')
    negative=np.flatnonzero(fault==0);negative=np.sort(np.random.default_rng(20260912).choice(negative,min(16000,len(negative)),replace=False))
    fit=np.sort(np.r_[np.flatnonzero(fault==1),negative]);heads={}
    for axis,labels in [('mechanism',MECH),('component',COMP)]:
        source=joblib.load(ROOT/f'data/eval/final_reason_comparison_20260924/temporal_{axis}_{choice[axis]}.joblib')
        assert names==source['feature_names'];original=list(z[axis+'_label_names']);y=np.asarray(z['y_'+axis],int)[:,[original.index(n) for n in labels]]
        for j,label in enumerate(labels):
            h=source['heads'][label];models=[]
            for old,cols in zip(h['models'],h['views']):
                if isinstance(old,ReasonRgfnEstimator):
                    m=ReasonRgfnEstimator(old.context,old.rules,old.seed).fit(x[fit][:,cols],y[fit,j],balanced(groups[fit],y[fit,j]),epochs=old.best_epoch)
                else:m=fit_estimator(x[fit][:,cols],y[fit,j],groups[fit])
                models.append(m)
            heads[f'{axis}:{label}']=dict(h,models=models)
            print(f'Refit chosen {choice[axis]} {axis}/{label}',flush=True)
    directory=MODELS/'reasons';directory.mkdir()
    version='validation-selected-episode-reasons-v3'
    manifest=dict(version=version,selected_families=choice,feature_count=len(names),trained_through=str(FREEZE),training_fault_hours=int(fault.sum()),
        selection='Temporal validation micro-F1 per axis; minimum-one both outputs; original episode targets.',july_used_for_selection=False)
    joblib.dump(dict(version=version,feature_names=names,heads=heads,manifest=manifest),directory/'reason_heads.joblib')
    manifest['model_sha256']=sha(directory/'reason_heads.joblib');(directory/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    from src.model.final_reason_codes import score,EPISODE_OUTPUT_POLICY
    score(JULY_RAW,JULY_REFS,OUT/'binary_predictions.parquet',OUT/'reasons',directory,EPISODE_OUTPUT_POLICY)
    write('reason_selection.json',manifest)

def forecasts():
    directory=MODELS/'forecasts';directory.mkdir(exist_ok=False)
    selected=pd.read_csv(ROOT/'data/eval/final_forecast_comparison_20260924/chosen.csv');horizons=tuple(selected.horizon_h.astype(int))
    metadata=pd.read_csv(ROOT/'data/merged/station_hourly_merged.csv',usecols=['station_id','elevation'])
    june=hf.build_health_forecast_dataset(pd.read_parquet(ROOT/'data/processed/station_health_scores.parquet'),station_metadata=metadata,horizons=horizons)
    july_scores=pd.read_parquet(ROOT/'data/eval/july_2026_health/station_health_scores_through_july.parquet')
    july=hf.build_health_forecast_dataset(july_scores,station_metadata=metadata,horizons=horizons)
    dashboard=pd.read_parquet(ROOT/'data/eval/july_2026_health_forecast/july_health_forecast_predictions.parquet')
    results=[];ledgers=[];manifest=[]
    for row in selected.itertuples():
        h=int(row.horizon_h);old=joblib.load(ROOT/f'data/eval/final_forecast_comparison_20260924/{row.model.replace(" ","_")}_{h}h.joblib')
        train=hf._regime_subset(hf.health_forecast_horizon_frame(june,h),'transmitting_origin')
        assert train.label_end_utc.max()<=FREEZE
        target=train.target_health_total.to_numpy(float)
        if old.final_policy!='direct_regression':target-=hf._baseline_level_predictions(train)[old.residual_baseline]
        w=hf._recency_weights(train,old.recency_half_life_days)
        if old.family in ('linear','ridge'):
            estimator=clone(old.estimator);estimator.fit(hf._categorical_model_frame(train,old.feature_columns),target,**({} if w is None else {'model__sample_weight':w}))
            model=copy.copy(old);model.estimator=estimator
        else:
            model=hf._fit_health_forecast_model(old.family,train,target,feature_columns=old.feature_columns,iterations=old.iterations,sample_weight=w)
            for name in ('alpha','final_policy','horizon_h','regime','feature_set','recency_half_life_days','iterations','residual_baseline'):setattr(model,name,getattr(old,name))
        path=directory/f'health_forecast_forecast_transmitting_origin_{h}h.joblib';joblib.dump(model,path)
        frame=hf.health_forecast_horizon_frame(july,h);frame=hf._regime_subset(frame[frame.hour_utc.ge(pd.Timestamp('2026-07-01',tz='UTC'))],'transmitting_origin')
        prediction=model.predict_health(frame);actual=frame.target_health_total.to_numpy(float)
        results.append(dict(horizon_h=h,model=row.model,rows=len(frame),**forecast_metrics(actual,prediction)))
        ledgers.append(pd.DataFrame(dict(horizon_h=h,model=row.model,station_id=frame.station_id,hour_utc=frame.hour_utc,actual=actual,prediction=prediction)))
        if h<=24:
            inference=hf.health_forecast_inference_frame(july,h).set_index(['station_id','hour_utc'])
            mask=dashboard.horizon_h.eq(h);keys=pd.MultiIndex.from_frame(dashboard.loc[mask,['station_id','hour_utc']])
            dashboard.loc[mask,'predicted_frozen_selected_policy']=model.predict_health(inference.loc[keys].reset_index())
        manifest.append(dict(horizon_h=h,model=row.model,validation_mae=row.mae,path=str(path.relative_to(ROOT)),sha256=sha(path),train_rows=len(train)))
        print(f'{h}h selected {row.model} refit; July MAE {results[-1]["mae"]:.3f}',flush=True)
    pd.DataFrame(results).to_csv(OUT/'july_forecast_metrics.csv',index=False);pd.concat(ledgers).to_parquet(OUT/'july_forecast_evaluation.parquet',index=False)
    dashboard.to_parquet(OUT/'dashboard_forecasts.parquet',index=False);write('forecast_selection.json',manifest)

def evaluate_reasons():
    from src.model.final_reason_codes import predict
    from src.model.reason_code_rebuild import multilabel_rows
    gate=pd.read_parquet(OUT/'binary_predictions.parquet')
    bundle=joblib.load(MODELS/'reasons/reason_heads.joblib')
    features,_=build_features(load_observations(JULY_RAW,JULY_REFS,gate.hour_utc.max()))
    oracle=predict(features,gate.assign(random_prediction=1),bundle)
    # Read targets only after predictions are fixed.
    ep=pd.read_csv(ROOT/'data/eval/july_2026_adjudicated_labels/episode_labels_adjudicated.csv')
    for c in ('start_hour','end_hour'):ep[c]=pd.to_datetime(ep[c],utc=True)
    ep=ep[ep.label_state.eq('fault')&ep.end_hour.ge(gate.hour_utc.min())&ep.start_hour.le(gate.hour_utc.max())]
    y=np.zeros((len(gate),len(MECH)+len(COMP)),int)
    for e in ep.itertuples():
        mask=gate.station_id.eq(e.station_id)&gate.hour_utc.between(e.start_hour,e.end_hour)
        for field,labels,offset in [('mechanisms',MECH,0),('components',COMP,len(MECH))]:
            for label in str(getattr(e,field)).split('|'):
                if label in labels:y[mask,offset+labels.index(label)]=1
    fault=gate.truth_fault.eq(1).to_numpy();alert=gate.random_prediction.eq(1).to_numpy();rows=[]
    for axis,labels,offset in [('mechanism',MECH,0),('component',COMP,len(MECH))]:
        np.testing.assert_array_equal(y[:,offset:offset+len(labels)].any(axis=1),fault)
        emitted=oracle[f'likely_{axis}s'].str.split(' | ',regex=False)
        p=np.column_stack([emitted.apply(lambda values:label in values).to_numpy(bool) for label in labels])
        for scope,mask,pp in [('all_reference_fault_hours',fault,p),('correctly_detected_fault_hours',fault&alert,p),
            ('all_alerts',alert,p),('end_to_end',np.ones(len(gate),bool),p&alert[:,None])]:
            _,ss=multilabel_rows(y[mask,offset:offset+len(labels)],pp[mask],gate.source_episode_ids.to_numpy()[mask],labels,dict(axis=axis,scope=scope))
            rows.append(ss)
    pd.DataFrame(rows).to_csv(OUT/'july_reason_metrics.csv',index=False)
    # Reapply unchanged annotation conditions to the new gate; never suppress alerts.
    from scripts.investigate_july_temperature import guard_masks
    cols=['station_id','hour_utc','temp_low_c_past_hour_z','temp_avg_c_past_hour_z','temp_high_c_past_hour_z',
        'r_temp_past_z','r_spatial_temp_past_z','physical_or_confirmed_stuck','temperature_stat','other_stat']
    context=pd.read_parquet(ROOT/'data/eval/july_temperature_investigation/july_shadow_predictions.parquet',columns=cols)
    pd.testing.assert_frame_equal(context[['station_id','hour_utc']],gate[['station_id','hour_utc']])
    notes=gate[['station_id','hour_utc','random_probability','random_prediction']].copy()
    flag=guard_masks(context)['temperature_context_reference_and_peers'].to_numpy()&alert
    notes['weather_consistent_temperature']=flag
    notes['weather_note']=np.where(flag,'Temperature pattern consistent with recent history, reference weather and neighbours. Alert retained; sensor fault not ruled out.','')
    notes.to_parquet(OUT/'weather_annotations.parquet',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['binary','reasons','forecasts','evaluate_reasons']);args=parser.parse_args()
    with threadpool_limits(limits=2):globals()[args.stage]()
