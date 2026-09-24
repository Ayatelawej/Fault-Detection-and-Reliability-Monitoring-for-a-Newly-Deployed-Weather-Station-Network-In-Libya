"""Validation-only search with causal added features; no test scoring or deployment."""
from pathlib import Path
from dataclasses import replace, asdict
import os
import sys
import json
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from sklearn.metrics import precision_recall_curve
from scripts.experiment_causal_feature_mix import run as prepare, history_features
from src.model.hourly_baseline import HourlyBaselineConfig, make_classifier, binary_metrics


def expanded_features(frame):
    columns={}
    for channel in frame:
        s=frame[channel]
        columns[f'{channel}:value']=s
        columns[f'{channel}:present']=s.notna().astype(float)
        for lag in (1,3,6,12,24):
            columns[f'{channel}:delta{lag}']=s-s.shift(lag)
        for window in (3,6,12,24):
            rolling=s.rolling(window,min_periods=window)
            columns[f'{channel}:variance{window}']=rolling.var()
            columns[f'{channel}:range{window}']=rolling.max()-rolling.min()
            columns[f'{channel}:flat_fraction{window}']=s.diff().abs().le(1e-6).astype(float).rolling(window,min_periods=window).mean()
        same=s.notna() & s.shift().notna() & s.diff().abs().le(1e-6)
        columns[f'{channel}:flat_hours']=s.groupby((~same).cumsum()).cumcount().add(1).clip(upper=168).where(s.notna())
    columns['hour_sin']=np.sin(2*np.pi*frame.index.hour/24)
    columns['hour_cos']=np.cos(2*np.pi*frame.index.hour/24)
    return pd.DataFrame(columns,index=frame.index)


def select(y,p):
    precision,recall,thresholds=precision_recall_curve(y,p)
    precision,recall=precision[:-1],recall[:-1]
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros_like(precision),where=(precision+recall)>0)
    minimum=np.minimum(np.minimum(precision,recall),f1)
    i=max(range(len(thresholds)),key=lambda i:(minimum[i],f1[i],precision[i],recall[i],-abs(thresholds[i]-.5)))
    return float(thresholds[i]),binary_metrics(y,p,float(thresholds[i]))


def run():
    out=ROOT/'data/eval/causal_validation_search_20260923'
    out.mkdir(parents=True,exist_ok=False)
    prepared=prepare(prepare_only=True)
    d=prepared['data']; y=prepared['labels']; splits=prepared['splits']
    x,names=prepared['variants']['plus_past_changes']
    raw=d['raw'].copy()
    channels=['temp_avg_c','temp_high_c','temp_low_c','humidity_avg_pct','pressure_max_hpa','pressure_min_hpa',
        'pressure_trend_hpa','windspeed_avg_kmh','windspeed_high_kmh','windspeed_low_kmh',
        'windgust_avg_kmh','windgust_high_kmh','windgust_low_kmh','precip_total_mm','precip_rate_mmh',
        'solar_radiation_high_wm2','uv_high']
    angle=np.deg2rad(raw.winddir_avg_deg)
    raw['winddir_sin']=np.sin(angle); raw['winddir_cos']=np.cos(angle)
    channels+=['winddir_sin','winddir_cos']
    frames=[]
    for station,part in raw.groupby('station_id',sort=False):
        part=part.set_index('hour_utc')[channels].sort_index()
        part=part.reindex(pd.date_range(part.index.min(),part.index.max(),freq='h',name='hour_utc'))
        features=expanded_features(part)
        cut=len(part)//2
        pd.testing.assert_frame_equal(features.iloc[:cut],expanded_features(part.iloc[:cut]))
        features['station_id']=station
        frames.append(features.reset_index().set_index(['station_id','hour_utc']))
    extra=pd.concat(frames).reindex(d['index']).iloc[d['retained']]
    # Do not retain duplicate raw/current and 1/3/6h-change columns already supplied.
    existing=set(names)
    allowed=[]
    for c in extra:
        ch,_,kind=c.partition(':')
        duplicate=(kind=='value' and 'raw:'+ch in existing) or (kind=='present' and 'raw:'+ch+'_present' in existing)
        duplicate |= kind.startswith('delta') and ('past:'+ch+'_change_'+kind[5:]+'h' in existing)
        duplicate |= kind=='flat_hours' and ('past:'+ch+'_unchanged_hours' in existing)
        if not duplicate: allowed.append(c)
    extra=extra[allowed]
    enriched=np.column_stack([x,extra.to_numpy(dtype=np.float32)])
    feature_sets={'previous':(x,names),'expanded':(enriched,names+['extra:'+c for c in extra])}
    cache=dict(feature_sets=feature_sets,y=y,splits=splits,
        stations=d['index'].get_level_values('station_id').to_numpy()[d['retained']],
        hours=d['index'].get_level_values('hour_utc').to_numpy()[d['retained']])
    joblib.dump(cache,out/'feature_cache.joblib')
    trials=[]
    # Predeclared capacity/weight combinations. Stop once all three validation metrics exceed .8.
    for feature_set in ('previous','expanded'):
        for leaves,iterations,weight in ((31,300,2),(31,300,4),(63,300,2),(63,300,4),(31,600,2),(31,600,4)):
            trials.append(dict(feature_set=feature_set,leaves=leaves,iterations=iterations,weight=weight))
    (out/'plan.json').write_text(json.dumps(dict(trials=trials,selection='validation minimum precision/recall/F1; threshold chosen from validation scores',
        test_scoring=False,history_prefix_checks_passed=True,partition_index_sha256=d['digests'],
        limitations='Repeated validation search; shared reference-label evidence and original cached statistical preprocessing remain. Not a full leakage audit. Production unchanged.'),indent=2),encoding='utf-8')
    rows=[]; best=-1
    for trial in trials:
        matrix,feature_names=feature_sets[trial['feature_set']]
        config=replace(HourlyBaselineConfig(),max_leaf_nodes=trial['leaves'],max_iter=trial['iterations'],fault_class_weight=trial['weight'])
        model=make_classifier(config)
        model.fit(matrix[splits['train']],y[splits['train']])
        p=model.predict_proba(matrix[splits['validation']])[:,1]
        threshold,metric=select(y[splits['validation']],p)
        minimum=min(metric[k] for k in ('precision','recall','f1'))
        row=dict(**trial,columns=matrix.shape[1],threshold=threshold,minimum=minimum,**metric)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out/'validation_search.csv',index=False)
        if minimum>best:
            best=minimum
            joblib.dump(dict(estimator=model,feature_names=feature_names,config=asdict(config),selection=row),out/'best_model.joblib')
            pd.DataFrame(dict(target=y[splits['validation']],probability=p)).to_csv(out/'best_validation_predictions.csv',index=False)
        print(json.dumps(row),flush=True)
        if minimum>.8:
            break
    print(f'Finished {len(rows)} trials. Best minimum validation score: {best:.6f}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        run()
