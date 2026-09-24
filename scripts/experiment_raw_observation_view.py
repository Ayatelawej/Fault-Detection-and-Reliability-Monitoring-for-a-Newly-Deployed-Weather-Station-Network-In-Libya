"""Small isolated fourth-view experiment on the existing blocked split."""
from pathlib import Path
import os
import sys
import json

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from scripts.experiment_blocked_fault_detection import blocked_split, validate_split, sha
from src.model.hourly_baseline import (
    load_hourly_tensor, flatten_hourly_features, _fault_groups,
    HourlyBaselineConfig, make_classifier, binary_metrics,
)
from src.model.hourly_calibration import CALIBRATION_THRESHOLDS

CHANNELS = ['temp_avg_c', 'humidity_avg_pct', 'dewpoint_avg_c',
            'pressure_max_hpa', 'windspeed_avg_kmh', 'windgust_avg_kmh',
            'winddir_avg_deg', 'precip_rate_mmh', 'precip_total_mm',
            'solar_radiation_high_wm2', 'uv_high']


def grid_select(y, branches, allow_raw):
    rows = []
    for f in range(11):
        for c in range(11-f):
            for r in range(11-f-c):
                o = 10-f-c-r
                if not allow_raw and o:
                    continue
                weights = np.array([f,c,r,o], dtype=float)/10
                score = branches @ weights
                for threshold in CALIBRATION_THRESHOLDS:
                    pred = score >= threshold
                    tp = int(np.sum(pred & (y == 1)))
                    fp = int(np.sum(pred & (y == 0)))
                    fn = int(np.sum(~pred & (y == 1)))
                    precision = tp/max(tp+fp,1)
                    recall = tp/max(tp+fn,1)
                    f1 = 2*tp/max(2*tp+fp+fn,1)
                    rows.append(dict(full=f/10,context=c/10,rules=r/10,raw=o/10,
                        threshold=threshold,precision=precision,recall=recall,f1=f1,
                        minimum=min(precision,recall,f1)))
    best = max(rows, key=lambda x:(x['minimum'],x['f1'],x['precision'],
        x['recall'],-x['raw'],x['full'],-x['threshold']))
    return best, rows


def run():
    out = ROOT/'data/eval/raw_observation_view_blocked_20260917'
    out.mkdir(exist_ok=False)
    path = ROOT/'data/eval/blocked_model_comparison_20260916_v2/models/ef_hgb.joblib'
    protected = [path, ROOT/'data/hourly_detection/one_hour_final/models/evidence_fusion/selected_ef_hgb_random_01h.joblib']
    before = {str(p):sha(p) for p in protected}
    bundle = joblib.load(path)
    model = bundle['estimator']
    z = load_hourly_tensor(ROOT/'data/hourly_detection/one_hour_final/hourly_detection_01h.npz')
    x,names,_ = flatten_hourly_features(z)
    assert names == bundle['feature_names']
    y = np.asarray(z['y_binary'],int)
    hours = pd.to_datetime(z['hour'],utc=True)
    keys = pd.MultiIndex.from_arrays([z['station_id'],hours],names=['station_id','hour_utc'])
    groups = np.array([f'normal:{s}:{str(t)[:10]}' for s,t in keys],object)
    for key,ii in _fault_groups(y,z['source_episode_ids']).items():
        groups[ii] = 'event:'+key
    splits = blocked_split(hours,groups)
    validate_split(hours,groups,splits)
    raw = pd.read_csv(ROOT/'data/merged/station_hourly_merged.csv',usecols=['station_id','hour_utc']+CHANNELS)
    raw.hour_utc = pd.to_datetime(raw.hour_utc,utc=True)
    raw = raw.set_index(['station_id','hour_utc'])
    assert raw.index.is_unique and keys.is_unique and keys.isin(raw.index).all()
    raw = raw.reindex(keys).apply(pd.to_numeric,errors='coerce')
    angle = np.deg2rad(raw.pop('winddir_avg_deg'))
    raw['winddir_sin'] = np.sin(angle)
    raw['winddir_cos'] = np.cos(angle)
    raw = raw.replace([np.inf,-np.inf],np.nan)
    raw = pd.concat([raw,raw.notna().astype(float).add_suffix('_present')],axis=1)
    obs = raw.to_numpy(dtype=np.float32)
    plan = dict(features=list(raw.columns),split_counts={p:len(ii) for p,ii in splits.items()},
        config=bundle['config'],weights='convex 0.1 steps',thresholds=list(CALIBRATION_THRESHOLDS),
        selection='February maximum min(precision, recall, F1); ties prefer less raw weight',
        limitations='Previously inspected blocked period; exploratory extension. Existing weak labels and feature preparation retained.',
        source_sha256=before)
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    extra = make_classifier(HourlyBaselineConfig(**bundle['config']))
    extra.fit(obs[splits['train']],y[splits['train']])
    joblib.dump(dict(estimator=extra,feature_names=list(raw.columns)),out/'raw_branch.joblib')
    def branches(ii):
        return np.column_stack([model.full_estimator.predict_proba(x[ii])[:,1],
            model.context_estimator.predict_proba(x[ii][:,model.context_indices])[:,1],
            model.rule_estimator.predict_proba(x[ii][:,model.rule_indices])[:,1],
            extra.predict_proba(obs[ii])[:,1]])
    v = splits['validation']
    vp = branches(v)
    choices = {}
    for name,allow in [('existing_three_views',False),('four_views_with_raw',True)]:
        choices[name],grid = grid_select(y[v],vp,allow)
        pd.DataFrame(grid).to_csv(out/f'{name}_validation_grid.csv',index=False)
    (out/'selection.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
    print('February selections frozen: '+json.dumps(choices),flush=True)
    test = splits['test']
    tp = branches(test)
    rows=[]
    ledger=pd.DataFrame(dict(row_index=test,station_id=z['station_id'][test],hour=hours[test],target=y[test]))
    for name,choice in choices.items():
        weights=np.array([choice[k] for k in ('full','context','rules','raw')])
        for part,ii,scores in [('validation',v,vp),('test',test,tp)]:
            rows.append(dict(model=name,partition=part,**binary_metrics(y[ii],scores@weights,choice['threshold'])))
        ledger[name]=tp@weights
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    ledger.to_parquet(out/'predictions.parquet',index=False)
    assert before == {str(p):sha(p) for p in protected}
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
