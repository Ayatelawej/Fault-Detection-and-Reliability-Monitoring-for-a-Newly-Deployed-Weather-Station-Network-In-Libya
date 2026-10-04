"""Bounded comparison of learned corrections to three health baselines."""
from pathlib import Path
import os
import sys
import json
import argparse

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from src.availability import health_forecast as hf
from src.availability.risk_eval import split_timestamp_partitions, regression_metrics


def run(long_horizon=False, requested_horizons=None, output=None):
    horizons=tuple(requested_horizons) if requested_horizons else ((48,72,168) if long_horizon else (12,24))
    out=output or ROOT/('data/eval/forecast_baseline_corrections_long_20260919' if long_horizon else 'data/eval/forecast_baseline_corrections_20260918')
    out.mkdir(exist_ok=False)
    hf.HEALTH_FORECAST_CATBOOST_PARAMETERS['thread_count']=2
    scores=pd.read_parquet(ROOT/'data/processed/station_health_scores.parquet')
    metadata=pd.read_csv(ROOT/'data/merged/station_hourly_merged.csv',usecols=['station_id','elevation'])
    bundle=hf.build_health_forecast_dataset(scores,station_metadata=metadata,horizons=horizons)
    results=[]; selections=[]; predictions=[]
    for h in horizons:
        suffix='health_forecast_long_horizon/current' if h>24 else 'health_forecast'
        expected=json.loads((ROOT/f'data/eval/{suffix}/health_forecast_split_digests.json').read_text())
        saved=joblib.load(ROOT/f'data/model/{suffix}/health_forecast_forecast_transmitting_origin_{h}h.joblib')
        frame=hf.health_forecast_horizon_frame(bundle,h)
        split=split_timestamp_partitions(frame,target_columns=('target_health_total','target_delta_health'),horizon_h=h)
        for p in ('train','validation','test'):
            assert hf._split_digest(split[p])==expected[str(h)][p],f'{h}h {p} membership changed'
        train,valid,test=[hf._regime_subset(split[p],'transmitting_origin') for p in ('train','validation','test')]
        combined=pd.concat([train,valid],ignore_index=True)
        for baseline,column in [('roll_forward','baseline_no_new_incident_level'),
                                ('persistence','baseline_persistence_level'),('trend','baseline_trend_level')]:
            def fit(part):
                return hf._fit_health_forecast_model(saved.family,part,
                    (part.target_health_total-part[column]).to_numpy(float),
                    feature_columns=saved.feature_columns,iterations=saved.iterations,
                    sample_weight=hf._recency_weights(part,saved.recency_half_life_days))
            model=fit(train)
            correction=model.predict(valid)
            grid=[]
            for alpha in hf.HEALTH_FORECAST_ALPHA_GRID:
                estimate=np.clip(valid[column].to_numpy(float)+alpha*correction,0,100)
                grid.append(dict(alpha=alpha,**regression_metrics(valid.target_health_total.to_numpy(float),estimate)))
            choice=min(grid,key=lambda r:(r['mae'],r['alpha']))
            selections.extend(dict(horizon=h,baseline=baseline,**r) for r in grid)
            final=fit(combined)
            estimate=np.clip(test[column].to_numpy(float)+choice['alpha']*final.predict(test),0,100)
            actual=test.target_health_total.to_numpy(float)
            results.append(dict(horizon=h,baseline=baseline,model=saved.family,iterations=saved.iterations,
                features=saved.feature_set,alpha=choice['alpha'],validation_mae=choice['mae'],
                validation_rmse=choice['rmse'],validation_r2=choice['r2'],
                baseline_test_mae=regression_metrics(actual,test[column].to_numpy(float))['mae'],
                **regression_metrics(actual,estimate),test_start=str(test.hour_utc.min()),test_end=str(test.hour_utc.max()),n=len(test)))
            predictions.append(pd.DataFrame(dict(horizon=h,baseline=baseline,station_id=test.station_id,
                hour=test.hour_utc,actual=actual,predicted=estimate)))
            joblib.dump(dict(model=final,baseline_column=column,alpha=choice['alpha']),out/f'{baseline}_{h}h.joblib')
            print(json.dumps(results[-1]),flush=True)
            pd.DataFrame(results).to_csv(out/'metrics.csv',index=False)

        control=predictions[-3].predicted.to_numpy()
        np.testing.assert_allclose(control,saved.predict_health(test),rtol=1e-7,atol=1e-7)
    pd.DataFrame(selections).to_csv(out/'validation_alpha_grid.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(out/'predictions.parquet',index=False)
    (out/'design.json').write_text(json.dumps(dict(
        horizons=list(horizons),population='transmitting origins; original purged temporal partitions verified by digest',
        method='Same saved regression family, features, iterations and recency; retrain each residual target; validation MAE selects alpha; refit train+validation',
        limitations='Exploratory previously evaluated May-June period. Hyperparameters originally selected for roll-forward; not a full baseline-specific search.',
        control_reproduces_saved_predictions=True),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--long-horizon',action='store_true')
    parser.add_argument('--horizons',type=int,nargs='+')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.long_horizon,args.horizons,args.output)
