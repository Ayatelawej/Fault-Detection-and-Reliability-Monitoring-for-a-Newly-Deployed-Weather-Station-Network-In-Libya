"""Four-family validation-first health comparison on unchanged purged splits."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from src.availability import health_forecast as hf
from src.availability.risk_eval import split_timestamp_partitions,regression_metrics
from scripts.experiment_report_linear_forecasts import prep

OUT=ROOT/'data/eval/final_forecast_comparison_20260924'

def metrics(y,p):
    m=regression_metrics(y,p)
    return dict(**m,normalized_score_pct=100-m['mae'],band_accuracy_pct=100*np.mean(np.searchsorted([40,60,80],y,side='right')==np.searchsorted([40,60,80],p,side='right')))

def run():
    OUT.mkdir(exist_ok=False);start=time.perf_counter();hf.HEALTH_FORECAST_CATBOOST_PARAMETERS['thread_count']=2
    previous=pd.read_csv(ROOT/'data/eval/report_linear_forecasts_20260924/metrics.csv')
    horizons=(1,3,6,12,24,48,72,96,120,144,168)
    bundle=hf.build_health_forecast_dataset(pd.read_parquet(ROOT/'data/processed/station_health_scores.parquet'),
        station_metadata=pd.read_csv(ROOT/'data/merged/station_hourly_merged.csv',usecols=['station_id','elevation']),horizons=horizons)
    results=[];selections=[];ledgers=[];grid=[]
    for h in horizons:
        folder='health_forecast' if h<=24 else 'health_forecast_long_horizon/current'
        saved=joblib.load(ROOT/f'data/model/{folder}/health_forecast_forecast_transmitting_origin_{h}h.joblib')
        expected=json.loads((ROOT/f'data/eval/{folder}/health_forecast_split_digests.json').read_text())[str(h)]
        frame=hf.health_forecast_horizon_frame(bundle,h)
        split=split_timestamp_partitions(frame,target_columns=('target_health_total','target_delta_health'),horizon_h=h)
        for p in ('train','validation','test'):assert hf._split_digest(split[p])==expected[p]
        tr,va,te=[hf._regime_subset(split[p],'transmitting_origin') for p in ('train','validation','test')]
        combined=pd.concat([tr,va],ignore_index=True);cols=saved.feature_columns
        column={'persistence':'baseline_persistence_level','no_new_incident_roll_forward':'baseline_no_new_incident_level','recent_trend_24h':'baseline_trend_level'}[saved.residual_baseline]
        for family in ('Linear regression','Ridge','HGB','CatBoost'):
            if family in ('Linear regression','Ridge'):
                choice=previous.loc[previous.horizon_h.eq(h)&previous.model.eq(family)].iloc[0].to_dict()
                form=choice['formulation'];alpha=choice['correction_weight'];strength=choice['ridge_strength']
                def linear_fit(part):
                    estimator=LinearRegression() if family=='Linear regression' else Ridge(alpha=strength,solver='lsqr',tol=1e-6)
                    pipeline=Pipeline([('prepare',prep(cols)),('model',estimator)])
                    target=part.target_health_total.to_numpy(float)-(part[column].to_numpy(float) if form=='residual' else 0)
                    w=hf._recency_weights(part,saved.recency_half_life_days)
                    pipeline.fit(part,target,**({} if w is None else {'model__sample_weight':w}))
                    return hf.FittedHealthForecastModel('linear' if family=='Linear regression' else 'ridge',cols,pipeline)
                model=linear_fit(tr);final=linear_fit(combined);iterations=None
            else:
                best=None
                for form in ('direct','residual'):
                    target=tr.target_health_total.to_numpy(float)-(tr[column].to_numpy(float) if form=='residual' else 0)
                    for iterations in (100,200,300):
                        model=hf._fit_health_forecast_model('catboost' if family=='CatBoost' else 'hist_gradient_boosting',tr,target,
                            feature_columns=cols,iterations=iterations,sample_weight=hf._recency_weights(tr,saved.recency_half_life_days))
                        raw=model.predict(va)
                        for alpha in ((1.,) if form=='direct' else hf.HEALTH_FORECAST_ALPHA_GRID):
                            pv=np.clip((va[column].to_numpy(float) if form=='residual' else 0)+alpha*raw,0,100)
                            vm=metrics(va.target_health_total.to_numpy(float),pv)
                            row=dict(horizon_h=h,model=family,formulation=form,iterations=iterations,alpha=alpha,**vm);grid.append(row)
                            key=(vm['mae'],iterations,alpha,form)
                            if best is None or key<best[0]:best=(key,model,row)
                _,model,choice=best;form=choice['formulation'];alpha=choice['alpha'];iterations=choice['iterations']
                target=combined.target_health_total.to_numpy(float)-(combined[column].to_numpy(float) if form=='residual' else 0)
                final=hf._fit_health_forecast_model('catboost' if family=='CatBoost' else 'hist_gradient_boosting',combined,target,
                    feature_columns=cols,iterations=iterations,sample_weight=hf._recency_weights(combined,saved.recency_half_life_days))
            for fitted in (model,final):
                fitted.alpha=alpha;fitted.final_policy='direct_regression' if form=='direct' else 'learned_residual'
                fitted.residual_baseline=saved.residual_baseline;fitted.horizon_h=h;fitted.regime='transmitting_origin'
                fitted.feature_set=saved.feature_set;fitted.recency_half_life_days=saved.recency_half_life_days;fitted.iterations=iterations
            validation_metrics=metrics(va.target_health_total.to_numpy(float),model.predict_health(va))
            if family in ('Linear regression','Ridge'):np.testing.assert_allclose(validation_metrics['mae'],choice['validation_mae'],atol=1e-7)
            for part,partframe,estimator in [('validation',va,model),('test',te,final)]:
                prediction=estimator.predict_health(partframe);actual=partframe.target_health_total.to_numpy(float)
                results.append(dict(horizon_h=h,model=family,partition=part,formulation=form,rows=len(partframe),**metrics(actual,prediction)))
                ledgers.append(pd.DataFrame(dict(horizon_h=h,model=family,partition=part,station_id=partframe.station_id,hour_utc=partframe.hour_utc,actual=actual,prediction=prediction)))
            selections.append(dict(horizon_h=h,model=family,formulation=form,iterations=iterations,alpha=alpha,validation_mae=validation_metrics['mae']))
            joblib.dump(final,OUT/f'{family.replace(" ","_")}_{h}h.joblib')
            print(f'{h}h {family}: val MAE {validation_metrics["mae"]:.4f}',flush=True)
            pd.DataFrame(results).to_csv(OUT/'metrics.csv',index=False)
    f=pd.DataFrame(results);v=f[f.partition.eq('validation')]
    chosen=v.sort_values(['mae','model']).groupby('horizon_h',sort=True).first().reset_index()
    chosen.to_csv(OUT/'chosen.csv',index=False);pd.DataFrame(selections).to_csv(OUT/'selection.csv',index=False)
    pd.DataFrame(grid).to_csv(OUT/'nonlinear_validation_grid.csv',index=False)
    pd.concat(ledgers).to_parquet(OUT/'predictions.parquet',index=False)
    (OUT/'audit.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-start,selection='Minimum validation MAE per horizon; ties by model name.',
        protocol='Shared numeric features/recency and original horizon-purged time splits. Direct or correction-to-fixed-baseline. Linear/Ridge reuse verified prior grid; HGB/CatBoost each search 100/200/300 iterations and correction weight.',
        bands='Critical <40; Degraded 40–<60; Watch 60–<80; Healthy >=80. Separate secondary metric.',
        limitations='Features/recency and baseline per horizon inherited from earlier development; not exhaustive feature search. Previously examined periods; not fresh prospective validation.'),indent=2),encoding='utf-8')
    print(chosen[['horizon_h','model','mae']].to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
