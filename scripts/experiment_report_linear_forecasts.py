"""Ridge and ordinary linear health forecasts on original purged partitions."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import Ridge,LinearRegression
from threadpoolctl import threadpool_limits
from src.availability import health_forecast as hf
from src.availability.risk_eval import split_timestamp_partitions,regression_metrics


def prep(cols):
    return ColumnTransformer([('numeric',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()),list(cols)),
        ('station',OneHotEncoder(handle_unknown='ignore',drop='first',sparse_output=False),['station_id'])])


def run():
    started=time.perf_counter();out=ROOT/'data/eval/report_linear_forecasts_20260924';out.mkdir(exist_ok=False)
    selected=pd.read_csv(ROOT/'data/eval/health_forecast_release_20260920/selected_test_metrics.csv')
    horizons=tuple(int(h) for h in selected.horizon_h)
    paths={h:ROOT/('data/model/health_forecast' if h<=24 else 'data/model/health_forecast_long_horizon/current')/f'health_forecast_forecast_transmitting_origin_{h}h.joblib' for h in horizons}
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()}
    bundle=hf.build_health_forecast_dataset(pd.read_parquet(ROOT/'data/processed/station_health_scores.parquet'),
        station_metadata=pd.read_csv(ROOT/'data/merged/station_hourly_merged.csv',usecols=['station_id','elevation']),horizons=horizons)
    rows=[];grid=[];predictions=[]
    for h in horizons:
        saved=joblib.load(paths[h]);folder='health_forecast' if h<=24 else 'health_forecast_long_horizon/current'
        expected=json.loads((ROOT/f'data/eval/{folder}/health_forecast_split_digests.json').read_text())[str(h)]
        frame=hf.health_forecast_horizon_frame(bundle,h)
        split=split_timestamp_partitions(frame,target_columns=('target_health_total','target_delta_health'),horizon_h=h)
        for p in ('train','validation','test'):assert hf._split_digest(split[p])==expected[p]
        tr,va,te=[hf._regime_subset(split[p],'transmitting_origin') for p in ('train','validation','test')]
        combined=pd.concat([tr,va],ignore_index=True)
        cols=saved.feature_columns
        p=prep(cols);xt=p.fit_transform(tr);xv=p.transform(va)
        finalprep=prep(cols);xf=finalprep.fit_transform(combined);xe=finalprep.transform(te)
        w=hf._recency_weights(tr,saved.recency_half_life_days);wf=hf._recency_weights(combined,saved.recency_half_life_days)
        column={'persistence':'baseline_persistence_level','no_new_incident_roll_forward':'baseline_no_new_incident_level','recent_trend_24h':'baseline_trend_level'}[saved.residual_baseline]
        yv=va.target_health_total.to_numpy(float);yt=te.target_health_total.to_numpy(float)
        control=saved.predict_health(te);cm=regression_metrics(yt,control);old=selected.loc[selected.horizon_h.eq(h)].iloc[0]
        np.testing.assert_allclose([cm[k] for k in ('mae','rmse','r2')],[old[k] for k in ('mae','rmse','r2')],atol=1e-7)
        rows.append(dict(horizon_h=h,model='Deployed '+saved.family,formulation='residual',validation_mae=old.validation_mae,
            test_mae=cm['mae'],test_rmse=cm['rmse'],test_r2=cm['r2'],test_rows=len(te)))
        for family in ('Linear regression','Ridge'):
            best=None
            for formulation in ('direct','residual'):
                y=tr.target_health_total.to_numpy(float)-(tr[column].to_numpy(float) if formulation=='residual' else 0)
                for strength in ((.1,1.,10.,100.,1000.) if family=='Ridge' else (0.,)):
                    m=Ridge(alpha=strength,solver='lsqr',tol=1e-6) if family=='Ridge' else LinearRegression()
                    m.fit(xt,y,sample_weight=w);pv=m.predict(xv)
                    for alpha in (hf.HEALTH_FORECAST_ALPHA_GRID if formulation=='residual' else (1.,)):
                        estimate=np.clip((va[column].to_numpy(float) if formulation=='residual' else 0)+alpha*pv,0,100)
                        vm=regression_metrics(yv,estimate)
                        choice=dict(horizon_h=h,model=family,formulation=formulation,ridge_strength=strength,correction_weight=alpha,**{f'validation_{k}':v for k,v in vm.items()})
                        grid.append(choice)
                        key=(vm['mae'],strength,alpha)
                        if best is None or key<best[0]:best=(key,choice)
            choice=best[1];formulation=choice['formulation'];strength=choice['ridge_strength'];alpha=choice['correction_weight']
            final=Ridge(alpha=strength,solver='lsqr',tol=1e-6) if family=='Ridge' else LinearRegression()
            target=combined.target_health_total.to_numpy(float)-(combined[column].to_numpy(float) if formulation=='residual' else 0)
            final.fit(xf,target,sample_weight=wf)
            estimate=np.clip((te[column].to_numpy(float) if formulation=='residual' else 0)+alpha*final.predict(xe),0,100)
            tm=regression_metrics(yt,estimate)
            rows.append(dict(**choice,test_mae=tm['mae'],test_rmse=tm['rmse'],test_r2=tm['r2'],test_rows=len(te),features=len(cols),recency_half_life_days=saved.recency_half_life_days))
            joblib.dump(dict(preprocessor=finalprep,model=final,selection=choice,baseline_column=column),out/f'{family.split()[0].lower()}_{h}h.joblib')
            predictions.append(pd.DataFrame(dict(horizon_h=h,model=family,station_id=te.station_id,hour_utc=te.hour_utc,actual=yt,prediction=estimate)))
            print(f'{h}h {family}: val MAE {choice["validation_mae"]:.3f}; test MAE {tm["mae"]:.3f}',flush=True)
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(grid).to_csv(out/'validation_grid.csv',index=False)
    pd.concat(predictions).to_parquet(out/'test_predictions.parquet',index=False)
    assert hashes=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()}
    (out/'audit.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-started,protected_unchanged=True,hashes=hashes,
        selection='Validation MAE chooses direct versus residual, Ridge strength and residual correction weight. Preprocessing fits train only; final models refit train+validation.',
        limitations='Same feature scope/recency as selected nonlinear models; not a full linear-specific feature search. Transmitting-origin only. Previously examined test periods; no deployment changes.'),indent=2),encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
