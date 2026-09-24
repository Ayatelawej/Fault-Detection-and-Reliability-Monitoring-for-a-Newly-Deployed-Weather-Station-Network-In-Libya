"""Matched direct-score versus deployed residual forecast pilot; no promotion."""
from pathlib import Path
import os
import sys
import json
import time
import hashlib
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from src.availability import health_forecast as hf
from src.availability.risk_eval import split_timestamp_partitions, regression_metrics


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    started=time.perf_counter()
    horizons=(24,48,72,168)
    out=ROOT/'data/eval/direct_health_forecast_20260924'
    out.mkdir(exist_ok=False)
    hf.HEALTH_FORECAST_CATBOOST_PARAMETERS['thread_count']=2
    release=ROOT/'data/eval/health_forecast_release_20260920/selected_test_metrics.csv'
    selected=pd.read_csv(release)
    model_paths={h:ROOT/('data/model/health_forecast' if h<=24 else 'data/model/health_forecast_long_horizon/current')/f'health_forecast_forecast_transmitting_origin_{h}h.joblib' for h in horizons}
    protected=list(model_paths.values())+[ROOT/'data/eval/july_2026_health_forecast/july_health_forecast_predictions.parquet']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    scores=pd.read_parquet(ROOT/'data/processed/station_health_scores.parquet')
    metadata=pd.read_csv(ROOT/'data/merged/station_hourly_merged.csv',usecols=['station_id','elevation'])
    bundle=hf.build_health_forecast_dataset(scores,station_metadata=metadata,horizons=horizons)
    rows=[]; predictions=[]; membership=[]
    for h in horizons:
        folder='health_forecast' if h<=24 else 'health_forecast_long_horizon/current'
        expected=json.loads((ROOT/f'data/eval/{folder}/health_forecast_split_digests.json').read_text())[str(h)]
        saved=joblib.load(model_paths[h])
        frame=hf.health_forecast_horizon_frame(bundle,h)
        split=split_timestamp_partitions(frame,target_columns=('target_health_total','target_delta_health'),horizon_h=h)
        for part in ('train','validation','test'):
            digest=hf._split_digest(split[part]); assert digest==expected[part]
            membership.append(dict(horizon=h,partition=part,digest=digest))
        train,valid,test=[hf._regime_subset(split[p],'transmitting_origin') for p in ('train','validation','test')]
        combined=pd.concat([train,valid],ignore_index=True)
        baseline_col={'persistence':'baseline_persistence_level','no_new_incident_roll_forward':'baseline_no_new_incident_level','recent_trend_24h':'baseline_trend_level'}[saved.residual_baseline]
        assert not any('baseline' in c for c in saved.feature_columns)
        def fit(part,direct):
            target=part.target_health_total.to_numpy(float)
            if not direct: target=target-part[baseline_col].to_numpy(float)
            return hf._fit_health_forecast_model(saved.family,part,target,feature_columns=saved.feature_columns,
                iterations=saved.iterations,sample_weight=hf._recency_weights(part,saved.recency_half_life_days))
        for mode in ('deployed_residual','direct_score'):
            direct=mode=='direct_score'
            model=fit(train,direct)
            vp=model.predict(valid)
            if not direct: vp=valid[baseline_col].to_numpy(float)+saved.alpha*vp
            vm=regression_metrics(valid.target_health_total.to_numpy(float),np.clip(vp,0,100))
            if direct:
                final=fit(combined,True)
                prediction=np.clip(final.predict(test),0,100)
                joblib.dump(dict(model=final,prediction_contract='clip(model.predict(frame),0,100); do not use residual predict_health'),out/f'direct_{h}h.joblib')
            else:
                prediction=saved.predict_health(test)
                expected_row=selected.loc[selected.horizon_h.eq(h)].iloc[0]
                np.testing.assert_allclose(vm['mae'],expected_row.validation_mae,atol=1e-7)
            actual=test.target_health_total.to_numpy(float)
            tm=regression_metrics(actual,prediction)
            if not direct:
                np.testing.assert_allclose([tm[k] for k in ('mae','rmse','r2')],[expected_row[k] for k in ('mae','rmse','r2')],atol=1e-7)
            rows.append(dict(horizon_h=h,method=mode,family=saved.family,features=len(saved.feature_columns),iterations=saved.iterations,
                validation_mae=vm['mae'],validation_rmse=vm['rmse'],validation_r2=vm['r2'],
                test_mae=tm['mae'],test_rmse=tm['rmse'],test_r2=tm['r2'],test_rows=len(test)))
            predictions.append(pd.DataFrame(dict(horizon_h=h,method=mode,station_id=test.station_id,hour_utc=test.hour_utc,actual=actual,prediction=prediction)))
            pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
            print(json.dumps(rows[-1]),flush=True)
    pd.concat(predictions).to_parquet(out/'test_predictions.parquet',index=False)
    pd.DataFrame(membership).to_csv(out/'split_digests.csv',index=False)
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in protected}
    (out/'audit.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-started,protected_unchanged=True,
        protected_hashes=before,control_validation_and_test_reproduced=True,horizons=horizons,
        design='Same model family, feature columns, iterations, recency weights, purged split; residual correction versus direct score target. Test models fit train+validation.',
        selection='Compare validation MAE; no deployment changes.',
        limitations='Model settings originally selected for residual learning; bounded pilot, not full direct-model search. Previously examined test periods. Transmitting-origin only.'),indent=2),encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=2): run()
