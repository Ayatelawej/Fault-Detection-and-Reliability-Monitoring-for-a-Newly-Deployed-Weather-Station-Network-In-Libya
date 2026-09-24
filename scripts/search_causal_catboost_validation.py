"""Small validation-only CatBoost comparison on the saved causal feature experiment."""
from pathlib import Path
import sys
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
import json
from catboost import CatBoostClassifier
from scripts.search_causal_binary_validation import select
from threadpoolctl import threadpool_limits

def run():
    out=ROOT/'data/eval/causal_validation_search_20260923'
    destination=out/'catboost_validation_search.csv'
    if destination.exists(): raise FileExistsError(destination)
    cache=joblib.load(out/'feature_cache.joblib')
    splits=cache['splits']; y=cache['y']; rows=[]; best=-1
    trials=[('previous',6,2),('expanded',6,2),('expanded',6,4),('expanded',8,2)]
    for feature_set,depth,weight in trials:
        x,names=cache['feature_sets'][feature_set]
        # Fit numerical missing-value medians on training rows only.
        train=pd.DataFrame(x[splits['train']])
        medians=train.median().fillna(0).to_numpy(dtype=np.float32)
        xt=np.where(np.isfinite(x[splits['train']]),x[splits['train']],medians)
        xv=np.where(np.isfinite(x[splits['validation']]),x[splits['validation']],medians)
        model=CatBoostClassifier(iterations=600,depth=depth,learning_rate=.05,l2_leaf_reg=5,
            loss_function='Logloss',class_weights=[1,weight],random_seed=2026,thread_count=2,
            allow_writing_files=False,verbose=False)
        model.fit(xt,y[splits['train']])
        p=model.predict_proba(xv)[:,1]
        threshold,metric=select(y[splits['validation']],p)
        minimum=min(metric[k] for k in ('precision','recall','f1'))
        row=dict(feature_set=feature_set,depth=depth,weight=weight,iterations=600,threshold=threshold,minimum=minimum,**metric)
        rows.append(row); pd.DataFrame(rows).to_csv(destination,index=False)
        if minimum>best:
            best=minimum
            joblib.dump(dict(estimator=model,feature_names=names,medians=medians,selection=row),out/'best_catboost_model.joblib')
            pd.DataFrame(dict(target=y[splits['validation']],probability=p)).to_csv(out/'best_catboost_validation_predictions.csv',index=False)
        print(json.dumps(row),flush=True)
        if minimum>.8: break
    print(f'CatBoost search complete. Best validation minimum: {best:.6f}',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2): run()
