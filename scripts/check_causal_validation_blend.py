"""Validation-only blend of saved HGB/CatBoost candidates, plus search summary."""
from pathlib import Path
import sys
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import json
from scripts.search_causal_binary_validation import select


def main():
    out=ROOT/'data/eval/causal_validation_search_20260923'
    hgb=pd.read_csv(out/'best_validation_predictions.csv')
    cat=pd.read_csv(out/'best_catboost_validation_predictions.csv')
    np.testing.assert_array_equal(hgb.target,cat.target)
    rows=[]
    for weight in np.linspace(0,1,21):
        p=weight*hgb.probability.to_numpy()+(1-weight)*cat.probability.to_numpy()
        threshold,m=select(hgb.target.to_numpy(),p)
        rows.append(dict(hgb_weight=float(weight),catboost_weight=float(1-weight),threshold=threshold,
            minimum=min(m[k] for k in ('precision','recall','f1')),**m))
    table=pd.DataFrame(rows)
    table.to_csv(out/'validation_blend_grid.csv',index=False)
    best=table.sort_values(['minimum','f1','precision','recall'],ascending=False).iloc[0].to_dict()
    (out/'blend_selection.json').write_text(json.dumps(best,indent=2),encoding='utf-8')
    print(json.dumps(best),flush=True)


if __name__=='__main__': main()
