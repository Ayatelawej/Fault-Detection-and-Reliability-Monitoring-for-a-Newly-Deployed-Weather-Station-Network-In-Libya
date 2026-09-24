"""Full-development refit of the validated episode-target configuration."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.model.final_reason_codes import (ROOT, FREEZE, EPISODE_MODEL_DIR, EPISODE_VERSION,
    EPISODE_OUTPUT_POLICY, load_observations, build_features)
from src.model.hourly_baseline import load_hourly_tensor, _fault_groups
from src.model.reason_code_rebuild import MECH, COMP, SEED, feature_views, fit_estimator, sha


def train_episode_release(model_dir=EPISODE_MODEL_DIR):
    model_dir=Path(model_dir)
    if model_dir.exists(): raise FileExistsError(f'Refusing to overwrite release: {model_dir}')
    inputs=dict(tensor=ROOT/'data/hourly_detection/one_hour_final/hourly_detection_01h.npz',
        raw=ROOT/'data/merged/station_hourly_merged.csv', references=ROOT/'data/features/external_residuals.parquet',
        selections=ROOT/'data/eval/episode_reasons_three_splits_20260924_v2/selection.json')
    hashes={k:sha(p) for k,p in inputs.items()}
    choices=[r for r in json.loads(inputs['selections'].read_text()) if r['split']=='temporal_holdout']
    z=load_hourly_tensor(inputs['tensor'])
    hours=pd.to_datetime(z['hour'],utc=True)
    assert hours.max()<=FREEZE
    index=pd.MultiIndex.from_arrays([z['station_id'],hours],names=['station_id','hour'])
    features,_=build_features(load_observations(inputs['raw'],inputs['references'],FREEZE))
    assert index.isin(features.index).all()
    names=list(features.columns); x=features.reindex(index).to_numpy('float32')
    fault=np.asarray(z['y_binary'],int)
    groups=np.array([f'normal:{s}:{str(t)[:10]}' for s,t in index],dtype=object)
    for key,ii in _fault_groups(fault,z['source_episode_ids']).items(): groups[ii]='event:'+key
    negative=np.flatnonzero(fault==0)
    if len(negative)>16000:
        negative=np.sort(np.random.default_rng(SEED).choice(negative,16000,replace=False))
    fit=np.sort(np.r_[np.flatnonzero(fault==1),negative])
    heads={}
    with threadpool_limits(limits=2):
        for axis,labels in [('mechanism',MECH),('component',COMP)]:
            original=list(z[axis+'_label_names'])
            yy=np.asarray(z['y_'+axis],int)[:,[original.index(n) for n in labels]]
            np.testing.assert_array_equal(yy.any(axis=1),fault.astype(bool))
            for j,label in enumerate(labels):
                choice=next(r for r in choices if r['axis']==axis and r['label']==label)
                if choice['status']!='fitted': raise ValueError(f'No frozen setting for {axis}:{label}')
                views=feature_views(names,axis,label)
                models=[fit_estimator(x[fit][:,cols],yy[fit,j],groups[fit]) for cols in views]
                heads[f'{axis}:{label}']=dict(models=models,views=views,
                    weights=np.asarray(choice['weights']),threshold=choice['threshold'])
                print(f'Refit episode head {axis}/{label}',flush=True)
    assert hashes=={k:sha(p) for k,p in inputs.items()}
    manifest=dict(version=EPISODE_VERSION,target='original_episode_reasons',output_policy=EPISODE_OUTPUT_POLICY,
        trained_through=str(hours.max()),feature_count=len(names),training_hours=len(index),
        training_fault_hours=int(fault.sum()),resolved_fault_hours=int(fault.sum()),input_hashes=hashes,
        selection='Fixed HGB settings; per-head fusion/thresholds selected on February in the blocked pilot; full development refit.',
        held_out_results='Three-split pilot scores belong to separately fitted models, not this full refit.',
        interpretation='Predict event-associated reference reasons using current/past features; not confirmation that each rule fires now.',
        limitation='Episode targets may be retrospectively established. Archived reference arrival times unverified.',
        seed=SEED,threads=2,july_used_for_fitting_or_threshold_selection=False)
    model_dir.mkdir(parents=True,exist_ok=False)
    joblib.dump(dict(version=EPISODE_VERSION,feature_names=names,heads=heads,manifest=manifest),model_dir/'reason_heads.joblib')
    manifest['model_sha256']=sha(model_dir/'reason_heads.joblib')
    (model_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (model_dir/'selection.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
    print(f'Saved {model_dir}',flush=True)
