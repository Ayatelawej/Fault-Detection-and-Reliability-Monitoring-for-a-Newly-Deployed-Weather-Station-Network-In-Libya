"""Bounded feature-mix comparison and validation-only rule reliance audit."""
from pathlib import Path
from dataclasses import replace, asdict
import os
import sys
import json
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='2'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from scripts.experiment_causal_rule_indicators import run as prepare
from scripts.experiment_raw_observation_view import CHANNELS
from src.model.hourly_baseline import HourlyBaselineConfig, make_classifier, binary_metrics
from src.model.hourly_calibration import calibrate_split, validation_grid_frame
from src.model.feature_spec import RULE_EVIDENCE_FLAGS


def history_features(frame):
    columns={}
    for channel in frame:
        s=frame[channel]
        for lag in (1,3,6):
            columns[f'{channel}_change_{lag}h']=s-s.shift(lag)
        same=s.notna() & s.shift().notna() & s.diff().abs().le(1e-6)
        run=(~same).cumsum()
        duration=s.groupby(run).cumcount().add(1).clip(upper=168).astype(float)
        columns[f'{channel}_unchanged_hours']=duration.where(s.notna())
    return pd.DataFrame(columns,index=frame.index)


def run(prepare_only=False):
    out=ROOT/'data/eval/causal_feature_mix_20260923'
    if not prepare_only:
        out.mkdir(parents=True,exist_ok=False)
    d=prepare(prepare_only=True)
    retained=d['retained']; splits=d['splits']; y=d['y'][retained]
    base=d['x'][:,d['base']]
    rules=d['rules'].loc[:,RULE_EVIDENCE_FLAGS].to_numpy(dtype=np.float32)
    base_names=[d['names'][i] for i in d['base']]+list(RULE_EVIDENCE_FLAGS)
    corrected=np.column_stack([base,rules])[retained]
    assert corrected.shape[1]==48
    raw=d['raw'].set_index(['station_id','hour_utc'])[CHANNELS].copy()
    angle=np.deg2rad(raw.pop('winddir_avg_deg'))
    raw['winddir_sin']=np.sin(angle); raw['winddir_cos']=np.cos(angle)
    raw=raw.replace([np.inf,-np.inf],np.nan)
    extra_frames=[]
    for station,part in raw.groupby(level='station_id',sort=False):
        part=part.droplevel('station_id').sort_index()
        part=part.reindex(pd.date_range(part.index.min(),part.index.max(),freq='h',name='hour_utc'))
        features=history_features(part)
        cut=len(part)//2
        pd.testing.assert_frame_equal(features.iloc[:cut],history_features(part.iloc[:cut]))
        features['station_id']=station
        extra_frames.append(features.reset_index().set_index(['station_id','hour_utc']))
    history=pd.concat(extra_frames).reindex(d['index'])
    raw=raw.reindex(d['index'])
    raw=pd.concat([raw,raw.notna().astype(float).add_suffix('_present')],axis=1)
    raw_values=raw.to_numpy(dtype=np.float32)[retained]
    history_values=history.to_numpy(dtype=np.float32)[retained]
    variants={
        'corrected_deduplicated':(corrected,base_names),
        'plus_raw':(np.column_stack([corrected,raw_values]),base_names+['raw:'+c for c in raw.columns]),
        'plus_past_changes':(np.column_stack([corrected,raw_values,history_values]),base_names+['raw:'+c for c in raw.columns]+['past:'+c for c in history.columns])}
    if prepare_only:
        return dict(variants=variants, data=d, labels=y, splits=splits)
    rows=[]; choices={}
    for key,(x,names) in variants.items():
        result=calibrate_split(x,y,splits,HourlyBaselineConfig())
        choice=result['best_balanced']; choices[key]=choice
        row=dict(variant=key,columns=x.shape[1],threshold=choice['threshold'],class_weight=choice['fault_class_weight'],validation_minimum=choice['validation_minimum_metric'])
        for part,metrics in [('validation',choice['validation']),('test',result['final_test'])]:
            row.update({f'{part}_{k}':metrics[k] for k in ('precision','recall','f1','accuracy')})
        rows.append(row)
        validation_grid_frame(result).to_csv(out/f'{key}_validation_grid.csv',index=False)
        pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
        print(json.dumps(row),flush=True)
    selected=max(choices,key=lambda k:(choices[k]['validation_minimum_metric'],choices[k]['validation']['f1'],choices[k]['validation']['precision'],choices[k]['validation']['recall']))
    x,names=variants[selected]; choice=choices[selected]
    config=replace(HourlyBaselineConfig(),fault_class_weight=float(choice['fault_class_weight']),threshold=float(choice['threshold']))
    model=make_classifier(config); model.fit(x[splits['train']],y[splits['train']])
    xv=x[splits['validation']]; yv=y[splits['validation']]
    baseline=binary_metrics(yv,model.predict_proba(xv)[:,1],config.threshold)
    np.testing.assert_allclose(baseline['f1'],choice['validation']['f1'],atol=1e-12)
    joblib.dump(dict(estimator=model,feature_names=names,config=asdict(config)),out/'selected_model.joblib')
    ii=retained[splits['test']]
    pd.DataFrame(dict(station_id=d['index'].get_level_values('station_id')[ii],hour=d['index'].get_level_values('hour_utc')[ii],
        target=y[splits['test']],probability=model.predict_proba(x[splits['test']])[:,1])).to_parquet(out/'selected_test_predictions.parquet',index=False)
    print(f'Selected by validation: {selected}. Auditing all 19 rule indicators on validation only.',flush=True)
    stations=d['index'].get_level_values('station_id').to_numpy()[retained][splits['validation']]
    station_rows=[np.flatnonzero(stations==s) for s in np.unique(stations)]
    rng=np.random.default_rng(2026); importance=[]
    for flag in RULE_EVIDENCE_FLAGS:
        column=names.index(flag); drops=[]
        for repeat in range(5):
            perturbed=xv.copy()
            for idx in station_rows:
                perturbed[idx,column]=xv[rng.permutation(idx),column]
            metric=binary_metrics(yv,model.predict_proba(perturbed)[:,1],config.threshold)
            drops.append(100*(baseline['f1']-metric['f1']))
        keep=[i for i in range(x.shape[1]) if i!=column]
        reduced=make_classifier(config)
        reduced.fit(x[splits['train']][:,keep],y[splits['train']])
        reduced_metric=binary_metrics(yv,reduced.predict_proba(xv[:,keep])[:,1],config.threshold)
        importance.append(dict(feature=flag,validation_active_hours=int(np.sum(xv[:,column]>.5)),
            permutation_f1_drop_pp=float(np.mean(drops)),permutation_repeat_std_pp=float(np.std(drops)),
            removal_refit_f1_drop_pp=100*(baseline['f1']-reduced_metric['f1'])))
        pd.DataFrame(importance).to_csv(out/'rule_importance.csv',index=False)
        print(f"Rule {len(importance)}/19: {flag}; shuffle drop {np.mean(drops):.2f} pp; removal/refit drop {importance[-1]['removal_refit_f1_drop_pp']:.2f} pp",flush=True)
    table=pd.DataFrame(rows); table['selected_by_validation']=table.variant.eq(selected)
    table.to_csv(out/'comparison.csv',index=False)
    (out/'design.json').write_text(json.dumps(dict(selected_by_validation=selected,
        partition_index_sha256=d['digests'],history_prefix_checks_passed=True,
        new_history='Past-only changes at 1/3/6 hours; consecutive unchanged readings within 1e-6, missing readings break runs, capped at 168h. Wind direction encoded as sine/cosine.',
        rule_audit='Selected model; five within-station validation shuffles per rule; one-column removal and refit at fixed selected settings, validation scored only. Correlated rules and derived group flags are not removed together; effects are not additive or causal.',
        limitations='Only HGB. Retrospective reference labels, existing continuous features and cached statistical fitting unchanged. Previously examined holdout; not a full leakage audit. No promise of independent hardware validity.',
        deployment_changed=False),indent=2),encoding='utf-8')
    print(table.to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        run()
