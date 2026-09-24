"""Split broad context into engineered observations and availability/context."""
from scripts.experiment_raw_observation_view import (
    ROOT, np, pd, joblib, json, threadpool_limits, load_hourly_tensor,
    flatten_hourly_features, _fault_groups, blocked_split, validate_split,
    sha, make_classifier, HourlyBaselineConfig, binary_metrics, grid_select,
)


def run():
    out = ROOT / 'data/eval/engineered_observation_view_blocked_20260917'
    out.mkdir(exist_ok=False)
    source = ROOT / 'data/eval/blocked_model_comparison_20260916_v2/models/ef_hgb.joblib'
    before = sha(source)
    bundle = joblib.load(source)
    model = bundle['estimator']
    z = load_hourly_tensor(ROOT / 'data/hourly_detection/one_hour_final/hourly_detection_01h.npz')
    x, names, feature_groups = flatten_hourly_features(z)
    assert names == bundle['feature_names']
    y = np.asarray(z['y_binary'], int)
    hours = pd.to_datetime(z['hour'], utc=True)
    groups = np.array([f'normal:{s}:{str(t)[:10]}' for s,t in zip(z['station_id'],hours)],object)
    for key, ii in _fault_groups(y,z['source_episode_ids']).items():
        groups[ii] = 'event:'+key
    splits = blocked_split(hours,groups)
    validate_split(hours,groups,splits)
    availability = {'continuous:n_neighbors_present_pressure','continuous:ext_n_valid_array'}
    observation_cols = np.array(sorted(int(i) for k,v in feature_groups.items()
        if k.startswith('continuous:') and k not in availability for i in v))
    context_cols = np.array(sorted(int(i) for k,v in feature_groups.items()
        if k.startswith(('mask:','time_since_last:','static:')) or k in availability for i in v))
    assert not np.intersect1d(observation_cols,context_cols).size
    np.testing.assert_array_equal(np.sort(np.r_[observation_cols,context_cols]),model.context_indices)
    plan = dict(observation_features=[names[i] for i in observation_cols],
        context_features=[names[i] for i in context_cols],config=bundle['config'],
        selection='February only; same balanced objective, 0.1 convex weights and 0.30-0.80 thresholds',
        limitations='Exploratory previously inspected March-April block; existing labels/preprocessing retained.',
        source_sha256=before)
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    fitted=[]
    for label,cols in [('observations',observation_cols),('context',context_cols)]:
        estimator=make_classifier(HourlyBaselineConfig(**bundle['config']))
        estimator.fit(x[splits['train']][:,cols],y[splits['train']])
        joblib.dump(dict(estimator=estimator,columns=cols),out/f'{label}.joblib')
        fitted.append(estimator)
    def branches(ii,new):
        full=model.full_estimator.predict_proba(x[ii])[:,1]
        rule=model.rule_estimator.predict_proba(x[ii][:,model.rule_indices])[:,1]
        context=(fitted[1].predict_proba(x[ii][:,context_cols])[:,1] if new else
            model.context_estimator.predict_proba(x[ii][:,model.context_indices])[:,1])
        obs=fitted[0].predict_proba(x[ii][:,observation_cols])[:,1] if new else np.zeros(len(ii))
        return np.column_stack([full,context,rule,obs])
    choices={}; valid_scores={}
    for name,new in [('existing_three_views',False),('split_four_views',True)]:
        valid_scores[name]=branches(splits['validation'],new)
        choice,grid=grid_select(y[splits['validation']],valid_scores[name],new)
        # The shared selector's fourth-column name is raw; this experiment uses engineered observations.
        choices[name]={('observations' if k=='raw' else k):v for k,v in choice.items()}
        pd.DataFrame(grid).rename(columns={'raw':'observations'}).to_csv(out/f'{name}_validation_grid.csv',index=False)
    (out/'selection.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
    print('February selections frozen: '+json.dumps(choices),flush=True)
    rows=[]
    test=splits['test']
    ledger=pd.DataFrame(dict(row_index=test,station_id=z['station_id'][test],hour=hours[test],target=y[test]))
    for name,new in [('existing_three_views',False),('split_four_views',True)]:
        choice=choices[name]
        weights=np.array([choice[k] for k in ('full','context','rules','observations')])
        test_scores=branches(test,new)@weights
        for part,scores in [('validation',valid_scores[name]@weights),('test',test_scores)]:
            rows.append(dict(model=name,partition=part,**binary_metrics(y[splits[part]],scores,choice['threshold'])))
        ledger[name]=test_scores
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    ledger.to_parquet(out/'predictions.parquet',index=False)
    assert sha(source)==before
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        run()
