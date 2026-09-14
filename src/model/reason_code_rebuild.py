"""Isolated, timestamp-aligned reason-code research; no production model writes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from src.model.feature_spec import MECHANISM_LABEL_NAMES, COMPONENT_LABEL_NAMES
from src.model.hourly_baseline import load_hourly_tensor, load_reason_code_manifest_splits, _fault_groups
from src.rules.channel_handlers import sensor_group_for_channel
from src.rules.config import PHYSICAL_LIMIT_RULES, STUCK_IGNORE_ZERO_CHANNELS, STUCK_SKIP_CHANNELS
from src.rules.physical_limits import physical_limit_flags

ROOT = Path(__file__).resolve().parents[2]
MECH = list(MECHANISM_LABEL_NAMES)
COMP = list(COMPONENT_LABEL_NAMES)
SEED = 20260912


def features_for_station(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Full hourly clock, past-only rolling features; no cached fitted detector flags."""
    raw = raw.sort_index().reindex(pd.date_range(raw.index.min(), raw.index.max(), freq='h'))
    channels = [c for c in raw if sensor_group_for_channel(c) in COMP]
    signals = {c: pd.to_numeric(raw[c], errors='coerce') for c in channels if c != 'winddir_avg_deg'}
    references = [c for c in raw if c.startswith('reference_')]
    if 'winddir_avg_deg' in raw:
        a = np.deg2rad(pd.to_numeric(raw.winddir_avg_deg, errors='coerce'))
        signals.update(winddir_sin=np.sin(a), winddir_cos=np.cos(a))
    out, evidence = {}, {}
    for c, v in signals.items():
        group = sensor_group_for_channel(c)
        prefix = group + '::' + c + '::'
        past = v.shift(1).rolling(168, min_periods=24)
        spread = past.std().clip(lower=0.1)
        roll = v.rolling(24, min_periods=24)
        var = roll.var().clip(lower=0)
        hard = physical_limit_flags(v, c)
        stuck = var.lt(1e-6).fillna(False)
        if c in STUCK_SKIP_CHANNELS:
            stuck[:] = False
        if c in STUCK_IGNORE_ZERO_CHANNELS:
            stuck &= roll.mean().abs().gt(1e-9)
        evidence[c] = {'spike_impossible': hard, 'stuck_flatline': stuck}
        fields = {
            'value': v, 'missing': v.isna().astype(float),
            'delta': v.diff(), 'delta24': v.diff(24),
            'past_z': ((v-past.mean())/spread).clip(-50, 50),
            'range6': v.rolling(6,min_periods=3).max()-v.rolling(6,min_periods=3).min(),
            'range24': roll.max()-roll.min(), 'variance24': var,
            'repeated24': v.diff().abs().lt(1e-6).rolling(24,min_periods=1).mean(),
            'coverage24': v.notna().rolling(24,min_periods=1).mean(),
            'hard': hard.astype(float), 'stuck': stuck.astype(float),
        }
        out.update({prefix+k: val for k,val in fields.items()})
    if 'winddir_sin' in signals:
        direction = evidence['winddir_sin']['stuck_flatline'] & evidence['winddir_cos']['stuck_flatline']
        speed_stuck = evidence['windspeed_avg_kmh']['stuck_flatline']
        calm = pd.to_numeric(raw.windspeed_avg_kmh, errors='coerce').abs().le(1)
        qualified = direction & ~speed_stuck & ~calm
        evidence['winddir_avg_deg'] = {'spike_impossible': pd.Series(False,index=raw.index),
                                      'stuck_flatline': qualified}
        out['wind_vane::context::calm'] = calm.astype(float)
        out['wind_vane::context::speed'] = pd.to_numeric(raw.windspeed_avg_kmh,errors='coerce')
        out['wind_vane::context::direction_pair_stuck'] = direction.astype(float)
        out['wind_vane::context::speed_stuck'] = speed_stuck.astype(float)
    for c in references:
        _,group,field=c.split('__')
        v=pd.to_numeric(raw[c],errors='coerce')
        past=v.shift(1).rolling(168,min_periods=24)
        prefix=group+'::reference_'+field+'::'
        out[prefix+'value']=v
        out[prefix+'missing']=v.isna().astype(float)
        out[prefix+'delta']=v.diff()
        out[prefix+'past_z']=((v-past.mean())/past.std().clip(lower=.1)).clip(-50,50)
        out[prefix+'mean24']=v.rolling(24,min_periods=12).mean()
        out[prefix+'std24']=v.rolling(24,min_periods=12).std()
    frame = pd.DataFrame(out, index=raw.index).replace([np.inf,-np.inf],np.nan).astype('float32')
    # Mechanism views summarize channel-local evidence, without station identity.
    for group in COMP:
        for suffix in ('hard','stuck','past_z','delta','range24','missing'):
            cols = [c for c in frame if c.startswith(group+'::') and c.endswith('::'+suffix)]
            if cols:
                frame[f'{group}::summary::{suffix}_max'] = frame[cols].abs().max(axis=1)
    return frame, evidence


def aligned_labels(raw, station_evidence, episodes, statistical, feature_index):
    """Reference episode membership AND evidence at this hour, with no backward fill.

    Statistical adjudication is retained as a retrospective weak reference only.
    Confirmed calibration intervals retain their historical interval meaning.
    No target columns or episode boundaries are predictor inputs.
    """
    target = np.zeros((len(feature_index),len(MECH)+len(COMP)),dtype=np.uint8)
    loc = pd.Series(np.arange(len(feature_index)),index=feature_index)
    stats = statistical.loc[statistical.full_gate_passed.fillna(False).astype(bool)]
    stat_times = {key:set(g.hour_utc) for key,g in stats.groupby(['station_id','raw_channel'])}
    for ep in episodes.loc[episodes.label_state.eq('fault')].itertuples():
        evidence = station_evidence[ep.station_id]
        ticks = evidence[next(iter(evidence))]['spike_impossible'].index
        within = ticks[(ticks>=ep.start_hour)&(ticks<=ep.end_hour)]
        for item in str(ep.fired_channels).split('|'):
            if '=' not in item:
                continue
            mechanism, channels = item.split('=',1)
            if mechanism not in MECH:
                continue
            for channel in channels.split('+'):
                component = sensor_group_for_channel(channel)
                if component not in COMP:
                    continue
                if mechanism == 'statistical_anomaly':
                    active = [t for t in within if t in stat_times.get((ep.station_id,channel),set())]
                elif mechanism == 'calibration_offset':
                    active = within
                else:
                    ev = evidence.get(channel,{}).get(mechanism)
                    active = [] if ev is None else within[ev.reindex(within).fillna(False).to_numpy(bool)]
                if len(active):
                    keys = pd.MultiIndex.from_arrays([[ep.station_id]*len(active),active])
                    positions = loc.reindex(keys).dropna().to_numpy(int)
                    target[positions,MECH.index(mechanism)] = 1
                    target[positions,len(MECH)+COMP.index(component)] = 1
    return target


def group_balanced_split(groups, y, original_fault, fractions=(.7,.15,.15)):
    """Greedy event allocation balancing code/event support, fault hours, and total rows.

    Groups are never divided. Split construction uses labels, never model performance.
    """
    unique, inverse = np.unique(groups,return_inverse=True)
    stats = np.zeros((len(unique),y.shape[1]+2),float)
    np.add.at(stats[:,:y.shape[1]],inverse,y)
    np.add.at(stats[:,-2],inverse,original_fault)
    np.add.at(stats[:,-1],inverse,1)
    # Code presence counts event support; hourly counts have a smaller weight.
    event = (stats[:,:y.shape[1]]>0).astype(float)
    values = np.c_[event,stats[:,:y.shape[1]],stats[:,-2:]]
    totals = values.sum(axis=0).clip(1)
    importance=np.r_[np.ones(y.shape[1]),np.full(y.shape[1],.5),20.,5.]
    rarity = (event / event.sum(axis=0).clip(1)).sum(axis=1)
    rng = np.random.default_rng(SEED)
    order = np.lexsort((rng.random(len(unique)),-stats[:,-2],-rarity))
    goals = np.asarray(fractions)[:,None]*totals
    used = np.zeros_like(goals); assignment=np.zeros(len(unique),int)
    # Incremental squared-error allocation against all class and row quotas.
    for g in order:
        delta = ((((used+values[g]-goals)/totals)**2-((used-goals)/totals)**2)*importance).sum(axis=1)
        k=int(delta.argmin());assignment[g]=k;used[k]+=values[g]
    # Improve the aggregate objective by whole-group moves, without looking at scores.
    for _ in range(8):
        changed=0
        for g in order:
            old=assignment[g]
            cost_before=(((used-goals)/totals)**2*importance).sum()
            best=old;best_cost=cost_before
            for new in range(3):
                if new==old:continue
                trial=used.copy();trial[old]-=values[g];trial[new]+=values[g]
                cost=(((trial-goals)/totals)**2*importance).sum()
                if cost<best_cost-1e-10:best=new;best_cost=cost
            if best!=old:
                used[old]-=values[g];used[best]+=values[g];assignment[g]=best;changed+=1
        if changed==0:break
    return {p:np.flatnonzero(assignment[inverse]==k) for k,p in enumerate(('train','validation','test'))}


def chronological_split(hours, groups):
    a=pd.Timestamp('2026-04-01',tz='UTC'); b=pd.Timestamp('2026-05-01',tz='UTC')
    h=pd.DatetimeIndex(hours)
    category=np.select([h<a,h<b],[0,1],default=2)
    boundaries=pd.DataFrame({'g':groups,'p':category}).groupby('g').p.nunique()
    crosses=np.isin(groups,boundaries[boundaries>1].index)
    # Seven-day history embargo; the deployment features themselves stay causal.
    gap=((h>=a)&(h<a+pd.Timedelta(days=7)))|((h>=b)&(h<b+pd.Timedelta(days=7)))
    return {p:np.flatnonzero((category==k)&~crosses&~gap) for k,p in enumerate(('train','validation','test'))}


def event_weights(groups):
    _,inverse,count=np.unique(groups,return_inverse=True,return_counts=True)
    weights=1.0/count[inverse]
    return weights/weights.mean()


def fit_estimator(x,y,groups,seed=SEED):
    if len(np.unique(y))<2:
        return float(y.mean()) if len(y) else 0.0
    w=event_weights(groups)
    # Balance positive and negative total event weight, rather than repeated hours.
    w[y==1] *= w[y==0].sum()/max(w[y==1].sum(),1e-12)
    w/=w.mean()
    m=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=15,min_samples_leaf=10,
        l2_regularization=2,learning_rate=.06,early_stopping=False,random_state=seed)
    m.fit(x,y,sample_weight=w)
    return m


def probability(model,x):
    return np.full(len(x),model) if isinstance(model,float) else model.predict_proba(x)[:,1]


def feature_views(names,axis,label):
    if axis=='component':
        allowed=[c for c in names if c.startswith(label+'::')]
        # wind-speed information is explicitly scoped to direction interpretation.
    else:
        allowed=[c for c in names if '::value' not in c]
    full=np.array([names.index(c) for c in allowed],int)
    rules=np.array([names.index(c) for c in allowed if any(s in c for s in
        ('::hard','::stuck','::repeated','::missing','::coverage','::context::','::summary::'))],int)
    context=np.setdiff1d(full,rules)
    return [full,context,rules]


def select_policy(y, probs, weights, folds):
    """Fixed compact convex grid; mean fold event-weighted F1. No test input."""
    candidates=[]
    for a in (0,.5,1):
        for b in (0,.5,1):
            if a+b>1:continue
            mix=np.array([a,b,1-a-b]);p=probs@mix
            for threshold in (.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95):
                pred=p>=threshold; scores=[]
                for f in np.unique(folds):
                    mask=folds==f; yy=y[mask]; ww=weights[mask]; pp=pred[mask]
                    tp=ww[(yy==1)&pp].sum();fp=ww[(yy==0)&pp].sum();fn=ww[(yy==1)&~pp].sum()
                    scores.append(2*tp/max(2*tp+fp+fn,1e-12))
                candidates.append((np.mean(scores),a,-threshold,mix,threshold))
    chosen=max(candidates,key=lambda t:t[:3])
    return chosen[3],chosen[4],float(chosen[0])


def multilabel_rows(y,pred,groups,names,metadata):
    p,r,f,s=precision_recall_fscore_support(y,pred,average=None,zero_division=0)
    rows=[dict(metadata,label=n,precision=float(p[i]),recall=float(r[i]),f1=float(f[i]),
               support=int(s[i]),positive_events=int(len(np.unique(groups[y[:,i]==1])))) for i,n in enumerate(names)]
    good=s>0
    micro=precision_recall_fscore_support(y,pred,average='micro',zero_division=0)
    summary=dict(metadata,rows=len(y),positive_labels=int(good.sum()),
        macro_f1=float(f[good].mean()) if good.any() else None,
        micro_precision=float(micro[0]),micro_recall=float(micro[1]),micro_f1=float(micro[2]),
        exact_match=float(accuracy_score(y,pred)),code_coverage=float(pred.any(axis=1).mean()))
    return rows,summary


def minimum_one(pred,probs,eligible_labels):
    result=pred.copy();empty=~result.any(axis=1)
    if eligible_labels.any() and empty.any():
        rank=probs.copy();rank[:,~eligible_labels]=-np.inf
        result[np.flatnonzero(empty),rank[empty].argmax(axis=1)]=1
    return result


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def run(output:Path):
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    inputs={
        'raw':ROOT/'data/merged/station_hourly_merged.csv',
        'episodes':ROOT/'data/labels/episode_labels.csv',
        'statistical':ROOT/'data/labels/statistical_anomaly_review.csv',
        'tensor':ROOT/'data/hourly_detection/one_hour_final/hourly_detection_01h.npz',
        'splits':ROOT/'data/hourly_detection/hourly_baseline_split_manifest.csv',
        'references':ROOT/'data/features/external_residuals.parquet',
    }
    before={k:sha(p) for k,p in inputs.items()}
    raw=pd.read_csv(inputs['raw']);raw.hour_utc=pd.to_datetime(raw.hour_utc,utc=True)
    refs=pd.read_parquet(inputs['references'],columns=['station_id','time_utc','r_pressure','r_temp','r_dewpoint','r_wind','r_solar'])
    refs=refs.rename(columns={'time_utc':'hour_utc',**{f'r_{k}':f'reference__{v}__{k}' for k,v in
        {'pressure':'barometer','temp':'thermo_hygrometer','dewpoint':'thermo_hygrometer','wind':'anemometer','solar':'light_uv'}.items()}})
    refs.hour_utc=pd.to_datetime(refs.hour_utc,utc=True)
    raw=raw.merge(refs,on=['station_id','hour_utc'],how='left',validate='one_to_one')
    frames=[]; evidence={};checks=[]
    for station,g in raw.groupby('station_id',sort=True):
        g=g.set_index('hour_utc'); ff,ev=features_for_station(g)
        cut=ff.index[int(len(ff)*.8)]
        truncated,_=features_for_station(g.loc[g.index<=cut])
        common=truncated.index.intersection(ff.index)
        pd.testing.assert_frame_equal(ff.loc[common],truncated.loc[common],check_exact=False,rtol=1e-5,atol=1e-6)
        checks.append({'station_id':station,'cutoff':str(cut),'prefix_invariant':True})
        ff['station_id']=station;ff['hour']=ff.index;frames.append(ff.set_index(['station_id','hour']))
        evidence[station]=ev
    features=pd.concat(frames).sort_index();names=features.columns.tolist()
    print(f'causal_features={len(names)} stations={len(checks)} prefix_checks_passed',flush=True)
    z=load_hourly_tensor(inputs['tensor'])
    index=pd.MultiIndex.from_arrays([z['station_id'],pd.to_datetime(z['hour'],utc=True)],names=['station_id','hour'])
    x=features.reindex(index).to_numpy('float32')
    ep=pd.read_csv(inputs['episodes']);ep.start_hour=pd.to_datetime(ep.start_hour,utc=True);ep.end_hour=pd.to_datetime(ep.end_hour,utc=True)
    stat=pd.read_csv(inputs['statistical']);stat.hour_utc=pd.to_datetime(stat.hour_utc,utc=True)
    y=aligned_labels(raw,evidence,ep,stat,index)
    fault=np.asarray(z['y_binary'],int)
    assert not y[fault==0].any()
    groups=np.array([f'normal:{s}:{str(t)[:10]}' for s,t in index],dtype=object)
    for k,ii in _fault_groups(fault,z['source_episode_ids']).items():groups[ii]='event:'+k
    splits={'random':load_reason_code_manifest_splits(z,inputs['splits'])['random'],
            'grouped':group_balanced_split(groups,y,fault),
            'chronological':chronological_split(index.get_level_values('hour'),groups)}
    memberships=[];support=[];split_audit=[]
    for scheme,sp in splits.items():
        for part,ii in sp.items():
            memberships.append(pd.DataFrame({'scheme':scheme,'partition':part,'row_index':ii,'group':groups[ii]}))
            support.append(dict(scheme=scheme,partition=part,hours=len(ii),fault_hours=int(fault[ii].sum()),
                resolved_fault_hours=int(y[ii].any(axis=1).sum()),
                unresolved_fault_hours=int(((fault[ii]==1)&~y[ii].any(axis=1)).sum()),
                first=str(index[ii].get_level_values('hour').min()),last=str(index[ii].get_level_values('hour').max())))
            for j,label in enumerate(MECH+COMP):
                split_audit.append(dict(scheme=scheme,partition=part,label=label,
                    positive_hours=int(y[ii,j].sum()),positive_events=len(np.unique(groups[ii[y[ii,j]==1]])),
                    positive_stations=len(np.unique(z['station_id'][ii[y[ii,j]==1]]))))
        train_groups=set(groups[sp['train']]);overlap=len(train_groups&set(groups[sp['test']]))
        if scheme!='random':assert overlap==0
        print(f'{scheme} train/test_group_overlap={overlap}',flush=True)
    pd.concat(memberships).to_csv(output/'split_membership.csv',index=False)
    pd.DataFrame(support).to_csv(output/'population.csv',index=False)
    pd.DataFrame(split_audit).to_csv(output/'label_support.csv',index=False)
    print(pd.DataFrame(support).to_string(index=False),flush=True)
    reference=pd.DataFrame(y,columns=MECH+COMP);reference.insert(0,'hour',index.get_level_values('hour'));reference.insert(0,'station_id',z['station_id']);reference['original_fault']=fault
    reference.to_parquet(output/'aligned_reference.parquet',index=False)
    feature_spec={axis:{label:[ [names[c] for c in view] for view in feature_views(names,axis,label)]
        for label in (MECH if axis=='mechanism' else COMP)} for axis in ('mechanism','component')}
    (output/'feature_spec.json').write_text(json.dumps(feature_spec,indent=2))
    rows=[];summaries=[];selection=[];ledgers=[];model_dir=output/'models';model_dir.mkdir()
    with threadpool_limits(limits=2):
        for scheme,sp in splits.items():
            train=sp['train'];val=sp['validation'];test=sp['test']
            # Baseline detector trained in this experiment for a coherent cascade.
            # It is not substituted for the frozen operational EF-HGB.
            gatecols=np.array([j for j,n in enumerate(names) if '::summary::' in n or '::context::' in n])
            gate=fit_estimator(x[train][:,gatecols],fault[train],groups[train])
            vp=probability(gate,x[val][:,gatecols]);gp=probability(gate,x[test][:,gatecols])
            gm,gt,_=select_policy(fault[val],np.repeat(vp[:,None],3,axis=1),event_weights(groups[val]),np.zeros(len(val),int))
            gatepred=gp>=gt
            from src.model.hourly_baseline import binary_metrics
            gate_metric=binary_metrics(fault[test],gp,gt)
            (output/f'gate_{scheme}.json').write_text(json.dumps(dict(threshold=gt,**gate_metric),indent=2))
            joblib.dump(dict(model=gate,columns=gatecols,threshold=gt,feature_names=names),model_dir/f'gate_{scheme}.joblib')
            for axis,labels,offset in [('mechanism',MECH,0),('component',COMP,len(MECH))]:
                yy=y[:,offset:offset+len(labels)]
                resolved=yy.any(axis=1)
                # Include non-fault negatives; unresolved fault hours have unknown codes.
                train_allowed=train[(fault[train]==0)|resolved[train]]
                rng=np.random.default_rng(SEED)
                positive=train_allowed[fault[train_allowed]==1]
                negative=train_allowed[fault[train_allowed]==0]
                if len(negative)>16000:negative=np.sort(rng.choice(negative,16000,replace=False))
                fitrows=np.sort(np.r_[positive,negative])
                preds=np.zeros((len(test),len(labels)),bool);probs=np.zeros_like(preds,dtype=float)
                available=np.zeros(len(labels),bool)
                for j,label in enumerate(labels):
                    target=yy[:,j];views=feature_views(names,axis,label)
                    support_events=len(np.unique(groups[fitrows[target[fitrows]==1]]))
                    available[j]=support_events>=3
                    if not available[j]:
                        selection.append(dict(scheme=scheme,axis=axis,label=label,status='insufficient_training_events',events=support_events))
                        continue
                    oof=np.zeros((len(fitrows),3));foldids=np.zeros(len(fitrows),int)
                    for fold,(fi,oi) in enumerate(GroupKFold(3).split(fitrows,groups=groups[fitrows])):
                        foldids[oi]=fold
                        for branch,cols in enumerate(views):
                            m=fit_estimator(x[fitrows[fi]][:,cols],target[fitrows[fi]],groups[fitrows[fi]])
                            oof[oi,branch]=probability(m,x[fitrows[oi]][:,cols])
                    mix,threshold,cvf1=select_policy(target[fitrows],oof,event_weights(groups[fitrows]),foldids)
                    models=[];pv=[];pt=[]
                    for branch,cols in enumerate(views):
                        m=fit_estimator(x[fitrows][:,cols],target[fitrows],groups[fitrows]);models.append(m)
                        pt.append(probability(m,x[test][:,cols]));pv.append(probability(m,x[val][:,cols]))
                    probs[:,j]=np.array(pt).T@mix;preds[:,j]=probs[:,j]>=threshold
                    vv=np.array(pv).T@mix
                    valid=(fault[val]==0)|resolved[val]
                    vm=binary_metrics(target[val[valid]],vv[valid],threshold)
                    selection.append(dict(scheme=scheme,axis=axis,label=label,status='fitted',events=support_events,
                        threshold=threshold,full_weight=mix[0],context_weight=mix[1],rules_weight=mix[2],
                        oof_event_f1=cvf1,validation_f1=vm['f1']))
                    joblib.dump(dict(models=models,views=views,weights=mix,threshold=threshold,feature_names=names),model_dir/f'{scheme}_{axis}_{label}.joblib')
                    print(f'{scheme} {axis} {label}: train_events={support_events} OOF_event_F1={cvf1:.3f} validation_F1={vm["f1"]:.3f}',flush=True)
                for scope in ('resolved_faults','detected_resolved_faults','cascade'):
                    mask=resolved[test] if scope=='resolved_faults' else ((resolved[test]&gatepred) if scope=='detected_resolved_faults' else ((fault[test]==0)|resolved[test]))
                    for policy in ('threshold','minimum_one'):
                        pp=preds.copy()
                        if policy=='minimum_one':pp=minimum_one(pp,probs,available)
                        if scope=='cascade':pp &= gatepred[:,None]
                        if not mask.any():continue
                        rr,ss=multilabel_rows(yy[test[mask]],pp[mask],groups[test[mask]],labels,
                            dict(scheme=scheme,axis=axis,scope=scope,policy=policy))
                        rows.extend(rr);summaries.append(ss)
                for j,label in enumerate(labels):
                    ledgers.append(pd.DataFrame(dict(scheme=scheme,axis=axis,label=label,
                        station_id=z['station_id'][test],hour=z['hour'][test],group=groups[test],
                        original_fault=fault[test],reference_resolved=resolved[test],target=yy[test,j],
                        probability=probs[:,j],prediction=preds[:,j],gate_prediction=gatepred)))
    pd.DataFrame(rows).to_csv(output/'per_label.csv',index=False)
    pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)
    pd.DataFrame(selection).to_csv(output/'selection.csv',index=False)
    pd.concat(ledgers).to_parquet(output/'predictions.parquet',index=False)
    assert before=={k:sha(p) for k,p in inputs.items()}
    audit=dict(input_hashes=before,source_files_unchanged=True,causality_checks=checks,
        chronological_boundaries=['2026-04-01','2026-05-01'],embargo_days=7,seed=SEED,
        threads=2,elapsed_seconds=time.monotonic()-start,
        target='current-hour evidence within accepted reference episodes; calibration retains confirmed interval',
        statistical_reference='historical contextual adjudication, not independently verified or claimed causal',
        external_reference='same-hour archived residual values only; rolling statistics rebuilt causally; archive arrival-time availability not verified',
        calibration_reference='four historical corroborated intervals; live confirmation availability unverified',
        selection='3-fold training-only grouped OOF, mean event-weighted F1; validation sanity only',
        cascade='new experimental HGB gate trained on same split; not frozen operational EF-HGB',
        limitation='New target population: not directly comparable to episode-broadcast scores; unresolved hours excluded from code metrics and reported in population.csv.')
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
