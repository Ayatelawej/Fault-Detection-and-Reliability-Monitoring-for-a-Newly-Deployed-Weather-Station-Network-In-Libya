"""Shared reason features, split utilities, head fitting and evaluation metrics."""
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

    event = (stats[:,:y.shape[1]]>0).astype(float)
    values = np.c_[event,stats[:,:y.shape[1]],stats[:,-2:]]
    totals = values.sum(axis=0).clip(1)
    importance=np.r_[np.ones(y.shape[1]),np.full(y.shape[1],.5),20.,5.]
    rarity = (event / event.sum(axis=0).clip(1)).sum(axis=1)
    rng = np.random.default_rng(SEED)
    order = np.lexsort((rng.random(len(unique)),-stats[:,-2],-rarity))
    goals = np.asarray(fractions)[:,None]*totals
    used = np.zeros_like(goals); assignment=np.zeros(len(unique),int)

    for g in order:
        delta = ((((used+values[g]-goals)/totals)**2-((used-goals)/totals)**2)*importance).sum(axis=1)
        k=int(delta.argmin());assignment[g]=k;used[k]+=values[g]

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
