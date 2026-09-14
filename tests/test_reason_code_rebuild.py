import numpy as np
import pandas as pd

from src.model.reason_code_rebuild import (
    features_for_station, aligned_labels, group_balanced_split,
    chronological_split, feature_views, minimum_one, MECH, COMP,
)


def station_frame():
    t=pd.date_range('2026-01-01',periods=220,freq='h',tz='UTC')
    return pd.DataFrame({'pressure_trend_hpa':np.sin(np.arange(220)),
        'windspeed_avg_kmh':np.ones(220)*5,'winddir_avg_deg':np.ones(220)*45},index=t)


def test_future_spike_does_not_label_earlier_episode_hours():
    raw=station_frame();raw.loc[raw.index[100],'pressure_trend_hpa']=30
    ff,ev=features_for_station(raw)
    ep=pd.DataFrame([dict(station_id='s',label_state='fault',start_hour=raw.index[90],
        end_hour=raw.index[100],fired_channels='spike_impossible=pressure_trend_hpa')])
    stat=pd.DataFrame(columns=['station_id','raw_channel','hour_utc','full_gate_passed'])
    index=pd.MultiIndex.from_product([['s'],raw.index])
    y=aligned_labels(raw,{'s':ev},ep,stat,index)
    assert y[:,MECH.index('spike_impossible')].sum()==1
    assert y[100,MECH.index('spike_impossible')]==1
    assert y[100,len(MECH)+COMP.index('barometer')]==1


def test_stuck_confirmation_and_features_do_not_look_forward():
    raw=station_frame();full,ev=features_for_station(raw)
    assert not ev['winddir_avg_deg']['stuck_flatline'].any()  # speed stuck too
    assert not ev['windspeed_avg_kmh']['stuck_flatline'].iloc[:23].any()
    assert ev['windspeed_avg_kmh']['stuck_flatline'].iloc[23]
    short,_=features_for_station(raw.iloc[:130])
    pd.testing.assert_frame_equal(full.iloc[:130],short)


def test_component_views_exclude_unrelated_pressure():
    names=['wind_vane::winddir_cos::value','wind_vane::winddir_cos::hard',
           'barometer::pressure_max_hpa::value','wind_vane::context::speed']
    full,context,rules=feature_views(names,'component','wind_vane')
    assert 2 not in full and 2 not in context and 2 not in rules


def test_chronological_split_removes_crossing_events_and_applies_gap():
    hours=pd.to_datetime(['2026-03-31','2026-04-02','2026-04-03','2026-04-09','2026-05-10'],utc=True)
    groups=np.array(['cross','cross','other','v','t'])
    s=chronological_split(hours,groups)
    assert not len(s['train'])
    assert s['validation'].tolist()==[3]
    assert s['test'].tolist()==[4]


def test_group_split_never_splits_event_and_covers_all_rows():
    groups=np.repeat(np.arange(30).astype(str),3)
    y=np.zeros((90,2),int);y[:45,0]=1;y[30:75,1]=1
    s=group_balanced_split(groups,y,y.any(axis=1).astype(int))
    assert len(np.unique(np.concatenate(list(s.values()))))==90
    for a,b in [('train','test'),('train','validation'),('validation','test')]:
        assert not set(groups[s[a]])&set(groups[s[b]])


def test_minimum_one_preserves_multilabel_and_never_chooses_unsupported():
    p=np.array([[.4,.9,.1],[.8,.2,.8]])
    pred=np.array([[False,False,False],[True,False,True]])
    result=minimum_one(pred,p,np.array([True,False,True]))
    assert result.tolist()==[[True,False,False],[True,False,True]]
