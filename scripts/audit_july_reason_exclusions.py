"""Trace excluded July reason-reference hours; no model fitting or deployment writes."""
from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.model.final_reason_codes import JULY_RAW, JULY_REFS, load_observations, build_features
from src.model.reason_code_rebuild import MECH, COMP, aligned_labels
from src.rules.channel_handlers import sensor_group_for_channel


def run():
    out = ROOT/'data/eval/july_reason_exclusion_audit_20260924'
    out.mkdir(exist_ok=False)
    ref = pd.read_parquet(ROOT/'data/eval/july_2026_reason_code_evaluation/aligned_reference.parquet')
    ref.hour_utc = pd.to_datetime(ref.hour_utc, utc=True)
    index = pd.MultiIndex.from_frame(ref[['station_id','hour_utc']])
    excluded = ref.truth_fault.astype(bool) & ~ref[MECH+COMP].any(axis=1)
    ep = pd.read_csv(ROOT/'data/eval/july_2026_adjudicated_labels/episode_labels_adjudicated.csv')
    for c in ('start_hour','end_hour'): ep[c] = pd.to_datetime(ep[c], utc=True)
    ep = ep.loc[ep.label_state.eq('fault') & ep.end_hour.ge(ref.hour_utc.min()) & ep.start_hour.le(ref.hour_utc.max())]
    stat = pd.read_parquet(ROOT/'data/eval/july_2026_reason_code_evaluation/reconstructed_statistical_evidence.parquet')
    stat.hour_utc = pd.to_datetime(stat.hour_utc, utc=True)
    raw = load_observations(JULY_RAW,JULY_REFS,pd.Timestamp('2026-07-31T23:00Z'))
    features, evidence = build_features(raw)
    rebuilt = aligned_labels(raw,evidence,ep,stat,index)
    np.testing.assert_array_equal(rebuilt,ref[MECH+COMP].to_numpy(int))
    print('Reproduced saved hourly labels exactly',flush=True)
    rawindex = raw.set_index(['station_id','hour_utc'])
    statindex = stat.set_index(['station_id','hour_utc','raw_channel'])
    details, episodes = [], []
    for e in ep.itertuples():
        member = ref.station_id.eq(e.station_id)&ref.hour_utc.between(e.start_hour,e.end_hour)&ref.truth_fault.astype(bool)
        lost = member & excluded
        if not lost.any(): continue
        episodes.append(dict(episode_id=e.episode_id,station_id=e.station_id,start=e.start_hour,end=e.end_hour,
            original_mechanisms=e.mechanisms,original_components=e.components,
            july_fault_hours=int(member.sum()),excluded_hours=int(lost.sum()),fired_channels=e.fired_channels))
        parsed=[]
        for item in str(e.fired_channels).split('|'):
            if '=' in item:
                m,cc=item.split('=',1)
                parsed.extend((m,c) for c in cc.split('+'))
        for r in ref.loc[lost].itertuples():
            for m,c in parsed or [('unmapped','unmapped')]:
                component=sensor_group_for_channel(c)
                note=''; failures=''
                if m not in MECH or component not in COMP:
                    cause='unmapped_mechanism_or_component'
                elif m=='calibration_offset':
                    raise AssertionError('Calibration interval unexpectedly excluded')
                elif m=='statistical_anomaly':
                    key=(r.station_id,r.hour_utc,c)
                    if key not in statindex.index:
                        cause='no_statistical_record_at_hour'
                    else:
                        ss=statindex.loc[[key]]
                        assert not ss.full_gate_passed.fillna(False).astype(bool).any()
                        cause='statistical_gate_not_passed_at_hour'
                        failures='|'.join(sorted(set(ss.gate_failure_reasons.dropna().astype(str))))
                else:
                    ev=evidence[r.station_id].get(c,{}).get(m)
                    if ev is None:
                        cause='missing_channel_evidence'
                    else:
                        assert not bool(ev.reindex([r.hour_utc]).fillna(False).iloc[0])
                        if m=='spike_impossible':
                            cause='no_physical_breach_at_hour'
                        else:
                            cause='causal_stuck_check_not_passed'
                            if c in rawindex.columns:
                                series=rawindex.loc[r.station_id,c].sort_index()
                                window=series.reindex(pd.date_range(r.hour_utc-pd.Timedelta(hours=23),r.hour_utc,freq='h'))
                                note=f'nonmissing_24h={window.notna().sum()};variance_24h={window.var()}'
                details.append(dict(station_id=r.station_id,hour_utc=r.hour_utc,episode_id=e.episode_id,
                    mechanism=m,channel=c,component=component,cause=cause,gate_failure_reasons=failures,detail=note))
    dd=pd.DataFrame(details)
    hourly=ref.loc[excluded,['station_id','hour_utc','source_episode_ids','detected_fault']].copy()
    for col in ('mechanism','component','cause','gate_failure_reasons'):
        values=dd.groupby(['station_id','hour_utc'])[col].agg(lambda v:'|'.join(sorted(set(str(x) for x in v if str(x)))))
        hourly[col]=[values.get((r.station_id,r.hour_utc),'unmapped') for r in hourly.itertuples()]
    assert len(hourly)==843
    hourly.to_csv(out/'excluded_hours.csv',index=False)
    dd.to_csv(out/'evidence_checks.csv',index=False)
    pd.DataFrame(episodes).to_csv(out/'episodes.csv',index=False)
    hourly.groupby(['station_id','mechanism','component','cause']).size().rename('hours').reset_index().to_csv(out/'breakdown.csv',index=False)
    audit=dict(excluded_hours=len(hourly),fault_hours=int(ref.truth_fault.sum()),
        reproduced_reference=True,stations=int(hourly.station_id.nunique()),episodes=len(episodes),
        causes=hourly.cause.value_counts().to_dict(),reason_combinations=hourly.mechanism.value_counts().to_dict())
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)
    print(hourly.groupby(['station_id','mechanism','component','cause']).size().to_string(),flush=True)

if __name__=='__main__': run()
