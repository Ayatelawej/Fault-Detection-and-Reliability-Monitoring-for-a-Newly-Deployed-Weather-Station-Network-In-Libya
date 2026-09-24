"""Evaluate the active July episode model on detected faults and all alerts."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.model.final_reason_codes import JULY_EPISODE_OUTPUT,JULY_GATE
from src.model.reason_code_rebuild import MECH,COMP,multilabel_rows,sha


def run():
    out=ROOT/'data/eval/episode_reason_alert_scopes_20260924';out.mkdir(exist_ok=False)
    path=JULY_EPISODE_OUTPUT/'reason_code_predictions.parquet';before=sha(path)
    pred=pd.read_parquet(path);gate=pd.read_parquet(JULY_GATE)
    pd.testing.assert_frame_equal(pred[['station_id','hour_utc']],gate[['station_id','hour_utc']])
    ep=pd.read_csv(ROOT/'data/eval/july_2026_adjudicated_labels/episode_labels_adjudicated.csv')
    for c in ('start_hour','end_hour'):ep[c]=pd.to_datetime(ep[c],utc=True)
    ep=ep[ep.label_state.eq('fault')&ep.end_hour.ge(gate.hour_utc.min())&ep.start_hour.le(gate.hour_utc.max())]
    fault=gate.truth_fault.eq(1).to_numpy();alert=gate.random_prediction.eq(1).to_numpy()
    y=np.zeros((len(gate),len(MECH)+len(COMP)),int)
    for e in ep.itertuples():
        member=gate.station_id.eq(e.station_id)&gate.hour_utc.between(e.start_hour,e.end_hour)
        for field,labels,offset in [('mechanisms',MECH,0),('components',COMP,len(MECH))]:
            for label in str(getattr(e,field)).split('|'):
                if label in labels:y[member,offset+labels.index(label)]=1
    for part in (y[:,:len(MECH)],y[:,len(MECH):]):np.testing.assert_array_equal(part.any(axis=1),fault)
    summaries=[];details=[]
    for axis,labels,offset in [('mechanism',MECH,0),('component',COMP,len(MECH))]:
        emitted=pred[f'likely_{axis}s'].fillna('').str.split(' | ',regex=False)
        p=np.column_stack([emitted.apply(lambda v:label in v).to_numpy(bool) for label in labels])
        for scope,mask in [('correctly_detected_fault_hours',fault&alert),('all_detector_alerts',alert),('complete_detector_to_reason_pipeline',np.ones(len(gate),bool))]:
            rr,ss=multilabel_rows(y[mask,offset:offset+len(labels)],p[mask],gate.source_episode_ids.to_numpy()[mask],labels,dict(axis=axis,scope=scope))
            summaries.append(ss);details.extend(rr)
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(details).to_csv(out/'per_label.csv',index=False)
    assert before==sha(path)
    (out/'audit.json').write_text(json.dumps(dict(reference_fault_hours=int(fault.sum()),detected_fault_hours=int((fault&alert).sum()),false_alert_hours=int((~fault&alert).sum()),
        reason_unknown_exclusions=0,no_refitting=True,source_sha256=before,target='original episode labels',
        limitation='Previously explored July extension; detected-true-fault scope excludes both missed faults and false alerts.'),indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=='__main__':run()
