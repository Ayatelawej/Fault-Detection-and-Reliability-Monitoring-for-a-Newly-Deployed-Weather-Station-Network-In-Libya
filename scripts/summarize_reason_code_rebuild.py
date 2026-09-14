"""Post-fit reporting, including old-target coverage and the frozen random EF-HGB gate."""
from pathlib import Path
import sys
import argparse
import json

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from src.model.hourly_baseline import load_hourly_tensor, flatten_hourly_features, load_reason_code_manifest_splits
from src.model.reason_code_rebuild import ROOT, MECH, COMP, multilabel_rows, minimum_one


def main(out):
    ledger=pd.read_parquet(out/'predictions.parquet')
    reference=pd.read_parquet(out/'aligned_reference.parquet')
    z=load_hourly_tensor(ROOT/'data/hourly_detection/one_hour_final/hourly_detection_01h.npz')
    x,names,_=flatten_hourly_features(z)
    splits=load_reason_code_manifest_splits(z,ROOT/'data/hourly_detection/hourly_baseline_split_manifest.csv')
    key=pd.MultiIndex.from_arrays([z['station_id'],pd.to_datetime(z['hour'],utc=True)])
    location=pd.Series(np.arange(len(key)),index=key)
    source=ROOT/'data/hourly_detection/one_hour_final/models/evidence_fusion/selected_ef_hgb_random_01h.joblib'
    bundle=joblib.load(source)
    assert names==bundle['feature_names']
    test=splits['random']['test']
    with threadpool_limits(limits=2):
        gate=bundle['estimator'].predict_proba(x[test])[:,1]>=bundle['config']['threshold']
    gate_map=pd.Series(gate,index=key[test])
    selection=pd.read_csv(out/'selection.csv')
    original_rows=[];original_summary=[];frozen_rows=[];frozen_summary=[];coverage=[]
    for scheme in ledger.scheme.unique():
        for axis,labels in [('mechanism',MECH),('component',COMP)]:
            f=ledger[(ledger.scheme==scheme)&(ledger.axis==axis)].copy()
            f.hour=pd.to_datetime(f.hour,utc=True)
            ordered=f[f.label==labels[0]].sort_values(['station_id','hour'])
            keys=pd.MultiIndex.from_frame(ordered[['station_id','hour']])
            ii=location.reindex(keys).to_numpy(int)
            predicted=np.column_stack([f[f.label==label].sort_values(['station_id','hour']).prediction.to_numpy(bool) for label in labels])
            probs=np.column_stack([f[f.label==label].sort_values(['station_id','hour']).probability.to_numpy(float) for label in labels])
            new_target=np.column_stack([f[f.label==label].sort_values(['station_id','hour']).target.to_numpy(int) for label in labels])
            original=np.asarray(z['y_'+axis][ii],int)
            fault=z['y_binary'][ii]==1;resolved=new_target.any(axis=1)
            groups=ordered.group.to_numpy(str)
            fitted=set(selection[(selection.scheme==scheme)&(selection.axis==axis)&(selection.status=='fitted')].label)
            available=np.array([l in fitted for l in labels])
            for policy in ('threshold','minimum_one'):
                pp=predicted if policy=='threshold' else minimum_one(predicted,probs,available)
                rr,ss=multilabel_rows(original[fault],pp[fault],groups[fault],labels,
                    dict(scheme=scheme,axis=axis,scope='original_episode_labels_on_all_fault_hours',policy=policy))
                original_rows.extend(rr);original_summary.append(ss)
                if scheme=='random':
                    detected=gate_map.reindex(keys).to_numpy(bool)
                    for scope in ('frozen_gate_resolved_alerts','frozen_gate_cascade'):
                        mask=resolved&detected if scope=='frozen_gate_resolved_alerts' else (~fault|resolved)
                        gated=pp if scope=='frozen_gate_resolved_alerts' else pp&detected[:,None]
                        rr,ss=multilabel_rows(new_target[mask],gated[mask],groups[mask],labels,
                            dict(scheme=scheme,axis=axis,scope=scope,policy=policy))
                        frozen_rows.extend(rr);frozen_summary.append(ss)
            for j,label in enumerate(labels):
                coverage.append(dict(scheme=scheme,axis=axis,label=label,
                    original_positive_hours=int(original[:,j].sum()),
                    timed_positive_hours=int(new_target[:,j].sum()),
                    timed_positive_events=len(np.unique(groups[new_target[:,j]==1]))))
    pd.DataFrame(original_rows).to_csv(out/'original_target_per_label.csv',index=False)
    pd.DataFrame(original_summary).to_csv(out/'original_target_summary.csv',index=False)
    pd.DataFrame(frozen_rows).to_csv(out/'frozen_ef_hgb_per_label.csv',index=False)
    pd.DataFrame(frozen_summary).to_csv(out/'frozen_ef_hgb_summary.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'target_coverage_comparison.csv',index=False)
    summary=pd.read_csv(out/'summary.csv');pop=pd.read_csv(out/'population.csv')
    shown=summary[(summary.scope=='resolved_faults')&(summary.policy=='minimum_one')]
    lines=['# Reason-code redesign experiment','',
        'The models use 322 newly built features, with local features per component and three-view HGB fusion. '
        'One or more codes can be returned. Training includes non-fault examples for rejection, excludes fault hours with unknown current-hour codes, '
        'balances event influence, and selects fusion/thresholds using three grouped folds inside training.', '',
        '## Current-hour reference results','',
        '| Split | Axis | Fault hours with timed targets | Macro F1 | Micro F1 | Exact label-set accuracy |',
        '|---|---|---:|---:|---:|---:|']
    for r in shown.itertuples():
        lines.append(f'| {r.scheme} | {r.axis} | {r.rows} | {100*r.macro_f1:.2f}% | {100*r.micro_f1:.2f}% | {100*r.exact_match:.2f}% |')
    lines+=['','Macro F1 averages labels with positive test support. The minimum-one policy retains multiple above-threshold codes and supplies the highest-scoring supported code if none crosses threshold. It is intended for the conditional known-fault task.', '',
        '## Coverage and interpretation','',
        'Targets changed: an episode-wide spike label is now attached only to its physical-evidence hours. '
        'Stuck evidence starts when a trailing 24-hour window confirms it; it is not backfilled. '
        'Statistical targets retain historical contextual adjudication at the evidence timestamp. '
        'Calibration retains the original corroborated intervals. These are weak reference labels, not independent hardware diagnoses.', '',
        '| Split | Original fault test hours | Hours with a timed code | Hours without a timed code |',
        '|---|---:|---:|---:|']
    for r in pop[pop.partition=='test'].itertuples():
        lines.append(f'| {r.scheme} | {r.fault_hours} | {r.resolved_fault_hours} | {r.unresolved_fault_hours} |')
    lines+=['','Missing timed labels are unknown, not proof of healthy sensors. They are excluded from current-hour code metrics, retained in the prediction ledger, and included in the secondary original-episode-label comparison. '
        'Old and new F1 values therefore cannot be treated as a controlled improvement on the same target.', '',
        'The original May wind-vane event must be checked in target_coverage_comparison.csv: a missing timed target is not a successful prediction. '
        'A high new wind-vane score alone cannot establish that the earlier 29-hour failure has been fixed.', '',
        '## Evaluation protocol','',
        '- Random: original 70/15/15 memberships; event overlap remains and is disclosed.',
        '- Grouped: whole connected fault events, and station-day blocks for negatives; allocated to balance code/event support and hourly populations. No train/test group overlap.',
        '- Chronological: train before 1 April 2026; validate 8–30 April; test 8 May–30 June. Crossing groups are removed and seven-day boundary gaps applied.',
        '- The all-station prefix-invariance checks test predictor calculations after future rows are removed. Archived external residual arrival times and historical statistical target baselines are not certified as live-available.',
        '- Models/thresholds were not tuned on the reported test scores. These are development experiments after prior inspection of this dataset, not a fresh prospective holdout.', '',
        '## Binary detector connection','',
        'summary.csv cascade rows use a newly fitted experimental HGB gate on the same splits. This is not the frozen operational EF-HGB and is not a replacement claim. '
        'frozen_ef_hgb_summary.csv separately evaluates the actual saved final random EF-HGB with the new reason-code heads on its original random-test membership. '
        'Using that saved model on the new grouped or chronological splits would contaminate the evaluation, so it is not done.', '',
        '## Artifacts','',
        '- per_label.csv / summary.csv: current-hour code results, including threshold and minimum-one policies.',
        '- label_support.csv / population.csv: positive hours, events, stations, and unresolved coverage.',
        '- original_target_summary.csv: diagnostic performance against the original episode-broadcast targets on all fault hours.',
        '- target_coverage_comparison.csv: code-by-code changes in target membership.',
        '- frozen_ef_hgb_summary.csv: actual saved binary EF-HGB cascade on random test.',
        '- selection.csv: fitted/unsupported status, genuine selected thresholds, fusion weights, OOF selection and validation metrics.',
        '- models/: saved estimators, feature order, view indices, thresholds, and weights.',
        '- predictions.parquet / split_membership.csv / aligned_reference.parquet: reproducible sample-level evidence.',
        '- audit.json: inputs and unchanged hashes, chronology, causality checks, and limitations.', '']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(shown.to_string(index=False));print(pd.DataFrame(frozen_summary).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
