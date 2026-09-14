"""Isolated June-trained inspection-class test; no production artifacts change."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from src.model.final_reason_codes import (ROOT, FREEZE, MODEL_DIR, JULY_RAW,
    JULY_REFS, JULY_OUTPUT, load_observations, build_features)
from src.model.reason_code_rebuild import (MECH, COMP, SEED, sha, feature_views,
    fit_estimator, probability, event_weights, select_policy, minimum_one, multilabel_rows)
from src.model.hourly_baseline import load_hourly_tensor, _fault_groups


def route_to_inspection(code_predictions, requested, detected):
    """Inspection is an exclusive alternative to specific codes, behind the gate."""
    routed = np.asarray(requested, bool) & np.asarray(detected, bool)
    codes = np.asarray(code_predictions, bool) & np.asarray(detected, bool)[:, None]
    codes[routed] = False
    return np.column_stack([codes, routed])


def run(output):
    if output.exists():
        raise FileExistsError(output)
    development = ROOT / 'data/eval/reason_code_rebuild_20260912_v2'
    july_eval = ROOT / 'data/eval/july_2026_reason_code_evaluation'
    paths = dict(model=MODEL_DIR / 'reason_heads.joblib', deployed=JULY_OUTPUT / 'reason_code_predictions.parquet',
        raw=ROOT / 'data/merged/station_hourly_merged.csv', references=ROOT / 'data/features/external_residuals.parquet',
        tensor=ROOT / 'data/hourly_detection/one_hour_final/hourly_detection_01h.npz',
        june_reference=development / 'aligned_reference.parquet', july_reference=july_eval / 'aligned_reference.parquet',
        july_raw=JULY_RAW, july_residuals=JULY_REFS)
    before = {k: sha(v) for k, v in paths.items()}
    dev_audit = json.loads((development / 'audit.json').read_text())
    for key in ('raw', 'references', 'tensor'):
        assert before[key] == dev_audit['input_hashes'][key]
    train_ref = pd.read_parquet(paths['june_reference'])
    train_ref.hour = pd.to_datetime(train_ref.hour, utc=True)
    assert train_ref.hour.max() <= FREEZE
    z = load_hourly_tensor(paths['tensor'])
    index = pd.MultiIndex.from_arrays([z['station_id'], pd.to_datetime(z['hour'], utc=True)], names=['station_id', 'hour'])
    train_ref = train_ref.set_index(['station_id', 'hour']).reindex(index)
    assert not train_ref[MECH + COMP + ['original_fault']].isna().any().any()
    fault = np.asarray(z['y_binary'], int)
    assert np.array_equal(fault, train_ref.original_fault.to_numpy(int))
    resolved = train_ref[MECH + COMP].any(axis=1).to_numpy()
    assert np.array_equal(train_ref[MECH].any(axis=1), train_ref[COMP].any(axis=1))
    target = ((fault == 1) & ~resolved).astype(int)
    groups = np.array([f'normal:{s}:{t:%Y-%m-%d}' for s, t in index], dtype=object)
    for key, ii in _fault_groups(fault, z['source_episode_ids']).items():
        groups[ii] = 'event:' + key
    positives = np.flatnonzero(fault == 1)  # Both resolved and unresolved faults.
    negatives = np.flatnonzero(fault == 0)
    if len(negatives) > 16000:
        negatives = np.sort(np.random.default_rng(SEED).choice(negatives, 16000, replace=False))
    fitrows = np.sort(np.r_[positives, negatives])
    features, _ = build_features(load_observations(paths['raw'], paths['references'], FREEZE))
    names = features.columns.tolist()
    x = features.reindex(index).to_numpy('float32')
    del features
    views = feature_views(names, 'mechanism', 'admin_inspection_required')
    oof = np.zeros((len(fitrows), 3))
    foldids = np.zeros(len(fitrows), int)
    for fold, (fi, oi) in enumerate(GroupKFold(3).split(fitrows, groups=groups[fitrows])):
        assert not set(groups[fitrows[fi]]) & set(groups[fitrows[oi]])
        foldids[oi] = fold
        for branch, cols in enumerate(views):
            model = fit_estimator(x[fitrows[fi]][:, cols], target[fitrows[fi]], groups[fitrows[fi]])
            oof[oi, branch] = probability(model, x[fitrows[oi]][:, cols])
        print(f'June grouped OOF fold {fold + 1}/3 complete', flush=True)
    mix, threshold, cv = select_policy(target[fitrows], oof, event_weights(groups[fitrows]), foldids)
    models = [fit_estimator(x[fitrows][:, cols], target[fitrows], groups[fitrows]) for cols in views]
    del x
    selection = dict(threshold=threshold, weights=mix.tolist(), oof_event_f1=cv,
        training_rows=len(fitrows), training_unknown_hours=int(target.sum()), seed=SEED,
        selection='One fixed EF-HGB router; 3-fold June grouped OOF; no July tuning')
    print('Inspection router frozen: ' + json.dumps(selection), flush=True)

    # Only now load July evaluation targets. They never enter fitting or selection.
    july = pd.read_parquet(paths['july_reference'])
    predictions = pd.read_parquet(paths['deployed'])
    assert july[['station_id', 'hour_utc']].equals(predictions[['station_id', 'hour_utc']])
    assert july.detected_fault.equals(predictions.random_prediction.eq(1))
    jindex = pd.MultiIndex.from_arrays([july.station_id, july.hour_utc], names=['station_id', 'hour'])
    features, _ = build_features(load_observations(JULY_RAW, JULY_REFS, july.hour_utc.max()))
    assert features.columns.tolist() == names
    xj = features.reindex(jindex).to_numpy('float32')
    gate = predictions.random_prediction.eq(1).to_numpy()
    score = np.full(len(july), np.nan)
    score[gate] = np.column_stack([probability(m, xj[gate][:, cols])
                                  for m, cols in zip(models, views, strict=True)]) @ mix
    request = score >= threshold
    known = july[MECH + COMP].any(axis=1).to_numpy()
    unknown = july.truth_fault.to_numpy(bool) & ~known
    assert np.array_equal(july[MECH].any(axis=1), july[COMP].any(axis=1))
    frozen = joblib.load(paths['model'])
    summaries, details = [], []
    ledger = july[['station_id', 'hour_utc', 'truth_fault', 'detected_fault']].copy()
    ledger['reference_admin_required'] = unknown
    ledger['admin_score'] = score
    ledger['predicted_admin_required'] = request & gate
    for axis, labels in [('mechanism', MECH), ('component', COMP)]:
        pp = predictions[[f'{axis}_score__{label}' for label in labels]].to_numpy(float)
        thresholds = np.array([frozen['heads'][f'{axis}:{label}']['threshold'] for label in labels])
        codes = minimum_one(pp >= thresholds, pp, np.ones(len(labels), bool)) & gate[:, None]
        target_aug = np.column_stack([july[labels].to_numpy(int), unknown])
        assert not target_aug[~july.truth_fault.to_numpy(bool)].any()
        for policy, routed in [('minimum_one_baseline', np.zeros(len(july), bool)), ('learned_inspection', request)]:
            predicted = route_to_inspection(codes, routed, gate)
            assert not predicted[~gate].any()
            assert not (predicted[:, -1] & predicted[:, :-1].any(axis=1)).any()
            for scope, mask, yy, prediction, label_names in [
                ('all_hours_reason_or_review', np.ones(len(july), bool), target_aug, predicted, labels + ['admin_inspection_required']),
                ('previous_population_specific_codes', ~unknown, target_aug[:, :-1], predicted[:, :-1], labels),
            ]:
                rr, ss = multilabel_rows(yy[mask], prediction[mask], np.arange(len(july))[mask], label_names,
                    dict(axis=axis, policy=policy, scope=scope))
                # Event counts are not claimed: here each row is an evaluation unit.
                for row in rr:
                    row.pop('positive_events', None)
                details.extend(rr)
                summaries.append(ss)
            ledger[f'{policy}_{axis}'] = [' | '.join(np.asarray(labels + ['admin_inspection_required'])[row]) for row in predicted]
    review_metrics = precision_recall_fscore_support(unknown, request & gate, average='binary', zero_division=0)[:3]
    audit = dict(selection=selection, july_hours=len(july), july_unknown_hours=int(unknown.sum()),
        july_alert_hours=int(gate.sum()), admin_requests=int((request & gate).sum()),
        correctly_routed_unknown=int((request & gate & unknown).sum()),
        resolved_faults_routed_to_admin=int((request & gate & known).sum()),
        false_fault_alerts_routed_to_admin=int((request & gate & ~july.truth_fault.to_numpy(bool)).sum()),
        inspection_precision=float(review_metrics[0]), inspection_recall=float(review_metrics[1]), inspection_f1=float(review_metrics[2]),
        input_hashes=before, production_unchanged=True,
        task='Unknown reference faults become an exclusive inspection class. Normal hours have no class. Router uses features only.',
        limitation='Inspection reference is label-resolution status, not independent human confirmation of inspection need. July is exploratory after repeated inspection; not a fresh confirmatory holdout.',
        comparison='Compare policies on identical all-hour augmented targets; do not equate augmented F1 with old exclusion-based F1.')
    assert before == {k: sha(v) for k, v in paths.items()}
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(summaries).to_csv(output / 'summary.csv', index=False)
    pd.DataFrame(details).to_csv(output / 'per_label.csv', index=False)
    ledger.to_parquet(output / 'predictions.parquet', index=False)
    joblib.dump(dict(models=models, views=views, feature_names=names, **selection), output / 'inspection_router.joblib')
    (output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2), flush=True)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/eval/july_admin_inspection_experiment')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)
