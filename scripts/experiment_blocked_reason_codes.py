"""Isolated March-April reason-head test using the saved blocked EF-HGB gate."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from scripts.experiment_chronological_fault_detection import blocked_split, validate_split
from src.model.hourly_baseline import load_hourly_tensor, _fault_groups, binary_metrics
from src.model.final_reason_codes import load_observations, build_features, FREEZE, apply_output_policy
from src.model.reason_code_rebuild import (
    MECH, COMP, SEED, sha, feature_views, fit_estimator, probability,
    event_weights, select_policy, multilabel_rows,
)


def training_rows(train, fault, resolved, limit=16000):
    positive = train[(fault[train] == 1) & resolved[train]]
    negative = train[fault[train] == 0]
    if len(negative) > limit:
        negative = np.sort(np.random.default_rng(SEED).choice(negative, limit, replace=False))
    return np.sort(np.r_[positive, negative])


def run(out):
    if out.exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    gate_dir = ROOT / 'data/eval/blocked_ef_hgb_20260914'
    original_dir = ROOT / 'data/eval/reason_code_rebuild_20260912_v2'
    paths = dict(raw=ROOT / 'data/merged/station_hourly_merged.csv',
        references=ROOT / 'data/features/external_residuals.parquet',
        tensor=ROOT / 'data/hourly_detection/one_hour_final/hourly_detection_01h.npz',
        labels=original_dir / 'aligned_reference.parquet',
        split=gate_dir / 'split_membership.csv', gate=gate_dir / 'predictions.parquet',
        gate_model=gate_dir / 'model.joblib', gate_report=gate_dir / 'report.json',
        deployed_reason=ROOT / 'data/model/reason_codes/final/reason_heads.joblib',
        deployed_binary=ROOT / 'data/hourly_detection/one_hour_final/models/evidence_fusion/selected_ef_hgb_random_01h.joblib',
        deployed_july=ROOT / 'data/eval/july_2026_reason_codes_mixed_v2/reason_code_predictions.parquet',
        dashboard=ROOT / 'src/dashboard/replay.py')
    hashes = {key: sha(path) for key, path in paths.items()}
    z = load_hourly_tensor(paths['tensor'])
    hours = pd.to_datetime(z['hour'], utc=True)
    index = pd.MultiIndex.from_arrays([z['station_id'], hours], names=['station_id', 'hour'])
    fault = np.asarray(z['y_binary'], int)
    groups = np.array([f'normal:{s}:{str(t)[:10]}' for s, t in index], dtype=object)
    for key, ii in _fault_groups(fault, z['source_episode_ids']).items():
        groups[ii] = 'event:' + key
    splits = blocked_split(hours, groups)
    validate_split(hours, groups, splits, chronological=False)
    saved_split = pd.read_csv(paths['split'])
    for part, ii in splits.items():
        np.testing.assert_array_equal(ii, saved_split.loc[saved_split.partition.eq(part), 'row_index'])
    reference = pd.read_parquet(paths['labels'])
    reference.hour = pd.to_datetime(reference.hour, utc=True)
    reference = reference.set_index(['station_id', 'hour']).reindex(index)
    assert not reference[MECH + COMP + ['original_fault']].isna().any().any()
    np.testing.assert_array_equal(reference.original_fault, fault)
    y = reference[MECH + COMP].to_numpy(int)
    assert not y[fault == 0].any()
    population = [dict(partition=p, hours=len(ii), fault_hours=int(fault[ii].sum()),
        resolved_fault_hours=int(y[ii].any(axis=1).sum()),
        unknown_fault_hours=int(((fault[ii] == 1) & ~y[ii].any(axis=1)).sum()))
        for p, ii in splits.items()]
    out.mkdir(parents=True)
    (out / 'models').mkdir()
    plan = dict(split='blocked March-April 2026; February validation; May-June included in training',
        population=population, input_hashes=hashes, seed=SEED, threads=2,
        reason_selection='Three grouped OOF folds inside training only; February sanity check, not reason selection',
        gate='Saved matching-split binary EF-HGB; both fixed 0.30 and February-selected 0.35 reported',
        policies=['threshold', 'minimum_one', 'mixed (minimum-one mechanism, threshold component)'],
        limitations=['Evidence-derived current-hour labels, not confirmed hardware diagnoses.',
            'Unknown reason fault hours excluded per axis from reason metrics, never treated as normal.',
            'Existing preprocessing and statistical reference timing limitations persist.',
            'Previously inspected development data and retrospectively chosen test period; exploratory, not prospective.',
            'Rolling predictors use observation history, including preceding held-out observations; model fitting excludes held-out rows.'])
    (out / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    pd.DataFrame(population).to_csv(out / 'population.csv', index=False)
    print('Blocked membership verified: ' + json.dumps(population), flush=True)
    features, _ = build_features(load_observations(paths['raw'], paths['references'], FREEZE))
    names = list(features.columns)
    assert names == joblib.load(paths['deployed_reason'])['feature_names']
    assert index.isin(features.index).all()
    x = features.reindex(index).to_numpy('float32')
    del features
    train, val, test = (splits[p] for p in ('train', 'validation', 'test'))
    heads, selection, support = {}, [], []
    for axis, labels in [('mechanism', MECH), ('component', COMP)]:
        yy = reference[labels].to_numpy(int)
        fitrows = training_rows(train, fault, yy.any(axis=1))
        assert not np.intersect1d(fitrows, np.r_[val, test]).size
        for j, label in enumerate(labels):
            target = yy[:, j]
            for partition, ii in splits.items():
                positive = ii[target[ii] == 1]
                support.append(dict(axis=axis, label=label, partition=partition,
                    hours=len(positive), events=len(np.unique(groups[positive])),
                    stations=len(np.unique(z['station_id'][positive]))))
            events = len(np.unique(groups[fitrows[target[fitrows] == 1]]))
            if events < 3:
                selection.append(dict(axis=axis, label=label, status='unsupported', events=events))
                print(f'{axis}/{label}: unsupported ({events} training events)', flush=True)
                continue
            views = feature_views(names, axis, label)
            oof = np.zeros((len(fitrows), 3)); folds = np.zeros(len(fitrows), int)
            for fold, (fi, oi) in enumerate(GroupKFold(3).split(fitrows, groups=groups[fitrows])):
                assert not set(groups[fitrows[fi]]) & set(groups[fitrows[oi]])
                folds[oi] = fold
                for branch, cols in enumerate(views):
                    estimator = fit_estimator(x[fitrows[fi]][:, cols], target[fitrows[fi]], groups[fitrows[fi]])
                    oof[oi, branch] = probability(estimator, x[fitrows[oi]][:, cols])
            weights, threshold, cv = select_policy(target[fitrows], oof, event_weights(groups[fitrows]), folds)
            models = [fit_estimator(x[fitrows][:, cols], target[fitrows], groups[fitrows]) for cols in views]
            head = dict(models=models, views=views, weights=weights, threshold=threshold, feature_names=names)
            heads[(axis, label)] = head
            vp = np.column_stack([probability(m, x[val][:, cols]) for m, cols in zip(models, views)]) @ weights
            known = (fault[val] == 0) | yy[val].any(axis=1)
            vm = binary_metrics(target[val[known]], vp[known], threshold)
            selection.append(dict(axis=axis, label=label, status='fitted', events=events,
                threshold=threshold, full_weight=weights[0], context_weight=weights[1], rules_weight=weights[2],
                oof_event_f1=cv, validation_f1=vm['f1']))
            joblib.dump(head, out / 'models' / f'{axis}_{label}.joblib')
            print(f'Fitted {axis}/{label}: {events} events, February F1 {vm["f1"]:.3f}', flush=True)
    pd.DataFrame(selection).to_csv(out / 'selection.csv', index=False)
    pd.DataFrame(support).to_csv(out / 'label_support.csv', index=False)
    print('All reason heads/thresholds frozen; scoring March-April now', flush=True)
    binary = pd.read_parquet(paths['gate']).set_index('row_index').loc[test]
    np.testing.assert_array_equal(binary.station_id, z['station_id'][test])
    np.testing.assert_array_equal(pd.DatetimeIndex(pd.to_datetime(binary.hour, utc=True)).asi8, hours[test].asi8)
    np.testing.assert_array_equal(binary.reference_fault, fault[test])
    all_rows, summaries, ledger = [], [], []
    for axis, labels in [('mechanism', MECH), ('component', COMP)]:
        yy = reference[labels].to_numpy(int)[test]
        resolved = yy.any(axis=1)
        scores = np.full((len(test), len(labels)), np.nan)
        thresholds = np.full(len(labels), np.inf)
        for j, label in enumerate(labels):
            if (axis, label) not in heads:
                continue
            head = joblib.load(out / 'models' / f'{axis}_{label}.joblib')
            scores[:, j] = np.column_stack([probability(m, x[test][:, cols])
                for m, cols in zip(head['models'], head['views'])]) @ head['weights']
            thresholds[j] = head['threshold']
        for policy in ('threshold', 'minimum_one', 'mixed'):
            mode = ('minimum_one' if axis == 'mechanism' else 'threshold') if policy == 'mixed' else policy
            for gate_name in ('conditional_known_fault', 'frozen_0.30', 'february_selected_0.35'):
                gate = (np.ones(len(test), bool) if gate_name == 'conditional_known_fault' else
                    binary['prediction_frozen' if gate_name == 'frozen_0.30' else 'prediction_february_selected'].to_numpy(bool))
                mask = resolved if gate_name == 'conditional_known_fault' else ((fault[test] == 0) | resolved)
                predicted = apply_output_policy(scores, thresholds, gate, mode)
                assert not predicted[~gate].any()
                rows, summary = multilabel_rows(yy[mask], predicted[mask], groups[test][mask], labels,
                    dict(axis=axis, policy=policy, gate=gate_name))
                all_rows.extend(rows); summaries.append(summary)
        for j, label in enumerate(labels):
            ledger.append(pd.DataFrame(dict(row_index=test, station_id=z['station_id'][test], hour=hours[test],
                group=groups[test], axis=axis, label=label, original_fault=fault[test],
                reference_resolved=resolved, target=yy[:, j], probability=scores[:, j], threshold=thresholds[j],
                available=(axis, label) in heads, prediction_threshold=scores[:, j] >= thresholds[j],
                gate_frozen=binary.prediction_frozen.to_numpy(bool),
                gate_february_selected=binary.prediction_february_selected.to_numpy(bool))))
    pd.DataFrame(all_rows).to_csv(out / 'per_label.csv', index=False)
    pd.DataFrame(summaries).to_csv(out / 'summary.csv', index=False)
    pd.concat(ledger).to_parquet(out / 'predictions.parquet', index=False)
    assert hashes == {key: sha(path) for key, path in paths.items()}
    audit = dict(**plan, elapsed_seconds=time.perf_counter() - started, feature_count=len(names),
        protected_unchanged=True, matching_binary_membership=True, group_overlap=0,
        heads_fitted=len(heads), scoring_from_serialized_heads=True)
    (out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/eval/blocked_reason_codes_20260914')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)
