"""Bounded EF-HGB temporal-transfer check; never replace deployed artifacts."""
from pathlib import Path
import argparse
import hashlib
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
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from src.model.hourly_baseline import (
    EvidenceFusedHgbClassifier, _fault_groups, binary_metrics,
    flatten_hourly_features, load_hourly_tensor,
)
from src.model.hourly_calibration import CALIBRATION_THRESHOLDS


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def select_threshold(labels, probability, original=0.30):
    rows = [dict(threshold=t, **binary_metrics(labels, probability, t))
            for t in CALIBRATION_THRESHOLDS]
    # Match the established balanced validation objective; no test input.
    best = max(rows, key=lambda r: (
        min(r['precision'], r['recall'], r['f1']), r['f1'],
        r['precision'], r['recall'], -abs(r['threshold'] - original)))
    return best, rows


def blocked_split(hours, groups):
    """February validation, March-April test, remaining months training."""
    h = pd.DatetimeIndex(hours)
    boundaries = pd.to_datetime(['2026-02-01', '2026-03-01', '2026-05-01'], utc=True)
    a, b, c = boundaries
    category = np.select([(h >= a) & (h < b), (h >= b) & (h < c)], [1, 2], default=0)
    counts = pd.DataFrame({'group': groups, 'partition': category}).groupby('group').partition.nunique()
    crosses = np.isin(groups, counts[counts > 1].index)
    gap = np.zeros(len(h), dtype=bool)
    for boundary in boundaries:
        gap |= (h >= boundary) & (h < boundary + pd.Timedelta(days=7))
    return {name: np.flatnonzero((category == k) & ~crosses & ~gap)
            for k, name in enumerate(('train', 'validation', 'test'))}


def validate_split(hours, groups, splits, chronological=False):
    h = pd.DatetimeIndex(hours)
    for name, indices in splits.items():
        if len(indices) == 0 or len(np.unique(indices)) != len(indices):
            raise ValueError(f'Empty or duplicated {name} membership')
    for left, right in (('train', 'validation'), ('validation', 'test'), ('train', 'test')):
        if np.intersect1d(splits[left], splits[right]).size:
            raise ValueError('Rows overlap across partitions')
        if chronological and h[splits[left]].max() >= h[splits[right]].min():
            raise ValueError('Non-chronological membership')
        if set(groups[splits[left]]) & set(groups[splits[right]]):
            raise ValueError('Connected groups cross partitions')


def run(out):
    mode = 'blocked'
    if out.exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    model_path = ROOT / 'data/hourly_detection/one_hour_final/models/evidence_fusion/selected_ef_hgb_random_01h.joblib'
    tensor_path = ROOT / 'data/hourly_detection/one_hour_final/hourly_detection_01h.npz'
    membership_path = ROOT / 'data/eval/reason_code_rebuild_20260912_v2/split_membership.csv'
    protected = [model_path, tensor_path, membership_path,
        ROOT / 'data/model/reason_codes/final/reason_heads.joblib',
        ROOT / 'data/eval/july_2026_reason_codes_mixed_v2/reason_code_predictions.parquet',
        ROOT / 'src/dashboard/replay.py', ROOT / 'scripts/run_dashboard.py']
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    bundle = joblib.load(model_path)
    original = bundle['estimator']
    z = load_hourly_tensor(tensor_path)
    x, names, _ = flatten_hourly_features(z)
    assert names == bundle['feature_names']
    y = np.asarray(z['y_binary'], int)
    hours = pd.to_datetime(z['hour'], utc=True)
    assert hours.max() < pd.Timestamp('2026-07-01', tz='UTC')
    groups = np.array([f'normal:{s}:{str(t)[:10]}'
                       for s, t in zip(z['station_id'], hours)], dtype=object)
    for key, indices in _fault_groups(y, z['source_episode_ids']).items():
        groups[indices] = 'event:' + key
    splits = blocked_split(hours, groups)
    validate_split(hours, groups, splits)
    validation_month = 'february'
    support = {}
    for name, ii in splits.items():
        assert len(np.unique(y[ii])) == 2
        support[name] = dict(hours=len(ii), faults=int(y[ii].sum()),
            normal=int((y[ii] == 0).sum()), first=str(hours[ii].min()), last=str(hours[ii].max()),
            stations=len(np.unique(z['station_id'][ii])),
            months=sorted(set(hours[ii].strftime('%Y-%m'))))
    out.mkdir(parents=True)
    plan = dict(mode=mode, configuration_source=str(model_path.relative_to(ROOT)),
        config=bundle['config'], fusion_weights=bundle['fusion_weights'],
        partitions=support, excluded_hours=len(y) - sum(len(ii) for ii in splits.values()),
        threshold_grid=list(CALIBRATION_THRESHOLDS),
        selection=f'{validation_month.title()} only: maximize min(precision, recall, F1), then F1/precision/recall',
        limitations=[
            'Existing historical feature tensor and weak reference labels are reused, not reconstructed as of each cutoff.',
            'Inherited retrospective statistical labels, preprocessing fit periods and backfilled stuck flags prevent a fully causal pipeline claim.',
            'Architecture, hyperparameters and fusion weights were previously selected on development data that includes later months.',
            'This is an exploratory temporal-transfer diagnostic, not independent prospective validation.',
            'Blocked mode trains on both sides of the held-out period; it is not chronological future prediction.',
        ], source_sha256=before)
    (out / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print('Plan frozen before fitting: ' + json.dumps(support), flush=True)
    estimators = []
    views = [np.arange(x.shape[1]), original.context_indices, original.rule_indices]
    for name, estimator, columns in zip(('full', 'context', 'rules'),
            (original.full_estimator, original.context_estimator, original.rule_estimator), views):
        fresh = clone(estimator)
        assert not hasattr(fresh, 'classes_')
        fresh.fit(x[splits['train']][:, columns], y[splits['train']])
        estimators.append(fresh)
        print(f'Fitted fresh {name} estimator ({fresh.n_iter_} iterations)', flush=True)
    model = EvidenceFusedHgbClassifier(*estimators, original.context_indices.copy(),
        original.rule_indices.copy(), original.full_weight, original.context_weight, original.rule_weight)
    valid = splits['validation']
    vp = model.predict_proba(x[valid])[:, 1]
    selected, grid = select_threshold(y[valid], vp, bundle['config']['threshold'])
    pd.DataFrame(grid).to_csv(out / 'validation_thresholds.csv', index=False)
    (out / 'selected_threshold.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    print(f'{validation_month.title()} threshold frozen before test scoring: ' + json.dumps(selected), flush=True)
    test = splits['test']
    tp = model.predict_proba(x[test])[:, 1]
    results = []
    for policy, threshold in [('frozen_0.30', bundle['config']['threshold']),
                              (f'{validation_month}_selected', selected['threshold'])]:
        for partition, ii, probability in [('validation', valid, vp), ('test', test, tp)]:
            results.append(dict(policy=policy, partition=partition, threshold=threshold,
                **binary_metrics(y[ii], probability, threshold),
                auroc=float(roc_auc_score(y[ii], probability)),
                auprc=float(average_precision_score(y[ii], probability))))
    pd.DataFrame(results).to_csv(out / 'metrics.csv', index=False)
    ledger = []
    for partition, ii, probability in [('validation', valid, vp), ('test', test, tp)]:
        ledger.append(pd.DataFrame(dict(partition=partition, row_index=ii,
            station_id=z['station_id'][ii], hour=hours[ii], group=groups[ii],
            reference_fault=y[ii], score=probability,
            prediction_frozen=probability >= bundle['config']['threshold'],
            **{f'prediction_{validation_month}_selected': probability >= selected['threshold']})))
    pd.concat(ledger).to_parquet(out / 'predictions.parquet', index=False)
    pd.concat([pd.DataFrame(dict(partition=name, row_index=ii, hour=hours[ii], group=groups[ii]))
               for name, ii in splits.items()]).to_csv(out / 'split_membership.csv', index=False)
    joblib.dump(dict(estimator=model, feature_names=names, config=bundle['config'],
        **{f'{validation_month}_selected_threshold': selected['threshold']}, exploratory_only=True), out / 'model.joblib')
    restored = joblib.load(out / 'model.joblib')['estimator']
    np.testing.assert_allclose(restored.predict_proba(x[test[:100]])[:, 1], tp[:100], rtol=0, atol=0)
    for result in results:
        if result['partition'] == 'test':
            p = tp >= result['threshold']
            assert result['tp'] == int((p & (y[test] == 1)).sum())
            assert result['fp'] == int((p & (y[test] == 0)).sum())
    after = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    assert before == after, 'Protected artifact changed'
    audit = dict(elapsed_seconds=time.perf_counter() - started, thread_limit=2,
        protected_unchanged=True, feature_count=x.shape[1],
        group_overlap=0,
        serialized_model_replay=True, results=results, **plan)
    (out / 'report.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('blocked',), default='blocked')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output or ROOT / 'data/eval/blocked_ef_hgb_20260914')
