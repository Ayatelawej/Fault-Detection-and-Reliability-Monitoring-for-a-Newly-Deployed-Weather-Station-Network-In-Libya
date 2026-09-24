"""Isolated blocked-holdout ablation; never modify deployed models or labels."""
from pathlib import Path
from dataclasses import asdict, replace
import os
import sys
import json
import hashlib

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from src.model.hourly_baseline import (load_hourly_tensor, filter_eligible_examples,
    flatten_hourly_features, _fault_groups, HourlyBaselineConfig, binary_metrics, make_classifier)
from src.model.hourly_calibration import calibrate_split, validation_grid_frame
from scripts.experiment_blocked_fault_detection import blocked_split, validate_split


def run():
    out = ROOT / 'data/eval/no_rule_indicators_blocked_20260923'
    out.mkdir(parents=True, exist_ok=False)
    examples, _ = filter_eligible_examples(load_hourly_tensor(ROOT / 'data/hourly_detection/one_hour_final/hourly_detection_01h.npz'))
    x, names, feature_groups = flatten_hourly_features(examples)
    y = np.asarray(examples['y_binary'], dtype=int)
    hours = pd.to_datetime(examples['hour'], utc=True)
    groups = np.array([f'normal:{s}:{str(h)[:10]}' for s, h in zip(examples['station_id'], hours)], dtype=object)
    for key, indices in _fault_groups(y, examples['source_episode_ids']).items():
        groups[indices] = 'event:' + key
    splits = blocked_split(hours, groups)
    validate_split(hours, groups, splits)
    retained = np.concatenate([splits[k] for k in ('train', 'validation', 'test')])
    remap = np.full(len(y), -1, dtype=int)
    remap[retained] = np.arange(len(retained))
    splits = {k: remap[v] for k, v in splits.items()}
    x, y = x[retained], y[retained]
    keep = sorted(i for key, indices in feature_groups.items() if not key.startswith('rule:') for i in indices)
    assert len(keep) == 29 and len(names) == 67
    old_path = ROOT / 'data/eval/blocked_model_comparison_20260916_v2/models/hgb.joblib'
    original = joblib.load(old_path)
    expected = pd.read_csv(ROOT / 'data/eval/blocked_model_comparison_20260916_v2/comparison.csv').set_index('model').loc['HGB']
    rows = []
    row = {'model': 'HGB with rule indicators', 'input_columns': len(names), 'threshold': float(expected.threshold), 'class_weight': float(expected.class_weight)}
    for partition in ('validation', 'test'):
        idx = splits[partition]
        metric = binary_metrics(y[idx], original['estimator'].predict_proba(x[idx])[:, 1], float(expected.threshold))
        for key in ('precision', 'recall', 'f1', 'accuracy'):
            np.testing.assert_allclose(metric[key], expected[f'{partition}_{key}'], atol=1e-12)
            row[f'{partition}_{key}'] = metric[key]
    rows.append(row)
    print('Original HGB metrics reproduced on identical partitions; calibrating 29-input ablation.', flush=True)
    result = calibrate_split(x[:, keep], y, splits, HourlyBaselineConfig())
    choice = result['best_balanced']
    row = {'model': 'HGB without rule indicators', 'input_columns': len(keep),
        'threshold': choice['threshold'], 'class_weight': choice['fault_class_weight']}
    for partition, metrics in [('validation', choice['validation']), ('test', result['final_test'])]:
        row.update({f'{partition}_{k}': metrics[k] for k in ('precision', 'recall', 'f1', 'accuracy')})
    rows.append(row)
    validation_grid_frame(result).to_csv(out / 'validation_grid.csv', index=False)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'comparison.csv', index=False)
    config = replace(HourlyBaselineConfig(), fault_class_weight=float(choice['fault_class_weight']), threshold=float(choice['threshold']))
    model = make_classifier(config)
    model.fit(x[splits['train']][:, keep], y[splits['train']])
    probabilities = model.predict_proba(x[splits['test']][:, keep])[:, 1]
    check = binary_metrics(y[splits['test']], probabilities, float(choice['threshold']))
    np.testing.assert_allclose(check['f1'], result['final_test']['f1'], atol=1e-12)
    joblib.dump({'estimator': model, 'config': asdict(config), 'feature_names': [names[i] for i in keep]}, out / 'hgb_without_rules.joblib')
    idx = retained[splits['test']]
    pd.DataFrame({'station_id': np.asarray(examples['station_id'])[idx], 'hour': np.asarray(examples['hour'])[idx],
        'target': y[splits['test']], 'prediction_score': probabilities}).to_parquet(out / 'test_predictions.parquet', index=False)
    manifest = {'selection': 'Original class-weight and threshold grids; maximize minimum validation precision, recall, F1',
        'split': 'February validation, March-April test, remaining eligible months training; original gaps and group exclusions',
        'partition_counts': {k: len(v) for k, v in splits.items()},
        'partition_index_sha256': {k: hashlib.sha256(retained[v].tobytes()).hexdigest() for k, v in splits.items()},
        'retained_features': [names[i] for i in keep], 'removed_features': [n for i, n in enumerate(names) if i not in keep],
        'control_reproduced': True,
        'limitations': 'Previously inspected test period. Labels and continuous engineered evidence unchanged. Removing indicators does not establish independent fault truth or audit upstream preprocessing.',
        'deployment_changed': False}
    (out / 'design.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
