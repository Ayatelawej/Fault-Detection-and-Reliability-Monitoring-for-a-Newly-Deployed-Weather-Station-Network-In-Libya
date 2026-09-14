"""Evaluate frozen July reason heads; never fit or modify deployment artifacts."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.model.final_reason_codes import (
    ROOT, MODEL_DIR, JULY_RAW, JULY_REFS, JULY_GATE, JULY_OUTPUT,
    load_observations, build_features, predict,
)
from src.model.reason_code_rebuild import MECH, COMP, aligned_labels, minimum_one, multilabel_rows, sha
from src.model.hourly_baseline import _fault_groups
from src.rules.frozen_statistics import load_frozen_rule_statistics
from src.rules.statistical_gate import build_contextual_cohort, build_statistical_evidence


def evaluate(output):
    if output.exists():
        raise FileExistsError(output)
    inputs = dict(model=MODEL_DIR / 'reason_heads.joblib', raw=JULY_RAW, references=JULY_REFS,
        gate=JULY_GATE, deployed=JULY_OUTPUT / 'reason_code_predictions.parquet',
        episodes=ROOT / 'data/eval/july_2026_adjudicated_labels/episode_labels_adjudicated.csv',
        scores=ROOT / 'data/eval/july_2026_features/statistical_anomaly_scores.parquet',
        frozen_statistics=ROOT / 'data/model/frozen_rule_statistics/frozen_rule_statistics.joblib')
    hashes = {k: sha(p) for k, p in inputs.items()}
    release = json.loads((MODEL_DIR / 'manifest.json').read_text())
    assert hashes['model'] == release['model_sha256']
    gate = pd.read_parquet(JULY_GATE)
    gate.hour_utc = pd.to_datetime(gate.hour_utc, utc=True)
    index = pd.MultiIndex.from_arrays([gate.station_id, gate.hour_utc], names=['station_id', 'hour'])
    fault = gate.truth_fault.to_numpy(int) == 1
    detected = gate.random_prediction.to_numpy(int) == 1
    episodes = pd.read_csv(inputs['episodes'])
    for col in ('start_hour', 'end_hour'):
        episodes[col] = pd.to_datetime(episodes[col], utc=True)
    episodes = episodes.loc[episodes.end_hour.ge(gate.hour_utc.min()) & episodes.start_hour.le(gate.hour_utc.max())]
    raw = load_observations(JULY_RAW, JULY_REFS, pd.Timestamp('2026-07-31T23:00Z'))
    features, evidence = build_features(raw)
    print('Current-hour features rebuilt; reconstructing retrospective July reference evidence', flush=True)

    # Context labels use the same month/hour leave-one-out historical definition
    # as development. This is retrospective reference construction, NOT inference.
    score_columns = ['station_id', 'hour_utc', 'channel', 'zscore', 'iforest_score',
                     'flag_zscore', 'flag_iforest', 'flag_physical', 'flag_stuck']
    all_scores = pd.read_parquet(inputs['scores'], columns=score_columns)
    all_scores.hour_utc = pd.to_datetime(all_scores.hour_utc, utc=True)
    july_month_scores = all_scores.loc[all_scores.hour_utc.dt.month.eq(7)].copy()
    del all_scores
    july_month_raw = raw.loc[raw.hour_utc.dt.month.eq(7)].copy()
    cohort = build_contextual_cohort(july_month_raw, july_month_scores)
    scores = july_month_scores.loc[july_month_scores.hour_utc.ge(gate.hour_utc.min())]
    frozen = load_frozen_rule_statistics(ROOT / 'data/model/frozen_rule_statistics')
    thresholds = frozen.evidence_thresholds
    del frozen
    stat = build_statistical_evidence(july_month_raw, scores, pd.read_parquet(JULY_REFS),
        cohort=cohort, frozen_thresholds=thresholds)
    y = aligned_labels(raw, evidence, episodes, stat, index)
    assert not y[~fault].any(), 'Hourly reasons must remain inside accepted fault membership'

    # Old episode-broadcast reference is reported separately, never substituted
    # for the redesigned current-hour target.
    old = np.zeros_like(y)
    for ep in episodes.loc[episodes.label_state.eq('fault')].itertuples():
        mask = gate.station_id.eq(ep.station_id) & gate.hour_utc.between(ep.start_hour, ep.end_hour)
        for axis, labels, offset in [('mechanisms', MECH, 0), ('components', COMP, len(MECH))]:
            for label in str(getattr(ep, axis)).split('|'):
                if label in labels:
                    old[mask, offset + labels.index(label)] = 1
    assert np.array_equal(old.any(axis=1), fault), 'Episode membership does not match frozen binary truth'
    groups = np.array([f'normal:{s}:{t:%Y-%m-%d}' for s, t in index], dtype=object)
    for key, ii in _fault_groups(fault.astype(int), gate.source_episode_ids.to_numpy()).items():
        groups[ii] = 'event:' + key

    bundle = joblib.load(inputs['model'])
    actual = predict(features, gate, bundle)
    saved = pd.read_parquet(inputs['deployed'])
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), saved.reset_index(drop=True))
    # Oracle-gate diagnostic asks the SAME frozen reason heads about known faults
    # missed by the detector. This does not replace the deployed ledger or policy.
    oracle = predict(features, gate.assign(random_prediction=1), bundle)
    rows, summaries = [], []
    for axis, labels, offset in [('mechanism', MECH, 0), ('component', COMP, len(MECH))]:
        yy = y[:, offset:offset + len(labels)]
        resolved = yy.any(axis=1)
        probs = oracle[[f'{axis}_score__{label}' for label in labels]].to_numpy(float)
        available = np.array([f'{axis}:{label}' in bundle['heads'] for label in labels])
        thresholds = np.array([bundle['heads'].get(f'{axis}:{label}', {}).get('threshold', np.inf) for label in labels])
        threshold_pred = probs >= thresholds
        for policy in ('threshold', 'minimum_one_diagnostic'):
            pp = threshold_pred if policy == 'threshold' else minimum_one(threshold_pred, probs, available)
            scopes = [
                ('known_resolved_faults', resolved, yy, pp),
                ('detected_resolved_faults', resolved & detected, yy, pp),
                ('end_to_end', ~fault | resolved, yy, pp & detected[:, None]),
                ('original_episode_fault_hours_diagnostic', fault, old[:, offset:offset + len(labels)], pp & detected[:, None]),
            ]
            for scope, mask, target, predicted in scopes:
                rr, ss = multilabel_rows(target[mask], predicted[mask], groups[mask], labels,
                    dict(axis=axis, policy=policy, scope=scope))
                rows.extend(rr)
                summaries.append(ss)
    summary = pd.DataFrame(summaries)
    detail = pd.DataFrame(rows)
    reference = pd.DataFrame(y, columns=MECH + COMP)
    reference.insert(0, 'hour_utc', gate.hour_utc)
    reference.insert(0, 'station_id', gate.station_id)
    reference['truth_fault'] = fault
    reference['detected_fault'] = detected
    reference['source_episode_ids'] = gate.source_episode_ids
    resolved = y.any(axis=1)
    audit = dict(input_hashes=hashes, no_refitting=True, no_threshold_changes=True,
        deployed_predictions_reproduced=True, rows=len(gate), fault_hours=int(fault.sum()),
        detected_fault_hours=int((fault & detected).sum()), false_alert_hours=int((~fault & detected).sum()),
        resolved_fault_hours=int(resolved.sum()), unresolved_fault_hours=int((fault & ~resolved).sum()),
        detected_resolved_fault_hours=int((resolved & detected).sum()),
        unresolved_alert_hours=int((fault & ~resolved & detected).sum()),
        label_reference='Accepted frozen July episode membership AND reconstructed same-hour evidence; not hardware ground truth',
        statistical_reference='Retrospective same-month/hour leave-one-out context (including July 2025 where present), frozen June detector thresholds and archived July external residuals',
        metric_scope='Unknown fault-hour reasons excluded from timed metrics and counted explicitly; minimum-one is diagnostic only',
        absent_positive_labels=[label for j, label in enumerate(MECH + COMP) if not y[:, j].any()])
    assert hashes == {k: sha(p) for k, p in inputs.items()}
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / 'summary.csv', index=False)
    detail.to_csv(output / 'per_label.csv', index=False)
    reference.to_parquet(output / 'aligned_reference.parquet', index=False)
    stat.to_parquet(output / 'reconstructed_statistical_evidence.parquet', index=False)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(detail[(detail.policy == 'threshold') & (detail.scope == 'end_to_end')].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/eval/july_2026_reason_code_evaluation')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        evaluate(args.output)
