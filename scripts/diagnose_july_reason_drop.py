"""Read-only model diagnosis; writes isolated tables, never tunes or refits."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.model.reason_code_rebuild import MECH, COMP, minimum_one, sha
from src.model.hourly_baseline import (load_hourly_tensor, flatten_hourly_features,
    load_reason_code_manifest_splits, binary_metrics, _fault_groups)
from src.model.final_reason_codes import ROOT, MODEL_DIR, JULY_GATE, JULY_OUTPUT


def run(out):
    if out.exists():
        raise FileExistsError(out)
    reason_path = MODEL_DIR / 'reason_heads.joblib'
    binary_path = ROOT / 'data/hourly_detection/one_hour_final/models/evidence_fusion/selected_ef_hgb_random_01h.joblib'
    inputs = [reason_path, binary_path, JULY_GATE, JULY_OUTPUT / 'reason_code_predictions.parquet']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    gate = pd.read_parquet(JULY_GATE)
    reasons = pd.read_parquet(inputs[-1])
    ref = pd.read_parquet(ROOT / 'data/eval/july_2026_reason_code_evaluation/aligned_reference.parquet')
    assert gate[['station_id', 'hour_utc']].equals(reasons[['station_id', 'hour_utc']])
    assert gate[['station_id', 'hour_utc']].equals(ref[['station_id', 'hour_utc']])
    bundle = joblib.load(reason_path)
    fault = gate.truth_fault.eq(1).to_numpy()
    detected = gate.random_prediction.eq(1).to_numpy()
    resolved = ref[MECH + COMP].any(axis=1).to_numpy()
    evaluated = ~fault | resolved
    station = gate[['station_id', 'hour_utc', 'random_probability']].copy()
    station['fp'] = ~fault & detected
    station['fn'] = fault & ~detected
    station['tp'] = fault & detected
    station['normal'] = ~fault
    station['resolved_missed'] = resolved & ~detected
    counts, score_distributions = [], []
    for axis, labels in [('mechanism', MECH), ('component', COMP)]:
        y = ref[labels].to_numpy(bool)
        probs = reasons[[f'{axis}_score__{label}' for label in labels]].to_numpy(float)
        thresholds = np.array([bundle['heads'][f'{axis}:{label}']['threshold'] for label in labels])
        prediction = minimum_one(probs >= thresholds, probs, np.ones(len(labels), bool)) & detected[:, None]
        for j, label in enumerate(labels):
            t, p = y[:, j], prediction[:, j]
            counts.append(dict(axis=axis, label=label, support=int(t.sum()),
                tp=int((t & p).sum()), fp_on_normal=int((~fault & p).sum()),
                fp_on_resolved_fault=int((resolved & ~t & p).sum()),
                fn_binary_gate=int((t & ~detected).sum()),
                fn_reason_head=int((t & detected & ~p).sum()),
                predictions_on_unknown=int((fault & ~resolved & p).sum())))
            active = t & detected
            if active.any():
                q = np.quantile(probs[active, j], [.1, .5, .9])
                score_distributions.append(dict(axis=axis, label=label, true_detected_hours=int(active.sum()),
                    threshold=thresholds[j], score_p10=q[0], score_median=q[1], score_p90=q[2],
                    fraction_below_threshold=float((probs[active, j] < thresholds[j]).mean())))
        station[f'{axis}_fp_labels'] = ((~y & prediction) & evaluated[:, None]).sum(axis=1)
    attribution = pd.DataFrame(counts)

    # Frozen binary model: verify saved probabilities and compare identical code
    # paths on development test versus the independent future-period ledger.
    june = load_hourly_tensor(ROOT / 'data/hourly_detection/one_hour_final/hourly_detection_01h.npz')
    x, names, _ = flatten_hourly_features(june)
    splits = load_reason_code_manifest_splits(june, ROOT / 'data/hourly_detection/hourly_baseline_split_manifest.csv')['random']
    combined = load_hourly_tensor(ROOT / 'data/eval/one_hour_candidate/tensors/hourly_detection_01h.npz')
    all_x, other_names, _ = flatten_hourly_features(combined)
    assert names == other_names
    combined_index = pd.MultiIndex.from_arrays([combined['station_id'], pd.to_datetime(combined['hour'], utc=True)])
    july_index = pd.MultiIndex.from_frame(gate[['station_id', 'hour_utc']])
    rows = pd.Series(np.arange(len(combined_index)), index=combined_index).reindex(july_index).to_numpy(int)
    xj = all_x[rows]
    june_index = pd.MultiIndex.from_arrays([june['station_id'], pd.to_datetime(june['hour'], utc=True)])
    prefix_rows = pd.Series(np.arange(len(combined_index)), index=combined_index).reindex(june_index).to_numpy(int)
    replayed_june = all_x[prefix_rows]
    differences = ~np.isclose(x, replayed_june, equal_nan=True, rtol=1e-5, atol=1e-6)
    prefix_check = pd.DataFrame(dict(feature=names, changed_rows=differences.sum(axis=0),
        missingness_changes=(np.isnan(x) != np.isnan(replayed_june)).sum(axis=0)))
    model = joblib.load(binary_path)['estimator']
    jp = model.predict_proba(xj)[:, 1]
    assert np.allclose(jp, gate.random_probability)
    test = splits['test']
    train = splits['train']
    val = splits['validation']
    testp = model.predict_proba(x[test])[:, 1]
    replayp = model.predict_proba(replayed_june[test])[:, 1]
    compare = []
    for scope, yy, pp in [('development_random_test', june['y_binary'][test], testp),
        ('same_test_July_preprocessing', june['y_binary'][test], replayp), ('july', fault, jp)]:
        metrics = binary_metrics(yy, pp, .3)
        metrics['scope'] = scope
        metrics['false_positive_rate'] = float(((pp >= .3) & (yy == 0)).sum() / (yy == 0).sum())
        compare.append(metrics)
    fg = _fault_groups(june['y_binary'], june['source_episode_ids'])
    train_set, test_set = set(train), set(test)
    shared_fault_test = sum(len(set(ii) & test_set) for ii in fg.values() if set(ii) & train_set)

    # Descriptive feature drift within REFERENCE-NORMAL rows only; not proof of
    # physical causes. Include missingness and excursions from training bounds.
    normal_train = x[train][june['y_binary'][train] == 0]
    normal_july = xj[~fault]
    fp_july = xj[~fault & detected]
    drift = []
    for j, name in enumerate(names):
        a, b, c = normal_train[:, j], normal_july[:, j], fp_july[:, j]
        af = a[np.isfinite(a)]
        if not len(af):
            continue
        q01, q25, q50, q75, q99 = np.quantile(af, [.01, .25, .5, .75, .99])
        scale = max(q75 - q25, 1e-6)
        drift.append(dict(feature=name, train_normal_median=q50, july_normal_median=float(np.nanmedian(b)),
            july_fp_median=float(np.nanmedian(c)), train_missing=float(np.isnan(a).mean()),
            july_missing=float(np.isnan(b).mean()), fp_missing=float(np.isnan(c).mean()),
            july_outside_train_1_99=float(((b < q01) | (b > q99)).mean()),
            fp_outside_train_1_99=float(((c < q01) | (c > q99)).mean()),
            shift_iqr=float(abs(np.nanmedian(b) - q50) / scale)))
    drift = pd.DataFrame(drift)

    feature_groups = {
        'statistical_rule_flags': [j for j, n in enumerate(names) if n.startswith('stat_') and (n.endswith('_any') or n.endswith('_rate'))],
        'statistical_continuous': [j for j, n in enumerate(names) if n.startswith(('stat_zscore_', 'stat_rolling_'))],
        'spatial_reference': [j for j, n in enumerate(names) if n.startswith(('z_spatial_', 'spatial_offset_', 'n_neighbors_present_'))],
        'external_reference': [j for j, n in enumerate(names) if n.startswith(('r_', 'offset_level_', 'z_pressure_', 'z_temp_', 'z_dewpoint_', 'z_wind_', 'rel_ratio_', 'ext_'))],
        'static': [j for j, n in enumerate(names) if n.startswith('ctx_') or n == 'spatial_isolated'],
    }
    # Pre-July validation permutation is a reliance diagnostic, not feature
    # removal, retraining, causal attribution, or July-driven feature selection.
    xv = x[val]
    base_f1 = binary_metrics(june['y_binary'][val], model.predict_proba(xv)[:, 1], .3)['f1']
    order = np.random.default_rng(20260912).permutation(len(val))
    reliance = []
    for group, cols in feature_groups.items():
        altered = xv.copy()
        altered[:, cols] = xv[order][:, cols]
        f1 = binary_metrics(june['y_binary'][val], model.predict_proba(altered)[:, 1], .3)['f1']
        reliance.append(dict(group=group, baseline_validation_f1=base_f1, permuted_f1=f1, f1_drop=base_f1 - f1))
    flags = xj[:, feature_groups['statistical_rule_flags']]
    station['any_rule_flag'] = (flags > 0).any(axis=1)
    station['full_branch_score'] = model.full_estimator.predict_proba(xj)[:, 1]
    station['rule_branch_score'] = model.rule_estimator.predict_proba(xj[:, model.rule_indices])[:, 1]

    # Known continuing calibration conditions were deliberately not new July
    # fault labels. Count overlap rather than silently relabeling false alerts.
    continuing = pd.read_csv(ROOT / 'data/eval/july_2026_adjudicated_labels/continuing_calibration_offset_decisions.csv')
    overlap = np.zeros(len(gate), bool)
    for row in continuing.itertuples():
        overlap |= (gate.station_id.eq(row.station_id) & gate.hour_utc.between(pd.Timestamp(row.start_hour), pd.Timestamp(row.end_hour))).to_numpy()
    audit = dict(input_hashes=hashes, frozen_predictions_reproduced=True, no_refitting_or_threshold_changes=True,
        development_fault_test_hours=int(june['y_binary'][test].sum()),
        development_fault_test_hours_sharing_training_event=int(shared_fault_test),
        development_hours_with_changed_replay_features=int(differences.any(axis=1).sum()),
        development_test_decisions_changed_by_replay_preprocessing=int(((testp >= .3) != (replayp >= .3)).sum()),
        july_false_alerts=int((~fault & detected).sum()),
        false_alerts_overlapping_separately_recorded_continuing_offsets=int((overlap & ~fault & detected).sum()),
        false_alerts_with_any_rule_flag=int(((flags > 0).any(axis=1) & ~fault & detected).sum()),
        note='Labels are weak references. Drift and permutation describe associations/model reliance, not verified physical causes. July has been repeatedly inspected.')
    assert hashes == {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    out.mkdir(parents=True, exist_ok=False)
    for filename, data in [('error_attribution', attribution), ('score_distributions', pd.DataFrame(score_distributions)),
        ('binary_comparison', pd.DataFrame(compare)), ('feature_drift', drift), ('preprocessing_parity', prefix_check), ('validation_feature_reliance', pd.DataFrame(reliance)),
        ('hour_diagnostics', station), ('station_errors', station.groupby('station_id')[['fp', 'fn', 'tp', 'normal', 'resolved_missed', 'mechanism_fp_labels', 'component_fp_labels']].sum().reset_index())]:
        data.to_csv(out / (filename + '.csv'), index=False)
    (out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('Attribution\n' + attribution.to_string(index=False))
    print('Binary comparison\n' + pd.DataFrame(compare).to_string(index=False))
    print('Head score distributions\n' + pd.DataFrame(score_distributions).to_string(index=False))
    print('Continuous feature drift\n' + drift[drift.feature.str.endswith('_t0')].sort_values('july_outside_train_1_99', ascending=False).head(10).to_string(index=False))
    print('Reliance\n' + pd.DataFrame(reliance).to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/eval/july_reason_drop_diagnosis_v2')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)
