"""Trace June/July label preservation and cached statistical baseline versions."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.model.final_reason_codes import ROOT, JULY_RAW, JULY_REFS
from src.model.reason_code_rebuild import sha
from src.rules.baselines import select_baseline
from src.rules.external_residuals import compute_external_residuals


def run(out):
    if out.exists():
        raise FileExistsError(out)
    june_labels = ROOT / 'data/labels/episode_labels.csv'
    july_labels = ROOT / 'data/eval/july_2026_adjudicated_labels/episode_labels_adjudicated.csv'
    protected = [june_labels, july_labels, ROOT / 'data/merged/station_hourly_merged.csv', JULY_RAW,
        ROOT / 'data/processed/statistical_anomaly_scores.parquet', ROOT / 'data/eval/july_2026_features/statistical_anomaly_scores.parquet',
        ROOT / 'data/features/external_residuals.parquet', JULY_REFS]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    a, b = pd.read_csv(june_labels), pd.read_csv(july_labels)
    for f in (a, b):
        for c in ('start_hour', 'end_hour'):
            f[c] = pd.to_datetime(f[c], utc=True)
    retained = b[b.start_hour.lt(pd.Timestamp('2026-07-01', tz='UTC'))]
    assert set(a.episode_id) == set(retained.episode_id)
    left, right = a.set_index('episode_id').sort_index(), retained.set_index('episode_id').sort_index()
    label_diff = {c: int(left[c].fillna('').ne(right[c].fillna('')).sum()) for c in left.columns}
    manifest = json.loads((july_labels.parent / 'freeze_manifest.json').read_text())
    reviewed = pd.read_csv(july_labels.parent / 'july_episode_adjudication.csv')
    raw = pd.read_csv(ROOT / 'data/merged/station_hourly_merged.csv')
    raw.hour_utc = pd.to_datetime(raw.hour_utc, utc=True)
    combined = pd.read_csv(JULY_RAW)
    combined.hour_utc = pd.to_datetime(combined.hour_utc, utc=True)
    original = raw.set_index(['station_id', 'hour_utc']).sort_index()
    replay = combined.set_index(['station_id', 'hour_utc']).reindex(original.index)
    raw_diff = {}
    for c in original:
        if pd.api.types.is_numeric_dtype(original[c]):
            raw_diff[c] = int((~np.isclose(original[c], replay[c], equal_nan=True, rtol=1e-7, atol=1e-8)).sum())
        else:
            raw_diff[c] = int(original[c].fillna('').ne(replay[c].fillna('')).sum())
    channels = ['temp_avg_c', 'windspeed_avg_kmh', 'solar_radiation_high_wm2']
    columns = ['station_id', 'hour_utc', 'channel', 'zscore']
    old_scores = pd.read_parquet(protected[4], columns=columns, filters=[('channel', 'in', channels)])
    new_scores = pd.read_parquet(protected[5], columns=columns, filters=[('channel', 'in', channels)])
    scores = old_scores.merge(new_scores, on=['station_id', 'hour_utc', 'channel'], suffixes=('_old', '_new'), validate='one_to_one')
    scores = scores.merge(raw[['station_id', 'hour_utc'] + channels], on=['station_id', 'hour_utc'], validate='many_to_one')
    may = raw[raw.hour_utc.lt(pd.Timestamp('2026-06-01', tz='UTC'))]
    evidence = []
    for (station, channel), g in scores.groupby(['station_id', 'channel']):
        may_base = select_baseline(may, station, channel)
        june_base = select_baseline(raw, station, channel)
        for phase, mask in [('before_june', g.hour_utc.lt(pd.Timestamp('2026-06-01', tz='UTC'))), ('june', g.hour_utc.ge(pd.Timestamp('2026-06-01', tz='UTC')))]:
            q = g[mask]
            for baseline_name, baseline in [('through_may', may_base), ('through_june', june_base)]:
                spread = baseline['baseline_spread']
                expected = (q[channel] - baseline['baseline_value']) / spread if spread > 0 else np.full(len(q), np.nan)
                evidence.append(dict(station_id=station, channel=channel, phase=phase, baseline=baseline_name,
                    center=baseline['baseline_value'], spread=spread, rows=len(q),
                    old_cache_mismatches=int((~np.isclose(q.zscore_old, expected, equal_nan=True, rtol=1e-5, atol=1e-6)).sum()),
                    july_cache_mismatches=int((~np.isclose(q.zscore_new, expected, equal_nan=True, rtol=1e-5, atol=1e-6)).sum())))
    ext_old, ext_new = pd.read_parquet(protected[6]), pd.read_parquet(protected[7])
    ext = ext_old.merge(ext_new, on=['station_id', 'time_utc'], suffixes=('_old', '_new'), validate='one_to_one')
    external = []
    for c in ['station_temp', 'ref_temp', 'r_temp', 'base_temp', 'bmad_temp', 'z_temp', 'r_pressure', 'base_pressure']:
        diff = ~np.isclose(ext[c + '_old'], ext[c + '_new'], equal_nan=True, rtol=1e-5, atol=1e-6)
        external.append(dict(column=c, compared_rows=len(ext), changed_rows=int(diff.sum()),
            first_change=str(ext.loc[diff, 'time_utc'].min()), last_change=str(ext.loc[diff, 'time_utc'].max())))
    seam_rows = []
    june_external = ext_old[ext_old.time_utc.between(pd.Timestamp('2026-06-01', tz='UTC'), pd.Timestamp('2026-06-30T23:00', tz='UTC'))]
    for station, g in june_external.groupby('station_id'):
        g = g.sort_values('time_utc').reset_index(drop=True)
        rebuilt = compute_external_residuals(g)
        for c in ['base_temp', 'bmad_temp', 'z_temp', 'base_pressure', 'bmad_pressure']:
            seam_rows.append(dict(station_id=station, column=c, rows=len(g),
                june_only_rebuild_mismatches=int((~np.isclose(g[c], rebuilt[c], equal_nan=True, rtol=1e-5, atol=1e-6)).sum())))
    report = dict(input_hashes=hashes, june_episodes=len(a), june_labels_changed=label_diff,
        raw_rows=len(raw), raw_columns_changed=raw_diff,
        july_adjudication_changes=int(reviewed.automatic_label_state.ne(reviewed.adjudicated_label_state).sum()),
        july_adjudication_decisions=reviewed.adjudication_decision.value_counts().to_dict(),
        frozen_label_hash_matches=sha(july_labels) == manifest['files_sha256']['episode_labels_adjudicated.csv'],
        labels_recorded_before_predictions=manifest['prediction_access_before_freeze'] is False,
        continuing_offset_policy=manifest['adjudication_contract']['continuing_offsets'],
        limitations='Existing artifact audit, not a complete label regeneration. Original automatic July source package is absent, so exact generating-code/parameter identity cannot be certified. June labels are preserved exactly; separate July adjudication is explicit.')
    assert hashes == {str(p.relative_to(ROOT)): sha(p) for p in protected}
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(evidence).to_csv(out / 'statistical_baseline_trace.csv', index=False)
    pd.DataFrame(external).to_csv(out / 'external_trace.csv', index=False)
    pd.DataFrame(seam_rows).to_csv(out / 'june_external_history_reset_check.csv', index=False)
    (out / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Labels/raw',json.dumps({k:v for k,v in report.items() if k!='input_hashes'}, indent=2))
    print(pd.DataFrame(evidence).groupby(['phase','baseline'])[['rows','old_cache_mismatches','july_cache_mismatches']].sum().to_string())
    print(pd.DataFrame(external).to_string(index=False))
    print('June-only external rebuild', pd.DataFrame(seam_rows).groupby('column')[['rows','june_only_rebuild_mismatches']].sum().to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/eval/july_contract_audit_v2')
    args = parser.parse_args()
    run(args.output)
