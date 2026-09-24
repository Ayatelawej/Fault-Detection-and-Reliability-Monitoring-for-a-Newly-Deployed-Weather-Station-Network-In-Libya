"""Promote verified forecast models, preserving previous canonical artifacts."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/eval/final_system_release_20260924'
MODELS = ROOT / 'data/model/final_system_20260924'


def run():
    checks = json.loads((OUT / 'verification.json').read_text())
    assert checks['dashboard_gate_joins'] and checks['validation_selection']
    assert len(checks['forecasts']) == 11
    manifest_path = OUT / 'release_manifest.json'
    if manifest_path.exists():
        raise FileExistsError('Release already published; previous backups must not be replaced')
    for name in ('station_operational_scorecard.csv', 'station_operational_scorecard_report.txt',
                 'station_operational_scorecard_invariants.json', 'station_operational_scorecard_causality.csv'):
        if not (OUT / name).is_file():
            raise FileNotFoundError(OUT / name)
    scorecard_checks = json.loads((OUT / 'station_operational_scorecard_invariants.json').read_text())
    assert scorecard_checks['all_delete_future_checks_passed']
    assert scorecard_checks['inconsistency_count'] == 0
    def backup(path):
        target = OUT / 'previous' / path.relative_to(ROOT)
        if path.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    forecasts = json.loads((OUT / 'forecast_selection.json').read_text())
    for row in forecasts:
        source = ROOT / row['path']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row['sha256']
    for row in forecasts:
        folder = 'health_forecast' if row['horizon_h'] <= 24 else 'health_forecast_long_horizon/current'
        destination = ROOT / 'data/model' / folder / Path(row['path']).name
        backup(destination)
        shutil.copy2(ROOT / row['path'], destination)
        row['canonical_path'] = str(destination.relative_to(ROOT))
    for folder in ('health_forecast', 'health_forecast_long_horizon/current'):
        path = ROOT / 'data/eval' / folder / 'health_forecast_model_manifest.json'
        if path.exists():
            backup(path)
            metadata = json.loads(path.read_text())
            metadata['transmitting_origin_release_override'] = str(manifest_path.relative_to(ROOT))
            metadata['historical_metadata_notice'] = 'Original experiments below are historical; the override identifies active forecast models.'
            path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for path in (ROOT / 'data/processed').glob('station_operational_scorecard*'):
        if path.is_file():
            backup(path)
    for name in ('station_operational_scorecard.csv', 'station_operational_scorecard_report.txt', 'station_operational_scorecard_invariants.json'):
        shutil.copy2(OUT / name, ROOT / 'data/processed' / name)
    shutil.copy2(OUT / 'station_operational_scorecard_causality.csv', ROOT / 'data/processed/station_operational_scorecard_delete_future_validation.csv')
    manifest = dict(release='20260924-validation-selected', binary=json.loads((OUT/'binary_selection.json').read_text()),
        reasons=json.loads((OUT/'reason_selection.json').read_text()), forecasts=forecasts,
        binary_model=str((MODELS/'binary.joblib').relative_to(ROOT)), reason_model=str((MODELS/'reasons/reason_heads.joblib').relative_to(ROOT)),
        evaluation_directory=str(OUT.relative_to(ROOT)), verification=checks,
        ledger_sha256={name: hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in
            ('binary_predictions.parquet', 'dashboard_forecasts.parquet', 'weather_annotations.parquet', 'reasons/reason_code_predictions.parquet')},
        selection='Development temporal validation only; no July fitting or policy tuning in this release.',
        limitations='Historical binary retrospective features remain. July was previously inspected, not an untouched holdout.',
        rollback='Previous canonical forecasts and scorecard outputs are under previous/. Older binary/reason releases are unchanged.')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(manifest_path)


if __name__ == '__main__':
    run()
