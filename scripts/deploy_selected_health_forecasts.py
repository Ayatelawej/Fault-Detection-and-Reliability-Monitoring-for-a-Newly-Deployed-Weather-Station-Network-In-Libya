"""Promote validation-selected residual forecasts, with backups and parity checks."""
from pathlib import Path
import sys
import json
import shutil
import hashlib
import os

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.availability import health_forecast as hf
from src.availability.risk_eval import regression_metrics
from scripts.summarize_forecast_baseline_corrections import SOURCES


def bands(values):
    return np.searchsorted([40., 60., 80.], np.asarray(values), side='right')


def finalize_metadata():
    release = ROOT / 'data/eval/health_forecast_release_20260920'
    release_manifest = release / 'manifest.json'
    manifest = json.loads(release_manifest.read_text())
    manifest['model_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in manifest['models']}
    manifest['selected_policies'] = pd.read_csv(release / 'selected_test_metrics.csv').to_dict(orient='records')
    release_manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for folder in ('health_forecast', 'health_forecast_long_horizon/current'):
        path = ROOT / 'data/eval' / folder / 'health_forecast_model_manifest.json'
        if path.exists():
            backup = release / 'previous' / path.relative_to(ROOT)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copy2(path, backup)
            manifest = json.loads(path.read_text())
            manifest['transmitting_origin_release_override'] = 'data/eval/health_forecast_release_20260920/manifest.json'
            manifest['historical_metadata_notice'] = 'Original experiment records below remain historical; use release override for deployed transmitting-origin model choices and metrics.'
            path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for path in (ROOT / 'data/processed').glob('station_operational_scorecard*'):
        backup = release / 'previous' / path.relative_to(ROOT)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and not backup.exists():
            shutil.copy2(path, backup)


def run():
    release = ROOT / 'data/eval/health_forecast_release_20260920'
    release.mkdir(exist_ok=False)
    def backup(path):
        if path.exists():
            target = release / 'previous' / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    selections = pd.read_csv(ROOT / 'data/eval/forecast_baseline_corrections_complete_20260919/validation_selected.csv')
    metadata = pd.read_csv(ROOT / 'data/merged/station_hourly_merged.csv', usecols=['station_id', 'elevation'])
    historical = hf.build_health_forecast_features(pd.read_parquet(ROOT / 'data/processed/station_health_scores.parquet'), station_metadata=metadata)
    staged, rows = [], []
    for row in selections.itertuples():
        h = int(row.horizon)
        source = next(ROOT / 'data/eval' / s for s in SOURCES if (ROOT / 'data/eval' / s / f'{row.baseline}_{h}h.joblib').exists())
        folder = 'health_forecast' if h <= 24 else 'health_forecast_long_horizon/current'
        destination = ROOT / 'data/model' / folder / f'health_forecast_forecast_transmitting_origin_{h}h.joblib'
        old = joblib.load(destination)
        experiment = joblib.load(source / f'{row.baseline}_{h}h.joblib')
        model = experiment['model']
        for name in ('horizon_h', 'regime', 'feature_set', 'recency_half_life_days', 'iterations'):
            setattr(model, name, getattr(old, name))
        model.alpha = float(experiment['alpha'])
        model.final_policy = 'learned_residual'
        model.residual_baseline = {'roll_forward': 'no_new_incident_roll_forward', 'persistence': 'persistence', 'trend': 'recent_trend_24h'}[row.baseline]
        saved = release / destination.name
        joblib.dump(model, saved)
        model = joblib.load(saved)
        p = pd.read_parquet(source / 'predictions.parquet')
        p = p[(p.horizon == h) & (p.baseline == row.baseline)]
        frame = hf.health_forecast_inference_frame(historical, h).set_index(['station_id', 'hour_utc'])
        keys = pd.MultiIndex.from_arrays([p.station_id, pd.to_datetime(p.hour, utc=True)])
        test = frame.loc[keys].reset_index()
        np.testing.assert_allclose(model.predict_health(test), p.predicted, atol=1e-7, rtol=1e-7)
        metrics = regression_metrics(p.actual.to_numpy(), p.predicted.to_numpy())
        rows.append(dict(horizon_h=h, baseline=row.baseline, model=model.family, alpha=model.alpha,
            validation_mae=row.validation_mae, **metrics,
            band_accuracy_pct=float(np.mean(bands(p.actual) == bands(p.predicted))*100),
            normalized_regression_score_pct=100-metrics['mae'], rows=len(p)))
        staged.append((saved, destination))
        print(f'{h}h: serialized model reproduces experiment predictions', flush=True)

    july_dir = ROOT / 'data/eval/july_2026_health_forecast'
    for path in july_dir.iterdir():
        if path.is_file():
            backup(path)
    july_path = july_dir / 'july_health_forecast_predictions.parquet'
    july = pd.read_parquet(july_path)
    july_features = hf.build_health_forecast_features(pd.read_parquet(ROOT / 'data/eval/july_2026_health/station_health_scores_through_july.parquet'), station_metadata=metadata)
    for saved, destination in staged:
        model = joblib.load(saved)
        h = model.horizon_h
        if h > 24:
            continue
        mask = july.horizon_h.eq(h)
        part = july.loc[mask]
        frame = hf.health_forecast_inference_frame(july_features, h).set_index(['station_id', 'hour_utc'])
        keys = pd.MultiIndex.from_arrays([part.station_id, pd.to_datetime(part.hour_utc, utc=True)])
        prediction = model.predict_health(frame.loc[keys].reset_index())
        assert np.isfinite(prediction).all()
        july.loc[mask, 'predicted_frozen_selected_policy'] = prediction
    # Only promote once all prediction checks have succeeded.
    for saved, destination in staged:
        backup(destination)
        shutil.copy2(saved, destination)
    july.to_parquet(july_path, index=False)
    metrics_path = july_dir / 'july_health_forecast_metrics.csv'
    metrics = pd.read_csv(metrics_path)
    for i, row in metrics.iterrows():
        if row['method'] != 'frozen_selected_policy':
            continue
        p = july[july.horizon_h.eq(row.horizon_h)]
        actual, predicted = p.target_health_total.to_numpy(), p.predicted_frozen_selected_policy.to_numpy()
        values = regression_metrics(actual, predicted)
        error = np.abs(actual-predicted)
        values.update({f'within_{n}_point_accuracy': float(np.mean(error <= n)) for n in (1, 5, 10)})
        values['health_band_accuracy'] = float(np.mean(bands(actual) == bands(predicted)))
        for label, column in [('persistence', 'predicted_persistence'), ('roll_forward', 'predicted_no_new_incident_roll_forward')]:
            baseline = regression_metrics(actual, p[column].to_numpy())
            for metric in ('mae', 'rmse'):
                values[f'{metric}_improvement_vs_{label}_pct'] = 100*(1-values[metric]/baseline[metric]) if baseline[metric] else np.nan
        for key, value in values.items():
            metrics.loc[i, key] = value
    metrics.to_csv(metrics_path, index=False)
    manifest_path = july_dir / 'july_health_forecast_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['contract'] = 'Validation-MAE-selected baseline-plus-regression policies; no July training or tuning. July is a previously inspected extension, not a new independent holdout.'
    manifest['release'] = str(release.relative_to(ROOT))
    manifest['models'] = {str(dest.relative_to(ROOT)): hashlib.sha256(dest.read_bytes()).hexdigest() for _, dest in staged if '/health_forecast/' in dest.as_posix()}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (july_dir / 'july_health_forecast_report.txt').write_text(manifest['contract']+'\n\n'+metrics.to_string(index=False)+'\n', encoding='utf-8')
    result = pd.DataFrame(rows)
    result.to_csv(release / 'selected_test_metrics.csv', index=False)
    (release / 'manifest.json').write_text(json.dumps({'selection': 'minimum validation MAE', 'test_population': 'transmitting-origin; original horizon-specific purged test partitions', 'regression_percentage': '100*(1-MAE/100); normalized score, not correctness accuracy', 'backup_directory': 'previous', 'models': [str(d.relative_to(ROOT)) for _, d in staged]}, indent=2), encoding='utf-8')
    print(result.to_string(index=False))
    finalize_metadata()


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
