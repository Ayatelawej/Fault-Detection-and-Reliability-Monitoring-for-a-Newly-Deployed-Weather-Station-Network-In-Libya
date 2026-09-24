"""Verify staged forecasts and dashboard joins before publishing a release."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.availability import health_forecast as hf
from src.dashboard.replay import load_replay_bundle, build_replay_snapshot

OUT = ROOT / 'data/eval/final_system_release_20260924'
MODELS = ROOT / 'data/model/final_system_20260924'


def run():
    comparisons = pd.read_csv(ROOT / 'data/eval/final_forecast_comparison_20260924/metrics.csv')
    assert len(comparisons) == 88
    assert np.isfinite(comparisons[['mae', 'rmse', 'r2', 'band_accuracy_pct']]).all().all()
    chosen = pd.read_csv(ROOT / 'data/eval/final_forecast_comparison_20260924/chosen.csv')
    for row in chosen.itertuples():
        candidates = comparisons[comparisons.partition.eq('validation') & comparisons.horizon_h.eq(row.horizon_h)]
        assert np.isclose(row.mae, candidates.mae.min())
    scores = pd.read_parquet(ROOT / 'data/eval/july_2026_health/station_health_scores_through_july.parquet')
    metadata = pd.read_csv(ROOT / 'data/merged/station_hourly_merged.csv', usecols=['station_id', 'elevation'])
    cutoff = pd.Timestamp('2026-07-15T12:00:00Z')
    full = hf.build_health_forecast_features(scores, station_metadata=metadata)
    prefix = hf.build_health_forecast_features(scores[scores.hour_utc.le(cutoff)], station_metadata=metadata)
    saved = pd.read_parquet(OUT / 'july_forecast_evaluation.parquet')
    audits = []
    for row in chosen.itertuples():
        h = int(row.horizon_h)
        model = joblib.load(MODELS / f'forecasts/health_forecast_forecast_transmitting_origin_{h}h.joblib')
        left = hf.health_forecast_inference_frame(full, h).set_index(['station_id', 'hour_utc'])
        right = hf.health_forecast_inference_frame(prefix, h).set_index(['station_id', 'hour_utc'])
        sample = saved[saved.horizon_h.eq(h) & saved.hour_utc.between(cutoff - pd.Timedelta(hours=24), cutoff)]
        assert len(sample) > 0
        keys = pd.MultiIndex.from_frame(sample[['station_id', 'hour_utc']])
        a = model.predict_health(left.loc[keys].reset_index())
        b = model.predict_health(right.loc[keys].reset_index())
        np.testing.assert_allclose(a, b, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(a, sample.prediction, atol=1e-8, rtol=1e-8)
        assert np.isfinite(a).all() and ((a >= 0) & (a <= 100)).all()
        audits.append(dict(horizon_h=h, rows=len(sample), prefix_invariant=True, serialized_parity=True))
        print(f'{h}h: future-truncation and serialized prediction checks passed', flush=True)
    bundle = load_replay_bundle(forecast_path=OUT/'dashboard_forecasts.parquet', detection_path=OUT/'binary_predictions.parquet',
        reason_path=OUT/'reasons/reason_code_predictions.parquet', weather_note_path=OUT/'weather_annotations.parquet')
    assert len(build_replay_snapshot(bundle, cutoff)) == 26
    report = dict(forecasts=audits, dashboard_gate_joins=True, validation_selection=True)
    (OUT/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
