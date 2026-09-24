import numpy as np
import pandas as pd
import pytest
from src.availability.health_forecast import FittedHealthForecastModel
from src.availability.build_network_outage_windows import _window_outage_class, assign_outage_class


@pytest.mark.parametrize('baseline,expected',[
    ('persistence',[55.,100.]),
    ('recent_trend_24h',[65.,75.]),
    ('no_new_incident_roll_forward',[75.,85.]),
])
def test_correction_uses_selected_baseline_and_clips(monkeypatch,baseline,expected):
    model=FittedHealthForecastModel('ridge',(),None,alpha=.5,residual_baseline=baseline)
    monkeypatch.setattr(model,'predict',lambda frame:np.array([10.,20.]))
    frame=pd.DataFrame({'baseline_persistence_level':[50.,95.],
        'baseline_trend_level':[60.,65.],'baseline_no_new_incident_level':[70.,75.]})
    np.testing.assert_allclose(model.predict_health(frame),expected)


def test_coordinated_labels_are_hour_independent_and_legacy_compatible():
    for hour in (0,12,22,23):
        assert _window_outage_class(pd.Timestamp(f'2026-01-01 {hour}:00',tz='UTC'))=='coordinated'
    events=pd.DataFrame({'start_utc':pd.to_datetime(['2026-01-01 22:00'],utc=True)})
    for old in ('network_midnight','network_other'):
        windows=pd.DataFrame({'outage_class':[old],
            'backfill_start_utc':events.start_utc,'backfill_end_utc':events.start_utc})
        assert assign_outage_class(events,windows).outage_class.tolist()==['coordinated']


def test_old_model_without_baseline_metadata_keeps_roll_forward(monkeypatch):
    model=FittedHealthForecastModel('ridge',(),None,alpha=.5)
    del model.residual_baseline
    monkeypatch.setattr(model,'predict',lambda frame:np.array([10.]))
    frame=pd.DataFrame({'baseline_persistence_level':[50.],
        'baseline_trend_level':[60.],'baseline_no_new_incident_level':[70.]})
    np.testing.assert_allclose(model.predict_health(frame),[75.])


def test_band_metric_boundaries():
    from scripts.deploy_selected_health_forecasts import bands
    np.testing.assert_array_equal(bands([0,39.99,40,59.99,60,79.99,80,100]),[0,0,1,1,2,2,3,3])


def test_direct_forecast_does_not_add_any_baseline(monkeypatch):
    model=FittedHealthForecastModel('ridge',(),None,alpha=.1,final_policy='direct_regression')
    monkeypatch.setattr(model,'predict',lambda frame:np.array([-4.,52.,104.]))
    np.testing.assert_array_equal(model.predict_health(pd.DataFrame(index=range(3))),[0.,52.,100.])
