import joblib
import numpy as np
import pandas as pd
from src.model.reason_rgfn_adapter import ReasonRgfnEstimator
from src.model.final_reason_codes import predict,SELECTED_VERSION
from test_final_reason_codes import fixture_inputs


def test_selected_reason_version_minimum_one_and_gate():
    x,gate,bundle=fixture_inputs(.2);bundle['version']=SELECTED_VERSION
    result=predict(x,gate,bundle)
    assert result.mechanism_status.tolist()==['not_applicable','likely','likely']
    assert result.component_status.tolist()==['not_applicable','likely','likely']
    assert result.reason_model_version.eq(SELECTED_VERSION).all()


def test_rgfn_adapter_serialization_and_no_future_dependence(tmp_path):
    rng=np.random.default_rng(2);x=rng.normal(size=(24,4)).astype('float32');x[0,0]=np.nan
    y=(np.nan_to_num(x[:,0])>0).astype(int)
    model=ReasonRgfnEstimator([0,1],[2,3]).fit(x,y,np.ones(24),epochs=2)
    expected=model.predict_proba(x)
    np.testing.assert_allclose(model.predict_proba(x[:8]),expected[:8],atol=1e-7)
    path=tmp_path/'model.joblib';joblib.dump(model,path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(x),expected,atol=0)
    assert np.isfinite(expected).all()
