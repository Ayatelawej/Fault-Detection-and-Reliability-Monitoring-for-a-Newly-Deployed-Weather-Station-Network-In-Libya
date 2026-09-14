from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

import src.model.final_reason_codes as final_reason_codes
from src.model.final_reason_codes import VERSION, apply_output_policy, predict
from src.dashboard.replay import event_reason_history, segment_predicted_fault_events, build_replay_snapshot
from test_availability import _synthetic_replay_bundle


def fixture_inputs(score=.8):
    hours = pd.date_range("2026-07-01", periods=3, freq="h", tz="UTC")
    x = pd.DataFrame({"barometer::pressure::hard": [0., 1., 1.]},
        index=pd.MultiIndex.from_product([["A"], hours], names=["station_id", "hour"]))
    gate = pd.DataFrame(dict(station_id="A", hour_utc=hours, random_probability=[.1, .8, .9],
                             random_prediction=[0, 1, 1]))
    head = dict(models=[score] * 3, views=[np.array([0])] * 3,
                weights=np.array([1., 0., 0.]), threshold=.6)
    bundle = dict(version=VERSION, feature_names=list(x), heads={
        "mechanism:spike_impossible": head, "component:barometer": head})
    return x, gate, bundle


def test_final_reasons_gate_abstain_and_ignore_truth():
    x, gate, bundle = fixture_inputs()
    output = predict(x, gate.assign(y_binary=1, source_episode_ids="future-label"), bundle)
    assert output.likely_mechanisms.tolist() == ["", "spike_impossible", "spike_impossible"]
    assert output.likely_components.tolist() == ["", "barometer", "barometer"]
    assert output.mechanism_status.iloc[0] == "not_applicable"
    assert "y_binary" not in output and "source_episode_ids" not in output
    x, gate, bundle = fixture_inputs(.2)
    output = predict(x, gate, bundle)
    assert output.likely_mechanisms.eq("").all()
    assert output.mechanism_status.tolist() == ["not_applicable", "insufficient_evidence", "insufficient_evidence"]


def test_final_reasons_preserve_multilabel_and_prefix():
    x, gate, bundle = fixture_inputs()
    bundle["heads"]["mechanism:stuck_flatline"] = bundle["heads"]["mechanism:spike_impossible"]
    output = predict(x, gate, bundle)
    assert output.likely_mechanisms.iloc[1] == "spike_impossible | stuck_flatline"
    pd.testing.assert_frame_equal(output.iloc[:2], predict(x.iloc[:2], gate.iloc[:2], bundle))


def test_mixed_policy_forces_only_mechanism_minimum_one():
    x, gate, bundle = fixture_inputs(.2)
    output = predict(x, gate, bundle,
        output_policy={"mechanism": "minimum_one", "component": "threshold"})
    assert output.likely_mechanisms.tolist() == ["", "spike_impossible", "spike_impossible"]
    assert output.mechanism_status.tolist() == ["not_applicable", "likely", "likely"]
    assert output.likely_components.eq("").all()
    assert output.component_status.tolist() == ["not_applicable", "insufficient_evidence", "insufficient_evidence"]


def test_output_policy_keeps_threshold_multilabel_and_rejects_bad_inputs():
    scores = np.array([[.8, .7], [.2, .3], [np.nan, np.nan]])
    gate = np.array([True, True, True])
    np.testing.assert_array_equal(apply_output_policy(scores, [.6, .6], gate, "minimum_one"),
                                  [[1, 1], [0, 1], [0, 0]])
    with pytest.raises(ValueError, match="Unsupported"):
        apply_output_policy(scores, [.6, .6], gate, "forced")


def test_cli_score_defaults_mixed_while_direct_predict_defaults_threshold(monkeypatch, tmp_path):
    x, gate, bundle = fixture_inputs(.2)
    assert predict(x, gate, bundle).mechanism_status.iloc[1] == "insufficient_evidence"
    called = {}
    monkeypatch.setattr(final_reason_codes, "score", lambda *args: called.update(args=args))
    final_reason_codes.main(["score", "--raw", str(tmp_path / "raw"), "--references", str(tmp_path / "refs"),
        "--detections", str(tmp_path / "gate"), "--output", str(tmp_path / "out")])
    assert called["args"][-1] == final_reason_codes.MIXED_OUTPUT_POLICY


def test_july_cli_policy_selects_versioned_defaults(monkeypatch):
    calls = []
    monkeypatch.setattr(final_reason_codes, "score", lambda *args: calls.append(args))
    final_reason_codes.main(["july"])
    final_reason_codes.main(["july", "--output-policy", "threshold"])
    assert calls[0][3] == final_reason_codes.JULY_MIXED_OUTPUT
    assert calls[0][-1] == final_reason_codes.MIXED_OUTPUT_POLICY
    assert calls[1][3] == final_reason_codes.JULY_OUTPUT
    assert calls[1][-1] == final_reason_codes.LEGACY_THRESHOLD_POLICY


def test_saved_mixed_rescore_rejects_unvalidated_source_manifest(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    source = tmp_path / "source"
    model_dir.mkdir(); source.mkdir()
    model = model_dir / "reason_heads.joblib"
    gate = tmp_path / "gate.parquet"
    model.write_bytes(b"frozen model"); gate.write_bytes(b"frozen gate")
    (model_dir / "manifest.json").write_text(json.dumps({"model_sha256": final_reason_codes.sha(model)}))
    (source / "scoring_manifest.json").write_text(json.dumps({
        "model_sha256": "altered", "input_hashes": {"detections": final_reason_codes.sha(gate)}}))
    monkeypatch.setattr(final_reason_codes, "MODEL_DIR", model_dir)
    monkeypatch.setattr(final_reason_codes, "JULY_GATE", gate)
    with pytest.raises(ValueError, match="different reason model"):
        final_reason_codes.rescore_saved_july_mixed(tmp_path / "new", source)


def test_final_reasons_reject_duplicate_or_missing_inputs():
    x, gate, bundle = fixture_inputs()
    with pytest.raises(ValueError, match="Duplicate"):
        predict(x, pd.concat([gate, gate]), bundle)
    with pytest.raises(ValueError, match="history"):
        predict(x.iloc[:1], gate, bundle)
    with pytest.raises(ValueError, match="schema"):
        predict(x.rename(columns={x.columns[0]: "wrong"}), gate, bundle)


def test_event_reasons_do_not_show_future_hours():
    x, gate, model = fixture_inputs()
    bundle = replace(_synthetic_replay_bundle(), reasons=predict(x, gate, model))
    event = segment_predicted_fault_events(gate, gate.hour_utc.iloc[1]).iloc[0]
    history = event_reason_history(bundle, event)
    assert history.hour_utc.tolist() == [gate.hour_utc.iloc[1]]


def test_snapshot_hides_reasons_for_full_outage():
    base = _synthetic_replay_bundle()
    reasons = base.detections.copy()
    reasons["likely_mechanisms"] = "stuck_flatline"
    reasons["likely_components"] = "anemometer"
    reasons["mechanism_status"] = reasons["component_status"] = "likely"
    snapshot = build_replay_snapshot(replace(base, reasons=reasons), "2026-07-01T01:00:00Z").set_index("station_id")
    assert snapshot.loc["A", "likely_mechanisms"] == "stuck flatline"
    assert snapshot.loc["B", "likely_components"] == ""
    assert snapshot.loc["B", "mechanism_status"] == "not_applicable"
