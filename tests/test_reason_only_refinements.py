import numpy as np

from scripts.experiment_reason_only_refinements import fallback, weather_codes, counts


def test_selective_fallback_margin_gate_and_missing():
    scores = np.array([[.6, .2], [.6, .55], [.95, .96], [.6, .1], [np.nan, np.nan]])
    gate = np.array([True, True, True, False, True])
    result = fallback(scores, np.array([.9, .9]), gate, floor=.5, margin=.2)
    np.testing.assert_array_equal(result, [[1, 0], [0, 0], [1, 1], [0, 0], [0, 0]])


def test_fallback_disabled_and_minimum_one():
    scores = np.array([[.6, .2], [.6, .55], [.95, .96]])
    thresholds = np.array([.9, .9])
    gate = np.ones(3, bool)
    assert not fallback(scores, thresholds, gate, enabled=False)[:2].any()
    np.testing.assert_array_equal(fallback(scores, thresholds, gate), [[1, 0], [1, 0], [1, 1]])


def test_weather_changes_only_target_code_and_never_gate_or_inputs():
    scores = np.array([[.95, .4, .8], [.95, .4, .8]])
    thresholds = np.array([.9, .7, .7])
    gate = np.array([True, False])
    original = scores.copy()
    pred, updated, limits = weather_codes(scores, thresholds, gate, 1, np.array([.9, .9]), 1., .5)
    np.testing.assert_array_equal(pred, [[1, 1, 1], [0, 0, 0]])
    np.testing.assert_array_equal(scores, original)
    np.testing.assert_array_equal(updated[:, [0, 2]], scores[:, [0, 2]])
    np.testing.assert_array_equal(thresholds, [.9, .7, .7])
    pred, _, _ = weather_codes(scores, thresholds, gate, 1, np.full(2, np.nan), 0., .01)
    np.testing.assert_array_equal(pred, fallback(scores, thresholds, gate, enabled=False))


def test_counts_penalize_false_reasons_and_misses():
    result = counts([[1, 0], [0, 1], [0, 0]], [[1, 0], [0, 0], [1, 0]])
    assert result['tp'] == result['fp'] == result['fn'] == 1
    assert result['f1'] == .5
