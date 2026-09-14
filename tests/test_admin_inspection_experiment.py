import numpy as np
import pytest
from sklearn.metrics import precision_recall_fscore_support

from scripts.experiment_admin_inspection import route_to_inspection


def test_review_replaces_codes_and_respects_binary_gate():
    codes = np.array([[1, 1], [1, 0], [0, 1]], bool)
    original = codes.copy()
    result = route_to_inspection(codes, [1, 0, 1], [1, 1, 0])
    assert result.tolist() == [[False, False, True], [True, False, False], [False, False, False]]
    np.testing.assert_array_equal(codes, original)


def test_review_does_not_get_free_credit_for_false_alerts_or_missed_faults():
    # One known reason, two unresolved faults, and two reference-normal hours.
    truth = np.array([[1, 0], [0, 1], [0, 1], [0, 0], [0, 0]])
    prediction = route_to_inspection(np.ones((5, 1)), [0, 1, 1, 1, 1], [1, 1, 0, 1, 0])
    p, r, f, _ = precision_recall_fscore_support(truth, prediction, average='micro')
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(2 / 3)
    assert f == pytest.approx(2 / 3)


def test_no_review_matches_specific_code_baseline():
    codes = np.array([[1, 0], [1, 1]], bool)
    result = route_to_inspection(codes, [0, 0], [1, 1])
    np.testing.assert_array_equal(result[:, :-1], codes)
    assert not result[:, -1].any()
