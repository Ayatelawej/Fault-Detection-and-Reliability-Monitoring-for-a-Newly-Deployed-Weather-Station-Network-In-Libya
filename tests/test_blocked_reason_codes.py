import numpy as np
from scripts.experiment_blocked_reason_codes import training_rows
from src.model.final_reason_codes import apply_output_policy


def test_fit_population_excludes_unknown_and_heldout_faults():
    train = np.array([0, 1, 2, 3])
    fault = np.array([0, 1, 1, 0, 1])
    resolved = np.array([False, True, False, False, True])
    assert training_rows(train, fault, resolved).tolist() == [0, 1, 3]


def test_negative_cap_is_deterministic():
    train = np.arange(20)
    fault = np.zeros(20, int); fault[0] = 1
    resolved = fault.astype(bool)
    first = training_rows(train, fault, resolved, limit=4)
    np.testing.assert_array_equal(first, training_rows(train, fault, resolved, limit=4))
    assert len(first) == 5 and 0 in first


def test_unsupported_heads_not_forced_and_gate_is_preserved():
    scores = np.array([[np.nan, .1], [np.nan, .9], [np.nan, np.nan]])
    result = apply_output_policy(scores, np.array([np.inf, .7]), np.array([True, False, True]), 'minimum_one')
    np.testing.assert_array_equal(result, [[False, True], [False, False], [False, False]])
