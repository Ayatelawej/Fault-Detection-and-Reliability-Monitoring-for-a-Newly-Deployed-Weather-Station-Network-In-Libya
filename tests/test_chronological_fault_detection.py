import numpy as np
import pandas as pd
import pytest
from scripts.experiment_chronological_fault_detection import blocked_split, select_threshold, validate_split
from src.model.reason_code_rebuild import chronological_split


def test_chronology_embargo_and_crossing_events():
    hours = pd.to_datetime(['2026-03-20', '2026-03-31', '2026-04-01',
        '2026-04-08', '2026-04-30', '2026-05-01', '2026-05-08'], utc=True)
    groups = np.array(['a', 'cross', 'cross', 'b', 'cross2', 'cross2', 'c'])
    split = chronological_split(hours, groups)
    assert split['train'].tolist() == [0]
    assert split['validation'].tolist() == [3]
    assert split['test'].tolist() == [6]
    validate_split(hours, groups, split)


def test_reject_group_leakage():
    hours = pd.to_datetime(['2026-03-20', '2026-04-10', '2026-05-10'], utc=True)
    with pytest.raises(ValueError, match='groups'):
        validate_split(hours, np.array(['same', 'b', 'same']),
                       {k: np.array([i]) for i, k in enumerate(('train', 'validation', 'test'))})


def test_threshold_selected_from_validation_scores():
    selected, grid = select_threshold(np.array([0, 0, 1, 1]), np.array([.1, .4, .6, .9]))
    assert selected['f1'] == 1
    assert .4 < selected['threshold'] <= .6
    assert len(grid) == 11


def test_blocked_periods_include_later_training_and_gaps():
    hours = pd.to_datetime(['2026-01-20', '2026-02-01', '2026-02-08',
        '2026-03-01', '2026-03-08', '2026-04-20', '2026-05-01',
        '2026-05-08', '2026-06-20'], utc=True)
    groups = np.array([str(i) for i in range(len(hours))])
    splits = blocked_split(hours, groups)
    assert splits['train'].tolist() == [0, 7, 8]
    assert splits['validation'].tolist() == [2]
    assert splits['test'].tolist() == [4, 5]
    validate_split(hours, groups, splits, chronological=False)
    with pytest.raises(ValueError, match='Non-chronological'):
        validate_split(hours, groups, splits)


def test_blocked_crossing_events_removed_on_both_sides():
    hours = pd.to_datetime(['2026-01-20', '2026-02-10', '2026-02-20',
        '2026-03-10', '2026-04-20', '2026-05-10', '2026-06-10'], utc=True)
    groups = np.array(['train', 'val', 'cross_val', 'cross_val', 'cross_test', 'cross_test', 'train2'])
    splits = blocked_split(hours, groups)
    assert splits['train'].tolist() == [0, 6]
    assert splits['validation'].tolist() == [1]
    assert splits['test'].tolist() == []
