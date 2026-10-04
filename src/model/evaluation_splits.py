"""Shared blocked partitions and validation-only threshold selection."""
import hashlib
import numpy as np
import pandas as pd
from src.model.hourly_baseline import binary_metrics
from src.model.hourly_calibration import CALIBRATION_THRESHOLDS

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def select_threshold(labels, probability, original=0.30):
    rows = [dict(threshold=t, **binary_metrics(labels, probability, t))
            for t in CALIBRATION_THRESHOLDS]

    best = max(rows, key=lambda r: (
        min(r['precision'], r['recall'], r['f1']), r['f1'],
        r['precision'], r['recall'], -abs(r['threshold'] - original)))
    return best, rows


def blocked_split(hours, groups):
    """February validation, March-April test, remaining months training."""
    h = pd.DatetimeIndex(hours)
    boundaries = pd.to_datetime(['2026-02-01', '2026-03-01', '2026-05-01'], utc=True)
    a, b, c = boundaries
    category = np.select([(h >= a) & (h < b), (h >= b) & (h < c)], [1, 2], default=0)
    counts = pd.DataFrame({'group': groups, 'partition': category}).groupby('group').partition.nunique()
    crosses = np.isin(groups, counts[counts > 1].index)
    gap = np.zeros(len(h), dtype=bool)
    for boundary in boundaries:
        gap |= (h >= boundary) & (h < boundary + pd.Timedelta(days=7))
    return {name: np.flatnonzero((category == k) & ~crosses & ~gap)
            for k, name in enumerate(('train', 'validation', 'test'))}


def validate_split(hours, groups, splits, chronological=False):
    h = pd.DatetimeIndex(hours)
    for name, indices in splits.items():
        if len(indices) == 0 or len(np.unique(indices)) != len(indices):
            raise ValueError(f'Empty or duplicated {name} membership')
    for left, right in (('train', 'validation'), ('validation', 'test'), ('train', 'test')):
        if np.intersect1d(splits[left], splits[right]).size:
            raise ValueError('Rows overlap across partitions')
        if chronological and h[splits[left]].max() >= h[splits[right]].min():
            raise ValueError('Non-chronological membership')
        if set(groups[splits[left]]) & set(groups[splits[right]]):
            raise ValueError('Connected groups cross partitions')
