from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.rules.config import ISOLATION_FOREST_CONTAMINATION


def score_isolation_forest(
    series: pd.Series,
    contamination: float = ISOLATION_FOREST_CONTAMINATION,
    random_state: int = 42,
    fitted_estimator: IsolationForest | None = None,
) -> pd.Series:
    values = series.astype(float)
    present_values = values.dropna()

    if present_values.empty:
        return pd.Series(0.0, index=series.index, name="score")

    training_values = present_values.to_numpy(dtype=float).reshape(-1, 1)
    if fitted_estimator is None:
        estimator = IsolationForest(
            contamination=contamination,
            random_state=random_state,
        )
        estimator.fit(training_values)
    else:
        estimator = fitted_estimator

    present_scores = -estimator.score_samples(training_values)
    scores = pd.Series(np.nan, index=series.index, name="score", dtype=float)
    scores.loc[present_values.index] = present_scores
    return scores.fillna(float(np.min(present_scores)))


def fit_isolation_forest(
    series: pd.Series,
    contamination: float = ISOLATION_FOREST_CONTAMINATION,
    random_state: int = 42,
) -> IsolationForest | None:
    present_values = series.astype(float).dropna()
    if present_values.empty:
        return None
    training_values = present_values.to_numpy(dtype=float).reshape(-1, 1)
    estimator = IsolationForest(contamination=contamination, random_state=random_state)
    estimator.fit(training_values)
    return estimator
