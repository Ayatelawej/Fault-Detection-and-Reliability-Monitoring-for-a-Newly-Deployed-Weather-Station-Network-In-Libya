"""Select labelled training faults plus a reproducible sample of normal hours."""
import numpy as np
from src.model.reason_code_rebuild import SEED

def training_rows(train, fault, resolved, limit=16000):
    positive = train[(fault[train] == 1) & resolved[train]]
    negative = train[fault[train] == 0]
    if len(negative) > limit:
        negative = np.sort(np.random.default_rng(SEED).choice(negative, limit, replace=False))
    return np.sort(np.r_[positive, negative])
