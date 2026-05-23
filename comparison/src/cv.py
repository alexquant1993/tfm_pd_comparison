from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def make_folds(y: pd.Series, n_splits: int = 10, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return frozen list of (train_idx, test_idx) for stratified K-fold CV."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_dummy = np.zeros((len(y), 1))
    return [(tr, te) for tr, te in skf.split(X_dummy, y)]
