import numpy as np
import pandas as pd
from src.cv import make_folds

def test_make_folds_basic():
    y = pd.Series([0, 1] * 500)
    folds = make_folds(y, n_splits=10, seed=42)
    assert len(folds) == 10
    for tr, te in folds:
        assert len(tr) + len(te) == 1000
        assert set(tr).isdisjoint(set(te))

def test_make_folds_stratified():
    y = pd.Series([0] * 700 + [1] * 300)
    folds = make_folds(y, n_splits=10, seed=42)
    for tr, te in folds:
        rate = y.iloc[te].mean()
        assert 0.25 <= rate <= 0.35

def test_make_folds_deterministic():
    y = pd.Series([0, 1] * 500)
    folds_a = make_folds(y, n_splits=10, seed=42)
    folds_b = make_folds(y, n_splits=10, seed=42)
    for (tra, tea), (trb, teb) in zip(folds_a, folds_b):
        assert np.array_equal(tra, trb)
        assert np.array_equal(tea, teb)
