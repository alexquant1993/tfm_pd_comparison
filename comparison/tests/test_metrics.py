import numpy as np
from sklearn.metrics import roc_auc_score
from src.metrics import compute_all, ks_stat

def test_compute_all_matches_sklearn_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    m = compute_all(y, p)
    assert abs(m["auc"] - roc_auc_score(y, p)) < 1e-12
    assert abs(m["gini"] - (2 * m["auc"] - 1)) < 1e-12
    assert 0.0 <= m["ks"] <= 1.0
    assert 0.0 <= m["brier"] <= 1.0
    assert m["logloss"] > 0

def test_ks_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert abs(ks_stat(y, p) - 1.0) < 1e-12

def test_logloss_clipping():
    y = np.array([0, 1])
    p = np.array([1.0, 0.0])
    m = compute_all(y, p)
    assert np.isfinite(m["logloss"])
