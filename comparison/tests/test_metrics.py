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

from src.metrics import delong_test

def test_delong_identical_returns_high_p():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    z, pval = delong_test(y, p, p)
    assert abs(z) < 1e-6
    assert pval > 0.99

def test_delong_better_model_significant():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 500)
    p_bad = rng.random(500)
    p_good = y + rng.normal(0, 0.3, 500)
    z, pval = delong_test(y, p_bad, p_good)
    assert pval < 0.01

from src.metrics import wilcoxon_paired, bonferroni

def test_wilcoxon_paired_returns_pval():
    np.random.seed(0)
    a = np.random.random(10)
    b = a + 0.05
    stat, p = wilcoxon_paired(a, b)
    assert p < 0.05

def test_bonferroni_clips_to_one():
    assert bonferroni(0.5, 5) == 1.0
    assert bonferroni(0.01, 7) == 0.07
