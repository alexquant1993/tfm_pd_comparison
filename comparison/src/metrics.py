"""Discrimination + scoring-rule metrics + paired statistical tests."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, log_loss

_EPS = 1e-15

def ks_stat(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))

def compute_all(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), _EPS, 1 - _EPS)
    auc = float(roc_auc_score(y_true, y_prob))
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_stat(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }

from scipy import stats

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _compute_midrank(positive[r])
        ty[r] = _compute_midrank(negative[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if sx.ndim == 0:
        sx = sx.reshape(1, 1)
        sy = sy.reshape(1, 1)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def delong_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> tuple[float, float]:
    """Two-sided paired DeLong test. Returns (z_statistic, p_value).
    z > 0 means model A has higher AUC than B.
    """
    y_true = np.asarray(y_true).astype(int)
    order = (-y_true).argsort(kind="stable")
    y_sorted = y_true[order]
    label_1_count = int(y_sorted.sum())
    preds = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])
    aucs, cov = _fast_delong(preds, label_1_count)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(pval)
