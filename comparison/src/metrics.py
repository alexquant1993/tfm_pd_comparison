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
