import csv
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.runner import run_one_fold_one_model, append_row

class _DummyModel:
    name = "stub"
    requires_string_labels = False
    def fit_predict(self, X_tr, y_tr, X_te, meta, seed=42):
        return np.full(len(X_te), 0.5)

def test_append_row_creates_header(tmp_path):
    path = tmp_path / "out.csv"
    append_row(path, {"model": "x", "fold": 0, "auc": 0.7})
    append_row(path, {"model": "x", "fold": 1, "auc": 0.8})
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 2

def test_run_one_fold_one_model_returns_metrics_and_proba():
    X_tr = pd.DataFrame({"a": range(80)})
    y_tr = pd.Series([0, 1] * 40)
    X_te = pd.DataFrame({"a": range(80, 100)})
    y_te = pd.Series([0, 1] * 10)
    out, proba = run_one_fold_one_model(_DummyModel(), X_tr, y_tr, X_te, y_te, meta={})
    assert "auc" in out and "logloss" in out
    assert out["model"] == "stub"
    assert proba.shape == (20,)
    assert out["n_train"] == 80 and out["n_test"] == 20

def test_run_one_fold_passes_correct_labels():
    """Regression test for the v1 label-alignment bug: ensure the labels
    metrics are computed against actually match y_test, not the first N rows
    of some larger y."""
    class _ProbaIsYModel:
        name = "perfect"
        requires_string_labels = False
        def fit_predict(self, X_tr, y_tr, X_te, meta, seed=42):
            # Return X_te's "a" column as proba (cheat: assumes y == a is perfect)
            return X_te["a"].values.astype(float) / X_te["a"].max()
    # y_te is monotone with X_te["a"] so AUC must be 1.0
    X_te = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
    y_te = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    X_tr = pd.DataFrame({"a": [0, 5]})
    y_tr = pd.Series([0, 1])
    out, _ = run_one_fold_one_model(_ProbaIsYModel(), X_tr, y_tr, X_te, y_te, meta={})
    assert out["auc"] == 1.0, f"label misalignment detected: got AUC={out['auc']}"
