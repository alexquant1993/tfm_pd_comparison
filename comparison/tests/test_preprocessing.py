from pathlib import Path
import numpy as np
import pandas as pd
from src.data_loader import load
from src.preprocessing import fold_safe_iqr_cap, CONTINUOUS_COLS_TO_CAP

REPO_ROOT = Path(__file__).resolve().parents[2]

def test_caps_derived_from_train_only():
    """If the test set has a huge outlier, it MUST still be capped at the train
    distribution's Q3 + 1.5*IQR — meaning the cap comes from train, not test."""
    X = pd.DataFrame({
        "Duration of Credit (month)": np.r_[np.arange(1, 41), [999.0]],   # train normal, test outlier
        "Credit Amount": np.arange(100, 140, dtype=float).tolist() + [1.0],
        "Age (years)": np.arange(20, 60, dtype=float).tolist() + [30.0],
        "other": np.zeros(41),
    })
    meta = {
        "Duration of Credit (month)": {"type": "continuous"},
        "Credit Amount": {"type": "continuous"},
        "Age (years)": {"type": "continuous"},
        "other": {"type": "nominal"},
    }
    train_idx = np.arange(40)
    test_idx = np.array([40])
    X_tr, X_te, caps = fold_safe_iqr_cap(X.iloc[train_idx], X.iloc[test_idx], meta)
    # test-row 999 must have been capped at the TRAIN distribution's upper cap
    expected_cap = np.quantile(np.arange(1, 41), 0.75) + 1.5 * (
        np.quantile(np.arange(1, 41), 0.75) - np.quantile(np.arange(1, 41), 0.25)
    )
    assert X_te["Duration of Credit (month)"].iloc[0] == expected_cap
    assert caps["Duration of Credit (month)"] == expected_cap

def test_only_upper_tail_capped():
    """Stage 02 uses upper-tail capping only — small values left alone."""
    X = pd.DataFrame({
        "Duration of Credit (month)": [1.0, 2.0, 3.0, 4.0, 100.0],
        "Credit Amount": [1.0] * 5,
        "Age (years)": [1.0] * 5,
        "other": [0] * 5,
    })
    meta = {
        "Duration of Credit (month)": {"type": "continuous"},
        "Credit Amount": {"type": "continuous"},
        "Age (years)": {"type": "continuous"},
        "other": {"type": "nominal"},
    }
    X_tr, X_te, caps = fold_safe_iqr_cap(X, X.iloc[[0]], meta)
    # smallest value unchanged
    assert X_tr["Duration of Credit (month)"].iloc[0] == 1.0
    # largest capped
    assert X_tr["Duration of Credit (month)"].iloc[-1] < 100.0

def test_only_three_continuous_columns_capped():
    """Stage 02 capped exactly Duration / Credit Amount / Age. Other cols untouched."""
    X, y, meta = load(repo_root=REPO_ROOT)
    X_tr, X_te, caps = fold_safe_iqr_cap(X.iloc[:800], X.iloc[800:], meta)
    assert set(caps.keys()) == set(CONTINUOUS_COLS_TO_CAP)
    # Untouched columns unchanged
    pd.testing.assert_series_equal(X_tr["Account Balance"], X.iloc[:800]["Account Balance"])
    pd.testing.assert_series_equal(X_te["Purpose"], X.iloc[800:]["Purpose"])

def test_idempotent_when_no_outliers_present():
    X = pd.DataFrame({
        "Duration of Credit (month)": [10.0] * 100,
        "Credit Amount": [500.0] * 100,
        "Age (years)": [35.0] * 100,
    })
    meta = {c: {"type": "continuous"} for c in X.columns}
    X_tr, X_te, caps = fold_safe_iqr_cap(X.iloc[:80], X.iloc[80:], meta)
    pd.testing.assert_frame_equal(X_tr, X.iloc[:80])
    pd.testing.assert_frame_equal(X_te, X.iloc[80:])
