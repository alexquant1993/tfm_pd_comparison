"""Fold-safe outlier capping. Caps are derived from the outer-training fold
only, then applied to both the outer-training and outer-test folds.

This replaces the leaky approach of using the pre-computed loans_clean.csv
which had caps fit on all 1,000 rows in Stage 02.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Stage 02 capped exactly these three continuous columns. We mirror that
# choice for fold-safe comparability with the report's recipe.
CONTINUOUS_COLS_TO_CAP = [
    "Duration of Credit (month)",
    "Credit Amount",
    "Age (years)",
]

def _upper_cap(values: np.ndarray) -> float:
    q1 = np.quantile(values, 0.25)
    q3 = np.quantile(values, 0.75)
    return float(q3 + 1.5 * (q3 - q1))

def fold_safe_iqr_cap(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    types_meta: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Apply IQR upper-tail capping derived from X_train to both X_train and X_test.

    Returns: (X_train_capped, X_test_capped, caps_dict)
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()
    caps: dict[str, float] = {}
    for col in CONTINUOUS_COLS_TO_CAP:
        if col not in X_tr.columns:
            continue
        cap = _upper_cap(X_tr[col].values)
        caps[col] = cap
        X_tr[col] = X_tr[col].clip(upper=cap)
        X_te[col] = X_te[col].clip(upper=cap)
    return X_tr, X_te, caps
