"""WoE + unregularised Logit champion — faithful to pd-autopilot stage 04a (MIV).

Spec §8 requirements honoured here:
- penalty=None (matches report's statsmodels-style logit)
- Explicit monotonic trends per variable (no "auto")
- CP solver with try/except fallback to MIP
- 8 champion variables fixed at the class level
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from optbinning import OptimalBinning
from sklearn.linear_model import LogisticRegression

CHAMPION_VARS = [
    "Account Balance",
    "Payment Status of Previous Credit",
    "Duration of Credit (month)",
    "Value Savings/Stocks",
    "Purpose",
    "Credit Amount",
    "Most valuable available asset",
    "Age (years)",
]

# Hard-coded from report stage 03. None = no monotonicity (nominal).
MONOTONIC_TRENDS: dict[str, str | None] = {
    "Account Balance":                   "descending",
    "Payment Status of Previous Credit": None,
    "Duration of Credit (month)":        "ascending",
    "Value Savings/Stocks":              "descending",
    "Purpose":                           None,
    "Credit Amount":                     "ascending",
    "Most valuable available asset":     "ascending",
    "Age (years)":                       "descending",
}


def _fit_one_binning(col: str, X_tr: pd.DataFrame, y_tr: pd.Series, meta: dict) -> OptimalBinning:
    info = meta[col]
    common = dict(
        name=col,
        dtype=info["dtype"],
        monotonic_trend=MONOTONIC_TRENDS[col],
        # If special_codes is an empty list, normalise to None for OptimalBinning.
        special_codes=info["special_codes"] or None,
    )
    try:
        ob = OptimalBinning(solver="cp", **common)
        ob.fit(X_tr[col].values, y_tr.values)
    except (ValueError, RuntimeError) as e:
        # Documented Stage-03 behaviour: CP solver occasionally fails on
        # certain ordinal-ascending cases; MIP is a stable fallback.
        print(f"[woe_logit] CP solver failed for {col!r} ({type(e).__name__}: {e}); "
              f"falling back to MIP solver", file=sys.stderr)
        ob = OptimalBinning(solver="mip", **common)
        ob.fit(X_tr[col].values, y_tr.values)
    return ob


def _fit_binning(X_tr: pd.DataFrame, y_tr: pd.Series, meta: dict) -> dict[str, OptimalBinning]:
    return {col: _fit_one_binning(col, X_tr, y_tr, meta) for col in CHAMPION_VARS}


def _transform(binners: dict[str, OptimalBinning], X: pd.DataFrame) -> np.ndarray:
    # metric_special="empirical" so special codes get their own bin's WoE (not 0).
    # Without this, optbinning silently maps special values to WoE=0 — which
    # collapses the dedicated bin and corrupts every coefficient downstream.
    cols = [
        binners[c].transform(
            X[c].values,
            metric="woe",
            metric_special="empirical",
            metric_missing="empirical",
        )
        for c in CHAMPION_VARS
    ]
    return np.column_stack(cols)


class WoELogitChampion:
    name = "woe_logit"
    requires_string_labels = False

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        self._binners = _fit_binning(X_train, y_train, types_meta)
        Z_tr = _transform(self._binners, X_train)
        Z_te = _transform(self._binners, X_test)
        # C=1e10 is effectively unregularised (the L2 penalty coefficient 1/C
        # is 10⁻¹⁰, which on 1000 rows × 8 features cannot meaningfully shift
        # any coefficient — the gradient at the MLE is ~unit-scale). We use
        # this instead of:
        #   - penalty=None: deprecated in sklearn 1.8, removed in 1.10
        #   - C=np.inf: triggers sklearn's internal penalty=None translation
        #               and the corresponding warning
        # The gating test (test_full_sample_refit_matches_report_coefficients,
        # ±0.05 tol vs model_params.json) passes identically with C=1e10.
        self._lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000, random_state=seed)
        self._lr.fit(Z_tr, y_train.values)
        return self._lr.predict_proba(Z_te)[:, 1]
