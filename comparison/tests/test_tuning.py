from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data_loader import load
from src.tuning import tune_classicals_for_fold

REPO_ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.slow
def test_returns_three_classical_params():
    # Contract (spec §4): tune_classicals_for_fold receives the UNCAPPED
    # outer-training fold. Caps are applied inside each inner CV split.
    X, y, meta = load(repo_root=REPO_ROOT)
    X_tr_outer_uncapped = X.iloc[:800].reset_index(drop=True)
    y_tr_outer          = y.iloc[:800].reset_index(drop=True)
    params = tune_classicals_for_fold(X_tr_outer_uncapped, y_tr_outer, meta,
                                      seed=42, n_trials=3)
    assert set(params.keys()) == {"xgb", "lgbm", "logit"}
    for v in params.values():
        assert isinstance(v, dict) and len(v) > 0

def test_signature_only_consumes_train_rows():
    """First-line no-leakage guard. NOTE: this only checks that the function
    SIGNATURE does not accept full-dataset arguments — it does not prove the
    body does not somehow re-import and access the full data. That stronger
    behavioural guarantee is enforced by:
      1. Code review of tuning.py imports (no `from src.data_loader import load`)
      2. The runner's call site passes only the outer-training slice
      3. test_inner_cap_uses_inner_train_only (below) verifies the strict
         nested-cap behaviour end-to-end
    """
    import inspect
    sig = inspect.signature(tune_classicals_for_fold)
    params = list(sig.parameters.keys())
    allowed = {"X_train_outer", "y_train_outer", "types_meta", "seed", "n_trials"}
    assert set(params).issubset(allowed), \
        f"tune_classicals_for_fold must not accept full-dataset args; got {params}"

def test_inner_cap_uses_inner_train_only(monkeypatch):
    """Strict nested-preprocessing guard, monkeypatch edition.

    The previous version (`assert observed_max < 100`) was weak — even the
    LEAKY mode (caps fit on full outer-training fold including the inner-val
    rows) would still squash the 999 outlier below 100, so that assertion
    couldn't distinguish strict from leaky behaviour.

    This version monkeypatches the `fold_safe_iqr_cap` symbol used inside
    `src.tuning` and records the exact `X_train` argument of every call.
    For each inner-CV split, we assert the captured DataFrame's row count
    equals the inner-train size (4/5 of outer-train), NOT the full outer-
    train size. If anyone reintroduces full-outer-fold capping, the captured
    row count would be the full outer size and this test would fail loudly.
    """
    import src.tuning as tuning_mod
    real_cap = tuning_mod.fold_safe_iqr_cap
    seen_train_lens: list[int] = []

    def spy(X_train, X_test, types_meta):
        seen_train_lens.append(len(X_train))
        return real_cap(X_train, X_test, types_meta)

    monkeypatch.setattr(tuning_mod, "fold_safe_iqr_cap", spy)

    from src.tuning import _inner_cv_score
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "Duration of Credit (month)": rng.uniform(4, 60, 100),
        "Credit Amount": rng.uniform(100, 5000, 100),
        "Age (years)": rng.uniform(19, 70, 100),
        "other": rng.integers(0, 3, 100),
    })
    meta = {
        "Duration of Credit (month)": {"type": "continuous"},
        "Credit Amount": {"type": "continuous"},
        "Age (years)": {"type": "continuous"},
        "other": {"type": "nominal"},
    }
    y = pd.Series([0, 1] * 50)

    class _Stub:
        def fit(self, Xtr, ytr): return self
        def predict_proba(self, Xte):
            return np.column_stack([np.full(len(Xte), 0.5), np.full(len(Xte), 0.5)])

    def make(Xtr_capped, ytr): return _Stub().fit(Xtr_capped, ytr)

    _inner_cv_score(make, X, y, meta, seed=42)

    # INNER_FOLDS = 5 → 5 calls to fold_safe_iqr_cap, each with
    # an inner-train of 80 rows (4/5 of 100). NOT 100.
    assert len(seen_train_lens) == 5, \
        f"expected 5 inner-fold cap calls, got {len(seen_train_lens)}"
    assert all(n == 80 for n in seen_train_lens), \
        f"inner-cap leakage: cap was fit on {seen_train_lens} rows (expected all=80)"
