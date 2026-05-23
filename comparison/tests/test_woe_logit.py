from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from src.data_loader import load
from src.preprocessing import fold_safe_iqr_cap
from src.models.woe_logit import WoELogitChampion, CHAMPION_VARS, MONOTONIC_TRENDS

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PARAMS = REPO_ROOT / "runs" / "2026-03-17_071354" / "pipeline" / "model_params.json"

def test_champion_vars_count():
    assert len(CHAMPION_VARS) == 8

def test_monotonic_trends_defined_for_all_vars():
    assert set(MONOTONIC_TRENDS.keys()) == set(CHAMPION_VARS)

def test_fit_predict_shape_and_range():
    X, y, meta = load(repo_root=REPO_ROOT)
    X_tr, X_te, _ = fold_safe_iqr_cap(X.iloc[:800], X.iloc[800:], meta)
    model = WoELogitChampion()
    proba = model.fit_predict(X_tr, y.iloc[:800], X_te, meta, seed=42)
    assert proba.shape == (200,)
    assert np.all((proba >= 0) & (proba <= 1))

def test_determinism():
    X, y, meta = load(repo_root=REPO_ROOT)
    X_tr, X_te, _ = fold_safe_iqr_cap(X.iloc[:800], X.iloc[800:], meta)
    m = WoELogitChampion()
    a = m.fit_predict(X_tr, y.iloc[:800], X_te, meta, seed=42)
    b = m.fit_predict(X_tr, y.iloc[:800], X_te, meta, seed=42)
    np.testing.assert_allclose(a, b)

def test_special_codes_isolated_and_routed_at_transform():
    """After fitting, each special-coded variable must (a) have a dedicated
    Special bin in the binner AND (b) actually route special values to that
    bin's empirical WoE at .transform() time — NOT silently map them to 0.

    Regression guard for the silent-WoE-zero bug: optbinning's default
    transform behaviour maps special codes to WoE=0 unless
    metric_special="empirical" is passed. Without that kwarg, special-coded
    rows are stripped of their bin information and every downstream
    coefficient is corrupted.
    """
    import numpy as np
    X, y, meta = load(repo_root=REPO_ROOT)
    X_tr, _, _ = fold_safe_iqr_cap(X, X.iloc[:1], meta)
    m = WoELogitChampion()
    m.fit_predict(X_tr, y, X_tr.iloc[:1], meta, seed=42)

    for col, special in [("Account Balance", 4),
                         ("Value Savings/Stocks", 5),
                         ("Most valuable available asset", 4)]:
        binner = m._binners[col]
        bin_table = binner.binning_table.build()

        # (a) Structural: the Special bin exists in the table
        special_rows = bin_table[bin_table["Bin"].astype(str).str.contains("Special")]
        assert len(special_rows) > 0, f"{col} missing Special bin in binning_table"

        # (b) Behavioural: transform(special_value) returns NON-ZERO WoE
        # (the WoE from the Special bin, not silently zeroed)
        special_arr = np.array([float(special)])
        woe_via_transform = binner.transform(
            special_arr, metric="woe", metric_special="empirical"
        )
        assert abs(woe_via_transform[0]) > 1e-6, (
            f"{col}: transform(special_code={special}) returned ~0, "
            f"indicating the special bin's WoE was silently dropped"
        )

        # (c) The transform-time WoE matches the binning_table's reported WoE
        # for the Special bin (within float precision)
        expected_special_woe = float(special_rows.iloc[0]["WoE"])
        assert abs(woe_via_transform[0] - expected_special_woe) < 1e-4, (
            f"{col}: transform returned {woe_via_transform[0]:.4f} but "
            f"binning_table Special row shows WoE={expected_special_woe:.4f}"
        )

def test_full_sample_refit_matches_report_coefficients():
    """GATING TEST: every coefficient (intercept + 8 slopes) must be within
    absolute tolerance 0.05 of the report's model_params.json."""
    if not MODEL_PARAMS.exists():
        pytest.skip(f"missing {MODEL_PARAMS}")
    report = json.loads(MODEL_PARAMS.read_text())
    expected_intercept = report.get("intercept")
    expected_coefs = report.get("coefficients") or report.get("coef")
    if expected_intercept is None or expected_coefs is None:
        pytest.skip("model_params.json does not contain intercept/coefficients in expected schema")

    X, y, meta = load(repo_root=REPO_ROOT)
    X_capped, _, _ = fold_safe_iqr_cap(X, X.iloc[:1], meta)
    m = WoELogitChampion()
    m.fit_predict(X_capped, y, X_capped.iloc[:1], meta, seed=42)

    actual_intercept = float(m._lr.intercept_[0])
    actual_coefs = dict(zip(CHAMPION_VARS, m._lr.coef_[0].tolist()))

    assert abs(actual_intercept - expected_intercept) <= 0.05, \
        f"intercept drift: actual={actual_intercept:.4f}, expected={expected_intercept:.4f}"
    for col in CHAMPION_VARS:
        diff = abs(actual_coefs[col] - expected_coefs[col])
        assert diff <= 0.05, \
            f"coef drift on {col}: actual={actual_coefs[col]:.4f}, expected={expected_coefs[col]:.4f}, diff={diff:.4f}"
