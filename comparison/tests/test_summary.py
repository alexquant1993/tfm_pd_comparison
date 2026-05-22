from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.summary import build_summary_md, DISCLAIMER

def _fake_inputs():
    """Build aligned inputs: SAME y per (fold_idx, test_idx) across models.

    The alignment is what the production parquet must guarantee, so the
    fixture must mirror it — otherwise the unit test would silently accept
    a broken alignment check.
    """
    df = pd.DataFrame({
        "model": ["woe_logit"] * 10 + ["xgb"] * 10,
        "fold_idx": list(range(10)) * 2,
        "auc": np.r_[np.linspace(0.78, 0.82, 10), np.linspace(0.79, 0.83, 10)],
        "gini": 0.6, "ks": 0.5, "brier": 0.17, "logloss": 0.5, "runtime_s": 1.0,
    })
    rng = np.random.default_rng(0)
    # Generate y ONCE per (fold, test_idx); reuse for every model.
    base = []
    for f in range(10):
        for i in range(100):
            base.append({"fold_idx": f, "test_idx": f * 100 + i,
                         "y": int(rng.integers(0, 2))})
    base_df = pd.DataFrame(base)
    rows = []
    for m in ["woe_logit", "xgb"]:
        for _, b in base_df.iterrows():
            rows.append({"model": m, "fold_idx": b["fold_idx"], "test_idx": b["test_idx"],
                         "y": b["y"], "proba": float(rng.random())})
    preds = pd.DataFrame(rows)
    return df, preds

def test_build_summary_includes_disclaimer():
    df, preds = _fake_inputs()
    md = build_summary_md(df, preds, champion="woe_logit")
    assert DISCLAIMER in md
    assert "woe_logit" in md and "xgb" in md
    assert "DeLong" in md and "Wilcoxon" in md

def test_alignment_check_rejects_misaligned_predictions():
    """If the predictions parquet is misaligned (challenger has different y
    for some (fold, test_idx) than the champion), summary must REFUSE to
    compute DeLong — silent miscalculation would be worse than no test."""
    df, preds = _fake_inputs()
    # Corrupt xgb's y for one row only
    mask = (preds["model"] == "xgb") & (preds["fold_idx"] == 0) & (preds["test_idx"] == 0)
    preds.loc[mask, "y"] = 1 - preds.loc[mask, "y"]
    with pytest.raises(AssertionError, match="misaligned"):
        build_summary_md(df, preds, champion="woe_logit")
