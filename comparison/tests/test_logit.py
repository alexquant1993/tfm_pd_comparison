from pathlib import Path
import numpy as np
from src.data_loader import load
from src.preprocessing import fold_safe_iqr_cap
from src.models.logit import PlainLogit

REPO_ROOT = Path(__file__).resolve().parents[2]

def test_plain_logit_runs():
    X, y, meta = load(repo_root=REPO_ROOT)
    X_tr, X_te, _ = fold_safe_iqr_cap(X.iloc[:800], X.iloc[800:], meta)
    m = PlainLogit()
    p = m.fit_predict(X_tr, y.iloc[:800], X_te, meta)
    assert p.shape == (200,)
    assert np.all((p >= 0) & (p <= 1))

def test_plain_logit_accepts_tuned_params():
    X, y, meta = load(repo_root=REPO_ROOT)
    X_tr, X_te, _ = fold_safe_iqr_cap(X.iloc[:800], X.iloc[800:], meta)
    m = PlainLogit(params={"C": 0.1})
    p = m.fit_predict(X_tr, y.iloc[:800], X_te, meta)
    assert p.shape == (200,)
