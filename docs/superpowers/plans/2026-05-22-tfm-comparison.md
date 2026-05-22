# TFM vs WoE-Logit PD Comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible 10-fold CV benchmark in `comparison/` that pits 4 tabular foundation models (TabPFN v2, TabICL, TabDPT, CARTE) against the `pd-autopilot` WoE-logit champion plus three classical baselines (XGB, LightGBM, plain logit) on German Credit (n=1,000), with paired statistical tests and two passes (defaults + light Optuna tuning for classicals only).

**Architecture:** Sibling directory `comparison/` with `uv`-managed environment. Thin notebook orchestrates a per-model wrapper for each of 8 models behind a uniform `PDModel` protocol. Shared modules for data loading, fold creation, and metrics. Per-fold CSV written row-by-row (crash-safe). Tests under `tests/` use `pytest`.

**Tech Stack:** `uv` (env mgmt), Python ≥ 3.10, pandas, numpy, scikit-learn, scipy, matplotlib, optbinning, xgboost, lightgbm, tabpfn, tabicl, tabdpt, carte-ai, optuna, jupyter, tqdm, pytest.

**Spec reference:** `docs/superpowers/specs/2026-05-22-tfm-comparison-design.md`

---

## Task 1: Bootstrap project with uv

**Files:**
- Create: `comparison/pyproject.toml` (via `uv init`)
- Create: `comparison/.gitignore`
- Create: `comparison/README.md`
- Create: `comparison/src/__init__.py`
- Create: `comparison/src/models/__init__.py`
- Create: `comparison/tests/__init__.py`
- Create: `comparison/notebooks/.gitkeep`
- Create: `comparison/results/figures/.gitkeep`

- [ ] **Step 1: Install uv if missing**

```bash
which uv || brew install uv
uv --version
```
Expected: prints a version like `uv 0.4.x`.

- [ ] **Step 2: Initialise the project**

```bash
cd comparison 2>/dev/null || mkdir comparison && cd comparison
uv init --python 3.12 --no-readme --no-workspace
```
This creates `pyproject.toml`, `.python-version`, and downloads CPython 3.12 into uv's cache.

- [ ] **Step 3: Add dependencies (core first, ML libs second)**

```bash
uv add pandas numpy scikit-learn scipy matplotlib tqdm
uv add optbinning xgboost lightgbm optuna
uv add jupyter ipykernel
uv add --dev pytest
```
Note: `tabpfn`, `tabicl`, `tabdpt`, `carte-ai` are added in their respective wrapper tasks because they each may need a GitHub install fallback.

- [ ] **Step 4: Create directory skeleton**

```bash
mkdir -p src/models tests notebooks results/figures
touch src/__init__.py src/models/__init__.py tests/__init__.py
touch notebooks/.gitkeep results/figures/.gitkeep
```

- [ ] **Step 5: Write `.gitignore`**

Create `comparison/.gitignore`:
```
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.pytest_cache/
results/per_fold_*.csv.lock
*.egg-info/
.DS_Store
```

- [ ] **Step 6: Write `README.md`**

Create `comparison/README.md`:
````markdown
# TFM vs WoE-Logit Comparison

Benchmarks 4 tabular foundation models against the pd-autopilot WoE-logit
champion on German Credit. See
[design spec](../docs/superpowers/specs/2026-05-22-tfm-comparison-design.md).

## Setup

```bash
brew install uv      # one-time
uv sync              # creates .venv from uv.lock
```

## Run

```bash
# Always use `uv run` so you get this project's venv, not your system/conda Python.
uv run pytest                       # tests
uv run python -m src.runner --pass defaults
uv run python -m src.runner --pass tuned
uv run jupyter lab notebooks/       # interactive analysis
```

## Outputs

`results/per_fold_defaults.csv`, `results/per_fold_tuned.csv`,
`results/summary_defaults.md`, `results/summary_tuned.md`, and figures
under `results/figures/`.
````

- [ ] **Step 7: Verify env works**

```bash
cd comparison
uv run python -c "import pandas, numpy, sklearn, optbinning, xgboost, lightgbm, optuna; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 8: Commit**

```bash
cd ..    # back to repo root
git add comparison/pyproject.toml comparison/uv.lock comparison/.python-version \
        comparison/.gitignore comparison/README.md \
        comparison/src/__init__.py comparison/src/models/__init__.py \
        comparison/tests/__init__.py \
        comparison/notebooks/.gitkeep comparison/results/figures/.gitkeep
git commit -m "feat(comparison): bootstrap uv project with core deps"
```

---

## Task 2: Data loader

**Files:**
- Create: `comparison/src/data_loader.py`
- Create: `comparison/tests/test_data_loader.py`

- [ ] **Step 1: Write the failing test**

Create `comparison/tests/test_data_loader.py`:
```python
from pathlib import Path
import pandas as pd
from src.data_loader import load, carte_decode

REPO_ROOT = Path(__file__).resolve().parents[2]

def test_load_shape_and_target():
    X, y, meta = load(repo_root=REPO_ROOT)
    assert X.shape == (1000, 20)
    assert y.shape == (1000,)
    assert set(y.unique()) == {0, 1}
    assert abs(y.mean() - 0.30) < 0.01   # 30% default rate
    assert "Creditability" not in X.columns

def test_meta_has_all_columns():
    X, y, meta = load(repo_root=REPO_ROOT)
    assert set(meta.keys()) == set(X.columns)
    for col, info in meta.items():
        assert info["type"] in {"nominal", "ordinal", "continuous"}
        assert info["dtype"] in {"numerical", "categorical"}

def test_carte_decode_replaces_purpose():
    X, y, meta = load(repo_root=REPO_ROOT)
    X_str = carte_decode(X, meta)
    # Purpose=3 must become "Radio/TV" (or similar string) — not the integer
    assert X_str["Purpose"].dtype == object
    assert not (X_str["Purpose"] == 3).any()
    assert "Radio/TV" in X_str["Purpose"].astype(str).str.cat(sep="|")

def test_carte_decode_keeps_continuous_numeric():
    X, y, meta = load(repo_root=REPO_ROOT)
    X_str = carte_decode(X, meta)
    assert pd.api.types.is_numeric_dtype(X_str["Age (years)"])
    assert pd.api.types.is_numeric_dtype(X_str["Credit Amount"])
```

- [ ] **Step 2: Run test, confirm failure**

```bash
cd comparison && uv run pytest tests/test_data_loader.py -v
```
Expected: ImportError on `src.data_loader`.

- [ ] **Step 3: Implement `data_loader.py`**

Create `comparison/src/data_loader.py`:
```python
"""Load German Credit dataset and metadata from the parent repo."""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

TARGET = "Creditability"

def _parse_encoding(encoding_str: str) -> dict[int, str]:
    """Parse the Encoding column into {int_code: human_label}.

    Format examples:
      '1=<0 DM (A11), 2=0-200 DM (A12), 3=>=200 DM/salary (A13). Special: 4=No checking account (A14)'
      '0=New car, 1=Used car, 2=Furniture, 3=Radio/TV, ...'
    """
    if not isinstance(encoding_str, str):
        return {}
    mapping: dict[int, str] = {}
    # Strip parenthetical UCI codes like "(A11)"
    cleaned = re.sub(r"\([^)]*\)", "", encoding_str)
    # Drop "Special:" prefix on tail items
    cleaned = cleaned.replace("Special:", ",")
    # Split on commas or periods
    for part in re.split(r"[,.]", cleaned):
        m = re.match(r"\s*(-?\d+)\s*=\s*(.+?)\s*$", part)
        if m:
            mapping[int(m.group(1))] = m.group(2).strip()
    return mapping

def _parse_special_codes(s: str) -> list[int]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [int(c.strip()) for c in s.split(",") if c.strip()]

def load(repo_root: Path | str) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load loans.csv + variable_types.csv from the parent repo.

    Returns:
        X: DataFrame (1000, 20)
        y: Series (1000,) of 0/1 int
        meta: dict[col_name, {type, dtype, monotonicity, special_codes, encoding}]
    """
    repo_root = Path(repo_root)
    df = pd.read_csv(repo_root / "data" / "loans.csv")
    types_df = pd.read_csv(repo_root / "data" / "variable_types.csv")

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    type_map = {"Nominal": "nominal", "Ordinal": "ordinal",
                "Continuous": "continuous", "Numerical": "continuous",
                "Binary": "nominal", "Categorical": "nominal"}

    meta: dict = {}
    for _, row in types_df.iterrows():
        col = row["Variable"]
        if col == TARGET or col not in X.columns:
            continue
        meta[col] = {
            "type": type_map.get(row["Type"], "continuous"),
            "dtype": row["dtype"] if isinstance(row["dtype"], str) else "numerical",
            "monotonicity": row["Monotonicity"] if isinstance(row["Monotonicity"], str) else "no",
            "special_codes": _parse_special_codes(row.get("SpecialCodes", "")),
            "encoding": _parse_encoding(row.get("Encoding", "")),
        }
    return X, y, meta

def carte_decode(X: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Replace integer codes with human-readable strings for nominal/ordinal columns.

    Continuous columns are passed through unchanged. CARTE consumes the resulting
    string-labelled DataFrame so its graph-attention mechanism can use the actual
    semantic labels rather than meaningless integer codes.
    """
    out = X.copy()
    for col, info in meta.items():
        if info["type"] == "continuous":
            continue
        encoding = info["encoding"]
        if not encoding:
            continue
        out[col] = out[col].map(lambda v: encoding.get(int(v), str(v)) if pd.notna(v) else v)
        out[col] = out[col].astype(object)
    return out
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
cd comparison && uv run pytest tests/test_data_loader.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/data_loader.py comparison/tests/test_data_loader.py
git commit -m "feat(comparison): data loader with CARTE string decoder"
```

---

## Task 3: Cross-validation folds

**Files:**
- Create: `comparison/src/cv.py`
- Create: `comparison/tests/test_cv.py`

- [ ] **Step 1: Write the failing test**

Create `comparison/tests/test_cv.py`:
```python
import numpy as np
import pandas as pd
from src.cv import make_folds

def test_make_folds_basic():
    y = pd.Series([0, 1] * 500)
    folds = make_folds(y, n_splits=10, seed=42)
    assert len(folds) == 10
    for tr, te in folds:
        assert len(tr) + len(te) == 1000
        assert set(tr).isdisjoint(set(te))

def test_make_folds_stratified():
    y = pd.Series([0] * 700 + [1] * 300)
    folds = make_folds(y, n_splits=10, seed=42)
    for tr, te in folds:
        rate = y.iloc[te].mean()
        assert 0.25 <= rate <= 0.35   # stratified ~ 0.30

def test_make_folds_deterministic():
    y = pd.Series([0, 1] * 500)
    folds_a = make_folds(y, n_splits=10, seed=42)
    folds_b = make_folds(y, n_splits=10, seed=42)
    for (tra, tea), (trb, teb) in zip(folds_a, folds_b):
        assert np.array_equal(tra, trb)
        assert np.array_equal(tea, teb)
```

- [ ] **Step 2: Run, confirm fail**

```bash
cd comparison && uv run pytest tests/test_cv.py -v
```

- [ ] **Step 3: Implement**

Create `comparison/src/cv.py`:
```python
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def make_folds(y: pd.Series, n_splits: int = 10, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return frozen list of (train_idx, test_idx) for stratified K-fold CV."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_dummy = np.zeros((len(y), 1))
    return [(tr, te) for tr, te in skf.split(X_dummy, y)]
```

- [ ] **Step 4: Run, confirm pass**

```bash
cd comparison && uv run pytest tests/test_cv.py -v
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/cv.py comparison/tests/test_cv.py
git commit -m "feat(comparison): stratified 10-fold CV with frozen seed"
```

---

## Task 4: Core metrics (AUC, Gini, KS, Brier, log-loss)

**Files:**
- Create: `comparison/src/metrics.py`
- Create: `comparison/tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `comparison/tests/test_metrics.py`:
```python
import numpy as np
from sklearn.metrics import roc_auc_score
from src.metrics import compute_all, ks_stat

def test_compute_all_matches_sklearn_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    m = compute_all(y, p)
    assert abs(m["auc"] - roc_auc_score(y, p)) < 1e-12
    assert abs(m["gini"] - (2 * m["auc"] - 1)) < 1e-12
    assert 0.0 <= m["ks"] <= 1.0
    assert 0.0 <= m["brier"] <= 1.0
    assert m["logloss"] > 0

def test_ks_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert abs(ks_stat(y, p) - 1.0) < 1e-12

def test_logloss_clipping():
    y = np.array([0, 1])
    p = np.array([1.0, 0.0])   # would be inf without clipping
    m = compute_all(y, p)
    assert np.isfinite(m["logloss"])
```

- [ ] **Step 2: Run, fail**

```bash
cd comparison && uv run pytest tests/test_metrics.py -v
```

- [ ] **Step 3: Implement**

Create `comparison/src/metrics.py`:
```python
"""Discrimination + scoring-rule metrics."""
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
```

- [ ] **Step 4: Run, pass**

```bash
cd comparison && uv run pytest tests/test_metrics.py -v
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/metrics.py comparison/tests/test_metrics.py
git commit -m "feat(comparison): core discrimination metrics with clipping"
```

---

## Task 5: DeLong paired test for AUC

**Files:**
- Modify: `comparison/src/metrics.py` (append DeLong functions)
- Modify: `comparison/tests/test_metrics.py` (append DeLong tests)

- [ ] **Step 1: Append failing tests to `test_metrics.py`**

Add to the bottom of `comparison/tests/test_metrics.py`:
```python
from src.metrics import delong_test

def test_delong_identical_returns_high_p():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    z, pval = delong_test(y, p, p)
    assert abs(z) < 1e-6
    assert pval > 0.99

def test_delong_better_model_significant():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 500)
    # bad model = noise; good model = y + noise (perfectly informative)
    p_bad = rng.random(500)
    p_good = y + rng.normal(0, 0.3, 500)
    z, pval = delong_test(y, p_bad, p_good)
    assert pval < 0.01
```

- [ ] **Step 2: Run, fail**

```bash
cd comparison && uv run pytest tests/test_metrics.py::test_delong_identical_returns_high_p -v
```

- [ ] **Step 3: Implement DeLong (Sun & Xu 2014 algorithm)**

Append to `comparison/src/metrics.py`:
```python
from scipy import stats

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Mid-rank, ties broken by averaging."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1   # 1-indexed mid-rank
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    """Sun & Xu 2014 fast DeLong implementation. Predictions shape (k, n), positives first."""
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
    order = (-y_true).argsort(kind="stable")   # positives first
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
```

- [ ] **Step 4: Run, pass**

```bash
cd comparison && uv run pytest tests/test_metrics.py -v
```
Expected: all metrics tests pass (5 total now).

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/metrics.py comparison/tests/test_metrics.py
git commit -m "feat(comparison): DeLong paired AUC test (Sun & Xu 2014)"
```

---

## Task 6: Wilcoxon + Bonferroni helpers

**Files:**
- Modify: `comparison/src/metrics.py`
- Modify: `comparison/tests/test_metrics.py`

- [ ] **Step 1: Append failing test**

Add to `comparison/tests/test_metrics.py`:
```python
from src.metrics import wilcoxon_paired, bonferroni

def test_wilcoxon_paired_returns_pval():
    np.random.seed(0)
    a = np.random.random(10)
    b = a + 0.05   # b consistently higher
    stat, p = wilcoxon_paired(a, b)
    assert p < 0.05

def test_bonferroni_clips_to_one():
    assert bonferroni(0.5, 5) == 1.0
    assert bonferroni(0.01, 7) == 0.07
```

- [ ] **Step 2: Run, fail**

```bash
cd comparison && uv run pytest tests/test_metrics.py -k "wilcoxon or bonferroni" -v
```

- [ ] **Step 3: Implement**

Append to `comparison/src/metrics.py`:
```python
def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test on per-fold metric differences."""
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue)

def bonferroni(pval: float, n_comparisons: int) -> float:
    return float(min(1.0, pval * n_comparisons))
```

- [ ] **Step 4: Run, pass**

```bash
cd comparison && uv run pytest tests/test_metrics.py -v
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/metrics.py comparison/tests/test_metrics.py
git commit -m "feat(comparison): Wilcoxon paired test + Bonferroni helper"
```

---

## Task 7: PDModel protocol

**Files:**
- Create: `comparison/src/models/base.py`
- Create: `comparison/tests/test_base.py`

- [ ] **Step 1: Write the failing test**

Create `comparison/tests/test_base.py`:
```python
import numpy as np
import pandas as pd
from src.models.base import PDModel

class _Dummy:
    name = "dummy"
    requires_string_labels = False
    def fit_predict(self, X_train, y_train, X_test, types_meta, seed=42):
        return np.full(len(X_test), 0.5)

def test_dummy_satisfies_protocol():
    m: PDModel = _Dummy()
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    out = m.fit_predict(X, y, X, {"a": {"type": "continuous"}})
    assert out.shape == (3,)
    assert np.all((out >= 0) & (out <= 1))
```

- [ ] **Step 2: Run, fail**

```bash
cd comparison && uv run pytest tests/test_base.py -v
```

- [ ] **Step 3: Implement**

Create `comparison/src/models/base.py`:
```python
from __future__ import annotations
from typing import Protocol
import numpy as np
import pandas as pd

class PDModel(Protocol):
    """Uniform interface every model wrapper implements."""
    name: str
    requires_string_labels: bool

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        types_meta: dict,
        seed: int = 42,
    ) -> np.ndarray:
        """Return P(default=1) for each row of X_test, shape (len(X_test),)."""
        ...
```

- [ ] **Step 4: Run, pass**

```bash
cd comparison && uv run pytest tests/test_base.py -v
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/models/base.py comparison/tests/test_base.py
git commit -m "feat(comparison): PDModel protocol"
```

---

## Task 8: WoE-logit champion wrapper (faithful to report)

**Files:**
- Create: `comparison/src/models/woe_logit.py`
- Create: `comparison/tests/test_woe_logit.py`

The wrapper fits OptimalBinning per fold using `dtype`/`monotonicity`/`special_codes` from `types_meta`, then sklearn LogisticRegression on the WoE-transformed 8 champion variables.

- [ ] **Step 1: Write the failing test**

Create `comparison/tests/test_woe_logit.py`:
```python
from pathlib import Path
import numpy as np
import pandas as pd
from src.data_loader import load
from src.models.woe_logit import WoELogitChampion, CHAMPION_VARS

REPO_ROOT = Path(__file__).resolve().parents[2]

def test_champion_vars_count():
    assert len(CHAMPION_VARS) == 8

def test_fit_predict_shape_and_range():
    X, y, meta = load(repo_root=REPO_ROOT)
    model = WoELogitChampion()
    # Use first 800 train, last 200 test
    proba = model.fit_predict(X.iloc[:800], y.iloc[:800], X.iloc[800:], meta, seed=42)
    assert proba.shape == (200,)
    assert np.all((proba >= 0) & (proba <= 1))

def test_full_sample_refit_matches_report_auc():
    """Sanity: full-sample fit must reproduce the report's in-sample AUC ~0.81."""
    from sklearn.metrics import roc_auc_score
    X, y, meta = load(repo_root=REPO_ROOT)
    model = WoELogitChampion()
    proba = model.fit_predict(X, y, X, meta, seed=42)
    auc = roc_auc_score(y, proba)
    # Report says 0.8109; allow tolerance for binning library version differences
    assert 0.80 < auc < 0.82, f"got AUC={auc:.4f}, expected ~0.81"

def test_determinism():
    X, y, meta = load(repo_root=REPO_ROOT)
    m = WoELogitChampion()
    a = m.fit_predict(X.iloc[:800], y.iloc[:800], X.iloc[800:], meta, seed=42)
    b = m.fit_predict(X.iloc[:800], y.iloc[:800], X.iloc[800:], meta, seed=42)
    np.testing.assert_allclose(a, b)
```

- [ ] **Step 2: Run, fail**

```bash
cd comparison && uv run pytest tests/test_woe_logit.py -v
```

- [ ] **Step 3: Implement**

Create `comparison/src/models/woe_logit.py`:
```python
"""WoE + Logit champion model — faithful to pd-autopilot stage 04a (MIV)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from optbinning import OptimalBinning
from sklearn.linear_model import LogisticRegression

# The 8 variables the report's MIV champion selected.
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

def _monotonic_trend(monotonicity_flag: str) -> str | None:
    """variable_types.csv encodes 'yes' (use auto) or 'no' (none). pdtoolkit's
    OptimalBinning accepts 'auto' / 'ascending' / 'descending' / None."""
    if isinstance(monotonicity_flag, str) and monotonicity_flag.strip().lower() == "yes":
        return "auto"
    return None

def _fit_binning(X_tr: pd.DataFrame, y_tr: pd.Series, meta: dict) -> dict[str, OptimalBinning]:
    binners: dict[str, OptimalBinning] = {}
    for col in CHAMPION_VARS:
        info = meta[col]
        ob = OptimalBinning(
            name=col,
            dtype=info["dtype"],
            monotonic_trend=_monotonic_trend(info["monotonicity"]),
            special_codes=info["special_codes"] or None,
            solver="cp",
        )
        ob.fit(X_tr[col].values, y_tr.values)
        binners[col] = ob
    return binners

def _transform(binners: dict[str, OptimalBinning], X: pd.DataFrame) -> np.ndarray:
    cols = [binners[c].transform(X[c].values, metric="woe") for c in CHAMPION_VARS]
    return np.column_stack(cols)

class WoELogitChampion:
    name = "woe_logit"
    requires_string_labels = False

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        binners = _fit_binning(X_train, y_train, types_meta)
        Z_tr = _transform(binners, X_train)
        Z_te = _transform(binners, X_test)
        lr = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
        lr.fit(Z_tr, y_train.values)
        return lr.predict_proba(Z_te)[:, 1]
```

- [ ] **Step 4: Run, pass**

```bash
cd comparison && uv run pytest tests/test_woe_logit.py -v
```
If `test_full_sample_refit_matches_report_auc` fails, do NOT proceed — debug the binning setup (special_codes, monotonic_trend, dtype) against the report's stage_03.md before continuing.

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/models/woe_logit.py comparison/tests/test_woe_logit.py
git commit -m "feat(comparison): WoE-logit champion wrapper with report-fidelity sanity test"
```

---

## Task 9: Plain logistic regression wrapper

**Files:**
- Create: `comparison/src/models/logit.py`
- Create: `comparison/tests/test_logit.py`

- [ ] **Step 1: Write the failing test**

Create `comparison/tests/test_logit.py`:
```python
from pathlib import Path
import numpy as np
from src.data_loader import load
from src.models.logit import PlainLogit

REPO_ROOT = Path(__file__).resolve().parents[2]

def test_plain_logit_runs():
    X, y, meta = load(repo_root=REPO_ROOT)
    m = PlainLogit()
    p = m.fit_predict(X.iloc[:800], y.iloc[:800], X.iloc[800:], meta)
    assert p.shape == (200,)
    assert np.all((p >= 0) & (p <= 1))
```

- [ ] **Step 2: Fail**

```bash
cd comparison && uv run pytest tests/test_logit.py -v
```

- [ ] **Step 3: Implement**

Create `comparison/src/models/logit.py`:
```python
"""Plain logistic regression on one-hot nominals + standardised continuous."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class PlainLogit:
    name = "logit"
    requires_string_labels = False

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        nominal = [c for c, i in types_meta.items() if i["type"] == "nominal" and c in X_train.columns]
        other = [c for c in X_train.columns if c not in nominal]
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal),
            ("num", StandardScaler(), other),
        ])
        pipe = Pipeline([("pre", pre),
                         ("lr", LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed))])
        pipe.fit(X_train, y_train.values)
        return pipe.predict_proba(X_test)[:, 1]
```

- [ ] **Step 4: Pass**

```bash
cd comparison && uv run pytest tests/test_logit.py -v
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/models/logit.py comparison/tests/test_logit.py
git commit -m "feat(comparison): plain logistic regression baseline"
```

---

## Task 10: XGBoost wrapper

**Files:**
- Create: `comparison/src/models/xgb.py`
- Create: `comparison/tests/test_xgb.py`

- [ ] **Step 1: Failing test**

Create `comparison/tests/test_xgb.py`:
```python
from pathlib import Path
import numpy as np
from src.data_loader import load
from src.models.xgb import XGBWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]

def test_xgb_runs():
    X, y, meta = load(repo_root=REPO_ROOT)
    m = XGBWrapper()
    p = m.fit_predict(X.iloc[:800], y.iloc[:800], X.iloc[800:], meta)
    assert p.shape == (200,)
    assert np.all((p >= 0) & (p <= 1))
```

- [ ] **Step 2: Fail**

```bash
cd comparison && uv run pytest tests/test_xgb.py -v
```

- [ ] **Step 3: Implement**

Create `comparison/src/models/xgb.py`:
```python
"""XGBoost with native categorical handling."""
from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

class XGBWrapper:
    name = "xgb"
    requires_string_labels = False

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def _to_cat(self, X: pd.DataFrame, types_meta: dict) -> pd.DataFrame:
        X = X.copy()
        for c, i in types_meta.items():
            if i["type"] == "nominal" and c in X.columns:
                X[c] = X[c].astype("category")
        return X

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        X_tr = self._to_cat(X_train, types_meta)
        X_te = self._to_cat(X_test, types_meta)
        # Align category levels of test to train
        for c in X_tr.columns:
            if str(X_tr[c].dtype) == "category":
                X_te[c] = X_te[c].astype(pd.CategoricalDtype(categories=X_tr[c].cat.categories))
        params = dict(
            tree_method="hist",
            enable_categorical=True,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
        )
        params.update(self.params)
        clf = XGBClassifier(**params)
        clf.fit(X_tr, y_train.values)
        return clf.predict_proba(X_te)[:, 1]
```

- [ ] **Step 4: Pass**

```bash
cd comparison && uv run pytest tests/test_xgb.py -v
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/models/xgb.py comparison/tests/test_xgb.py
git commit -m "feat(comparison): XGBoost wrapper with native categorical handling"
```

---

## Task 11: LightGBM wrapper

**Files:**
- Create: `comparison/src/models/lgbm.py`
- Create: `comparison/tests/test_lgbm.py`

- [ ] **Step 1: Failing test**

Create `comparison/tests/test_lgbm.py`:
```python
from pathlib import Path
import numpy as np
from src.data_loader import load
from src.models.lgbm import LGBMWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]

def test_lgbm_runs():
    X, y, meta = load(repo_root=REPO_ROOT)
    m = LGBMWrapper()
    p = m.fit_predict(X.iloc[:800], y.iloc[:800], X.iloc[800:], meta)
    assert p.shape == (200,)
    assert np.all((p >= 0) & (p <= 1))
```

- [ ] **Step 2: Fail**

```bash
cd comparison && uv run pytest tests/test_lgbm.py -v
```

- [ ] **Step 3: Implement**

Create `comparison/src/models/lgbm.py`:
```python
"""LightGBM with categorical_feature index list."""
from __future__ import annotations
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

class LGBMWrapper:
    name = "lgbm"
    requires_string_labels = False

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        cat_cols = [c for c, i in types_meta.items() if i["type"] == "nominal" and c in X_train.columns]
        cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]
        X_tr = X_train.copy()
        X_te = X_test.copy()
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype("category")
            X_te[c] = X_te[c].astype(pd.CategoricalDtype(categories=X_tr[c].cat.categories))
        params = dict(
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        params.update(self.params)
        clf = LGBMClassifier(**params)
        clf.fit(X_tr, y_train.values, categorical_feature=cat_idx)
        return clf.predict_proba(X_te)[:, 1]
```

- [ ] **Step 4: Pass**

```bash
cd comparison && uv run pytest tests/test_lgbm.py -v
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/models/lgbm.py comparison/tests/test_lgbm.py
git commit -m "feat(comparison): LightGBM wrapper with categorical_feature support"
```

---

## Task 12: TabPFN v2 wrapper

**Files:**
- Create: `comparison/src/models/tabpfn.py`
- Create: `comparison/tests/test_tabpfn.py`

- [ ] **Step 1: Add dep**

```bash
cd comparison && uv add tabpfn
```
If PyPI install fails, fall back to: `uv add "tabpfn @ git+https://github.com/PriorLabs/TabPFN.git"`.

- [ ] **Step 2: Failing test**

Create `comparison/tests/test_tabpfn.py`:
```python
from pathlib import Path
import numpy as np
import pytest
from src.data_loader import load
from src.models.tabpfn import TabPFNWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.slow
def test_tabpfn_runs():
    X, y, meta = load(repo_root=REPO_ROOT)
    m = TabPFNWrapper()
    p = m.fit_predict(X.iloc[:800], y.iloc[:800], X.iloc[800:], meta)
    assert p.shape == (200,)
    assert np.all((p >= 0) & (p <= 1))
```

- [ ] **Step 3: Fail**

```bash
cd comparison && uv run pytest tests/test_tabpfn.py -v -m slow
```

- [ ] **Step 4: Implement**

Create `comparison/src/models/tabpfn.py`:
```python
"""TabPFN v2 wrapper."""
from __future__ import annotations
import numpy as np
import pandas as pd

def _device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

class TabPFNWrapper:
    name = "tabpfn_v2"
    requires_string_labels = False

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        from tabpfn import TabPFNClassifier
        cat_idx = [X_train.columns.get_loc(c)
                   for c, i in types_meta.items()
                   if i["type"] == "nominal" and c in X_train.columns]
        clf = TabPFNClassifier(
            device=_device(),
            categorical_features_indices=cat_idx or None,
            random_state=seed,
        )
        clf.fit(X_train.values, y_train.values)
        return clf.predict_proba(X_test.values)[:, 1]
```

- [ ] **Step 5: Pass**

```bash
cd comparison && uv run pytest tests/test_tabpfn.py -v -m slow
```
First run downloads the TabPFN v2 weights from Hugging Face (~100 MB).

- [ ] **Step 6: Register the `slow` marker**

Edit `comparison/pyproject.toml`, append to `[tool.pytest.ini_options]` (create the section if absent):
```toml
[tool.pytest.ini_options]
markers = [
    "slow: heavy TFM tests; opt in with -m slow",
]
```

- [ ] **Step 7: Commit**

```bash
cd .. && git add comparison/src/models/tabpfn.py comparison/tests/test_tabpfn.py \
                comparison/pyproject.toml comparison/uv.lock
git commit -m "feat(comparison): TabPFN v2 wrapper (MPS/CUDA/CPU auto-device)"
```

---

## Task 13: TabICL wrapper

**Files:**
- Create: `comparison/src/models/tabicl.py`
- Create: `comparison/tests/test_tabicl.py`

- [ ] **Step 1: Add dep**

```bash
cd comparison && uv add tabicl
```
If unavailable on PyPI: `uv add "tabicl @ git+https://github.com/soda-inria/tabicl.git"`.

- [ ] **Step 2: Failing test**

Create `comparison/tests/test_tabicl.py`:
```python
from pathlib import Path
import numpy as np
import pytest
from src.data_loader import load
from src.models.tabicl import TabICLWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.slow
def test_tabicl_runs():
    X, y, meta = load(repo_root=REPO_ROOT)
    m = TabICLWrapper()
    p = m.fit_predict(X.iloc[:800], y.iloc[:800], X.iloc[800:], meta)
    assert p.shape == (200,)
    assert np.all((p >= 0) & (p <= 1))
```

- [ ] **Step 3: Fail**

```bash
cd comparison && uv run pytest tests/test_tabicl.py -v -m slow
```

- [ ] **Step 4: Implement**

Create `comparison/src/models/tabicl.py`:
```python
"""TabICL wrapper (in-context tabular learner)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.models.tabpfn import _device

class TabICLWrapper:
    name = "tabicl"
    requires_string_labels = False

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        from tabicl import TabICLClassifier
        cat_idx = [X_train.columns.get_loc(c)
                   for c, i in types_meta.items()
                   if i["type"] == "nominal" and c in X_train.columns]
        clf = TabICLClassifier(device=_device(), random_state=seed)
        # TabICL accepts categorical_features kwarg in fit() on recent versions.
        try:
            clf.fit(X_train.values, y_train.values, categorical_features=cat_idx or None)
        except TypeError:
            clf.fit(X_train.values, y_train.values)
        return clf.predict_proba(X_test.values)[:, 1]
```

- [ ] **Step 5: Pass**

```bash
cd comparison && uv run pytest tests/test_tabicl.py -v -m slow
```

- [ ] **Step 6: Commit**

```bash
cd .. && git add comparison/src/models/tabicl.py comparison/tests/test_tabicl.py \
                comparison/pyproject.toml comparison/uv.lock
git commit -m "feat(comparison): TabICL wrapper"
```

---

## Task 14: TabDPT wrapper

**Files:**
- Create: `comparison/src/models/tabdpt.py`
- Create: `comparison/tests/test_tabdpt.py`

- [ ] **Step 1: Add dep**

```bash
cd comparison && uv add tabdpt
```
Fallback: `uv add "tabdpt @ git+https://github.com/layer6ai-labs/TabDPT.git"`.

- [ ] **Step 2: Failing test**

Create `comparison/tests/test_tabdpt.py`:
```python
from pathlib import Path
import numpy as np
import pytest
from src.data_loader import load
from src.models.tabdpt import TabDPTWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.slow
def test_tabdpt_runs():
    X, y, meta = load(repo_root=REPO_ROOT)
    m = TabDPTWrapper()
    p = m.fit_predict(X.iloc[:800], y.iloc[:800], X.iloc[800:], meta)
    assert p.shape == (200,)
    assert np.all((p >= 0) & (p <= 1))
```

- [ ] **Step 3: Fail**

```bash
cd comparison && uv run pytest tests/test_tabdpt.py -v -m slow
```

- [ ] **Step 4: Implement**

Create `comparison/src/models/tabdpt.py`:
```python
"""TabDPT wrapper (in-context tabular foundation model from Layer 6 AI)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.models.tabpfn import _device

class TabDPTWrapper:
    name = "tabdpt"
    requires_string_labels = False

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        from tabdpt import TabDPTClassifier
        clf = TabDPTClassifier(device=_device())
        clf.fit(X_train.values, y_train.values)
        proba = clf.predict_proba(X_test.values)
        # Some versions return (n, 2), others (n,) for binary; normalise:
        if proba.ndim == 2:
            return proba[:, 1]
        return proba
```

- [ ] **Step 5: Pass**

```bash
cd comparison && uv run pytest tests/test_tabdpt.py -v -m slow
```

- [ ] **Step 6: Commit**

```bash
cd .. && git add comparison/src/models/tabdpt.py comparison/tests/test_tabdpt.py \
                comparison/pyproject.toml comparison/uv.lock
git commit -m "feat(comparison): TabDPT wrapper"
```

---

## Task 15: CARTE wrapper (string-decoded)

**Files:**
- Create: `comparison/src/models/carte.py`
- Create: `comparison/tests/test_carte.py`

- [ ] **Step 1: Add deps**

```bash
cd comparison && uv add carte-ai torch
```
PyG is a transitive dep of carte-ai; if `uv add carte-ai` fails on missing PyG wheels, install torch first then: `uv add "carte-ai @ git+https://github.com/soda-inria/carte.git"`.

- [ ] **Step 2: Failing test**

Create `comparison/tests/test_carte.py`:
```python
from pathlib import Path
import numpy as np
import pytest
from src.data_loader import load, carte_decode
from src.models.carte import CARTEWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.slow
def test_carte_requires_string_labels_flag():
    assert CARTEWrapper.requires_string_labels is True

@pytest.mark.slow
def test_carte_runs_on_decoded_input():
    X, y, meta = load(repo_root=REPO_ROOT)
    X_str = carte_decode(X, meta)
    m = CARTEWrapper()
    p = m.fit_predict(X_str.iloc[:800], y.iloc[:800], X_str.iloc[800:], meta)
    assert p.shape == (200,)
    assert np.all((p >= 0) & (p <= 1))
```

- [ ] **Step 3: Fail**

```bash
cd comparison && uv run pytest tests/test_carte.py -v -m slow
```

- [ ] **Step 4: Implement**

Create `comparison/src/models/carte.py`:
```python
"""CARTE wrapper — consumes string-decoded inputs from carte_decode()."""
from __future__ import annotations
import numpy as np
import pandas as pd

class CARTEWrapper:
    name = "carte"
    requires_string_labels = True

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        # carte-ai's public API as of 2026:
        from carte_ai import CARTEClassifier   # PyPI package name is carte-ai → module carte_ai
        clf = CARTEClassifier(random_state=seed)
        clf.fit(X_train, y_train.values)
        proba = clf.predict_proba(X_test)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba
```
Note: if the import path differs (e.g. `from carte import CARTEClassifier`), adjust to match the installed package's `__init__.py`. Check with `uv run python -c "import carte_ai; print(dir(carte_ai))"`.

- [ ] **Step 5: Pass**

```bash
cd comparison && uv run pytest tests/test_carte.py -v -m slow
```

- [ ] **Step 6: Commit**

```bash
cd .. && git add comparison/src/models/carte.py comparison/tests/test_carte.py \
                comparison/pyproject.toml comparison/uv.lock
git commit -m "feat(comparison): CARTE wrapper with string-decoded inputs"
```

---

## Task 16: Runner (per-fold CSV writer)

**Files:**
- Create: `comparison/src/runner.py`
- Create: `comparison/tests/test_runner.py`

- [ ] **Step 1: Failing test**

Create `comparison/tests/test_runner.py`:
```python
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.runner import run_one_fold, append_row

def test_append_row_creates_header(tmp_path):
    path = tmp_path / "out.csv"
    append_row(path, {"model": "x", "fold": 0, "auc": 0.7})
    append_row(path, {"model": "x", "fold": 1, "auc": 0.8})
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["auc"] == "0.7"

def test_run_one_fold_returns_metrics():
    class _M:
        name = "stub"
        requires_string_labels = False
        def fit_predict(self, X_tr, y_tr, X_te, meta, seed=42):
            return np.full(len(X_te), 0.5)
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series([0, 1] * 50)
    out = run_one_fold(_M(), X, y, np.arange(80), np.arange(80, 100), meta={})
    assert "auc" in out and "logloss" in out
    assert out["model"] == "stub"
    assert out["fold_idx"] == -1  # unset, runner passes fold_idx separately
```

- [ ] **Step 2: Fail**

```bash
cd comparison && uv run pytest tests/test_runner.py -v
```

- [ ] **Step 3: Implement**

Create `comparison/src/runner.py`:
```python
"""Orchestration: loop over (model, fold), write per-fold CSV row-by-row."""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import load, carte_decode
from src.cv import make_folds
from src.metrics import compute_all

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

def run_one_fold(model, X, y, train_idx, test_idx, meta) -> dict:
    if getattr(model, "requires_string_labels", False):
        X_tr_in = X.iloc[train_idx]
        X_te_in = X.iloc[test_idx]
    else:
        X_tr_in = X.iloc[train_idx]
        X_te_in = X.iloc[test_idx]
    t0 = time.perf_counter()
    proba = model.fit_predict(X_tr_in, y.iloc[train_idx], X_te_in, meta)
    runtime = time.perf_counter() - t0
    m = compute_all(y.iloc[test_idx].values, proba)
    m.update(model=model.name, n_train=len(train_idx), n_test=len(test_idx),
             runtime_s=runtime, fold_idx=-1)
    return m

def _build_models(pass_name: str, tuned_params: dict | None = None):
    from src.models.woe_logit import WoELogitChampion
    from src.models.logit import PlainLogit
    from src.models.xgb import XGBWrapper
    from src.models.lgbm import LGBMWrapper
    from src.models.tabpfn import TabPFNWrapper
    from src.models.tabicl import TabICLWrapper
    from src.models.tabdpt import TabDPTWrapper
    from src.models.carte import CARTEWrapper

    tuned_params = tuned_params or {}
    return [
        WoELogitChampion(),
        PlainLogit(),
        XGBWrapper(params=tuned_params.get("xgb")),
        LGBMWrapper(params=tuned_params.get("lgbm")),
        TabPFNWrapper(),
        TabICLWrapper(),
        TabDPTWrapper(),
        CARTEWrapper(),
    ]

def run_pass(pass_name: str, n_splits: int = 10, seed: int = 42, tuned_params: dict | None = None):
    """Execute all models on all folds, writing per_fold_<pass_name>.csv."""
    X, y, meta = load(REPO_ROOT)
    X_carte = carte_decode(X, meta)
    folds = make_folds(y, n_splits=n_splits, seed=seed)
    out_path = RESULTS_DIR / f"per_fold_{pass_name}.csv"
    if out_path.exists():
        out_path.unlink()
    models = _build_models(pass_name, tuned_params)
    for model in models:
        X_use = X_carte if getattr(model, "requires_string_labels", False) else X
        for fold_i, (tr, te) in enumerate(tqdm(folds, desc=model.name)):
            row = run_one_fold(model, X_use, y, tr, te, meta)
            row["fold_idx"] = fold_i
            row["pass"] = pass_name
            append_row(out_path, row)
    print(f"Wrote {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pass", dest="pass_name", choices=["defaults", "tuned"], required=True)
    args = p.parse_args()
    if args.pass_name == "defaults":
        run_pass("defaults")
    else:
        from src.tuning import tune_all_classicals
        tuned = tune_all_classicals(seed=42)
        run_pass("tuned", tuned_params=tuned)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Pass**

```bash
cd comparison && uv run pytest tests/test_runner.py -v
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/runner.py comparison/tests/test_runner.py
git commit -m "feat(comparison): runner with per-fold CSV writer"
```

---

## Task 17: Optuna tuning for classicals (pass 2)

**Files:**
- Create: `comparison/src/tuning.py`
- Create: `comparison/tests/test_tuning.py`

- [ ] **Step 1: Failing test**

Create `comparison/tests/test_tuning.py`:
```python
from pathlib import Path
from src.tuning import tune_xgb, tune_lgbm

REPO_ROOT = Path(__file__).resolve().parents[2]

def test_tune_xgb_returns_params():
    params = tune_xgb(n_trials=5, seed=42)   # small for test speed
    assert "n_estimators" in params or "max_depth" in params

def test_tune_lgbm_returns_params():
    params = tune_lgbm(n_trials=5, seed=42)
    assert isinstance(params, dict) and len(params) > 0
```

- [ ] **Step 2: Fail**

```bash
cd comparison && uv run pytest tests/test_tuning.py -v
```

- [ ] **Step 3: Implement**

Create `comparison/src/tuning.py`:
```python
"""Optuna nested CV tuning for the classical baselines (pass 2 only)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.data_loader import load

REPO_ROOT = Path(__file__).resolve().parents[2]
N_TRIALS = 50
INNER_FOLDS = 5

def _inner_cv_score(make_clf, X, y, seed: int) -> float:
    skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = make_clf()
        clf.fit(X.iloc[tr], y.iloc[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
    return float(np.mean(aucs))

def tune_xgb(n_trials: int = N_TRIALS, seed: int = 42) -> dict:
    from src.models.xgb import XGBWrapper
    from xgboost import XGBClassifier
    X, y, meta = load(REPO_ROOT)
    nominal = [c for c, i in meta.items() if i["type"] == "nominal"]
    X_cat = X.copy()
    for c in nominal:
        X_cat[c] = X_cat[c].astype("category")

    def objective(trial):
        params = dict(
            tree_method="hist",
            enable_categorical=True,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        )
        return _inner_cv_score(lambda: XGBClassifier(**params), X_cat, y, seed)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def tune_lgbm(n_trials: int = N_TRIALS, seed: int = 42) -> dict:
    from lightgbm import LGBMClassifier
    X, y, meta = load(REPO_ROOT)
    nominal = [c for c, i in meta.items() if i["type"] == "nominal"]
    cat_idx = [X.columns.get_loc(c) for c in nominal]
    X_cat = X.copy()
    for c in nominal:
        X_cat[c] = X_cat[c].astype("category")

    def objective(trial):
        params = dict(
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", -1, 12),
            num_leaves=trial.suggest_int("num_leaves", 8, 128),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        )
        def make():
            clf = LGBMClassifier(**params)
            # categorical_feature is passed at fit time
            class _Wrap:
                def fit(self, Xtr, ytr):
                    clf.fit(Xtr, ytr, categorical_feature=cat_idx)
                    return self
                def predict_proba(self, Xte):
                    return clf.predict_proba(Xte)
            return _Wrap()
        return _inner_cv_score(make, X_cat, y, seed)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def tune_logit(n_trials: int = N_TRIALS, seed: int = 42) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    X, y, meta = load(REPO_ROOT)
    nominal = [c for c, i in meta.items() if i["type"] == "nominal"]
    other = [c for c in X.columns if c not in nominal]

    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        def make():
            pre = ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal),
                ("num", StandardScaler(), other),
            ])
            return Pipeline([("pre", pre),
                             ("lr", LogisticRegression(solver="lbfgs", max_iter=2000,
                                                       C=C, random_state=seed))])
        return _inner_cv_score(make, X, y, seed)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def tune_all_classicals(seed: int = 42) -> dict:
    return {
        "xgb": tune_xgb(seed=seed),
        "lgbm": tune_lgbm(seed=seed),
        "logit": tune_logit(seed=seed),
    }
```

Note: `PlainLogit` and `LGBMWrapper` currently accept `params` only for some fields. For pass 2 to apply tuned LR `C`, extend `PlainLogit.__init__` to accept a `params` dict and forward to `LogisticRegression`. Edit `src/models/logit.py`:

Add after `class PlainLogit:`:
```python
    def __init__(self, params: dict | None = None):
        self.params = params or {}
```
And in `fit_predict`, replace the `LogisticRegression(...)` line with:
```python
        lr_kwargs = dict(solver="lbfgs", max_iter=2000, random_state=seed)
        lr_kwargs.update(self.params)
        pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(**lr_kwargs))])
```
Then in `src/runner.py::_build_models`, update the `PlainLogit()` instantiation to `PlainLogit(params=tuned_params.get("logit"))`.

- [ ] **Step 4: Pass**

```bash
cd comparison && uv run pytest tests/test_tuning.py -v
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/tuning.py comparison/tests/test_tuning.py \
                comparison/src/models/logit.py comparison/src/runner.py
git commit -m "feat(comparison): Optuna tuning for XGB/LGBM/logit (pass 2)"
```

---

## Task 18: Execute pass 1 (defaults)

**Files:**
- Execution only — produces `comparison/results/per_fold_defaults.csv`

- [ ] **Step 1: Confirm WoE sanity check passes**

```bash
cd comparison && uv run pytest tests/test_woe_logit.py -v
```
Must show `test_full_sample_refit_matches_report_auc PASSED`. Do not proceed otherwise.

- [ ] **Step 2: Run all fast (non-slow) tests once**

```bash
cd comparison && uv run pytest -v -m "not slow"
```
Expected: all green.

- [ ] **Step 3: Run pass 1**

```bash
cd comparison && uv run python -m src.runner --pass defaults
```
Expected wall-clock on M2 Max: ≤ 45 min (CARTE dominant). Output: `comparison/results/per_fold_defaults.csv` with 80 rows (8 models × 10 folds).

- [ ] **Step 4: Quick eyeball check**

```bash
cd comparison && uv run python -c "
import pandas as pd
df = pd.read_csv('results/per_fold_defaults.csv')
print(df.groupby('model')[['auc','gini','ks','brier','logloss','runtime_s']].mean().round(4))
print('row count:', len(df))
"
```
Expected: 80 rows; AUCs in [0.5, 0.9]; runtime column shows CARTE >> others.

- [ ] **Step 5: Commit results**

```bash
cd .. && git add comparison/results/per_fold_defaults.csv
git commit -m "results(comparison): pass 1 per-fold metrics (defaults)"
```

---

## Task 19: Execute pass 2 (tuned classicals)

- [ ] **Step 1: Run pass 2**

```bash
cd comparison && uv run python -m src.runner --pass tuned
```
Expected wall-clock: ≤ 60 min (Optuna tuning dominates). Output: `comparison/results/per_fold_tuned.csv` with 80 rows. WoE-logit and TFM rows are copied from pass 1 logic (runner re-runs everything; that's fine — costs little extra time for the non-tuned models).

- [ ] **Step 2: Eyeball**

```bash
cd comparison && uv run python -c "
import pandas as pd
a = pd.read_csv('results/per_fold_defaults.csv').groupby('model')['auc'].mean()
b = pd.read_csv('results/per_fold_tuned.csv').groupby('model')['auc'].mean()
print(pd.DataFrame({'defaults': a, 'tuned': b, 'delta': b - a}).round(4))
"
```
Expected: woe_logit, tabpfn_v2, tabicl, tabdpt, carte deltas ≈ 0 (only sampling noise); xgb, lgbm, logit deltas likely positive.

- [ ] **Step 3: Commit results**

```bash
cd .. && git add comparison/results/per_fold_tuned.csv
git commit -m "results(comparison): pass 2 per-fold metrics (tuned classicals)"
```

---

## Task 20: Summary markdown generator

**Files:**
- Create: `comparison/src/summary.py`
- Create: `comparison/tests/test_summary.py`

- [ ] **Step 1: Failing test**

Create `comparison/tests/test_summary.py`:
```python
from pathlib import Path
import pandas as pd
import numpy as np
from src.summary import build_summary_md

def test_build_summary_md_contains_models(tmp_path):
    df = pd.DataFrame({
        "model": ["woe_logit"] * 10 + ["xgb"] * 10,
        "fold_idx": list(range(10)) * 2,
        "auc": np.r_[np.linspace(0.78, 0.82, 10), np.linspace(0.79, 0.83, 10)],
        "gini": 0.6,
        "ks": 0.5,
        "brier": 0.17,
        "logloss": 0.5,
        "runtime_s": 1.0,
    })
    # Also need per-fold predictions for DeLong; for the unit test we skip DeLong:
    md = build_summary_md(df, paired_predictions=None, n_comparisons=1)
    assert "woe_logit" in md and "xgb" in md
    assert "AUC" in md
```

- [ ] **Step 2: Fail**

```bash
cd comparison && uv run pytest tests/test_summary.py -v
```

- [ ] **Step 3: Implement**

Create `comparison/src/summary.py`:
```python
"""Render per-fold CSV into a thesis-ready summary markdown."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src.metrics import delong_test, wilcoxon_paired, bonferroni

CHAMPION = "woe_logit"
METRICS = ["auc", "gini", "ks", "brier", "logloss"]

CONTEXT = """## Context (not part of paired comparison)
- Report dev AUC (in-sample): 0.8109
- Report 10-fold CV AUC: 0.8065
- Report bootstrap AUC: 0.8026
"""

def build_summary_md(df: pd.DataFrame,
                     paired_predictions: dict | None,
                     n_comparisons: int) -> str:
    g = df.groupby("model")
    agg = g[METRICS].agg(["mean", "std"]).round(4)
    rows = []
    for model in df["model"].unique():
        cells = []
        for m in METRICS:
            mu, sd = agg.loc[model, (m, "mean")], agg.loc[model, (m, "std")]
            cells.append(f"{mu:.4f} ± {sd:.4f}")
        if model == CHAMPION or paired_predictions is None:
            tail = "reference"
        else:
            y, p_c, p_m = paired_predictions["y"], paired_predictions[CHAMPION], paired_predictions[model]
            _, p_delong = delong_test(y, p_c, p_m)
            p_bonf = bonferroni(p_delong, n_comparisons)
            tail = f"p={p_delong:.3f} (bonf={p_bonf:.3f})"
        rows.append(f"| {model} | " + " | ".join(cells) + f" | {tail} |")
    header = "| Model | " + " | ".join(m.upper() for m in METRICS) + " | vs Champion (DeLong) |"
    sep = "|" + "|".join(["---"] * (len(METRICS) + 2)) + "|"
    return "\n".join([
        "## Headline (10-fold CV)",
        "",
        header,
        sep,
        *rows,
        "",
        CONTEXT,
    ])

def main():
    here = Path(__file__).resolve().parents[1]
    for pass_name in ["defaults", "tuned"]:
        csv_path = here / "results" / f"per_fold_{pass_name}.csv"
        if not csv_path.exists():
            print(f"skip {pass_name}: missing {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        # DeLong needs per-fold predictions; the runner doesn't store them yet.
        # For the markdown we use Wilcoxon on per-fold AUCs as the paired test.
        md = _build_with_wilcoxon(df)
        out = here / "results" / f"summary_{pass_name}.md"
        out.write_text(md)
        print(f"wrote {out}")

def _build_with_wilcoxon(df: pd.DataFrame) -> str:
    g = df.groupby("model")
    models = list(df["model"].unique())
    n_comp = len([m for m in models if m != CHAMPION])
    agg = g[METRICS].agg(["mean", "std"]).round(4)
    champion_auc = df[df.model == CHAMPION].sort_values("fold_idx")["auc"].values
    rows = []
    for model in models:
        cells = []
        for m in METRICS:
            mu, sd = agg.loc[model, (m, "mean")], agg.loc[model, (m, "std")]
            cells.append(f"{mu:.4f} ± {sd:.4f}")
        if model == CHAMPION:
            tail = "reference"
        else:
            ch_auc = df[df.model == model].sort_values("fold_idx")["auc"].values
            try:
                _, p_w = wilcoxon_paired(champion_auc, ch_auc)
            except ValueError:
                p_w = 1.0
            p_bonf = bonferroni(p_w, n_comp)
            tail = f"Wilcoxon p={p_w:.3f} (bonf={p_bonf:.3f})"
        rows.append(f"| {model} | " + " | ".join(cells) + f" | {tail} |")
    header = "| Model | " + " | ".join(m.upper() for m in METRICS) + " | vs Champion |"
    sep = "|" + "|".join(["---"] * (len(METRICS) + 2)) + "|"
    return "\n".join(["## Headline (10-fold CV)", "", header, sep, *rows, "", CONTEXT])

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Pass**

```bash
cd comparison && uv run pytest tests/test_summary.py -v
```

- [ ] **Step 5: Generate summaries**

```bash
cd comparison && uv run python -m src.summary
```
Expected: writes `results/summary_defaults.md` and `results/summary_tuned.md`.

- [ ] **Step 6: Commit**

```bash
cd .. && git add comparison/src/summary.py comparison/tests/test_summary.py \
                comparison/results/summary_defaults.md comparison/results/summary_tuned.md
git commit -m "feat(comparison): summary markdown with Wilcoxon paired test"
```

---

## Task 21: Persist per-fold predictions (enables DeLong in analysis notebook)

**Files:**
- Modify: `comparison/src/runner.py`

The summary uses Wilcoxon (only needs per-fold AUCs). For DeLong we need the actual probabilities. Add a sidecar parquet file.

- [ ] **Step 1: Modify runner to also store predictions**

Edit `comparison/src/runner.py`. In `run_pass`, after the inner loop's `append_row`, also save predictions. Replace the inner `for fold_i` block with:

```python
        per_fold_preds = {}
        for fold_i, (tr, te) in enumerate(tqdm(folds, desc=model.name)):
            row = run_one_fold(model, X_use, y, tr, te, meta)
            row["fold_idx"] = fold_i
            row["pass"] = pass_name
            append_row(out_path, row)
            # capture probabilities for this fold
            proba = model.fit_predict(X_use.iloc[tr], y.iloc[tr], X_use.iloc[te], meta)
            per_fold_preds[fold_i] = pd.DataFrame({
                "test_idx": te, "y": y.iloc[te].values, "proba": proba,
            })
        all_preds = pd.concat([df.assign(fold_idx=i) for i, df in per_fold_preds.items()])
        all_preds["model"] = model.name
        pred_path = RESULTS_DIR / f"predictions_{pass_name}.parquet"
        if pred_path.exists():
            existing = pd.read_parquet(pred_path)
            all_preds = pd.concat([existing, all_preds], ignore_index=True)
        all_preds.to_parquet(pred_path, index=False)
```
Note: this calls `fit_predict` twice per fold (once for metrics, once for stored proba). That doubles runtime. Better optimisation: refactor `run_one_fold` to return both metrics and proba. Do that instead:

Replace `run_one_fold` with:
```python
def run_one_fold(model, X, y, train_idx, test_idx, meta) -> tuple[dict, np.ndarray]:
    t0 = time.perf_counter()
    proba = model.fit_predict(X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx], meta)
    runtime = time.perf_counter() - t0
    m = compute_all(y.iloc[test_idx].values, proba)
    m.update(model=model.name, n_train=len(train_idx), n_test=len(test_idx),
             runtime_s=runtime, fold_idx=-1)
    return m, proba
```

And in `run_pass`:
```python
        per_fold_preds = []
        for fold_i, (tr, te) in enumerate(tqdm(folds, desc=model.name)):
            row, proba = run_one_fold(model, X_use, y, tr, te, meta)
            row["fold_idx"] = fold_i
            row["pass"] = pass_name
            append_row(out_path, row)
            per_fold_preds.append(pd.DataFrame({
                "test_idx": te, "y": y.iloc[te].values, "proba": proba,
                "fold_idx": fold_i, "model": model.name,
            }))
        all_preds = pd.concat(per_fold_preds, ignore_index=True)
        pred_path = RESULTS_DIR / f"predictions_{pass_name}.parquet"
        if pred_path.exists():
            existing = pd.read_parquet(pred_path)
            all_preds = pd.concat([existing, all_preds], ignore_index=True)
        all_preds.to_parquet(pred_path, index=False)
```

Also update `test_runner.py::test_run_one_fold_returns_metrics` — `run_one_fold` now returns a tuple:
```python
    out, proba = run_one_fold(_M(), X, y, np.arange(80), np.arange(80, 100), meta={})
    assert "auc" in out and "logloss" in out
    assert proba.shape == (20,)
```

- [ ] **Step 2: Pass tests**

```bash
cd comparison && uv run pytest tests/test_runner.py -v
```

- [ ] **Step 3: Re-run both passes**

```bash
cd comparison && rm -f results/predictions_*.parquet results/per_fold_*.csv
uv run python -m src.runner --pass defaults
uv run python -m src.runner --pass tuned
```

- [ ] **Step 4: Regenerate summaries**

```bash
cd comparison && uv run python -m src.summary
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/runner.py comparison/tests/test_runner.py \
                comparison/results/per_fold_defaults.csv comparison/results/per_fold_tuned.csv \
                comparison/results/predictions_defaults.parquet \
                comparison/results/predictions_tuned.parquet \
                comparison/results/summary_defaults.md comparison/results/summary_tuned.md
git commit -m "feat(comparison): persist per-fold predictions for paired tests"
```

---

## Task 22: Analysis notebook (ROC overlay, reliability, boxplot, forest plot, runtime)

**Files:**
- Create: `comparison/notebooks/01_results_analysis.ipynb`

- [ ] **Step 1: Build the notebook**

Create `comparison/notebooks/01_results_analysis.ipynb` with the following cells (write as `.py` first then `jupytext --to notebook`, or build the JSON directly). Cell-by-cell content:

**Cell 1 — imports:**
```python
%matplotlib inline
import sys; sys.path.insert(0, "..")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve
from src.metrics import delong_test, wilcoxon_paired, bonferroni
RES = Path("../results")
PASS = "defaults"   # change to "tuned" and re-run cells for second pass
df = pd.read_csv(RES / f"per_fold_{PASS}.csv")
preds = pd.read_parquet(RES / f"predictions_{PASS}.parquet")
models = list(df["model"].unique())
CHAMPION = "woe_logit"
```

**Cell 2 — ROC overlay (concatenated across folds):**
```python
fig, ax = plt.subplots(figsize=(7, 7))
for m in models:
    sub = preds[preds.model == m]
    fpr, tpr, _ = roc_curve(sub["y"], sub["proba"])
    ax.plot(fpr, tpr, label=m, lw=1.5)
ax.plot([0, 1], [0, 1], "k--", lw=0.5)
ax.set(xlabel="FPR", ylabel="TPR", title=f"ROC overlay (10-fold concat, {PASS})")
ax.legend(loc="lower right", fontsize=8)
fig.savefig(RES / "figures" / f"roc_overlay_{PASS}.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Cell 3 — Reliability curves:**
```python
fig, ax = plt.subplots(figsize=(7, 7))
for m in models:
    sub = preds[preds.model == m]
    prob_true, prob_pred = calibration_curve(sub["y"], sub["proba"], n_bins=10, strategy="quantile")
    ax.plot(prob_pred, prob_true, marker="o", label=m, lw=1.2)
ax.plot([0, 1], [0, 1], "k--", lw=0.5)
ax.set(xlabel="Predicted P(default)", ylabel="Observed P(default)", title=f"Reliability ({PASS})")
ax.legend(fontsize=8)
fig.savefig(RES / "figures" / f"reliability_{PASS}.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Cell 4 — AUC boxplot across folds:**
```python
fig, ax = plt.subplots(figsize=(8, 5))
data = [df[df.model == m]["auc"].values for m in models]
ax.boxplot(data, labels=models, vert=True)
ax.set(ylabel="AUC", title=f"Per-fold AUC distribution ({PASS})")
plt.xticks(rotation=30, ha="right")
fig.savefig(RES / "figures" / f"auc_boxplot_{PASS}.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Cell 5 — Forest plot of paired diffs vs champion (with DeLong CI):**
```python
ch = preds[preds.model == CHAMPION].sort_values(["fold_idx", "test_idx"])
y_all = ch["y"].values
p_ch = ch["proba"].values

rows = []
for m in models:
    if m == CHAMPION: continue
    sub = preds[preds.model == m].sort_values(["fold_idx", "test_idx"])
    p_m = sub["proba"].values
    z, p = delong_test(y_all, p_m, p_ch)   # z>0 ⇒ challenger > champion
    # rough 95% CI on the AUC diff: invert z = diff / se → se = diff/z, then ±1.96·se
    from sklearn.metrics import roc_auc_score
    diff = roc_auc_score(y_all, p_m) - roc_auc_score(y_all, p_ch)
    se = abs(diff / z) if z != 0 else 0.05
    rows.append((m, diff, diff - 1.96 * se, diff + 1.96 * se, p))
fdf = pd.DataFrame(rows, columns=["model", "diff", "lo", "hi", "p"])

fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(fdf))))
ax.errorbar(fdf["diff"], range(len(fdf)),
            xerr=[fdf["diff"] - fdf["lo"], fdf["hi"] - fdf["diff"]],
            fmt="o", color="black")
ax.axvline(0, color="grey", ls="--")
ax.set_yticks(range(len(fdf)))
ax.set_yticklabels([f"{r.model} (p={r.p:.3f})" for _, r in fdf.iterrows()])
ax.set_xlabel("AUC difference vs woe_logit champion")
ax.set_title(f"Forest plot of paired AUC diffs ({PASS})")
fig.savefig(RES / "figures" / f"forest_{PASS}.png", dpi=150, bbox_inches="tight")
plt.show()
fdf
```

**Cell 6 — Runtime table:**
```python
rt = df.groupby("model")["runtime_s"].agg(["mean", "sum"]).round(2)
rt.columns = ["mean_s_per_fold", "total_s"]
rt.sort_values("total_s", ascending=False)
```

**Cell 7 — Save final markdown summary with both Wilcoxon and DeLong p-values:**
```python
n_comp = len(models) - 1
lines = ["## Paired tests vs woe_logit", "",
         "| Model | DeLong p | DeLong p (bonf) | Wilcoxon p | Wilcoxon p (bonf) |",
         "|---|---|---|---|---|"]
ch_aucs = df[df.model == CHAMPION].sort_values("fold_idx")["auc"].values
for m in models:
    if m == CHAMPION: continue
    sub = preds[preds.model == m].sort_values(["fold_idx", "test_idx"])
    _, pd_ = delong_test(y_all, sub["proba"].values, p_ch)
    ch_m_aucs = df[df.model == m].sort_values("fold_idx")["auc"].values
    try:
        _, pw = wilcoxon_paired(ch_aucs, ch_m_aucs)
    except ValueError:
        pw = 1.0
    lines.append(f"| {m} | {pd_:.3f} | {bonferroni(pd_, n_comp):.3f} | {pw:.3f} | {bonferroni(pw, n_comp):.3f} |")
out = RES / f"paired_tests_{PASS}.md"
out.write_text("\n".join(lines))
print(f"wrote {out}")
```

- [ ] **Step 2: Execute notebook for both passes**

```bash
cd comparison && uv run jupyter nbconvert --to notebook --execute notebooks/01_results_analysis.ipynb --inplace
# Edit cell 1 (PASS = "tuned") and re-execute, OR use papermill:
uv run python -m pip install papermill   # one-off, optional
```
Or simpler: duplicate the notebook to `02_results_analysis_tuned.ipynb`, change cell 1's `PASS = "tuned"`, execute both:
```bash
cd comparison
cp notebooks/01_results_analysis.ipynb notebooks/02_results_analysis_tuned.ipynb
# manually edit cell 1 of 02_..._tuned.ipynb so PASS="tuned" (sed-friendly:
sed -i '' 's/PASS = "defaults"/PASS = "tuned"/' notebooks/02_results_analysis_tuned.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/01_results_analysis.ipynb --inplace
uv run jupyter nbconvert --to notebook --execute notebooks/02_results_analysis_tuned.ipynb --inplace
```

- [ ] **Step 3: Verify outputs**

```bash
ls comparison/results/figures/
```
Expected files: `roc_overlay_defaults.png`, `roc_overlay_tuned.png`, `reliability_defaults.png`, `reliability_tuned.png`, `auc_boxplot_defaults.png`, `auc_boxplot_tuned.png`, `forest_defaults.png`, `forest_tuned.png`, plus `paired_tests_defaults.md` and `paired_tests_tuned.md` in `results/`.

- [ ] **Step 4: Commit**

```bash
cd .. && git add comparison/notebooks/01_results_analysis.ipynb \
                comparison/notebooks/02_results_analysis_tuned.ipynb \
                comparison/results/figures/ \
                comparison/results/paired_tests_defaults.md \
                comparison/results/paired_tests_tuned.md
git commit -m "feat(comparison): analysis notebook with ROC/reliability/boxplot/forest"
```

---

## Task 23: Final benchmark orchestration notebook

**Files:**
- Create: `comparison/notebooks/benchmark.ipynb`

A thin "run the whole thing" notebook for the thesis appendix. The runner already does the heavy lifting; this notebook is a documented entry-point.

- [ ] **Step 1: Build the notebook**

Create `comparison/notebooks/benchmark.ipynb` with these cells:

**Cell 1 — markdown:**
```markdown
# TFM vs WoE-Logit Benchmark

Reproducible entry-point for the comparison defined in
[2026-05-22-tfm-comparison-design.md](../../docs/superpowers/specs/2026-05-22-tfm-comparison-design.md).

Runs both passes and renders the summary tables inline.
```

**Cell 2 — env check:**
```python
%matplotlib inline
import sys; sys.path.insert(0, "..")
from pathlib import Path
import pandas as pd, numpy as np
RES = Path("../results")
print("Python:", sys.version.split()[0])
import torch; print("torch:", torch.__version__, "MPS:", torch.backends.mps.is_available())
```

**Cell 3 — markdown:**
```markdown
## Pass 1 — defaults
Run from a terminal (slow, do not block the kernel):

```bash
cd comparison && uv run python -m src.runner --pass defaults
```
```

**Cell 4 — load and render pass 1 summary:**
```python
df1 = pd.read_csv(RES / "per_fold_defaults.csv")
df1.groupby("model")[["auc","gini","ks","brier","logloss","runtime_s"]].agg(["mean","std"]).round(4)
```

**Cell 5 — markdown:**
```markdown
## Pass 2 — tuned classicals
```bash
cd comparison && uv run python -m src.runner --pass tuned
```
```

**Cell 6 — load and render pass 2 summary:**
```python
df2 = pd.read_csv(RES / "per_fold_tuned.csv")
df2.groupby("model")[["auc","gini","ks","brier","logloss","runtime_s"]].agg(["mean","std"]).round(4)
```

**Cell 7 — markdown:**
```markdown
## Headline comparison
```

**Cell 8 — defaults vs tuned delta:**
```python
a = df1.groupby("model")["auc"].mean()
b = df2.groupby("model")["auc"].mean()
pd.DataFrame({"defaults": a, "tuned": b, "delta": b - a}).round(4).sort_values("tuned", ascending=False)
```

**Cell 9 — display summary MDs inline:**
```python
from IPython.display import Markdown, display
display(Markdown((RES / "summary_defaults.md").read_text()))
display(Markdown((RES / "summary_tuned.md").read_text()))
display(Markdown((RES / "paired_tests_defaults.md").read_text()))
display(Markdown((RES / "paired_tests_tuned.md").read_text()))
```

- [ ] **Step 2: Execute (assuming runner already produced CSVs)**

```bash
cd comparison && uv run jupyter nbconvert --to notebook --execute notebooks/benchmark.ipynb --inplace
```

- [ ] **Step 3: Commit**

```bash
cd .. && git add comparison/notebooks/benchmark.ipynb
git commit -m "feat(comparison): top-level benchmark notebook"
```

---

## Task 24: Final acceptance check

- [ ] **Step 1: Clean test run**

```bash
cd comparison && uv run pytest -v
```
Expected: all non-slow tests pass. Slow tests skipped without `-m slow`.

- [ ] **Step 2: Slow test pass (TFMs)**

```bash
cd comparison && uv run pytest -v -m slow
```
Expected: all 4 TFM wrapper smoke tests pass.

- [ ] **Step 3: WoE sanity reconfirmation**

```bash
cd comparison && uv run pytest tests/test_woe_logit.py::test_full_sample_refit_matches_report_auc -v
```
Expected: PASS, AUC in [0.80, 0.82].

- [ ] **Step 4: Artefact inventory**

```bash
ls -la comparison/results/ comparison/results/figures/
```
Expected files exist:
- `per_fold_defaults.csv` (80 rows)
- `per_fold_tuned.csv` (80 rows)
- `predictions_defaults.parquet`
- `predictions_tuned.parquet`
- `summary_defaults.md`
- `summary_tuned.md`
- `paired_tests_defaults.md`
- `paired_tests_tuned.md`
- 8 figures under `figures/`

- [ ] **Step 5: Tag**

```bash
cd .. && git tag -a comparison-v1 -m "TFM vs WoE-logit comparison complete"
```

- [ ] **Step 6: Final commit (if anything outstanding)**

```bash
git status
# if clean: nothing to do
```

---

## Self-review checklist (run after writing this plan)

- [x] Spec coverage: §1 goal → all tasks; §3 lineup → 8 wrappers in Tasks 8–15; §4 protocol → Task 3 (CV) + Task 17 (Optuna) + Tasks 18, 19 (execution); §5 metrics & tests → Tasks 4, 5, 6 + Task 22 (forest, DeLong); §6 architecture → directory in Task 1; §7 interface → Task 7; §8 woe_logit faithfulness → Task 8 incl. sanity test; §9 environment → Task 1; §10 reproducibility → seeds in Tasks 3, 8–17; §11 artefacts → Tasks 20, 22; §13 acceptance → Task 24.
- [x] No placeholders: every code step contains complete code.
- [x] Type consistency: `PDModel.fit_predict` signature identical across base.py, every wrapper, and runner.
- [x] Frequent commits: 22 commits across the plan.
- [x] TDD: every implementation task has failing-test → pass-test sequence.
