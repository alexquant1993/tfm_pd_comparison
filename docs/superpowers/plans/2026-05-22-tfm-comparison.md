# TFM vs WoE-Logit PD Comparison — Implementation Plan (v2, post-Codex revision)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible 10-fold CV benchmark in `comparison/` that pits 4 tabular foundation models (TabPFN v2, TabICL, TabDPT, CARTE) against a fold-safe re-implementation of the `pd-autopilot` WoE-logit champion plus three classical baselines (XGB, LightGBM, plain logit) on German Credit (n=1,000), with paired DeLong + Wilcoxon tests and two passes (defaults + properly nested Optuna tuning for classicals only).

**Architecture:** Sibling directory `comparison/` with `uv`-managed environment. Thin notebook orchestrates a per-model wrapper for each of 8 models behind a uniform `PDModel` protocol. Shared modules for fold-safe preprocessing, fold creation, metrics, and tuning. Per-fold CSV + predictions parquet written row-by-row from the very first runner version (crash-safe). Tests under `tests/` use `pytest`.

**Tech Stack:** `uv` (env mgmt), Python ≥ 3.10, pandas, numpy, scikit-learn, scipy, matplotlib, optbinning, xgboost, lightgbm, tabpfn, tabicl, tabdpt, carte-ai, optuna, jupyter, tqdm, pytest, pyarrow.

**Spec reference:** `docs/superpowers/specs/2026-05-22-tfm-comparison-design.md`

**Changes from v1 of this plan (Codex round-1, see commit `842af30`):**
- New Task 3: `preprocessing.py` with `fold_safe_iqr_cap`.
- Task 2 (`data_loader`) now fixes the `_parse_special_codes` int/float/NaN bug and adds an explicit special-codes test.
- Task 9 (`woe_logit`): switches to `LogisticRegression(penalty=None)`; hard-codes monotonic trends per variable; adds CP→MIP solver fallback; the sanity test is now coefficient-level vs `model_params.json` (±0.05) instead of an AUC band.
- Task 17 (`runner`): applies `fold_safe_iqr_cap` inside the fold loop and persists predictions parquet from v1 (no retrofit). Outer loop reordered to fold-first/model-second so per-fold preprocessing happens once.
- Task 18 (`tuning`): rewritten as `tune_classicals_for_fold(X_train_outer, y_train_outer, types_meta, seed)` invoked inside the runner's outer-fold loop; inner 5-fold CV lives entirely inside the outer-training fold. The single-shot `tune_xgb()` / `tune_lgbm()` / `tune_logit()` API is gone.
- Task 21 (`summary`): computes DeLong + Wilcoxon from the predictions parquet (both already exist when summary runs); narrow-claim disclaimer rendered into every `summary_*.md` header.
- v1 Task 21 (retrofit predictions persistence) deleted — rolled into v2 Task 17.
- Acceptance criteria expanded to 10 items, each tied to a named test or artefact.

**Additional changes (Codex round-2):**
- **CRITICAL bug fix in Task 17 runner:** old `run_one_fold_one_model` accepted full-dataset `(X, y, train_idx, test_idx)` and called `y.iloc[train_idx]` with locally-built `np.arange(...)` indices, silently scoring against the first N rows of the full `y` rather than the fold's labels. Every metric after fold 0 would have been invalid. v2 takes already-sliced `(X_train, y_train, X_test, y_test)` and a regression test (`test_run_one_fold_passes_correct_labels`) guards against recurrence.
- **Crash-safe parquet** (Task 17): predictions are now written as per-fold parts to `results/predictions_<pass>_parts/fold_NN.parquet` immediately after each fold completes, then combined into the canonical parquet at the end. A mid-pass crash leaves completed folds intact and recoverable.
- **Strict nested inner cap** (Task 18 + spec §4): `tune_classicals_for_fold` now receives the **uncapped** outer-training fold and recomputes `fold_safe_iqr_cap` inside each inner CV split — so inner-validation rows never influence the cap thresholds they are scored against. A new behavioural test `test_inner_cap_uses_inner_train_only` verifies this end-to-end.
- **Brier / log-loss relabelled as secondary** (Task 21 + spec §5): they are calibration-sensitive proper scoring rules, not pure discrimination metrics. Headline tables now visually group them as "Brier (sec.)" / "LOGLOSS (sec.)" with a legend, and the disclaimer mentions the distinction.
- **Shell-precedence fix** (Task 1): `cd comparison 2>/dev/null || mkdir comparison && cd comparison` → `mkdir -p comparison && cd comparison`.

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
mkdir -p comparison && cd comparison
uv init --python 3.12 --no-readme --no-workspace
```
Creates `pyproject.toml`, `.python-version`, and downloads CPython 3.12 into uv's cache.

- [ ] **Step 3: Add dependencies (core first, ML libs second)**

```bash
uv add pandas numpy scikit-learn scipy matplotlib tqdm pyarrow
uv add optbinning xgboost lightgbm optuna
uv add jupyter ipykernel
uv add --dev pytest
```
`pyarrow` is added from the start because the runner (Task 17) persists predictions parquet from its first version — predictions persistence is a first-class spec requirement, not a retrofit. TFM packages (`tabpfn`, `tabicl`, `tabdpt`, `carte-ai`) are added in their respective wrapper tasks because each may need a GitHub fallback.

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

**Scope (narrow):** Pure discrimination only. Does NOT support the claim that
"TFMs are better PD models" — that would require calibration, rating grades,
PSI, and OOT validation, all out of scope here.

## Setup

```bash
brew install uv      # one-time
uv sync              # creates .venv from uv.lock
```

## Run

```bash
# Always use `uv run` so you get this project's venv, not your system/conda Python.
uv run pytest -m "not slow"          # fast tests
uv run pytest -m slow                # TFM smoke tests (downloads weights ~1 GB)
uv run python -m src.runner --pass defaults
uv run python -m src.runner --pass tuned
uv run python -m src.summary         # build summary_*.md from CSVs + parquets
uv run jupyter lab notebooks/        # interactive analysis
```

## Outputs

- `results/per_fold_defaults.csv`, `results/per_fold_tuned.csv` — metrics × fold × model
- `results/predictions_defaults.parquet`, `results/predictions_tuned.parquet` — OOF predictions
- `results/summary_defaults.md`, `results/summary_tuned.md` — paired tests, mean ± std tables
- `results/figures/` — ROC overlays, reliability curves, AUC boxplots, forest plots
````

- [ ] **Step 7: Verify env works**

```bash
cd comparison
uv run python -c "import pandas, numpy, sklearn, optbinning, xgboost, lightgbm, optuna, pyarrow; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 8: Commit**

```bash
cd ..
git add comparison/pyproject.toml comparison/uv.lock comparison/.python-version \
        comparison/.gitignore comparison/README.md \
        comparison/src/__init__.py comparison/src/models/__init__.py \
        comparison/tests/__init__.py \
        comparison/notebooks/.gitkeep comparison/results/figures/.gitkeep
git commit -m "feat(comparison): bootstrap uv project with core deps + pyarrow"
```

---

## Task 2: Data loader (with fixed special-codes parsing)

**Files:**
- Create: `comparison/src/data_loader.py`
- Create: `comparison/tests/test_data_loader.py`

The critical fix vs v1 of this plan: `_parse_special_codes` must accept int, float, NaN, and "4,5"-style strings uniformly. The CSV stores values like `4` (int), `4,5` (string), `NaN` (empty); pandas reads the column as `float64` if all-numeric-or-empty, or `object` if any cell has a comma.

- [ ] **Step 1: Write the failing test**

Create `comparison/tests/test_data_loader.py`:
```python
from pathlib import Path
import pandas as pd
import numpy as np
from src.data_loader import load, carte_decode, _parse_special_codes

REPO_ROOT = Path(__file__).resolve().parents[2]

def test_load_shape_and_target():
    X, y, meta = load(repo_root=REPO_ROOT)
    assert X.shape == (1000, 20)
    assert y.shape == (1000,)
    assert set(y.unique()) == {0, 1}
    assert abs(y.mean() - 0.30) < 0.01
    assert "Creditability" not in X.columns

def test_meta_has_all_columns():
    X, y, meta = load(repo_root=REPO_ROOT)
    assert set(meta.keys()) == set(X.columns)
    for col, info in meta.items():
        assert info["type"] in {"nominal", "ordinal", "continuous"}
        assert info["dtype"] in {"numerical", "categorical"}

def test_parse_special_codes_handles_all_types():
    # The bug v1 had: only str was handled.
    assert _parse_special_codes("4") == [4]
    assert _parse_special_codes("4,5") == [4, 5]
    assert _parse_special_codes(4) == [4]
    assert _parse_special_codes(4.0) == [4]
    assert _parse_special_codes(np.nan) == []
    assert _parse_special_codes(None) == []
    assert _parse_special_codes("") == []

def test_special_codes_actually_loaded_for_three_known_vars():
    X, y, meta = load(repo_root=REPO_ROOT)
    assert meta["Account Balance"]["special_codes"] == [4]
    assert meta["Value Savings/Stocks"]["special_codes"] == [5]
    assert meta["Most valuable available asset"]["special_codes"] == [4]
    # And vars without special codes report []
    assert meta["Purpose"]["special_codes"] == []

def test_carte_decode_replaces_purpose():
    X, y, meta = load(repo_root=REPO_ROOT)
    X_str = carte_decode(X, meta)
    assert X_str["Purpose"].dtype == object
    assert not (X_str["Purpose"] == 3).any()
    assert "Radio/TV" in X_str["Purpose"].astype(str).str.cat(sep="|")

def test_carte_decode_keeps_continuous_numeric():
    X, y, meta = load(repo_root=REPO_ROOT)
    X_str = carte_decode(X, meta)
    assert pd.api.types.is_numeric_dtype(X_str["Age (years)"])
    assert pd.api.types.is_numeric_dtype(X_str["Credit Amount"])
```

- [ ] **Step 2: Run, confirm failure**

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
import math
from pathlib import Path
import pandas as pd
import numpy as np

TARGET = "Creditability"

def _parse_encoding(encoding_str) -> dict[int, str]:
    """Parse the Encoding column into {int_code: human_label}.

    Format examples (raw CSV cells):
      '1=<0 DM (A11), 2=0-200 DM (A12), 3=>=200 DM/salary (A13). Special: 4=No checking account (A14)'
      '0=New car, 1=Used car, 2=Furniture, 3=Radio/TV, ...'
    """
    if not isinstance(encoding_str, str):
        return {}
    mapping: dict[int, str] = {}
    cleaned = re.sub(r"\([^)]*\)", "", encoding_str)
    cleaned = cleaned.replace("Special:", ",")
    for part in re.split(r"[,.]", cleaned):
        m = re.match(r"\s*(-?\d+)\s*=\s*(.+?)\s*$", part)
        if m:
            mapping[int(m.group(1))] = m.group(2).strip()
    return mapping

def _parse_special_codes(s) -> list[int]:
    """Parse a SpecialCodes cell that may arrive as int, float, NaN, str, or None.

    Examples:
      "4"     → [4]
      "4,5"   → [4, 5]
      4       → [4]
      4.0     → [4]
      NaN     → []
      None    → []
      ""      → []
    """
    if s is None:
        return []
    # NaN check (works for float NaN)
    if isinstance(s, float) and math.isnan(s):
        return []
    # Integer cell from pandas
    if isinstance(s, (int, np.integer)):
        return [int(s)]
    # Float cell from pandas (most common when SpecialCodes col is read as float64)
    if isinstance(s, float):
        return [int(s)]
    # String cell (including comma-separated like "4,5")
    if isinstance(s, str):
        s = s.strip()
        if not s:
            return []
        return [int(c.strip()) for c in s.split(",") if c.strip()]
    return []

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
            "special_codes": _parse_special_codes(row.get("SpecialCodes")),
            "encoding": _parse_encoding(row.get("Encoding")),
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

- [ ] **Step 4: Run tests, confirm all pass**

```bash
cd comparison && uv run pytest tests/test_data_loader.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/data_loader.py comparison/tests/test_data_loader.py
git commit -m "feat(comparison): data loader with int/float/NaN special-codes parsing"
```

---

## Task 3: Fold-safe preprocessing (NEW)

**Files:**
- Create: `comparison/src/preprocessing.py`
- Create: `comparison/tests/test_preprocessing.py`

`fold_safe_iqr_cap` computes Q1, Q3, IQR, and the upper cap (`Q3 + 1.5·IQR`) from the outer-training fold only, then applies the cap to both outer-training and outer-test. This mirrors Stage 02's recipe but eliminates the leakage caused by computing caps on all 1,000 rows.

- [ ] **Step 1: Write the failing test**

Create `comparison/tests/test_preprocessing.py`:
```python
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
        "Credit Amount": np.arange(100, 141, dtype=float).tolist() + [1.0],
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
```

- [ ] **Step 2: Run, confirm fail**

```bash
cd comparison && uv run pytest tests/test_preprocessing.py -v
```

- [ ] **Step 3: Implement `preprocessing.py`**

Create `comparison/src/preprocessing.py`:
```python
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
```

- [ ] **Step 4: Run, confirm pass**

```bash
cd comparison && uv run pytest tests/test_preprocessing.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/preprocessing.py comparison/tests/test_preprocessing.py
git commit -m "feat(comparison): fold-safe IQR upper-tail capping"
```

---

## Task 4: Cross-validation folds

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
        assert 0.25 <= rate <= 0.35

def test_make_folds_deterministic():
    y = pd.Series([0, 1] * 500)
    folds_a = make_folds(y, n_splits=10, seed=42)
    folds_b = make_folds(y, n_splits=10, seed=42)
    for (tra, tea), (trb, teb) in zip(folds_a, folds_b):
        assert np.array_equal(tra, trb)
        assert np.array_equal(tea, teb)
```

- [ ] **Step 2: Fail**

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

- [ ] **Step 4: Pass**

```bash
cd comparison && uv run pytest tests/test_cv.py -v
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/cv.py comparison/tests/test_cv.py
git commit -m "feat(comparison): stratified 10-fold CV with frozen seed"
```

---

## Task 5: Core metrics (AUC, Gini, KS, Brier, log-loss)

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
    p = np.array([1.0, 0.0])
    m = compute_all(y, p)
    assert np.isfinite(m["logloss"])
```

- [ ] **Step 2: Fail**

```bash
cd comparison && uv run pytest tests/test_metrics.py -v
```

- [ ] **Step 3: Implement**

Create `comparison/src/metrics.py`:
```python
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
```

- [ ] **Step 4: Pass**

```bash
cd comparison && uv run pytest tests/test_metrics.py -v
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/metrics.py comparison/tests/test_metrics.py
git commit -m "feat(comparison): core discrimination metrics with clipping"
```

---

## Task 6: DeLong paired test for AUC

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
    p_bad = rng.random(500)
    p_good = y + rng.normal(0, 0.3, 500)
    z, pval = delong_test(y, p_bad, p_good)
    assert pval < 0.01
```

- [ ] **Step 2: Fail**

```bash
cd comparison && uv run pytest tests/test_metrics.py::test_delong_identical_returns_high_p -v
```

- [ ] **Step 3: Implement DeLong (Sun & Xu 2014)**

Append to `comparison/src/metrics.py`:
```python
from scipy import stats

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
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
    order = (-y_true).argsort(kind="stable")
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

- [ ] **Step 4: Pass**

```bash
cd comparison && uv run pytest tests/test_metrics.py -v
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/metrics.py comparison/tests/test_metrics.py
git commit -m "feat(comparison): DeLong paired AUC test (Sun & Xu 2014)"
```

---

## Task 7: Wilcoxon + Bonferroni helpers

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
    b = a + 0.05
    stat, p = wilcoxon_paired(a, b)
    assert p < 0.05

def test_bonferroni_clips_to_one():
    assert bonferroni(0.5, 5) == 1.0
    assert bonferroni(0.01, 7) == 0.07
```

- [ ] **Step 2: Fail**

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

- [ ] **Step 4: Pass**

```bash
cd comparison && uv run pytest tests/test_metrics.py -v
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/metrics.py comparison/tests/test_metrics.py
git commit -m "feat(comparison): Wilcoxon paired test + Bonferroni helper"
```

---

## Task 8: PDModel protocol

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

- [ ] **Step 2: Fail**

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
    """Uniform interface every model wrapper implements.

    The runner applies fold-safe preprocessing BEFORE calling fit_predict, so
    X_train and X_test arrive already capped. Wrappers should not re-cap.
    """
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

- [ ] **Step 4: Pass**

```bash
cd comparison && uv run pytest tests/test_base.py -v
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/models/base.py comparison/tests/test_base.py
git commit -m "feat(comparison): PDModel protocol"
```

---

## Task 9: WoE-logit champion wrapper (coefficient-faithful to report)

**Files:**
- Create: `comparison/src/models/woe_logit.py`
- Create: `comparison/tests/test_woe_logit.py`

Spec §8 requires: `penalty=None` LR, explicit monotonic trends per variable, CP→MIP solver fallback, coefficient-level sanity vs `model_params.json` within ±0.05.

- [ ] **Step 1: Write the failing tests**

Create `comparison/tests/test_woe_logit.py`:
```python
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

def test_special_codes_isolated():
    """After fitting, the three special-coded variables must have a dedicated
    bin for their special value. Verifies _parse_special_codes wiring is wired
    through to OptimalBinning."""
    X, y, meta = load(repo_root=REPO_ROOT)
    X_tr, _, _ = fold_safe_iqr_cap(X, X.iloc[:1], meta)
    m = WoELogitChampion()
    m.fit_predict(X_tr, y, X_tr.iloc[:1], meta, seed=42)
    # After fit_predict, binners are stored on the instance
    for col, special in [("Account Balance", 4),
                         ("Value Savings/Stocks", 5),
                         ("Most valuable available asset", 4)]:
        binner = m._binners[col]
        # OptimalBinning exposes the special bin via the splits object
        # The simplest check: there exists a bin label corresponding to the special code
        bin_table = binner.binning_table.build()
        assert any("Special" in str(row) for row in bin_table["Bin"].astype(str)), \
            f"{col} missing Special bin (special_code={special})"

def test_full_sample_refit_matches_report_coefficients():
    """GATING TEST: every coefficient (intercept + 8 slopes) must be within
    absolute tolerance 0.05 of the report's model_params.json."""
    if not MODEL_PARAMS.exists():
        pytest.skip(f"missing {MODEL_PARAMS}")
    report = json.loads(MODEL_PARAMS.read_text())
    # The report's model_params.json should contain a 'coefficients' or similar key.
    # If the structure differs, adapt this loader. Expected schema:
    #   {"intercept": -0.8475, "coefficients": {"Account Balance": -0.7848, ...}}
    expected_intercept = report.get("intercept")
    expected_coefs = report.get("coefficients") or report.get("coef")
    if expected_intercept is None or expected_coefs is None:
        pytest.skip("model_params.json does not contain intercept/coefficients in expected schema")

    X, y, meta = load(repo_root=REPO_ROOT)
    # Apply the same fold-safe cap recipe to the full sample (treating the full
    # 1000 rows as if they were the training fold) to mirror what the runner does.
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
```

- [ ] **Step 2: Fail**

```bash
cd comparison && uv run pytest tests/test_woe_logit.py -v
```

- [ ] **Step 3: Implement**

Create `comparison/src/models/woe_logit.py`:
```python
"""WoE + unregularised Logit champion — faithful to pd-autopilot stage 04a (MIV).

Spec §8 requirements honoured here:
- penalty=None (matches report's statsmodels-style logit)
- Explicit monotonic trends per variable (no "auto")
- CP solver with try/except fallback to MIP
- 8 champion variables fixed at the class level
"""
from __future__ import annotations
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
        special_codes=info["special_codes"] or None,
    )
    # Try CP first; on failure (some ordinal-ascending edge cases) fall back to MIP.
    try:
        ob = OptimalBinning(solver="cp", **common)
        ob.fit(X_tr[col].values, y_tr.values)
    except Exception:
        ob = OptimalBinning(solver="mip", **common)
        ob.fit(X_tr[col].values, y_tr.values)
    return ob

def _fit_binning(X_tr: pd.DataFrame, y_tr: pd.Series, meta: dict) -> dict[str, OptimalBinning]:
    return {col: _fit_one_binning(col, X_tr, y_tr, meta) for col in CHAMPION_VARS}

def _transform(binners: dict[str, OptimalBinning], X: pd.DataFrame) -> np.ndarray:
    cols = [binners[c].transform(X[c].values, metric="woe") for c in CHAMPION_VARS]
    return np.column_stack(cols)

class WoELogitChampion:
    name = "woe_logit"
    requires_string_labels = False

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        self._binners = _fit_binning(X_train, y_train, types_meta)
        Z_tr = _transform(self._binners, X_train)
        Z_te = _transform(self._binners, X_test)
        # penalty=None ⇒ unregularised, matches the report's coefficients
        self._lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000, random_state=seed)
        self._lr.fit(Z_tr, y_train.values)
        return self._lr.predict_proba(Z_te)[:, 1]
```

- [ ] **Step 4: Run and pass**

```bash
cd comparison && uv run pytest tests/test_woe_logit.py -v
```

If `test_full_sample_refit_matches_report_coefficients` fails or skips because `model_params.json` does not store coefficients in the expected schema: **stop**, open `runs/2026-03-17_071354/pipeline/model_params.json`, identify the actual key structure, and adapt the test's loader to match. Then re-run. Do NOT proceed to TFM tasks until this gating test passes — it's the contract that proves the incumbent is faithful.

If `test_special_codes_isolated` fails: the `_parse_special_codes` fix from Task 2 didn't propagate; debug there first.

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/models/woe_logit.py comparison/tests/test_woe_logit.py
git commit -m "feat(comparison): WoE-logit champion (penalty=None, explicit trends, coef sanity)"
```

---

## Task 10: Plain logistic regression wrapper

**Files:**
- Create: `comparison/src/models/logit.py`
- Create: `comparison/tests/test_logit.py`

- [ ] **Step 1: Failing test**

Create `comparison/tests/test_logit.py`:
```python
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

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        nominal = [c for c, i in types_meta.items() if i["type"] == "nominal" and c in X_train.columns]
        other = [c for c in X_train.columns if c not in nominal]
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal),
            ("num", StandardScaler(), other),
        ])
        lr_kwargs = dict(solver="lbfgs", max_iter=2000, random_state=seed)
        lr_kwargs.update(self.params)
        pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(**lr_kwargs))])
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
git commit -m "feat(comparison): plain logistic regression baseline (params-aware)"
```

---

## Task 11: XGBoost wrapper

**Files:**
- Create: `comparison/src/models/xgb.py`
- Create: `comparison/tests/test_xgb.py`

- [ ] **Step 1: Failing test**

Create `comparison/tests/test_xgb.py`:
```python
from pathlib import Path
import numpy as np
from src.data_loader import load
from src.preprocessing import fold_safe_iqr_cap
from src.models.xgb import XGBWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]

def test_xgb_runs():
    X, y, meta = load(repo_root=REPO_ROOT)
    X_tr, X_te, _ = fold_safe_iqr_cap(X.iloc[:800], X.iloc[800:], meta)
    m = XGBWrapper()
    p = m.fit_predict(X_tr, y.iloc[:800], X_te, meta)
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
git commit -m "feat(comparison): XGBoost wrapper (params-aware, native categorical)"
```

---

## Task 12: LightGBM wrapper

**Files:**
- Create: `comparison/src/models/lgbm.py`
- Create: `comparison/tests/test_lgbm.py`

- [ ] **Step 1: Failing test**

Create `comparison/tests/test_lgbm.py`:
```python
from pathlib import Path
import numpy as np
from src.data_loader import load
from src.preprocessing import fold_safe_iqr_cap
from src.models.lgbm import LGBMWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]

def test_lgbm_runs():
    X, y, meta = load(repo_root=REPO_ROOT)
    X_tr, X_te, _ = fold_safe_iqr_cap(X.iloc[:800], X.iloc[800:], meta)
    m = LGBMWrapper()
    p = m.fit_predict(X_tr, y.iloc[:800], X_te, meta)
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
git commit -m "feat(comparison): LightGBM wrapper (params-aware, categorical_feature)"
```

---

## Task 13: TabPFN v2 wrapper

**Files:**
- Create: `comparison/src/models/tabpfn.py`
- Create: `comparison/tests/test_tabpfn.py`

- [ ] **Step 1: Add dep**

```bash
cd comparison && uv add tabpfn
```
Fallback: `uv add "tabpfn @ git+https://github.com/PriorLabs/TabPFN.git"`.

- [ ] **Step 2: Failing test**

Create `comparison/tests/test_tabpfn.py`:
```python
from pathlib import Path
import numpy as np
import pytest
from src.data_loader import load
from src.preprocessing import fold_safe_iqr_cap
from src.models.tabpfn import TabPFNWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.slow
def test_tabpfn_runs():
    X, y, meta = load(repo_root=REPO_ROOT)
    X_tr, X_te, _ = fold_safe_iqr_cap(X.iloc[:800], X.iloc[800:], meta)
    m = TabPFNWrapper()
    p = m.fit_predict(X_tr, y.iloc[:800], X_te, meta)
    assert p.shape == (200,)
    assert np.all((p >= 0) & (p <= 1))
```

- [ ] **Step 3: Register the `slow` marker**

Edit `comparison/pyproject.toml`, append (create the section if absent):
```toml
[tool.pytest.ini_options]
markers = [
    "slow: heavy TFM tests; opt in with -m slow",
]
```

- [ ] **Step 4: Fail**

```bash
cd comparison && uv run pytest tests/test_tabpfn.py -v -m slow
```

- [ ] **Step 5: Implement**

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

- [ ] **Step 6: Pass**

```bash
cd comparison && uv run pytest tests/test_tabpfn.py -v -m slow
```
First run downloads TabPFN v2 weights from HF (~100 MB).

- [ ] **Step 7: Commit**

```bash
cd .. && git add comparison/src/models/tabpfn.py comparison/tests/test_tabpfn.py \
                comparison/pyproject.toml comparison/uv.lock
git commit -m "feat(comparison): TabPFN v2 wrapper (MPS/CUDA/CPU auto-device)"
```

---

## Task 14: TabICL wrapper

**Files:**
- Create: `comparison/src/models/tabicl.py`
- Create: `comparison/tests/test_tabicl.py`

- [ ] **Step 1: Add dep**

```bash
cd comparison && uv add tabicl
```
Fallback: `uv add "tabicl @ git+https://github.com/soda-inria/tabicl.git"`.

- [ ] **Step 2: Failing test**

Create `comparison/tests/test_tabicl.py`:
```python
from pathlib import Path
import numpy as np
import pytest
from src.data_loader import load
from src.preprocessing import fold_safe_iqr_cap
from src.models.tabicl import TabICLWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.slow
def test_tabicl_runs():
    X, y, meta = load(repo_root=REPO_ROOT)
    X_tr, X_te, _ = fold_safe_iqr_cap(X.iloc[:800], X.iloc[800:], meta)
    m = TabICLWrapper()
    p = m.fit_predict(X_tr, y.iloc[:800], X_te, meta)
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

## Task 15: TabDPT wrapper

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
from src.preprocessing import fold_safe_iqr_cap
from src.models.tabdpt import TabDPTWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.slow
def test_tabdpt_runs():
    X, y, meta = load(repo_root=REPO_ROOT)
    X_tr, X_te, _ = fold_safe_iqr_cap(X.iloc[:800], X.iloc[800:], meta)
    m = TabDPTWrapper()
    p = m.fit_predict(X_tr, y.iloc[:800], X_te, meta)
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

## Task 16: CARTE wrapper (string-decoded)

**Files:**
- Create: `comparison/src/models/carte.py`
- Create: `comparison/tests/test_carte.py`

- [ ] **Step 1: Add deps**

```bash
cd comparison && uv add carte-ai torch
```
Fallback: `uv add "carte-ai @ git+https://github.com/soda-inria/carte.git"`.

- [ ] **Step 2: Failing test**

Create `comparison/tests/test_carte.py`:
```python
from pathlib import Path
import numpy as np
import pytest
from src.data_loader import load, carte_decode
from src.preprocessing import fold_safe_iqr_cap
from src.models.carte import CARTEWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.slow
def test_carte_requires_string_labels_flag():
    assert CARTEWrapper.requires_string_labels is True

@pytest.mark.slow
def test_carte_runs_on_decoded_input():
    X, y, meta = load(repo_root=REPO_ROOT)
    # Apply fold-safe cap THEN string-decode (decode is a no-op on continuous cols)
    X_tr, X_te, _ = fold_safe_iqr_cap(X.iloc[:800], X.iloc[800:], meta)
    X_tr_str = carte_decode(X_tr, meta)
    X_te_str = carte_decode(X_te, meta)
    m = CARTEWrapper()
    p = m.fit_predict(X_tr_str, y.iloc[:800], X_te_str, meta)
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
        # PyPI package name is carte-ai → module carte_ai
        from carte_ai import CARTEClassifier
        clf = CARTEClassifier(random_state=seed)
        clf.fit(X_train, y_train.values)
        proba = clf.predict_proba(X_test)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba
```
If the import path differs, check with `uv run python -c "import carte_ai; print(dir(carte_ai))"` and adjust.

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

## Task 17: Runner (fold-safe + predictions parquet from v1)

**Files:**
- Create: `comparison/src/runner.py`
- Create: `comparison/tests/test_runner.py`

This is the heart of the orchestration. The outer loop is **folds-first, models-second** so:
1. `fold_safe_iqr_cap` runs once per fold, not once per (fold × model).
2. In Pass 2, `tune_classicals_for_fold` is called once per outer fold (yielding params shared by all three classicals on that fold).
3. Predictions parquet is written from this very first version — no retrofit.

- [ ] **Step 1: Failing test**

Create `comparison/tests/test_runner.py`:
```python
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
    # If the bug recurred (using y.iloc[range(N)] on full y), AUC would be
    # essentially random because labels would be from rows 0..N-1 of a full y
    # that bears no relation to the test indices.
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
```

- [ ] **Step 2: Fail**

```bash
cd comparison && uv run pytest tests/test_runner.py -v
```

- [ ] **Step 3: Implement**

Create `comparison/src/runner.py`:
```python
"""Orchestration: outer loop over folds, inner loop over models.

Per-fold pipeline (in order):
  1. Slice X, y by (train_idx, test_idx) from RAW loans.csv.
  2. Apply fold_safe_iqr_cap.
  3. (Pass 2 only) Run tune_classicals_for_fold on the capped training fold.
  4. For each model: invoke fit_predict, append metrics row to CSV,
     append predictions to parquet.
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import load, carte_decode
from src.preprocessing import fold_safe_iqr_cap
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

def run_one_fold_one_model(model, X_train, y_train, X_test, y_test, meta) -> tuple[dict, np.ndarray]:
    """Fit on already-sliced training data, predict on already-sliced test data.

    The caller is responsible for slicing — this function does NOT do iloc on
    full-dataset indices. (Previous version had a label-alignment bug where it
    used local positional indices on the full y, which silently returned the
    wrong labels for every fold after fold 0.)
    """
    t0 = time.perf_counter()
    proba = model.fit_predict(X_train, y_train, X_test, meta)
    runtime = time.perf_counter() - t0
    m = compute_all(y_test.values, proba)
    m.update(model=model.name, n_train=len(X_train), n_test=len(X_test),
             runtime_s=runtime, fold_idx=-1)
    return m, proba

def _build_models_for_fold(pass_name: str, tuned_params: dict | None) -> list:
    """Construct one instance per model for this outer fold.

    For Pass 1: all models with defaults.
    For Pass 2: classicals use tuned_params from this fold's nested study;
                woe_logit + 4 TFMs unchanged (their pass-1 rows will be reused).
    """
    from src.models.woe_logit import WoELogitChampion
    from src.models.logit import PlainLogit
    from src.models.xgb import XGBWrapper
    from src.models.lgbm import LGBMWrapper
    from src.models.tabpfn import TabPFNWrapper
    from src.models.tabicl import TabICLWrapper
    from src.models.tabdpt import TabDPTWrapper
    from src.models.carte import CARTEWrapper

    tp = tuned_params or {}
    return [
        WoELogitChampion(),
        PlainLogit(params=tp.get("logit")),
        XGBWrapper(params=tp.get("xgb")),
        LGBMWrapper(params=tp.get("lgbm")),
        TabPFNWrapper(),
        TabICLWrapper(),
        TabDPTWrapper(),
        CARTEWrapper(),
    ]

# Which models are unchanged between Pass 1 and Pass 2 → their pass-2 rows are
# copied from the pass-1 outputs rather than re-computed.
COPY_FROM_PASS1 = {"woe_logit", "tabpfn_v2", "tabicl", "tabdpt", "carte"}

def run_pass(pass_name: str, n_splits: int = 10, seed: int = 42) -> None:
    X, y, meta = load(REPO_ROOT)
    folds = make_folds(y, n_splits=n_splits, seed=seed)

    out_csv = RESULTS_DIR / f"per_fold_{pass_name}.csv"
    out_parquet = RESULTS_DIR / f"predictions_{pass_name}.parquet"
    parts_dir = RESULTS_DIR / f"predictions_{pass_name}_parts"
    # Clean prior outputs (full re-run)
    if out_csv.exists():
        out_csv.unlink()
    if out_parquet.exists():
        out_parquet.unlink()
    if parts_dir.exists():
        for f in parts_dir.iterdir():
            f.unlink()
        parts_dir.rmdir()
    parts_dir.mkdir(parents=True, exist_ok=True)

    pass1_preds = None
    pass1_metrics = None
    if pass_name == "tuned":
        # Reuse pass-1 outputs for the unchanged models
        p1_parquet = RESULTS_DIR / "predictions_defaults.parquet"
        p1_csv = RESULTS_DIR / "per_fold_defaults.csv"
        if not (p1_parquet.exists() and p1_csv.exists()):
            raise RuntimeError("Pass 2 requires Pass 1 outputs. Run --pass defaults first.")
        pass1_preds = pd.read_parquet(p1_parquet)
        pass1_metrics = pd.read_csv(p1_csv)

    for fold_i, (tr, te) in enumerate(folds):
        # Step 1: slice raw, uncapped data and labels by the fold's positional indices
        X_tr_raw = X.iloc[tr].reset_index(drop=True)
        y_tr     = y.iloc[tr].reset_index(drop=True)
        X_te_raw = X.iloc[te].reset_index(drop=True)
        y_te     = y.iloc[te].reset_index(drop=True)

        # Step 2: outer fold-safe cap (caps fit on X_tr_raw, applied to both)
        X_tr_capped, X_te_capped, _ = fold_safe_iqr_cap(X_tr_raw, X_te_raw, meta)
        # CARTE inputs: cap (numeric only) then string-decode
        X_tr_carte = carte_decode(X_tr_capped, meta)
        X_te_carte = carte_decode(X_te_capped, meta)

        # Step 3: nested Optuna tuning (Pass 2 only).
        # Tuning receives the UNCAPPED outer-training fold; it recomputes caps
        # inside each inner split (strict nested preprocessing — see spec §4).
        tuned_params = None
        if pass_name == "tuned":
            from src.tuning import tune_classicals_for_fold
            tuned_params = tune_classicals_for_fold(X_tr_raw, y_tr, meta, seed=seed)

        # Step 4: per-model fits on outer-capped data
        models = _build_models_for_fold(pass_name, tuned_params)
        fold_preds: list[pd.DataFrame] = []
        for model in tqdm(models, desc=f"fold {fold_i+1}/{n_splits}", leave=False):
            # Pass 2 shortcut: reuse pass-1 rows for unchanged models
            if pass_name == "tuned" and model.name in COPY_FROM_PASS1:
                row = pass1_metrics[(pass1_metrics["model"] == model.name) &
                                    (pass1_metrics["fold_idx"] == fold_i)].iloc[0].to_dict()
                row["pass"] = "tuned"
                append_row(out_csv, row)
                copy_preds = pass1_preds[(pass1_preds["model"] == model.name) &
                                         (pass1_preds["fold_idx"] == fold_i)].copy()
                fold_preds.append(copy_preds)
                continue

            # Choose feature view (CARTE wants strings)
            if getattr(model, "requires_string_labels", False):
                X_tr_use, X_te_use = X_tr_carte, X_te_carte
            else:
                X_tr_use, X_te_use = X_tr_capped, X_te_capped

            row, proba = run_one_fold_one_model(
                model, X_tr_use, y_tr, X_te_use, y_te, meta,
            )
            row["fold_idx"] = fold_i
            row["pass"] = pass_name
            append_row(out_csv, row)

            fold_preds.append(pd.DataFrame({
                "model": model.name,
                "fold_idx": fold_i,
                "test_idx": te,
                "y": y_te.values,
                "proba": proba,
            }))

        # Crash-safe: write this fold's predictions as a parquet part immediately.
        # If the run dies mid-pass, all completed folds' parts are intact and
        # can be combined with the snippet below.
        part_path = parts_dir / f"fold_{fold_i:02d}.parquet"
        pd.concat(fold_preds, ignore_index=True).to_parquet(part_path, index=False)

    # Combine parts into the canonical predictions parquet
    parts = sorted(parts_dir.glob("fold_*.parquet"))
    combined = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    combined.to_parquet(out_parquet, index=False)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_parquet} (combined from {len(parts)} parts in {parts_dir})")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pass", dest="pass_name", choices=["defaults", "tuned"], required=True)
    args = p.parse_args()
    run_pass(args.pass_name)

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
git commit -m "feat(comparison): runner with fold-safe preprocessing + predictions parquet"
```

---

## Task 18: Per-fold nested Optuna tuning (classicals)

**Files:**
- Create: `comparison/src/tuning.py`
- Create: `comparison/tests/test_tuning.py`

Spec §4: 10 outer folds × 3 classicals × 50 trials × inner 5-fold CV. Critically, `tune_classicals_for_fold` receives **only** the outer-training fold — never the full dataset.

- [ ] **Step 1: Failing test**

Create `comparison/tests/test_tuning.py`:
```python
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data_loader import load
from src.preprocessing import fold_safe_iqr_cap
from src.tuning import tune_classicals_for_fold

REPO_ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.slow
def test_returns_three_classical_params():
    X, y, meta = load(repo_root=REPO_ROOT)
    X_tr, _, _ = fold_safe_iqr_cap(X.iloc[:800], X.iloc[800:], meta)
    params = tune_classicals_for_fold(X_tr, y.iloc[:800], meta, seed=42, n_trials=3)
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

def test_inner_cap_uses_inner_train_only():
    """Strict nested-preprocessing guard. Inside _inner_cv_score, the cap
    applied to inner-validation rows must come from inner-train only — NOT
    from the full outer-training fold.

    Setup: build a tiny outer-train of 25 rows where one row has a huge
    outlier in 'Duration of Credit (month)'. If the outlier ends up in an
    inner-val fold but the cap is computed from inner-train only (which does
    not see that outlier), the outlier value gets capped at the inner-train
    distribution's Q3+1.5*IQR. We verify that by calling _inner_cv_score with
    a make_fn that records the X_te['Duration...'].max() it actually sees.
    """
    from src.tuning import _inner_cv_score
    X = pd.DataFrame({
        "Duration of Credit (month)": list(range(1, 25)) + [999.0],
        "Credit Amount": [500.0] * 25,
        "Age (years)": [30.0] * 25,
        "other": [0] * 25,
    })
    meta = {
        "Duration of Credit (month)": {"type": "continuous"},
        "Credit Amount": {"type": "continuous"},
        "Age (years)": {"type": "continuous"},
        "other": {"type": "nominal"},
    }
    y = pd.Series([0, 1] * 12 + [0])
    observed_max: list[float] = []

    class _Recorder:
        def fit(self, Xtr, ytr): self._x = Xtr; return self
        def predict_proba(self, Xte):
            observed_max.append(float(Xte["Duration of Credit (month)"].max()))
            return np.column_stack([1 - np.full(len(Xte), 0.5), np.full(len(Xte), 0.5)])

    def make(Xtr_capped, ytr): return _Recorder().fit(Xtr_capped, ytr)

    _inner_cv_score(make, X, y, meta, seed=42)
    # If 999 ever leaked into the inner-val set uncapped, observed_max would
    # contain a value >= 999 (because the outer cap was never applied).
    # With strict per-inner-split caps, it must be capped to something far
    # smaller than 999.
    assert all(m < 100 for m in observed_max), \
        f"inner-val rows saw uncapped outliers, suggesting inner cap leaked: {observed_max}"
```

- [ ] **Step 2: Fail**

```bash
cd comparison && uv run pytest tests/test_tuning.py -v -m slow
cd comparison && uv run pytest tests/test_tuning.py::test_signature_only_consumes_train_rows -v
```

- [ ] **Step 3: Implement**

Create `comparison/src/tuning.py`:
```python
"""Per-outer-fold nested Optuna tuning for the classical baselines.

Called by the runner ONCE per outer fold (Pass 2). Receives ONLY the outer-
training rows — and crucially, receives them **uncapped**. Caps are recomputed
inside each inner CV split (strict nested preprocessing, spec §4) so that inner-
validation rows never influence the cap thresholds they are later scored against.
"""
from __future__ import annotations
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.preprocessing import fold_safe_iqr_cap

N_TRIALS_DEFAULT = 50
INNER_FOLDS = 5

optuna.logging.set_verbosity(optuna.logging.WARNING)

def _inner_cv_score(
    make_clf_and_fit,
    X_uncapped: pd.DataFrame,
    y: pd.Series,
    types_meta: dict,
    seed: int,
) -> float:
    """5-fold inner CV with caps refit per inner split.

    The cap is computed from inner-train rows only and applied to both inner-
    train and inner-validation. This prevents inner-val from influencing
    its own cap threshold.
    """
    skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X_uncapped, y):
        X_inner_tr_uncapped = X_uncapped.iloc[tr].reset_index(drop=True)
        X_inner_te_uncapped = X_uncapped.iloc[te].reset_index(drop=True)
        # Strict nested cap
        X_inner_tr, X_inner_te, _ = fold_safe_iqr_cap(
            X_inner_tr_uncapped, X_inner_te_uncapped, types_meta,
        )
        y_inner_tr = y.iloc[tr].reset_index(drop=True)
        y_inner_te = y.iloc[te].reset_index(drop=True)
        clf = make_clf_and_fit(X_inner_tr, y_inner_tr)
        p = clf.predict_proba(X_inner_te)[:, 1]
        aucs.append(roc_auc_score(y_inner_te, p))
    return float(np.mean(aucs))

def _cast_nominals_to_category(X: pd.DataFrame, types_meta: dict) -> pd.DataFrame:
    """Cast nominal columns to pandas category dtype. Done inside each inner
    fit so that train/val category levels are consistent within the fit."""
    X = X.copy()
    for c, i in types_meta.items():
        if i["type"] == "nominal" and c in X.columns:
            X[c] = X[c].astype("category")
    return X

def _align_test_categories(X_te: pd.DataFrame, X_tr: pd.DataFrame) -> pd.DataFrame:
    X_te = X_te.copy()
    for c in X_tr.columns:
        if str(X_tr[c].dtype) == "category":
            X_te[c] = X_te[c].astype(pd.CategoricalDtype(categories=X_tr[c].cat.categories))
    return X_te

def _tune_xgb(X_uncapped: pd.DataFrame, y: pd.Series, types_meta: dict,
              seed: int, n_trials: int) -> dict:
    from xgboost import XGBClassifier

    def objective(trial):
        params = dict(
            tree_method="hist", enable_categorical=True, random_state=seed, n_jobs=-1,
            eval_metric="logloss",
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        )
        def make(Xtr_capped, ytr):
            Xtr_cat = _cast_nominals_to_category(Xtr_capped, types_meta)
            clf = XGBClassifier(**params)
            clf.fit(Xtr_cat, ytr)
            # Stash for later prediction with aligned categories
            clf._train_cats = Xtr_cat
            orig_predict_proba = clf.predict_proba
            def predict_proba(Xte):
                Xte_cat = _cast_nominals_to_category(Xte, types_meta)
                Xte_cat = _align_test_categories(Xte_cat, clf._train_cats)
                return orig_predict_proba(Xte_cat)
            clf.predict_proba = predict_proba
            return clf
        return _inner_cv_score(make, X_uncapped, y, types_meta, seed)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def _tune_lgbm(X_uncapped: pd.DataFrame, y: pd.Series, types_meta: dict,
               seed: int, n_trials: int) -> dict:
    from lightgbm import LGBMClassifier
    nominal_cols = [c for c, i in types_meta.items()
                    if i["type"] == "nominal" and c in X_uncapped.columns]
    cat_idx = [X_uncapped.columns.get_loc(c) for c in nominal_cols]

    def objective(trial):
        params = dict(
            random_state=seed, n_jobs=-1, verbosity=-1,
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
        def make(Xtr_capped, ytr):
            Xtr_cat = _cast_nominals_to_category(Xtr_capped, types_meta)
            clf = LGBMClassifier(**params)
            clf.fit(Xtr_cat, ytr, categorical_feature=cat_idx)
            clf._train_cats = Xtr_cat
            orig_predict_proba = clf.predict_proba
            def predict_proba(Xte):
                Xte_cat = _cast_nominals_to_category(Xte, types_meta)
                Xte_cat = _align_test_categories(Xte_cat, clf._train_cats)
                return orig_predict_proba(Xte_cat)
            clf.predict_proba = predict_proba
            return clf
        return _inner_cv_score(make, X_uncapped, y, types_meta, seed)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def _tune_logit(X_uncapped: pd.DataFrame, y: pd.Series, types_meta: dict,
                seed: int, n_trials: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    nominal = [c for c, i in types_meta.items()
               if i["type"] == "nominal" and c in X_uncapped.columns]
    other = [c for c in X_uncapped.columns if c not in nominal]

    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        def make(Xtr_capped, ytr):
            pre = ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal),
                ("num", StandardScaler(), other),
            ])
            pipe = Pipeline([("pre", pre),
                             ("lr", LogisticRegression(solver="lbfgs", max_iter=2000,
                                                       C=C, random_state=seed))])
            pipe.fit(Xtr_capped, ytr)
            return pipe
        return _inner_cv_score(make, X_uncapped, y, types_meta, seed)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def tune_classicals_for_fold(
    X_train_outer: pd.DataFrame,
    y_train_outer: pd.Series,
    types_meta: dict,
    seed: int = 42,
    n_trials: int = N_TRIALS_DEFAULT,
) -> dict[str, dict]:
    """Run Optuna for each of {xgb, lgbm, logit} on the outer-training fold only.

    ``X_train_outer`` MUST be the **uncapped** outer-training fold. Caps are
    recomputed inside each inner CV split so inner-validation rows never
    influence the caps they are later scored against (spec §4 strict nested
    preprocessing).

    Returns:
        {"xgb": best_params, "lgbm": best_params, "logit": best_params}
    """
    return {
        "xgb":   _tune_xgb(X_train_outer, y_train_outer, types_meta, seed, n_trials),
        "lgbm":  _tune_lgbm(X_train_outer, y_train_outer, types_meta, seed, n_trials),
        "logit": _tune_logit(X_train_outer, y_train_outer, types_meta, seed, n_trials),
    }
```

- [ ] **Step 4: Pass**

```bash
cd comparison && uv run pytest tests/test_tuning.py -v -m slow
cd comparison && uv run pytest tests/test_tuning.py::test_signature_only_consumes_train_rows -v
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/src/tuning.py comparison/tests/test_tuning.py
git commit -m "feat(comparison): per-outer-fold nested Optuna tuning (no leakage)"
```

---

## Task 19: Execute pass 1 (defaults)

**Files:** execution only — produces `comparison/results/per_fold_defaults.csv` and `predictions_defaults.parquet`.

- [ ] **Step 1: Gating sanity check**

```bash
cd comparison && uv run pytest tests/test_woe_logit.py::test_full_sample_refit_matches_report_coefficients -v
```
Must show PASSED. Do not proceed otherwise — debug Task 9 first.

- [ ] **Step 2: All non-slow tests**

```bash
cd comparison && uv run pytest -v -m "not slow"
```
Expected: all green.

- [ ] **Step 3: Run pass 1**

```bash
cd comparison && uv run python -m src.runner --pass defaults
```
Expected wall-clock on M2 Max: ≤ 45 min (CARTE dominant). Outputs: `comparison/results/per_fold_defaults.csv` (80 rows) and `predictions_defaults.parquet` (8 models × 1,000 rows = 8,000 rows).

- [ ] **Step 4: Eyeball**

```bash
cd comparison && uv run python -c "
import pandas as pd
df = pd.read_csv('results/per_fold_defaults.csv')
preds = pd.read_parquet('results/predictions_defaults.parquet')
print(df.groupby('model')[['auc','gini','ks','brier','logloss','runtime_s']].mean().round(4))
print('CSV rows:', len(df), '| Parquet rows:', len(preds))
"
```
Expected: 80 CSV rows; 8,000 parquet rows; AUCs in [0.5, 0.9]; CARTE runtime >> others.

- [ ] **Step 5: Commit results**

```bash
cd .. && git add comparison/results/per_fold_defaults.csv comparison/results/predictions_defaults.parquet
git commit -m "results(comparison): pass 1 (defaults) — per-fold metrics + OOF predictions"
```

---

## Task 20: Execute pass 2 (tuned classicals, nested Optuna)

- [ ] **Step 1: Run pass 2**

```bash
cd comparison && uv run python -m src.runner --pass tuned
```
Expected wall-clock on M2 Max: 1–2.5 hours (10 outer folds × 3 classicals × 50 trials × 5 inner fits dominates). Outputs: `per_fold_tuned.csv` (80 rows; 50 = 5 unchanged-model rows reused from pass 1 + 30 freshly tuned classical rows; total still 80) and `predictions_tuned.parquet` (8,000 rows).

- [ ] **Step 2: Eyeball delta**

```bash
cd comparison && uv run python -c "
import pandas as pd
a = pd.read_csv('results/per_fold_defaults.csv').groupby('model')['auc'].mean()
b = pd.read_csv('results/per_fold_tuned.csv').groupby('model')['auc'].mean()
print(pd.DataFrame({'defaults': a, 'tuned': b, 'delta': b - a}).round(4))
"
```
Expected: woe_logit + 4 TFMs deltas exactly 0 (rows copied verbatim); xgb/lgbm/logit deltas likely ≥ 0.

- [ ] **Step 3: Commit results**

```bash
cd .. && git add comparison/results/per_fold_tuned.csv comparison/results/predictions_tuned.parquet
git commit -m "results(comparison): pass 2 (nested Optuna tuning) — per-fold metrics + OOF predictions"
```

---

## Task 21: Summary markdown with DeLong + Wilcoxon + disclaimer

**Files:**
- Create: `comparison/src/summary.py`
- Create: `comparison/tests/test_summary.py`

- [ ] **Step 1: Failing test**

Create `comparison/tests/test_summary.py`:
```python
from pathlib import Path
import numpy as np
import pandas as pd
from src.summary import build_summary_md, DISCLAIMER

def _fake_inputs():
    df = pd.DataFrame({
        "model": ["woe_logit"] * 10 + ["xgb"] * 10,
        "fold_idx": list(range(10)) * 2,
        "auc": np.r_[np.linspace(0.78, 0.82, 10), np.linspace(0.79, 0.83, 10)],
        "gini": 0.6, "ks": 0.5, "brier": 0.17, "logloss": 0.5, "runtime_s": 1.0,
    })
    rng = np.random.default_rng(0)
    rows = []
    for m in ["woe_logit", "xgb"]:
        for f in range(10):
            for i in range(100):
                rows.append({"model": m, "fold_idx": f, "test_idx": f * 100 + i,
                             "y": rng.integers(0, 2), "proba": rng.random()})
    preds = pd.DataFrame(rows)
    return df, preds

def test_build_summary_includes_disclaimer():
    df, preds = _fake_inputs()
    md = build_summary_md(df, preds, champion="woe_logit")
    assert DISCLAIMER in md
    assert "woe_logit" in md and "xgb" in md
    assert "DeLong" in md and "Wilcoxon" in md
```

- [ ] **Step 2: Fail**

```bash
cd comparison && uv run pytest tests/test_summary.py -v
```

- [ ] **Step 3: Implement**

Create `comparison/src/summary.py`:
```python
"""Render per-fold CSV + predictions parquet into a thesis-ready summary."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src.metrics import delong_test, wilcoxon_paired, bonferroni

CHAMPION = "woe_logit"
# Primary discrimination metrics — the headline of the benchmark.
DISCRIMINATION_METRICS = ["auc", "gini", "ks"]
# Secondary probability-quality metrics — sensitive to calibration as well as
# discrimination. Reported for completeness but NOT part of the primary claim.
PROBABILITY_QUALITY_METRICS = ["brier", "logloss"]
METRICS = DISCRIMINATION_METRICS + PROBABILITY_QUALITY_METRICS

DISCLAIMER = (
    "> **Scope of claim (narrow):** This benchmark addresses *discriminatory power only*. "
    "It does NOT support the broader claim that \"TFMs are better PD models\" — that would "
    "require calibration quality, rating-grade homogeneity/heterogeneity, PSI stability, and "
    "out-of-time validation, all of which the `pd-autopilot` pipeline provides for the incumbent "
    "but are out of scope here. Brier and log-loss are reported as **probability-quality "
    "(secondary)** metrics — they combine discrimination with calibration, so a challenger that "
    "ties on AUC but loses on Brier is producing well-ranked but mis-scaled probabilities, "
    "which is a calibration story outside this benchmark's primary scope."
)

CONTEXT = """## Context (not directly comparable — different preprocessing path / fold seed)
- Report dev AUC (in-sample, loans_clean): 0.8109
- Report 10-fold CV AUC (loans_clean): 0.8065
- Report bootstrap AUC: 0.8026
"""

def _paired_tests(df: pd.DataFrame, preds: pd.DataFrame, champion: str) -> dict[str, dict]:
    """For each non-champion model, compute DeLong and Wilcoxon p-values."""
    models = list(df["model"].unique())
    n_comp = sum(1 for m in models if m != champion)

    ch_preds = preds[preds.model == champion].sort_values(["fold_idx", "test_idx"])
    y_all = ch_preds["y"].values
    p_ch = ch_preds["proba"].values
    ch_aucs = df[df.model == champion].sort_values("fold_idx")["auc"].values

    out: dict[str, dict] = {}
    for m in models:
        if m == champion:
            continue
        m_preds = preds[preds.model == m].sort_values(["fold_idx", "test_idx"])
        p_m = m_preds["proba"].values
        _, p_delong = delong_test(y_all, p_m, p_ch)
        m_aucs = df[df.model == m].sort_values("fold_idx")["auc"].values
        try:
            _, p_wilcox = wilcoxon_paired(m_aucs, ch_aucs)
        except ValueError:
            p_wilcox = 1.0
        out[m] = {
            "delong_p": p_delong,
            "delong_bonf": bonferroni(p_delong, n_comp),
            "wilcoxon_p": p_wilcox,
            "wilcoxon_bonf": bonferroni(p_wilcox, n_comp),
        }
    return out

def build_summary_md(df: pd.DataFrame, preds: pd.DataFrame, champion: str = CHAMPION) -> str:
    g = df.groupby("model")
    agg = g[METRICS].agg(["mean", "std"]).round(4)
    tests = _paired_tests(df, preds, champion)

    # Build the header in two visually grouped sections.
    discrim_hdr = " | ".join(m.upper() for m in DISCRIMINATION_METRICS)
    proba_hdr   = " | ".join(m.upper() + " (sec.)" for m in PROBABILITY_QUALITY_METRICS)
    header = (f"| Model | {discrim_hdr} | {proba_hdr} | "
              f"DeLong p | DeLong (bonf) | Wilcoxon p | Wilcoxon (bonf) |")
    sep = "|" + "|".join(["---"] * (len(METRICS) + 5)) + "|"

    rows = []
    for model in df["model"].unique():
        cells = [f"{agg.loc[model, (m, 'mean')]:.4f} ± {agg.loc[model, (m, 'std')]:.4f}" for m in METRICS]
        if model == champion:
            tail = ["reference"] * 4
        else:
            t = tests[model]
            tail = [f"{t['delong_p']:.3f}", f"{t['delong_bonf']:.3f}",
                    f"{t['wilcoxon_p']:.3f}", f"{t['wilcoxon_bonf']:.3f}"]
        rows.append("| " + model + " | " + " | ".join(cells) + " | " + " | ".join(tail) + " |")

    legend = (
        "*Primary discrimination metrics: AUC, Gini, KS. "
        "Probability-quality (secondary) metrics: Brier, log-loss — sensitive to calibration, "
        "not part of the primary claim (see disclaimer above).*"
    )
    return "\n".join([
        DISCLAIMER, "",
        "## Headline (10-fold CV)", "",
        header, sep, *rows, "",
        legend, "",
        CONTEXT,
    ])

def main():
    here = Path(__file__).resolve().parents[1]
    for pass_name in ["defaults", "tuned"]:
        csv_path = here / "results" / f"per_fold_{pass_name}.csv"
        pred_path = here / "results" / f"predictions_{pass_name}.parquet"
        if not (csv_path.exists() and pred_path.exists()):
            print(f"skip {pass_name}: missing artefact(s)")
            continue
        df = pd.read_csv(csv_path)
        preds = pd.read_parquet(pred_path)
        md = build_summary_md(df, preds)
        out = here / "results" / f"summary_{pass_name}.md"
        out.write_text(md)
        print(f"wrote {out}")

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
Outputs `results/summary_defaults.md` and `results/summary_tuned.md` — each starts with the disclaimer.

- [ ] **Step 6: Commit**

```bash
cd .. && git add comparison/src/summary.py comparison/tests/test_summary.py \
                comparison/results/summary_defaults.md comparison/results/summary_tuned.md
git commit -m "feat(comparison): summary MD with DeLong + Wilcoxon + narrow-claim disclaimer"
```

---

## Task 22: Analysis notebook (ROC overlay, reliability, boxplot, forest plot, runtime)

**Files:**
- Create: `comparison/notebooks/01_results_analysis.ipynb`

- [ ] **Step 1: Build the notebook**

Create `comparison/notebooks/01_results_analysis.ipynb` with these cells. The simplest way: write a Python script and convert via `jupytext`, or build the `.ipynb` JSON directly. Cell contents (each cell separated):

**Cell 1 — imports:**
```python
%matplotlib inline
import sys; sys.path.insert(0, "..")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from src.metrics import delong_test, wilcoxon_paired, bonferroni
RES = Path("../results")
PASS = "defaults"   # change to "tuned" and re-execute the cells for second pass
df = pd.read_csv(RES / f"per_fold_{PASS}.csv")
preds = pd.read_parquet(RES / f"predictions_{PASS}.parquet")
models = list(df["model"].unique())
CHAMPION = "woe_logit"
```

**Cell 2 — ROC overlay:**
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

**Cell 4 — AUC boxplot:**
```python
fig, ax = plt.subplots(figsize=(8, 5))
data = [df[df.model == m]["auc"].values for m in models]
ax.boxplot(data, labels=models, vert=True)
ax.set(ylabel="AUC", title=f"Per-fold AUC distribution ({PASS})")
plt.xticks(rotation=30, ha="right")
fig.savefig(RES / "figures" / f"auc_boxplot_{PASS}.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Cell 5 — Forest plot of paired diffs:**
```python
ch = preds[preds.model == CHAMPION].sort_values(["fold_idx", "test_idx"])
y_all = ch["y"].values
p_ch = ch["proba"].values

rows = []
for m in models:
    if m == CHAMPION: continue
    sub = preds[preds.model == m].sort_values(["fold_idx", "test_idx"])
    p_m = sub["proba"].values
    z, p = delong_test(y_all, p_m, p_ch)
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

**Cell 7 — Render saved summary inline:**
```python
from IPython.display import Markdown, display
display(Markdown((RES / f"summary_{PASS}.md").read_text()))
```

- [ ] **Step 2: Duplicate for tuned pass**

```bash
cd comparison
cp notebooks/01_results_analysis.ipynb notebooks/02_results_analysis_tuned.ipynb
# Update cell 1 in the duplicate so PASS="tuned"
sed -i '' 's/PASS = "defaults"/PASS = "tuned"/' notebooks/02_results_analysis_tuned.ipynb
```

- [ ] **Step 3: Execute both**

```bash
cd comparison
uv run jupyter nbconvert --to notebook --execute notebooks/01_results_analysis.ipynb --inplace
uv run jupyter nbconvert --to notebook --execute notebooks/02_results_analysis_tuned.ipynb --inplace
```

- [ ] **Step 4: Verify**

```bash
ls comparison/results/figures/
```
Expected: `roc_overlay_defaults.png`, `roc_overlay_tuned.png`, `reliability_defaults.png`, `reliability_tuned.png`, `auc_boxplot_defaults.png`, `auc_boxplot_tuned.png`, `forest_defaults.png`, `forest_tuned.png`.

- [ ] **Step 5: Commit**

```bash
cd .. && git add comparison/notebooks/01_results_analysis.ipynb \
                comparison/notebooks/02_results_analysis_tuned.ipynb \
                comparison/results/figures/
git commit -m "feat(comparison): analysis notebooks (ROC/reliability/boxplot/forest)"
```

---

## Task 23: Top-level benchmark notebook

**Files:**
- Create: `comparison/notebooks/benchmark.ipynb`

- [ ] **Step 1: Build the notebook**

Create `comparison/notebooks/benchmark.ipynb` with these cells:

**Cell 1 — markdown:**
```markdown
# TFM vs WoE-Logit Benchmark

Reproducible entry-point for the comparison defined in
[2026-05-22-tfm-comparison-design.md](../../docs/superpowers/specs/2026-05-22-tfm-comparison-design.md).

**Scope (narrow):** Pure discrimination only. Does NOT support the claim that
"TFMs are better PD models" — see disclaimer in `summary_*.md` for full caveats.
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
## How to run
From a terminal:
```bash
cd comparison
uv run python -m src.runner --pass defaults
uv run python -m src.runner --pass tuned
uv run python -m src.summary
```
```

**Cell 4 — load + render pass 1:**
```python
df1 = pd.read_csv(RES / "per_fold_defaults.csv")
df1.groupby("model")[["auc","gini","ks","brier","logloss","runtime_s"]].agg(["mean","std"]).round(4)
```

**Cell 5 — load + render pass 2:**
```python
df2 = pd.read_csv(RES / "per_fold_tuned.csv")
df2.groupby("model")[["auc","gini","ks","brier","logloss","runtime_s"]].agg(["mean","std"]).round(4)
```

**Cell 6 — defaults vs tuned delta:**
```python
a = df1.groupby("model")["auc"].mean()
b = df2.groupby("model")["auc"].mean()
pd.DataFrame({"defaults": a, "tuned": b, "delta": b - a}).round(4).sort_values("tuned", ascending=False)
```

**Cell 7 — display summary MDs inline:**
```python
from IPython.display import Markdown, display
display(Markdown((RES / "summary_defaults.md").read_text()))
display(Markdown((RES / "summary_tuned.md").read_text()))
```

- [ ] **Step 2: Execute**

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

- [ ] **Step 1: All non-slow tests pass**

```bash
cd comparison && uv run pytest -v -m "not slow"
```

- [ ] **Step 2: All slow tests pass**

```bash
cd comparison && uv run pytest -v -m slow
```

- [ ] **Step 3: Gating sanity test**

```bash
cd comparison && uv run pytest tests/test_woe_logit.py::test_full_sample_refit_matches_report_coefficients -v
```
Expected: PASSED — every coefficient within ±0.05.

- [ ] **Step 4: Spec acceptance criteria — by-number check**

| # | Criterion | Verification |
|---|---|---|
| 1 | `uv sync && uv run pytest tests/` passes | Steps 1+2 above |
| 2 | `test_full_sample_refit_matches_report_coefficients` passes | Step 3 |
| 3 | `test_special_codes_isolated` passes | `uv run pytest tests/test_woe_logit.py::test_special_codes_isolated -v` |
| 4 | `test_signature_only_consumes_train_rows` + `test_inner_cap_uses_inner_train_only` (no-leakage guards) | `uv run pytest tests/test_tuning.py -k "signature_only or inner_cap_uses" -v` |
| 5 | `test_caps_derived_from_train_only` + `test_run_one_fold_passes_correct_labels` (label-alignment regression) | `uv run pytest tests/test_preprocessing.py::test_caps_derived_from_train_only tests/test_runner.py::test_run_one_fold_passes_correct_labels -v` |
| 6 | per_fold CSVs have 80 rows each | `wc -l results/per_fold_*.csv` |
| 7 | predictions parquets have 8,000 rows each; per-fold parts directories exist with 10 parts each | `uv run python -c "import pandas as pd; print(len(pd.read_parquet('results/predictions_defaults.parquet')), len(pd.read_parquet('results/predictions_tuned.parquet')))"; ls results/predictions_defaults_parts/ results/predictions_tuned_parts/ | wc -l` |
| 8 | Summary MDs begin with disclaimer + contain both DeLong and Wilcoxon | `head -3 results/summary_*.md; grep -l 'DeLong' results/summary_*.md; grep -l 'Wilcoxon' results/summary_*.md` |
| 9 | Analysis notebook executes end-to-end | Already done in Task 22 step 3 |
| 10 | Wall-clock ≤ 3 hours combined | Recorded during Tasks 19 + 20 |

Run each check. Any failure → fix before tagging.

- [ ] **Step 5: Tag**

```bash
cd .. && git tag -a comparison-v1 -m "TFM vs WoE-logit comparison complete (post-Codex revision)"
```

- [ ] **Step 6: Final commit if anything outstanding**

```bash
git status
# if clean: done
```

---

## Self-review checklist

- [x] **Spec coverage:**
  - §1 narrow-claim → README + Task 21 disclaimer + Task 23 cell 1
  - §2 fold-safe preprocessing → Task 3 + Task 17 step 3
  - §2 no `loans_clean.csv` → Task 17 reads raw `loans.csv`
  - §3 8-model lineup → Tasks 9–16
  - §4 nested Optuna → Task 18; Task 17 calls it per outer fold
  - §4 coef-level sanity → Task 9 step 1 + Task 19 step 1
  - §5 predictions parquet first-class → Task 17 from v1 (no retrofit)
  - §5 DeLong + Wilcoxon in summary → Task 21
  - §6 architecture matches → file paths in Tasks 1–21
  - §7 PDModel protocol → Task 8
  - §8 woe_logit faithfulness → Task 9 (penalty=None, explicit trends, CP/MIP fallback)
  - §9 env with pyarrow → Task 1 step 3
  - §10 reproducibility → seeds in Tasks 4, 9–18
  - §11 reporting artefacts → Tasks 21, 22
  - §12 non-goals — no 8-var view → respected throughout (challengers always see all 20)
  - §13 acceptance criteria 1–10 → mapped 1:1 in Task 24 step 4
- [x] **No placeholders:** every code step contains complete code.
- [x] **Type consistency:** `PDModel.fit_predict` signature identical across `base.py`, all 8 wrappers, runner; `tune_classicals_for_fold` signature locked by the leakage-guard test.
- [x] **Frequent commits:** 24 commits across the plan.
- [x] **TDD:** every implementation task has failing-test → pass-test sequence.
- [x] **v1 issues addressed:** every Codex finding (commit `842af30` message) has a corresponding task change documented in the header diff.
