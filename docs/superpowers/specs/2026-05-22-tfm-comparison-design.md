# TFM vs WoE-Logit PD Comparison — Design

**Date:** 2026-05-22
**Author:** Anderson Arroyo (with Claude)
**Status:** Approved — ready for implementation plan
**Project:** `pd-autopilot` / `comparison/`

## 1. Goal

Benchmark four tabular foundation models (TFMs) against the WoE-logistic-regression champion produced by `pd-autopilot` (run `2026-03-17_071354`) on the German Credit dataset, with classical gradient-boosting and plain-logit baselines as reference points.

**Research question (locked):** *Pure discrimination — can a TFM beat WoE-logit on AUC/Gini/KS/Brier/log-loss under matched cross-validation?* Calibration, rating-grade construction, and regulatory validation tests are explicitly out of scope.

## 2. Dataset and incumbent

- **Data:** `data/loans.csv` (1,000 rows, 20 predictors, target `Creditability`, 30% default rate). Already fully numeric (categoricals integer-coded).
- **Metadata:** `data/variable_types.csv` is the single source of truth for column type (`nominal` / `ordinal` / `continuous`), monotonicity expectation, special codes, and UCI string labels.
- **Incumbent model (from report):** MIV-stepwise WoE-logit on 8 variables: Account Balance, Payment Status of Previous Credit, Duration of Credit, Value Savings/Stocks, Purpose, Credit Amount, Most Valuable Available Asset, Age. Published metrics: in-sample AUC 0.8109, 10-fold CV AUC 0.8065, bootstrap AUC 0.8026.

## 3. Model lineup (8 models)

| Group | Model | Library |
|---|---|---|
| Incumbent | `woe_logit` (MIV champion, 8 vars, refit in-fold) | `pdtoolkit` + `optbinning` + `sklearn` |
| Classical | `logit` (plain LR on raw + one-hot) | `sklearn` |
| Classical | `xgb` (XGBClassifier, native categorical) | `xgboost` |
| Classical | `lgbm` (LGBMClassifier, categorical_feature=) | `lightgbm` |
| TFM | `tabpfn_v2` | `tabpfn` |
| TFM | `tabicl` | `tabicl` |
| TFM | `tabdpt` | `tabdpt` |
| TFM | `carte` (string-decoded inputs) | `carte-ai` |

## 4. Protocol

- **Cross-validation:** 10-fold stratified on `Creditability`, fixed seed (42), folds frozen across all models. Same `(train_idx, test_idx)` tuples passed to every wrapper.
- **Features:** raw `loans.csv` for all challengers; type information passed via `types_meta` derived from `variable_types.csv`. CARTE additionally receives string-decoded inputs (e.g. `Purpose=3` → `"radio/TV"`) built from the `Encoding` column.
- **Two passes:**
  - **Pass 1 — defaults.** Every model with library defaults. TFMs have no real hyperparameters, so for them defaults = tuned.
  - **Pass 2 — light tuning, classicals only.** XGB / LGBM / plain logit each get an Optuna study (50 trials, TPE sampler, seed 42) with nested inner 5-fold CV on each outer training fold; objective = mean inner-fold AUC; best params refit on outer-train, scored on outer-test. WoE-logit and TFMs unchanged between passes — their pass-1 per-fold rows are copied verbatim into `per_fold_tuned.csv` so the file still has 80 rows (8 models × 10 folds) and the pass-1-vs-pass-2 delta is cleanly attributable to classical tuning.
- **Sanity check (mandatory before any TFM runs):** one-off `woe_logit` refit on full 1,000 rows must reproduce the report's coefficients to ≈3 decimals; if it doesn't, halt and debug binning setup.

## 5. Metrics and statistical tests

**Per-fold (10 rows per model in CSV):** AUC, Gini (= 2·AUC − 1), KS, Brier, log-loss, n_train, n_test.

**Aggregate:** mean ± std across folds; rough 95% band as `1.96·std/√10` (flagged as approximate — folds are not strictly independent).

**Paired tests vs `woe_logit` champion:**
1. **DeLong** (Sun & Xu 2014, ~40-LOC port, no extra dep) on concatenated predictions across all 10 folds.
2. **Wilcoxon signed-rank** on the 10 per-fold AUC differences.
3. **Bonferroni correction** for 7 challengers vs champion; both raw and corrected p reported.

Disagreement between DeLong and Wilcoxon is itself surfaced in the analysis notebook (usually flags "great on most folds, tanks one").

## 6. Architecture

**Approach B — notebook + per-model wrappers.** Thin orchestration notebook, each model isolated in its own module behind a uniform interface.

```
comparison/
  pyproject.toml             # uv-managed
  uv.lock                    # committed lockfile
  README.md

  src/
    data_loader.py           # load loans.csv + variable_types.csv → X, y, types_meta
                             # carte_decode() builds string-labelled DataFrame for CARTE
    cv.py                    # make_folds(n_splits=10, seed=42) → frozen list of (tr, te)
    metrics.py               # auc, gini, ks, brier, log_loss, delong_test, wilcoxon
    runner.py                # outer loop over (model, fold); writes per_fold CSV row-by-row

    models/
      base.py                # PDModel Protocol
      woe_logit.py           # pdtoolkit OptimalBinning refit in-fold + sklearn LR
      logit.py               # one-hot nominals + standardised continuous + LR
      xgb.py                 # XGBClassifier(enable_categorical=True, tree_method="hist")
      lgbm.py                # LGBMClassifier(categorical_feature=[indices])
      tabpfn.py              # TabPFNClassifier(device="mps" if available)
      tabicl.py              # TabICLClassifier
      tabdpt.py              # TabDPTClassifier
      carte.py               # CARTE with requires_string_labels=True

  notebooks/
    benchmark.ipynb           # orchestrator: load → run → render
    01_results_analysis.ipynb # ROC overlay, reliability curves, AUC boxplot, forest plot

  results/
    per_fold_defaults.csv
    per_fold_tuned.csv
    summary_defaults.md
    summary_tuned.md
    figures/

  tests/
    test_cv.py                # folds stratified, deterministic, disjoint
    test_metrics.py           # AUC matches sklearn; DeLong sanity
    test_data_loader.py       # row count, target rate, no leakage
```

## 7. Model interface contract

```python
class PDModel(Protocol):
    name: str                          # e.g. "tabpfn_v2"
    requires_string_labels: bool       # True only for CARTE

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,            # 0/1
        X_test: pd.DataFrame,
        types_meta: dict,              # {col: "nominal"|"ordinal"|"continuous"}
        seed: int = 42,
    ) -> np.ndarray:                   # shape (len(X_test),), P(default=1)
```

Deterministic given the seed. Orchestrator never touches model internals. New TFM = one new file.

## 8. `woe_logit` faithfulness to the report

| Element | Setting | Per-fold behaviour |
|---|---|---|
| Variable set | Fixed at the 8 champion vars from `model_params.json` | **Not re-selected per fold** — that would conflate "TFM vs incumbent" with "incumbent feature drift" |
| Binning | `optbinning.OptimalBinning` (CP solver, MIP fallback) | Refit on each fold's training rows only |
| `dtype`, monotonicity, `special_codes` | From `variable_types.csv` (matches stage 03) | Same every fold |
| Encoding | WoE (pdtoolkit convention) | Train bins applied to test fold |
| Classifier | `sklearn.linear_model.LogisticRegression`, default solver, no regularisation tuning | Refit on WoE-transformed train fold |
| Prediction | `predict_proba(X_te_woe)[:, 1]` | P(default = 1) |

The report's published 0.8065 likely fits binning once on all rows and only refits logit per fold — mildly optimistic. Our in-fold refit will probably land slightly lower; that lower number is the honest baseline TFMs must beat.

## 9. Environment

- **`uv`** managed project (`pyproject.toml` + committed `uv.lock`).
- Python ≥ 3.10 (CARTE requirement).
- Initial dep set: `pandas numpy scikit-learn scipy matplotlib optbinning xgboost lightgbm tabpfn tabicl tabdpt carte-ai optuna jupyter tqdm`.
- The local `src/pdtoolkit/` package from the parent repo is imported by `woe_logit.py` via a relative `sys.path` insert (no separate install). Package names for the TFMs to be verified at implementation time: `tabicl` and `tabdpt` may resolve to GitHub installs rather than PyPI; `carte-ai` is the PyPI name for CARTE.
- Hardware: local M2 Max, 32 GB unified, MPS where supported. No Colab.
- Hugging Face: public weights only, no account / no payment. First call to each TFM auto-downloads to `~/.cache/huggingface/`.

## 10. Reproducibility

- Single seed (42) used for fold creation, sklearn estimators, Optuna sampler, every TFM that exposes a seed argument.
- Per-fold CSV written row-by-row during execution (crash-safe; partial results survive).
- `uv sync` on any machine reproduces exact environment.
- All result CSVs and summary MD committed to git.

## 11. Reporting artefacts

- `results/summary_defaults.md` and `summary_tuned.md` — headline tables with mean ± std AUC/Gini/KS/Brier/log-loss and paired DeLong p-values (raw + Bonferroni).
- Three context rows always shown: report's in-sample 0.8109, report's 10-fold CV 0.8065, this-run full-sample `woe_logit` refit (sanity).
- `notebooks/01_results_analysis.ipynb` outputs: ROC overlay, reliability curves, AUC boxplot, forest plot of paired diffs, runtime table.

## 12. Explicit non-goals

- No Hosmer-Lemeshow on challengers (no calibration step).
- No rating-grade construction for TFMs.
- No PSI / out-of-time stability (no OOT data).
- No SHAP / explainability work.
- No pipeline integration (`comparison/` stays a sibling directory, not a new autopilot stage).

## 13. Acceptance criteria

1. `cd comparison && uv sync && uv run pytest tests/` passes.
2. `woe_logit` full-sample sanity refit matches report coefficients to ≈3 decimals.
3. Both `per_fold_defaults.csv` and `per_fold_tuned.csv` produced, 80 rows each (8 models × 10 folds).
4. Both summary MDs and the analysis notebook execute end-to-end without errors.
5. Total wall-clock runtime on M2 Max ≤ 2 hours for pass 1 + pass 2 combined.
