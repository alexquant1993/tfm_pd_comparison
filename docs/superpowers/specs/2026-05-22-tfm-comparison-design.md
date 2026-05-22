# TFM vs WoE-Logit PD Comparison — Design

**Date:** 2026-05-22
**Last revised:** 2026-05-22 (post-Codex review)
**Author:** Anderson Arroyo (with Claude)
**Status:** Approved — ready for implementation plan
**Project:** `pd-autopilot` / `comparison/`

## 1. Goal

Benchmark four tabular foundation models (TFMs) against a fold-safe re-implementation of the `pd-autopilot` WoE-logit champion (run `2026-03-17_071354`) on the German Credit dataset, with classical gradient-boosting and plain-logit baselines as reference points.

**Research question (locked):** *Pure discrimination — can a TFM beat WoE-logit on AUC/Gini/KS/Brier/log-loss under matched cross-validation?* Calibration, rating-grade construction, and regulatory validation tests are explicitly out of scope.

**Scope of claim (narrow):** This benchmark addresses *discriminatory power only*. It does NOT support the broader claim that "TFMs are better PD models" — that would require calibration quality, rating-grade homogeneity/heterogeneity, PSI stability, and out-of-time validation, all of which the `pd-autopilot` pipeline provides for the incumbent but are out of scope here. The thesis chapter, results notebook, and every generated `summary_*.md` must reproduce this caveat verbatim in their headers.

## 2. Dataset and incumbent

- **Data:** `data/loans.csv` (1,000 rows, 20 predictors, target `Creditability`, 30% default rate). Already fully numeric (categoricals integer-coded). This raw file is the **sole** input — we do **not** consume `runs/2026-03-17_071354/data/loans_clean.csv`, because its IQR caps were computed on all 1,000 rows in Stage 02 and therefore carry information from every fold's test rows. Using it would be cross-fold leakage.
- **Fold-safe preprocessing:** IQR-based outlier capping (1.5× IQR, upper tail only — matching Stage 02's recipe) is applied **inside each outer CV fold**: caps are computed from the outer-training rows only and applied to both the outer-training and outer-test rows. Helper lives in `src/preprocessing.py::fold_safe_iqr_cap(X_train, X_test, types_meta)`. Continuous columns capped: `Duration of Credit (month)`, `Credit Amount`, `Age (years)` (the same three Stage 02 capped).
- **Metadata:** `data/variable_types.csv` is the single source of truth for column type (`nominal` / `ordinal` / `continuous`), monotonicity expectation, special codes, and UCI string labels.
- **Incumbent model (from report):** MIV-stepwise WoE-logit on 8 variables: Account Balance, Payment Status of Previous Credit, Duration of Credit, Value Savings/Stocks, Purpose, Credit Amount, Most Valuable Available Asset, Age. Published metrics: in-sample AUC 0.8109, 10-fold CV AUC 0.8065, bootstrap AUC 0.8026. **These published numbers are reported as context only** — they were produced from `loans_clean.csv` and so are *not* directly comparable to our fold-safe re-implementation. The primary baseline TFMs must beat is our in-fold re-trained `woe_logit`, not the report's 0.8065.

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
- **Per-fold pipeline (applied identically before every model wrapper runs):**
  1. Slice `X[train_idx]`, `X[test_idx]` from raw `loans.csv`.
  2. `fold_safe_iqr_cap` — fit IQR caps on the three continuous columns from outer-train rows, apply to both outer-train and outer-test.
  3. Pass the resulting `(X_tr, y_tr, X_te, y_te, types_meta)` to the wrapper's `fit_predict`.
- **Features:** all 20 raw columns for all challengers (no manual selection). Type information passed via `types_meta` from `variable_types.csv`. CARTE additionally receives string-decoded inputs (e.g. `Purpose=3` → `"radio/TV"`) built from the `Encoding` column. The incumbent `woe_logit` wrapper internally restricts to its 8 champion variables (part of the model definition).
- **Two passes:**
  - **Pass 1 — defaults.** Every model with library defaults. TFMs have no real hyperparameters, so for them defaults = tuned.
  - **Pass 2 — properly nested tuning, classicals only.** For each outer fold, for each of {XGB, LGBM, plain logit}: run an Optuna study (50 trials, TPE sampler, seed 42) on the outer-training fold only, scoring trials by mean AUC over an inner 5-fold stratified CV that lives entirely inside the outer-training fold. The best params from that outer-fold study are refit on the entire outer-training fold and scored on the outer-test fold. Total Optuna budget: 10 outer folds × 3 classicals × 50 trials × 5 inner folds = 7,500 inner-fold fits. WoE-logit and TFMs unchanged between passes — their pass-1 per-fold rows are copied verbatim into `per_fold_tuned.csv` so the file still has 80 rows (8 models × 10 folds) and the pass-1-vs-pass-2 delta is cleanly attributable to classical tuning. The single-shot `tune_xgb()` / `tune_lgbm()` / `tune_logit()` style is explicitly forbidden — it leaks outer-test rows into hyperparameter selection.
- **Sanity check (mandatory before any TFM runs):** one-off `woe_logit` refit on full 1,000 rows must reproduce the report's **coefficients** from `runs/2026-03-17_071354/pipeline/model_params.json` within an absolute tolerance of 0.05 per coefficient (intercept + 8 slopes; tolerance is generous to absorb solver and library-version drift but tight enough to catch the L2-vs-unregularised mistake). The historical AUC-only sanity check was insufficient and is removed. If any coefficient deviates by more than 0.05, halt and debug binning setup, monotonic trends, or the regularisation choice before continuing.

## 5. Metrics and statistical tests

**Per-fold (10 rows per model in CSV):** AUC, Gini (= 2·AUC − 1), KS, Brier, log-loss, n_train, n_test, runtime_s, fold_idx, pass.

**Per-fold out-of-fold predictions (separate file, written alongside the metrics CSV from v1 of the runner):** `predictions_<pass>.parquet`, long-format columns `model, fold_idx, test_idx, y, proba`. This is a first-class artefact — both `summary_<pass>.md` (Task 20) and the analysis notebook (Task 22) consume it. DeLong cannot be computed without it; retro-fitting persistence later is forbidden.

**Aggregate:** mean ± std across folds; rough 95% band as `1.96·std/√10` (flagged as approximate — folds are not strictly independent).

**Paired tests vs `woe_logit` champion (computed in `summary.py`, not deferred to the notebook):**
1. **DeLong** (Sun & Xu 2014, ~40-LOC port, no extra dep) on concatenated out-of-fold predictions across all 10 folds.
2. **Wilcoxon signed-rank** on the 10 per-fold AUC differences.
3. **Bonferroni correction** for 7 challengers vs champion; both raw and corrected p reported for each test.

Both tests appear in `summary_<pass>.md` headline tables (not just the notebook). Disagreement between DeLong and Wilcoxon is surfaced in the analysis notebook (usually flags "great on most folds, tanks one").

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
                             # _parse_special_codes() handles str / int / float / NaN uniformly
    preprocessing.py         # fold_safe_iqr_cap(X_train, X_test, types_meta) → (X_tr', X_te')
                             # caps continuous cols at train Q3 + 1.5·IQR (upper tail only)
    cv.py                    # make_folds(n_splits=10, seed=42) → frozen list of (tr, te)
    metrics.py               # auc, gini, ks, brier, log_loss, delong_test, wilcoxon, bonferroni
    tuning.py                # tune_classicals_for_fold(X_tr, y_tr, types_meta, seed)
                             # → dict {"xgb": params, "lgbm": params, "logit": params}
                             # called inside the runner's outer-fold loop (Pass 2 only)
    runner.py                # outer loop over (fold, model); writes per_fold CSV + predictions parquet
    summary.py               # builds summary_<pass>.md with DeLong + Wilcoxon paired tests

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
    predictions_defaults.parquet   # long-format OOF predictions (first-class artefact)
    predictions_tuned.parquet
    summary_defaults.md            # includes DeLong + Wilcoxon (not deferred to notebook)
    summary_tuned.md
    figures/

  tests/
    test_cv.py                # folds stratified, deterministic, disjoint
    test_metrics.py           # AUC matches sklearn; DeLong sanity; Wilcoxon
    test_data_loader.py       # row count, target rate, special_codes parsed as int
    test_preprocessing.py     # caps come from train only; identical when X_test ⊂ X_train ranges
    test_woe_logit.py         # coefficient-level reproduction of report's model_params.json
    test_tuning.py            # nested CV signature; no leakage from outer-test rows
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
| Binning | `optbinning.OptimalBinning` with **explicit** monotonic trends per variable (see below); `solver="cp"` with try/except fallback to `solver="mip"` (mirrors the documented Stage 03 behaviour where the CP solver failed on certain ordinal-ascending cases and the autopilot fell back automatically). | Refit on each fold's training rows only |
| `dtype` | From `variable_types.csv` (`"numerical"` for ordinals + continuous, `"categorical"` for nominals) | Same every fold |
| Monotonic trend (per variable, hard-coded from report stage 03) | `Account Balance`: `descending`; `Payment Status of Previous Credit`: `None` (nominal); `Duration of Credit (month)`: `ascending`; `Value Savings/Stocks`: `descending`; `Purpose`: `None` (nominal); `Credit Amount`: `ascending`; `Most valuable available asset`: `ascending`; `Age (years)`: `descending`. No `"auto"` — every direction is explicit and reproducible. | Same every fold |
| `special_codes` | Parsed by `_parse_special_codes()` which accepts int, float, NaN, and comma-separated strings. Per variable: `Account Balance` → `[4]`; `Value Savings/Stocks` → `[5]`; `Most valuable available asset` → `[4]`. Others → `None`. | Same every fold; verified by unit test |
| Encoding | WoE (pdtoolkit convention, positive WoE = lower risk) | Train bins applied to test fold |
| Classifier | `sklearn.linear_model.LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)` — **unregularised** to match the report's statsmodels-style logit. Default L2 regularisation is forbidden as it silently shifts coefficients toward zero and breaks reproducibility against `model_params.json`. | Refit on WoE-transformed train fold |
| Prediction | `predict_proba(X_te_woe)[:, 1]` | P(default = 1) |

**Coefficient-level sanity test (gating, must pass before any TFM runs):** A `test_full_sample_refit_matches_report_coefficients` test loads `runs/2026-03-17_071354/pipeline/model_params.json`, refits `woe_logit` on the full 1,000 rows (no CV, but applying the same fold-safe IQR cap recipe to the full sample as if it were the training fold), and asserts each of the 9 coefficients (intercept + 8 slopes) is within absolute tolerance 0.05 of the reported value. AUC-only checks are insufficient — different coefficient sets can yield the same AUC. The report's `model_params.json` contains the canonical coefficient set; that is the gate.

The report's published 0.8065 was computed on `loans_clean.csv` with bins fit on all 1,000 rows (mildly leaky). Our fold-safe in-fold refit will likely land lower; that lower number is the honest baseline TFMs must beat.

## 9. Environment

- **`uv`** managed project (`pyproject.toml` + committed `uv.lock`).
- Python ≥ 3.10 (CARTE requirement).
- Initial dep set: `pandas numpy scikit-learn scipy matplotlib optbinning xgboost lightgbm tabpfn tabicl tabdpt carte-ai optuna jupyter tqdm pyarrow`. `pyarrow` is required from day one for the predictions parquet artefact (predictions persistence is not optional — see §5).
- The local `src/pdtoolkit/` package from the parent repo is imported by `woe_logit.py` via a relative `sys.path` insert (no separate install). Package names for the TFMs to be verified at implementation time: `tabicl` and `tabdpt` may resolve to GitHub installs rather than PyPI; `carte-ai` is the PyPI name for CARTE.
- Hardware: local M2 Max, 32 GB unified, MPS where supported. No Colab.
- Hugging Face: public weights only, no account / no payment. First call to each TFM auto-downloads to `~/.cache/huggingface/`.

## 10. Reproducibility

- Single seed (42) used for fold creation, sklearn estimators, Optuna sampler, every TFM that exposes a seed argument.
- Per-fold CSV written row-by-row during execution (crash-safe; partial results survive).
- `uv sync` on any machine reproduces exact environment.
- All result CSVs and summary MD committed to git.

## 11. Reporting artefacts

- `results/summary_defaults.md` and `summary_tuned.md` — every file begins with the **narrow-claim disclaimer** from §1 reproduced verbatim, then a headline table with mean ± std AUC/Gini/KS/Brier/log-loss and **both** DeLong and Wilcoxon paired p-values (raw + Bonferroni). DeLong and Wilcoxon are computed in `src/summary.py` from the predictions parquet — they are not deferred to the notebook.
- Three context rows always shown beneath the headline: report's in-sample 0.8109, report's 10-fold CV 0.8065, this-run full-sample `woe_logit` refit (sanity). The context rows are clearly flagged as not directly comparable to the in-fold benchmark rows (different preprocessing path, different fold seed).
- `notebooks/01_results_analysis.ipynb` consumes the same predictions parquet and adds: ROC overlay, reliability curves, AUC boxplot, forest plot of paired diffs, runtime table.

## 12. Explicit non-goals

- No Hosmer-Lemeshow on challengers (no calibration step).
- No rating-grade construction for TFMs.
- No PSI / out-of-time stability (no OOT data).
- No SHAP / explainability work.
- No pipeline integration (`comparison/` stays a sibling directory, not a new autopilot stage).
- **No 8-variable view for challengers.** Challengers always see all 20 columns. The "feature access" confound (Codex review point 7) is acknowledged in the thesis discussion text but not addressed with a secondary experiment — limiting TFMs to 8 vars would itself be unfair since their value proposition is "no manual feature selection". The incumbent uses 8 vars because that *is* the incumbent.
- **No use of `loans_clean.csv`.** All preprocessing happens fold-safely inside the runner (§2, §4). The Stage-02 cleaned file is leaky and ignored.

## 13. Acceptance criteria

1. `cd comparison && uv sync && uv run pytest tests/` passes (non-slow tests).
2. `test_full_sample_refit_matches_report_coefficients` passes — every coefficient within ±0.05 of `model_params.json`.
3. `test_special_codes_isolated` passes — Account Balance, Value Savings/Stocks, and Most Valuable Available Asset each have a dedicated special-code bin in the fitted OptimalBinning output.
4. `test_no_leakage_in_tuning` passes — proves the tuning function called inside outer fold N never sees rows from outer fold N's test set.
5. `test_fold_safe_caps` passes — caps applied to outer-test are derived solely from outer-train.
6. Both `per_fold_defaults.csv` and `per_fold_tuned.csv` produced, 80 rows each (8 models × 10 folds).
7. Both `predictions_defaults.parquet` and `predictions_tuned.parquet` produced; each contains exactly `n_models × n_rows = 8 × 1000 = 8000` rows (every observation appears once per model as its OOF prediction).
8. Both summary MDs begin with the narrow-claim disclaimer verbatim and contain both DeLong and Wilcoxon paired p-values.
9. Analysis notebook executes end-to-end without errors.
10. Total wall-clock runtime on M2 Max ≤ 3 hours for pass 1 + pass 2 combined (pass 2 raised from 2h to accommodate proper nested Optuna).
