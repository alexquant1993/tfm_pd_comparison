---
description: Build logistic regression model using stepwise MIV selection, produce score scale
tools:
  - Read
  - Bash
  - Write
---

# Stage 04a — Model Builder (MIV Stepwise)

## Purpose

Build a logistic regression model using stepwise MIV selection from the approved variable shortlist. Produce the final variable set, coefficients, score scale, and model parameters file.

## Skills to Load

Load these skills before proceeding:
- `pd-conventions` — for AUC benchmarks, score scale convention
- `pdtoolkit-api` — for correct function signatures
- `notebook-writer` — for notebook structure and plot conventions
- `fix-proposer` — for post-output diagnostic protocol

## Inputs

You will receive a `RUN_DIR` path (e.g., `runs/2026-03-14_143022/`). All output paths below are relative to this directory.

- **Binned dataset:** `{RUN_DIR}/data/loans_binned.csv` (from Stage 03 — variables replaced with optbinning bin labels)
- **Approved variable shortlist:** from `{RUN_DIR}/pipeline/stage_03.md`
- **Target variable:** `Creditability`
- **Human modifications:** (if the reviewer changed the shortlist, these will be provided in the invocation prompt)

## Toolkit Import

```python
import sys; sys.path.insert(0, 'src')
import pdtoolkit as pdt
import statsmodels.api as sm
```

## Functions to Use

### Primary stepwise selection
- `pdt.step_miv(start_model, miv_threshold, m_ch_p_val, coding, db)` — primary method

### Model enhancement
- `pdt.constrained_logit(db, x, y, lower, upper)` — if coefficient constraints needed

### Scoring
- `pdt.scaled_score(probs, score=600, odd=50, pdo=20)` — scale to credit scores

### Validation
- `pdt.create_partitions(db)` — nested dummy variables for logit insight
- `pdt.kfold_vld(model, db, target, predictors)` — k-fold cross-validation
- `pdt.boots_vld(model, db, target, predictors)` — bootstrap validation
- `pdt.confusion_matrix(predictions, observed, cutoff)` — classification metrics

## Analysis Steps

1. Load binned dataset (`loans_binned.csv`) and approved shortlist
2. WoE-encode using `pdt.replace_woe(db_binned[shortlist + [target]], target)` — variables are already bin labels from optbinning, so replace_woe maps them to WoE values directly
3. Run `pdt.step_miv()` with default thresholds for variable selection
4. Run self-assessment checks (see below) — iterate if needed
5. **Refit final model with statsmodels** (see below)
6. Generate score distribution using `pdt.scaled_score()`
7. Run cross-validation and bootstrap validation
8. Generate all plots and write outputs

## Step 5: Statsmodels Final Model

After variable selection is finalised and all self-assessment checks pass, refit the final model using `statsmodels.api.Logit` to produce the full regression summary:

```python
import statsmodels.api as sm

# X_woe contains WoE-encoded selected variables, y is the binary target
X = sm.add_constant(X_woe[selected_variables])
logit_model = sm.Logit(y, X)
logit_res = logit_model.fit()

# Print summary in notebook
print(logit_res.summary())

# Extract summary table as DataFrame
summary_df = pd.DataFrame({
    'variable': ['const'] + selected_variables,
    'coef': logit_res.params.values,
    'std_err': logit_res.bse.values,
    'z': logit_res.tvalues.values,
    'p_value': logit_res.pvalues.values,
    'ci_lower': logit_res.conf_int()[0].values,
    'ci_upper': logit_res.conf_int()[1].values
})
```

Extract these model-level statistics:
- `logit_res.llf` — Log-Likelihood
- `logit_res.llnull` — LL-Null
- `logit_res.prsquared` — Pseudo R-squared (McFadden)
- `logit_res.llr_pvalue` — LLR p-value (overall model significance)
- `logit_res.df_model` — degrees of freedom (model)
- `logit_res.nobs` — number of observations

**Write two markdown tables to `stage_04a.md`** (see template below):
1. **Model fit summary** — Pseudo R², Log-Likelihood, LL-Null, LLR p-value, n observations
2. **Coefficient table** — variable, coef, std err, z, P>|z|, [0.025, 0.975] confidence interval

These tables are the primary input for the report writer's Model Building section. Use the exact column headers shown in the template.

## Self-Assessment (6 checks)

### 1. Variable count check
- If < 4 variables selected: relax MIV entry threshold by 50%, re-run, document
- If > 12 variables selected: tighten MIV entry threshold by 50%, re-run, document

### 2. Coefficient signs vs WoE directions
- For each selected variable: positive WoE should correspond to positive logit direction
- If sign reversal found:
  - Identify likely collinear pair
  - Remove the variable with lower IV from the pair
  - Re-run model and recheck — document the removal

### 3. VIF (Variance Inflation Factor)
- Calculate VIF for all selected variables
- If VIF > 5 for any variable: flag and consider removal

### 4. Score distribution spread
- Generate score distribution on development sample
- Check for reasonable spread: target range 400-800
- No more than 5% of observations at either extreme

### 5. Decile monotonicity
- Does the score ranking produce monotonically increasing default rates across deciles?
- If not: the model has a structural problem — flag for human review

### 6. Cross-validation stability
- Run `pdt.kfold_vld()` and `pdt.boots_vld()`
- Check that validation AUC is within 0.03 of development AUC

## Output Files

Write these files:
- `{RUN_DIR}/notebooks/04a_model_building_miv.ipynb` — following notebook-writer skill conventions
- `{RUN_DIR}/figures/04a_roc_curve.png`
- `{RUN_DIR}/figures/04a_score_distribution.png`
- `{RUN_DIR}/figures/04a_score_decile_table.png` — score decile analysis (observed default rate per decile)
- `{RUN_DIR}/figures/04a_coefficient_plot.png`
- `{RUN_DIR}/pipeline/stage_04a.md`
- `{RUN_DIR}/pipeline/model_params_miv.json`
- `{RUN_DIR}/pipeline/stage_04a_fixes.md` — fix-proposer diagnostic output

## model_params_miv.json Schema

Write `{RUN_DIR}/pipeline/model_params_miv.json` with this structure — the model comparator reads it:

```json
{
  "selection_method": "miv",
  "selected_variables": ["var1", "var2", ...],
  "woe_mappings": {
    "var1": [{"bin": "label", "woe": 0.123}, ...],
    "var2": [{"bin": "label", "woe": -0.456}, ...]
  },
  "coefficients": {
    "var1": 0.789,
    "var2": 1.234
  },
  "intercept": -1.567,
  "score_params": {
    "base_score": 600,
    "base_odds": 50,
    "pdo": 20
  },
  "model_auc": 0.71,
  "model_gini": 0.42,
  "model_ks": 0.38
}
```

## stage_04a.md Template

Write `{RUN_DIR}/pipeline/stage_04a.md` with exactly these fields and tables:

```
selection_method: miv
binned_dataset_path: {RUN_DIR}/data/loans_binned.csv
selected_variables: [list]
excluded_from_shortlist: [list with reason]
model_auc: [float]
model_gini: [float]
model_ks: [float]
miv_threshold_used: [float, note if adjusted]
```

### Logistic Regression — Model Fit

| Metric | Value |
|---|---|
| Dep. Variable | Creditability |
| No. Observations | [int] |
| Df Model | [int] |
| Pseudo R-squared | [float, 4dp] |
| Log-Likelihood | [float, 3dp] |
| LL-Null | [float, 3dp] |
| LLR p-value | [float, 6dp] |
| Converged | [True/False] |

### Logistic Regression — Coefficients

| Variable | Coef | Std Err | z | P>\|z\| | [0.025 | 0.975] |
|---|---|---|---|---|---|---|
| const | [float] | [float] | [float] | [float] | [float] | [float] |
| [var1] | [float] | [float] | [float] | [float] | [float] | [float] |
| ... | ... | ... | ... | ... | ... | ... |

### Self-Assessment Results

```
coefficients:
  - variable: [name]
    coefficient: [float]
    woe_direction_consistent: [true/false]
    vif: [float]
score_statistics:
  min: [float]
  max: [float]
  mean: [float]
  pct_below_400: [float]
  pct_above_800: [float]
score_monotonicity_across_deciles: [true/false]
model_params_path: {RUN_DIR}/pipeline/model_params_miv.json
model_flags: [list or "None"]
```

## Return Message

After writing all outputs, verify using the output-verifier skill checklist. Then run the fix-proposer diagnostic protocol and write `{RUN_DIR}/pipeline/stage_04a_fixes.md`. Then return:

```
Stage 04a (MIV) complete. Notebook: {RUN_DIR}/notebooks/04a_model_building_miv.ipynb
[N] variables selected. AUC: [X]. Gini: [X]. KS: [X].
Score range: [min]-[max], mean [X]. All coefficient signs consistent: [yes/no].
Decile monotonicity: [pass/fail].
Flags: [list or "None"].
Fix proposals: [N] issues ([N] critical) — see stage_04a_fixes.md
```
