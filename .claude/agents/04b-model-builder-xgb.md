---
description: Build logistic regression model using XGBoost feature importance for variable selection
tools:
  - Read
  - Bash
  - Write
---

# Stage 04b — Model Builder (XGBoost Feature Importance)

## Purpose

Build a logistic regression model by using XGBoost feature importance to select variables from the approved shortlist. XGBoost is used **only for variable ranking** — the final model is logistic regression on WoE-encoded features (regulatory requirement).

## Skills to Load

Load these skills before proceeding:
- `pd-conventions` — for AUC benchmarks, score scale convention
- `pdtoolkit-api` — for correct function signatures
- `notebook-writer` — for notebook structure and plot conventions
- `fix-proposer` — for post-output diagnostic protocol

## Execution Logging

Append milestone entries to `{RUN_DIR}/pipeline/execution.log` using:
```bash
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] [stage-04b] message" >> {RUN_DIR}/pipeline/execution.log
```

Log these milestones:
1. After variable selection: `XGBoost selection | n_selected={N} | cumulative_threshold={X}`
2. After model fit: `Model fit | auc={X} | gini={X} | ks={X} | signs_consistent={yes/no}`

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
from xgboost import XGBClassifier
```

## Functions to Use

### XGBoost variable selection
- `XGBClassifier` from `xgboost` — for feature importance ranking

### pdtoolkit functions
- `pdt.replace_woe(db, target)` — WoE-encode binned variables
- `pdt.scaled_score(probs, score=600, odd=50, pdo=20)` — scale to credit scores
- `pdt.constrained_logit(db, x, y, lower, upper)` — if coefficient constraints needed
- `pdt.kfold_vld(model, db, target, predictors)` — k-fold cross-validation
- `pdt.boots_vld(model, db, target, predictors)` — bootstrap validation
- `pdt.confusion_matrix(predictions, observed, cutoff)` — classification metrics

## Analysis Steps

1. Load binned dataset (`loans_binned.csv`) and approved shortlist
2. WoE-encode using `pdt.replace_woe(db_binned[shortlist + [target]], target)`
3. **Fit XGBoost for feature importance** (see below)
4. **Select variables by cumulative importance** (see below)
5. Run self-assessment checks — iterate if needed
6. **Refit final logistic regression with statsmodels** (see below)
7. Generate score distribution using `pdt.scaled_score()`
8. Run cross-validation and bootstrap validation
9. Generate all plots and write outputs

## Step 3: XGBoost Feature Importance

Train an XGBoost classifier on the WoE-encoded features to obtain feature importance rankings:

```python
from xgboost import XGBClassifier
import pandas as pd

# Fit XGBoost on WoE-encoded shortlisted variables
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_woe[shortlist], y)

# Extract gain-based feature importances (best proxy for predictive contribution)
importances = pd.Series(
    xgb_model.feature_importances_,
    index=shortlist
).sort_values(ascending=False)

# Display importance ranking in notebook
print("XGBoost Feature Importances (gain-based):")
print(importances.to_string())
```

**Why gain-based importance:** Gain measures the average improvement in loss function when a feature is used for splitting. This aligns with predictive contribution and is the most relevant metric for variable selection in a credit scoring context.

## Step 4: Variable Selection by Cumulative Importance

Select variables using a cumulative importance threshold:

```python
# Cumulative importance
cum_importance = importances.cumsum() / importances.sum()

# Select variables covering 90% of cumulative importance
selected_mask = cum_importance <= 0.90
# Always include the variable that crosses the 90% threshold
selected_mask.iloc[selected_mask.sum()] = True if selected_mask.sum() < len(selected_mask) else selected_mask

selected_variables = importances[selected_mask].index.tolist()

# Enforce min 4, max 12 variables
if len(selected_variables) < 4:
    # Take top 4 regardless of cumulative threshold
    selected_variables = importances.head(4).index.tolist()
elif len(selected_variables) > 12:
    # Take top 12 regardless of cumulative threshold
    selected_variables = importances.head(12).index.tolist()
```

**Document in the notebook:** Show the cumulative importance curve and mark the cutoff point.

## Step 6: Statsmodels Final Model

After variable selection and self-assessment, refit with statsmodels (identical to MIV approach):

```python
import statsmodels.api as sm

X = sm.add_constant(X_woe[selected_variables])
logit_model = sm.Logit(y, X)
logit_res = logit_model.fit()

print(logit_res.summary())

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

Extract model-level statistics:
- `logit_res.llf` — Log-Likelihood
- `logit_res.llnull` — LL-Null
- `logit_res.prsquared` — Pseudo R-squared (McFadden)
- `logit_res.llr_pvalue` — LLR p-value
- `logit_res.df_model` — degrees of freedom
- `logit_res.nobs` — number of observations

## Self-Assessment (6 checks)

### 1. Variable count check
- If < 4 variables: lower cumulative importance threshold to 0.80 and re-select
- If > 12 variables: raise cumulative importance threshold to 0.95 and re-select

### 2. Coefficient signs vs WoE directions
- For each selected variable: positive WoE should correspond to positive logit direction
- If sign reversal found:
  - **Remove the reversed variable from the XGBoost-selected set**
  - **Add the next-ranked variable from the XGBoost importance list** (the highest-importance variable not yet selected)
  - Re-run model and recheck — document the swap
  - This is critical: XGBoost may select variable combinations that work well in a tree ensemble but produce multicollinearity in logistic regression

### 3. VIF (Variance Inflation Factor)
- Calculate VIF for all selected variables
- If VIF > 5 for any variable: remove the variable with lower XGBoost importance from the collinear pair, replace with next-ranked variable

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
- `{RUN_DIR}/notebooks/04b_model_building_xgb.ipynb` — following notebook-writer skill conventions
- `{RUN_DIR}/figures/04b_roc_curve.png`
- `{RUN_DIR}/figures/04b_score_distribution.png`
- `{RUN_DIR}/figures/04b_score_decile_table.png`
- `{RUN_DIR}/figures/04b_coefficient_plot.png`
- `{RUN_DIR}/figures/04b_xgb_importance.png` — XGBoost feature importance bar chart with cumulative line
- `{RUN_DIR}/pipeline/stage_04b.md`
- `{RUN_DIR}/pipeline/model_params_xgb.json`
- `{RUN_DIR}/pipeline/stage_04b_fixes.md` — fix-proposer diagnostic output

## model_params_xgb.json Schema

Write `{RUN_DIR}/pipeline/model_params_xgb.json` with this structure:

```json
{
  "selection_method": "xgboost_importance",
  "selected_variables": ["var1", "var2", ...],
  "xgb_importances": {
    "var1": 0.35,
    "var2": 0.22,
    "var3": 0.15
  },
  "cumulative_importance_threshold": 0.90,
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

## stage_04b.md Template

Write `{RUN_DIR}/pipeline/stage_04b.md` with exactly these fields and tables:

```
selection_method: xgboost_importance
binned_dataset_path: {RUN_DIR}/data/loans_binned.csv
selected_variables: [list]
excluded_from_shortlist: [list with reason]
model_auc: [float]
model_gini: [float]
model_ks: [float]
cumulative_importance_threshold: [float, note if adjusted]
xgb_hyperparameters: n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8
```

### XGBoost Feature Importances

| Variable | Importance (Gain) | Cumulative % | Selected |
|---|---|---|---|
| [var1] | [float] | [float] | [yes/no] |
| ... | ... | ... | ... |

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
model_params_path: {RUN_DIR}/pipeline/model_params_xgb.json
model_flags: [list or "None"]
```

## Return Message

After writing all outputs, verify using the output-verifier skill checklist. Then run the fix-proposer diagnostic protocol and write `{RUN_DIR}/pipeline/stage_04b_fixes.md`. Then return:

```
Stage 04b (XGBoost) complete. Notebook: {RUN_DIR}/notebooks/04b_model_building_xgb.ipynb
[N] variables selected (cumulative importance threshold: [X]). AUC: [X]. Gini: [X]. KS: [X].
Score range: [min]-[max], mean [X]. All coefficient signs consistent: [yes/no].
Decile monotonicity: [pass/fail].
Flags: [list or "None"].
Fix proposals: [N] issues ([N] critical) — see stage_04b_fixes.md
```
