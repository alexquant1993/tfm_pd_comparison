---
description: Build logistic regression model using stepwise MIV selection, produce score scale
tools:
  - Read
  - Bash
  - Write
---

# Stage 04 — Model Builder

## Purpose

Build a logistic regression model using stepwise MIV selection from the approved variable shortlist. Produce the final variable set, coefficients, score scale, and model parameters file.

## Skills to Load

Load these skills before proceeding:
- `pd-conventions` — for AUC benchmarks, score scale convention
- `pdtoolkit-api` — for correct function signatures
- `notebook-writer` — for notebook structure and plot conventions

## Inputs

You will receive a `RUN_DIR` path (e.g., `runs/2026-03-14_143022/`). All output paths below are relative to this directory.

- **Clean dataset:** `{RUN_DIR}/data/loans_clean.csv`
- **Approved variable shortlist:** from `{RUN_DIR}/pipeline/stage_03.md`
- **Target variable:** `Creditability`
- **Human modifications:** (if the reviewer changed the shortlist, these will be provided in the invocation prompt)

## Toolkit Import

```python
import sys; sys.path.insert(0, 'src')
import pdtoolkit as pdt
```

## Functions to Use

### Primary stepwise selection
- `pdt.step_miv(start_model, miv_threshold, m_ch_p_val, coding, db)` — primary method
- `pdt.step_fwd(start_model, p_value, coding, db, risk_factors, target)` — alternative forward stepwise
- `pdt.step_rpc(start_model, risk_profile, ...)` — alternative with risk profile check

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

1. Load clean dataset and approved shortlist
2. Prepare WoE-encoded dataset for the shortlist variables
3. Run `pdt.step_miv()` with default thresholds
4. Run self-assessment checks (see below) — iterate if needed
5. Generate score distribution using `pdt.scaled_score()`
6. Run cross-validation and bootstrap validation
7. Generate all plots and write outputs

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
- `{RUN_DIR}/notebooks/04_model_building.ipynb` — following notebook-writer skill conventions
- `{RUN_DIR}/figures/04_roc_curve.png`
- `{RUN_DIR}/figures/04_score_distribution.png`
- `{RUN_DIR}/figures/04_score_decile_table.png` — score decile analysis (observed default rate per decile)
- `{RUN_DIR}/figures/04_coefficient_plot.png`
- `{RUN_DIR}/pipeline/stage_04.md`
- `{RUN_DIR}/pipeline/model_params.json`

## model_params.json Schema

Write `{RUN_DIR}/pipeline/model_params.json` with this structure — stages 05 and 06 depend on it:

```json
{
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

## stage_04.md Template

Write `{RUN_DIR}/pipeline/stage_04.md` with exactly these fields:

```
clean_dataset_path: {RUN_DIR}/data/loans_clean.csv
selected_variables: [list]
excluded_from_shortlist: [list with reason]
model_auc: [float]
model_gini: [float]
model_ks: [float]
coefficients:
  - variable: [name]
    coefficient: [float]
    woe_direction_consistent: [true/false]
    vif: [float]
miv_threshold_used: [float, note if adjusted]
score_statistics:
  min: [float]
  max: [float]
  mean: [float]
  pct_below_400: [float]
  pct_above_800: [float]
score_monotonicity_across_deciles: [true/false]
model_params_path: {RUN_DIR}/pipeline/model_params.json
model_flags: [list or "None"]
```

## Return Message

After writing all outputs, verify using the output-verifier skill checklist. Then return:

```
Stage 04 complete. Notebook: {RUN_DIR}/notebooks/04_model_building.ipynb
[N] variables selected. AUC: [X]. Gini: [X]. KS: [X].
Score range: [min]-[max], mean [X]. All coefficient signs consistent: [yes/no].
Decile monotonicity: [pass/fail].
Flags: [list or "None"].
Awaiting human review before proceeding to Stage 05.
```
