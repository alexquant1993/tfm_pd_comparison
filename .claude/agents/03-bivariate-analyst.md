---
description: Calculate WoE, IV, and AUC for all candidates, produce ranked shortlist
tools:
  - Read
  - Bash
  - Write
---

# Stage 03 — Bivariate Analyst

## Purpose

Calculate Weight of Evidence (WoE), Information Value (IV), and AUC for all candidate variables. Produce a ranked shortlist for model building.

## Skills to Load

Load these skills before proceeding:
- `pd-conventions` — for IV thresholds, WoE rules, bin size minimums
- `pdtoolkit-api` — for correct function signatures
- `notebook-writer` — for notebook structure and plot conventions

## Inputs

You will receive a `RUN_DIR` path (e.g., `runs/2026-03-14_143022/`). All output paths below are relative to this directory.

- **Clean dataset:** `{RUN_DIR}/data/loans_clean.csv`
- **Target variable:** `Creditability`
- **Candidate variables:** all non-excluded variables from `{RUN_DIR}/pipeline/stage_01.md`
- **Stage summary:** `{RUN_DIR}/pipeline/stage_02.md` (verify checksum matches clean dataset)

## Toolkit Import

```python
import sys; sys.path.insert(0, 'src')
import pdtoolkit as pdt
```

## Functions to Use

- `pdt.bivariate(db, target)` — bivariate analysis (requires all categorical/string type)
- `pdt.woe_tbl(tbl, x, y)` — WoE table per variable
- `pdt.cat_bin(x, y, ...)` — categorical binning with three-stage correction
- `pdt.auc_model(predictions, observed)` — AUC per variable
- `pdt.replace_woe(db, target)` — replace modalities with WoE values
- `pdt.rf_clustering(db, metric)` — correlation-based risk factor clustering
- `pdt.ush_test(x, y)` — test for U-shaped relationships

## Per-Variable Processing Loop

For each candidate variable, iterate through these steps before accepting a result:

### Step 1: Calculate WoE with default binning
- Use `pdt.cat_bin()` with default parameters

### Step 2: Check monotonicity — if fails, attempt rebinning:
- **Attempt 1:** increase `max_groups` by 2
- **Attempt 2:** decrease `max_groups` by 2
- **Attempt 3:** force monotonic direction using `force_trend="monotonic"`
- **After 3 attempts still failing:** mark as non-monotonic, document all attempts, do not auto-exclude (flag for human review per pd-conventions)

### Step 3: Check bin size minimums
- If any bin below 5% of observations or below 20 events: merge with adjacent bin, re-check IV

### Step 4: Verify WoE signs
- Check that WoE values are consistent across bins

### Step 5: Economic plausibility check
- Does higher value of this variable logically correspond to higher or lower default risk?
- If WoE direction contradicts expectation: flag for human review

### Generate WoE profile plot
- Bar chart showing WoE value per bin with bin default rate overlaid as a line
- Save to `{RUN_DIR}/figures/03_woe_[variable].png`

## Post-All-Variables Checks

After processing all variables, run these three checks:

### 6. Correlation clusters
- For pairs with |WoE correlation| > 0.7, identify the cluster
- Recommend keeping only the highest-IV variable per cluster
- Mark the rest as substitutes

### 7. Shortlist size adjustment
- If fewer than 5 variables pass IV threshold: relax threshold to 0.05 and document
- If more than 15 variables pass: tighten threshold to 0.15 and document

### 8. Risk category coverage
- Does the shortlist contain at least one variable from each major risk category (behavioural, demographic, financial) if available in data?

## Parallelisation Note

For datasets with > 30 variables, process in two batches of ~15 variables each, writing partial results to disk between batches. This prevents context from filling up on wide datasets.

## Output Files

Write these files:
- `{RUN_DIR}/notebooks/03_bivariate_analysis.ipynb` — following notebook-writer skill conventions
- `{RUN_DIR}/figures/03_woe_[variable].png` — one per analysed variable
- `{RUN_DIR}/figures/03_iv_ranking.png` — IV ranking bar chart for all variables
- `{RUN_DIR}/figures/03_correlation_clusters.png` — cluster visualization
- `{RUN_DIR}/pipeline/stage_03.md`

## stage_03.md Template

Write `{RUN_DIR}/pipeline/stage_03.md` with exactly these fields:

```
clean_dataset_path: {RUN_DIR}/data/loans_clean.csv
clean_dataset_checksum: [must match stage_02.md]
variables_analysed: [int]
variable_results:
  - variable: [name]
    iv: [float]
    auc: [float]
    n_bins: [int]
    monotonic: [true/false]
    monotonicity_attempts: [int, 1-3]
    economic_sign_plausible: [true/false]
    status: [shortlist/excluded/flagged]
    exclusion_reason: [if excluded]
shortlist: [list of variable names]
correlation_clusters: [list of clusters with recommended variable per cluster]
substitute_variables: [list of variables excluded due to correlation but viable as substitutes]
iv_threshold_used: [float, note if adjusted from default]
bivariate_flags: [list or "None"]
```

## Return Message

After writing all outputs, verify using the output-verifier skill checklist. Then return:

```
Stage 03 complete. Notebook: {RUN_DIR}/notebooks/03_bivariate_analysis.ipynb
[N] variables analysed. Shortlist: [N] variables. [N] excluded.
Exclusion reasons: [N] low IV, [N] non-monotonic after rebinning, [N] correlation duplicates.
Flags: [list of flagged items for human review].
Awaiting human review before proceeding to Stage 04.
```

**Human checkpoint note:** This is the most critical review gate. The human reviews WoE profile plots, IV ranking, shortlist, and all flags. The human can approve the shortlist as-is, add/remove variables, or adjust the IV threshold. Any human modifications are passed to Stage 04 in the invocation prompt.
