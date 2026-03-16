---
description: Calculate WoE, IV, and AUC for all candidates using OptimalBinning, produce ranked shortlist
tools:
  - Read
  - Bash
  - Write
---

# Stage 03 — Bivariate Analyst

## Purpose

Calculate Weight of Evidence (WoE), Information Value (IV), and AUC for all candidate variables using constraint-programming optimal binning. Produce a ranked shortlist for model building.

## Skills to Load

Load these skills before proceeding:
- `pd-conventions` — for IV thresholds, WoE rules, bin size minimums
- `pdtoolkit-api` — for correct function signatures (includes OptimalBinning reference)
- `notebook-writer` — for notebook structure and plot conventions
- `fix-proposer` — for post-output diagnostic protocol

## Execution Logging

Append milestone entries to `{RUN_DIR}/pipeline/execution.log` using:
```bash
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] [stage-03] message" >> {RUN_DIR}/pipeline/execution.log
```

Log these milestones:
1. After binning all variables: `Binning complete | n_analysed={N}`
2. After shortlisting: `Shortlist finalised | n_shortlisted={N} | iv_threshold={X} | min_iv={X} | max_iv={X}`

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
from optbinning import OptimalBinning
import numpy as np
import pandas as pd
```

## Functions to Use

### OptimalBinning (primary binning method)
- `OptimalBinning(name, dtype, solver="cp", monotonic_trend="auto", min_bin_size=0.05, max_n_bins=10)` — constraint-programming optimal binning
  - `.fit(x_values, y_values)` — fit optimal bins (x: numpy array, y: 0/1 numpy array)
  - `.transform(x_values, metric="bins")` — get bin label strings for each observation
  - `.transform(x_values, metric="woe")` — get WoE float values for each observation
  - `.binning_table.build()` — DataFrame with Bin, Count, Count (%), Non-event, Event, Event rate, WoE, IV, JS
  - `.binning_table.analysis()` — dict with keys: iv, gini, ks, quality_score
  - `.binning_table.plot(metric="woe")` — plot WoE profile
  - `.splits` — optimal split points (numerical only)
  - `.status` — solver status: "OPTIMAL", "FEASIBLE", etc.

### pdtoolkit (supporting functions)
- `pdt.woe_tbl(tbl, x, y)` — WoE table per variable (use on bin labels for pdt-compatible output)
- `pdt.auc_model(predictions, observed)` — AUC per variable
- `pdt.rf_clustering(db, metric)` — correlation-based risk factor clustering
- `pdt.ush_test(x, y)` — test for U-shaped relationships

## Per-Variable Processing Loop

For each candidate variable, iterate through these steps:

### Step 1: Detect dtype
- If the variable is numeric (`pd.api.types.is_numeric_dtype`): use `dtype="numerical"`
- If categorical/string: use `dtype="categorical"` and ensure values are strings

### Step 2: Fit OptimalBinning
```python
optb = OptimalBinning(
    name=var,
    dtype=dtype,
    solver="cp",
    monotonic_trend="auto",
    min_bin_size=0.05,
    max_n_bins=10
)
optb.fit(df[var].values, df[target].values)
```

### Step 3: Check solver status — if not OPTIMAL/FEASIBLE, retry:
- **Attempt 2:** try `monotonic_trend="auto_asc_desc"`
- **Attempt 3:** try `monotonic_trend="ascending"` and `monotonic_trend="descending"` separately, pick the one with higher IV
- If all attempts produce non-optimal status: flag variable, document status, do not auto-exclude

### Step 4: Extract metrics
```python
table = optb.binning_table.build()
metrics = optb.binning_table.analysis()
iv = metrics['iv']
gini = metrics['gini']
```

### Step 5: Transform to bin labels
```python
bin_labels = optb.transform(df[var].values, metric="bins")
df_binned[var] = bin_labels
```

### Step 6: Compute pdt-compatible WoE table
Create a temporary DataFrame with bin labels and target, then use `pdt.woe_tbl()`:
```python
tmp = pd.DataFrame({var: bin_labels, target: df[target].values})
woe_table = pdt.woe_tbl(tmp, x=var, y=target)
```
This produces the standard pdt format with columns: bin, no, ng, nb, pct_o, dr, woe, iv_b, iv_s.

### Step 7: Economic plausibility check
- Does higher value of this variable logically correspond to higher or lower default risk?
- If WoE direction contradicts expectation: flag for human review

### Step 8: Generate WoE profile plot
- Use OptimalBinning's built-in plot method — do NOT write custom matplotlib code:
  ```python
  optb.binning_table.plot(
      metric="woe",
      savefig=f'{RUN_DIR}/figures/03_woe_{safe_name}.png'
  )
  plt.close()
  ```
- Optionally also generate an event rate plot for additional insight in the notebook:
  ```python
  optb.binning_table.plot(metric="event_rate")
  plt.close()
  ```

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
- `{RUN_DIR}/data/loans_binned.csv` — clean dataset with all analyzed variables replaced by optbinning bin labels (target column unchanged)
- `{RUN_DIR}/figures/03_woe_[variable].png` — one per analysed variable
- `{RUN_DIR}/figures/03_iv_ranking.png` — IV ranking bar chart for all variables
- `{RUN_DIR}/figures/03_correlation_clusters.png` — cluster visualization
- `{RUN_DIR}/pipeline/stage_03.md`
- `{RUN_DIR}/pipeline/stage_03_fixes.md` — fix-proposer diagnostic output

## stage_03.md Template

Write `{RUN_DIR}/pipeline/stage_03.md` with exactly these fields:

```
clean_dataset_path: {RUN_DIR}/data/loans_clean.csv
clean_dataset_checksum: [must match stage_02.md]
binning_method: optbinning (OptimalBinning, monotonic_trend="auto")
binned_dataset_path: {RUN_DIR}/data/loans_binned.csv
binned_dataset_checksum: [md5]
variables_analysed: [int]
variable_results:
  - variable: [name]
    iv: [float]
    auc: [float]
    n_bins: [int]
    monotonic: [true/false]
    optbinning_status: [OPTIMAL/FEASIBLE/etc]
    monotonicity_attempts: [int, usually 1]
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

After writing all outputs, verify using the output-verifier skill checklist. Then run the fix-proposer diagnostic protocol and write `{RUN_DIR}/pipeline/stage_03_fixes.md`. Then return:

```
Stage 03 complete. Notebook: {RUN_DIR}/notebooks/03_bivariate_analysis.ipynb
[N] variables analysed. Shortlist: [N] variables. [N] excluded.
Exclusion reasons: [N] low IV, [N] correlation duplicates.
Flags: [list of flagged items for human review].
Fix proposals: [N] issues ([N] critical) — see stage_03_fixes.md
Awaiting human review before proceeding to Stage 04.
```

**Human checkpoint note:** This is the most critical review gate. The human reviews WoE profile plots, IV ranking, shortlist, and all flags. The human can approve the shortlist as-is, add/remove variables, or adjust the IV threshold. Any human modifications are passed to Stage 04 in the invocation prompt.
