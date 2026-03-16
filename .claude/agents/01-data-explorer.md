---
description: Analyse raw data quality and document every variable's status before transformation
tools:
  - Read
  - Bash
  - Write
---

# Stage 01 — Data Explorer

## Purpose

Analyse raw data quality and document every variable's status before any transformation. Produce a comprehensive univariate report with distribution statistics, missing value analysis, and correlation assessment.

## Skills to Load

Load these skills before proceeding:
- `pd-conventions` — for all threshold and classification decisions
- `pdtoolkit-api` — for correct function signatures
- `notebook-writer` — for notebook structure and plot conventions
- `fix-proposer` — for post-output diagnostic protocol

## Execution Logging

Append milestone entries to `{RUN_DIR}/pipeline/execution.log` using:
```bash
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] [stage-01] message" >> {RUN_DIR}/pipeline/execution.log
```

Log these milestones:
1. After loading dataset: `Dataset loaded | n_obs={N} | n_vars={N} | default_rate={X}%`
2. After completing analysis: `Analysis complete | impute_outliers={N} | exclude={N} | high_corr_pairs={N}`

## Inputs

You will receive a `RUN_DIR` path (e.g., `runs/2026-03-14_143022/`). All output paths below are relative to this directory.

- **Dataset path:** `data/loans.csv` (project root, shared across runs)
- **Target variable:** `Creditability` (1 = default)

## Toolkit Import

```python
import sys; sys.path.insert(0, 'src')
import pdtoolkit as pdt
```

## Functions to Use

- `pdt.univariate(db)` — univariate statistics for all variables
- `pdt.nzv(db)` — detect near-zero variance risk factors

Additionally use standard pandas/numpy for:
- Missing value counts and rates
- Correlation matrix (numeric variables)
- Distribution plots

## Analysis Steps

1. Load the dataset and identify variable types (numeric vs categorical)
2. Run `pdt.univariate()` to get distribution statistics per variable
3. Run `pdt.nzv()` to identify near-zero variance variables
4. Calculate missing value rates for all variables
5. Compute correlation matrix for numeric variables
6. Identify highly correlated pairs (|r| > 0.7) — this informs Stage 03 and reduces wasted computation on redundant variables
7. For each variable, document a recommended action: keep / impute-special-cases / impute-outliers / exclude
8. Generate plots:
   - Missing rates bar chart → `{RUN_DIR}/figures/01_missing_rates.png`
   - Distribution plots for all variables → `{RUN_DIR}/figures/01_distributions.png`
   - Correlation matrix heatmap → `{RUN_DIR}/figures/01_correlation_matrix.png`

## Self-Assessment (before writing notebook)

Check all four conditions before proceeding to write outputs:

1. **Every variable has a documented action** (keep / impute-special-cases / impute-outliers / exclude)
2. **For variables flagged for imputation:** the missing pattern is documented (random vs systematic)
3. **For variables flagged for exclusion:** the reason is stated (high missingness / zero variance / wrong type)
4. **Outlier thresholds are justified** (IQR vs percentile choice explained)

If any check fails, revise the analysis before writing.

## Output Files

Write these files:
- `{RUN_DIR}/notebooks/01_data_quality.ipynb` — following notebook-writer skill conventions
- `{RUN_DIR}/figures/01_missing_rates.png`
- `{RUN_DIR}/figures/01_distributions.png`
- `{RUN_DIR}/figures/01_correlation_matrix.png`
- `{RUN_DIR}/pipeline/stage_01.md`
- `{RUN_DIR}/pipeline/stage_01_fixes.md` — fix-proposer diagnostic output

## stage_01.md Template

Write `{RUN_DIR}/pipeline/stage_01.md` with exactly these fields:

```
dataset_path: data/loans.csv
n_observations: [int]
n_variables: [int]
target_variable: Creditability
target_default_rate: [float]
numeric_variables: [list]
categorical_variables: [list]
variables_to_impute_sc: [list with method per variable]
variables_to_impute_outliers: [list with method per variable]
variables_to_exclude: [list with reason per variable]
high_correlation_pairs: [list of (var1, var2, r) tuples]
data_quality_flags: [list or "None"]
```

## Return Message

After writing all outputs, verify using the output-verifier skill checklist. Then run the fix-proposer diagnostic protocol and write `{RUN_DIR}/pipeline/stage_01_fixes.md`. Then return a message in this format:

```
Stage 01 complete. Notebook: {RUN_DIR}/notebooks/01_data_quality.ipynb
[N] variables analysed. [N] need special case imputation. [N] need outlier imputation.
[N] recommended for exclusion. [N] high-correlation pairs flagged.
Fix proposals: [N] issues ([N] critical) — see stage_01_fixes.md
Awaiting human review before proceeding to Stage 02.
```
