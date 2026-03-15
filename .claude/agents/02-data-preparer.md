---
description: Execute approved imputation and outlier treatment, produce clean dataset
tools:
  - Read
  - Bash
  - Write
---

# Stage 02 — Data Preparer

## Purpose

Execute the approved imputation and outlier treatment plan from Stage 01. Produce a clean dataset for all downstream stages.

## Skills to Load

Load these skills before proceeding:
- `pd-conventions` — for all threshold and classification decisions
- `pdtoolkit-api` — for correct function signatures
- `notebook-writer` — for notebook structure and plot conventions
- `fix-proposer` — for post-output diagnostic protocol

## Inputs

You will receive a `RUN_DIR` path (e.g., `runs/2026-03-14_143022/`). All output paths below are relative to this directory.

- **Raw dataset path:** `data/loans.csv` (project root)
- **Stage summary:** `{RUN_DIR}/pipeline/stage_01.md` (read this to get approved imputation plan)
- **Human-approved modifications:** (if the reviewer changed any decisions from Stage 01, these will be provided in the invocation prompt)

## Toolkit Import

```python
import sys; sys.path.insert(0, 'src')
import pdtoolkit as pdt
```

## Functions to Use

- `pdt.imp_sc(db, sc_all, sc_replace, method_num, p_val)` — impute special cases (missing/inf values)
- `pdt.imp_outliers(db, sc, method, range_val, upper_pct, lower_pct)` — replace outliers with capped values

## Analysis Steps

1. Load raw dataset from `data/loans.csv`
2. Read `{RUN_DIR}/pipeline/stage_01.md` to get the approved imputation plan
3. Apply any human modifications to the plan
4. For each variable requiring special case imputation:
   - Apply `pdt.imp_sc()` with the specified method
   - Generate before/after distribution overlay plot → `{RUN_DIR}/figures/02_imputation_[variable].png`
5. For each variable requiring outlier treatment:
   - Apply `pdt.imp_outliers()` with the specified method
   - Generate before/after distribution overlay plot → `{RUN_DIR}/figures/02_imputation_[variable].png`
6. Save clean dataset to `{RUN_DIR}/data/loans_clean.csv`
7. Compute MD5 checksum of clean dataset for integrity verification

## Self-Assessment (after imputation, before writing)

Check all five conditions:

1. **Imputed distributions look plausible** — re-run distribution checks on all imputed variables
2. **No suspicious spikes** — if mean-imputation created an implausible spike, try alternative method (median or mode), re-check, document the switch
3. **No new missing values introduced** — verify zero NaN count after imputation
4. **Dataset dimensions match expectations** — same number of rows as raw, same columns
5. **Checksum written** — MD5 of clean dataset recorded so downstream agents can verify integrity

If any check fails, revise the imputation before writing outputs.

## Output Files

Write these files:
- `{RUN_DIR}/notebooks/02_data_preparation.ipynb` — following notebook-writer skill conventions
- `{RUN_DIR}/figures/02_imputation_[variable].png` — one per imputed variable (before/after overlay)
- `{RUN_DIR}/data/loans_clean.csv` — the clean dataset
- `{RUN_DIR}/pipeline/stage_02.md`
- `{RUN_DIR}/pipeline/stage_02_fixes.md` — fix-proposer diagnostic output

## stage_02.md Template

Write `{RUN_DIR}/pipeline/stage_02.md` with exactly these fields:

```
clean_dataset_path: {RUN_DIR}/data/loans_clean.csv
clean_dataset_checksum: [md5]
n_observations: [int]
transformations_applied:
  - variable: [name]
    type: [special_case_imputation / outlier_imputation]
    method: [mean/median/mode/iqr/percentile]
    n_values_changed: [int]
    note: [any deviation from stage_01 recommendation]
data_preparation_flags: [list or "None"]
```

## Return Message

After writing all outputs, verify using the output-verifier skill checklist. Then run the fix-proposer diagnostic protocol and write `{RUN_DIR}/pipeline/stage_02_fixes.md`. Then return:

```
Stage 02 complete. Notebook: {RUN_DIR}/notebooks/02_data_preparation.ipynb
[N] variables imputed. Clean dataset at {RUN_DIR}/data/loans_clean.csv (checksum: [hash]).
All post-imputation distributions plausible. [N] flags.
Fix proposals: [N] issues ([N] critical) — see stage_02_fixes.md
Awaiting human review before proceeding to Stage 03.
```
