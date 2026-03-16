---
description: Run full model validation suite, produce validation report
tools:
  - Read
  - Bash
  - Write
---

# Stage 06 — Validator

## Purpose

Run the full model validation suite. Produce a comprehensive validation report with an overall pass/fail assessment.

**IMPORTANT:** This stage does NOT iterate to improve the model — the model is fixed. The agent iterates only on *completeness and precision of reporting*.

## Skills to Load

Load these skills before proceeding:
- `pd-conventions` — for validation thresholds, regulatory flags
- `pdtoolkit-api` — for correct function signatures
- `notebook-writer` — for notebook structure and plot conventions
- `fix-proposer` — for post-output diagnostic protocol

## Execution Logging

Append milestone entries to `{RUN_DIR}/pipeline/execution.log` using:
```bash
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] [stage-06] message" >> {RUN_DIR}/pipeline/execution.log
```

Log this milestone:
1. After validation: `Validation complete | overall={PASS/PASS WITH FLAGS/FAIL} | dp={PASS/FAIL} | pp={PASS/FAIL} | flags={N}`

## Inputs

You will receive a `RUN_DIR` path (e.g., `runs/2026-03-14_143022/`). All output paths below are relative to this directory.

- **Binned dataset:** `{RUN_DIR}/data/loans_binned.csv` (from Stage 03 — variables replaced with optbinning bin labels)
- **Model parameters:** `{RUN_DIR}/pipeline/model_params.json`
- **Stage summary:** `{RUN_DIR}/pipeline/stage_05.md`

## Toolkit Import

```python
import sys; sys.path.insert(0, 'src')
import pdtoolkit as pdt
```

## Functions to Use

- `pdt.dp_testing(app_port, def_ind, pdc, auc_test, alternative, alpha)` — discriminatory power test
- `pdt.pp_testing(rating_label, pdc, no, nb, alpha)` — predictive power tests (binomial, Jeffreys, z-score, Hosmer-Lemeshow)
- `pdt.power(rating_label, pdc, no, nb, alpha, sim_num)` — Monte Carlo power of PP tests
- `pdt.homogeneity(app_port, def_ind, rating, segment, segment_num, alpha)` — homogeneity test
- `pdt.heterogeneity(app_port, def_ind, rating, alpha)` — heterogeneity test
- `pdt.psi(base, target, bins, alpha)` — Population Stability Index
- `pdt.segment_vld(model, db, target, predictors, min_leaf, alpha)` — segment validation
- `pdt.fairness_vld(db, sensitive, obs_outcome, mod_outcome, ...)` — fairness validation (if sensitive attributes available)

## Analysis Steps

1. Load clean dataset, model parameters, and calibration results from `{RUN_DIR}/pipeline/stage_05.md`
2. Reconstruct scores and grade assignments for all observations
3. Run discriminatory power tests (AUC, Gini, KS)
4. Run predictive power tests per grade
5. Run homogeneity test
6. Run heterogeneity test
7. Calculate PSI between 80/20 random split (development vs holdout proxy)
8. Run stability analysis: split into two random halves, compute AUC on each
9. Run segment validation
10. If time variable available in data: run backtesting (train on earlier, test on later)
11. If sensitive attributes available: run fairness validation
12. Compile overall assessment
13. Generate all plots and write outputs

## Self-Assessment (4 checks on reporting quality)

### 1. Marginal results bootstrap
- For any test result with p-value within 0.01 of its threshold:
  - Run the test on a bootstrapped sample (n=1000) to get a confidence interval around the p-value
  - Report both point estimate and CI

### 2. Small-grade homogeneity power
- For any homogeneity failure at a specific grade:
  - Check whether it is a small grade (< 5% of portfolio)
  - If small: note that the test may lack statistical power

### 3. Stability half-split
- Split development sample into two random halves
- Run AUC on each half
- If |AUC_half1 - AUC_half2| > 0.05: flag as potential overfitting

### 4. Overall assessment statement
- Produce a single clear statement: PASS / PASS WITH FLAGS / FAIL
- List all flags for regulatory documentation

## Output Files

Write these files:
- `{RUN_DIR}/notebooks/06_validation.ipynb` — following notebook-writer skill conventions
- `{RUN_DIR}/figures/06_roc_curve.png`
- `{RUN_DIR}/figures/06_ks_plot.png`
- `{RUN_DIR}/figures/06_homogeneity_test.png`
- `{RUN_DIR}/figures/06_pp_test.png`
- `{RUN_DIR}/figures/06_stability.png`
- `{RUN_DIR}/pipeline/stage_06.md`
- `{RUN_DIR}/pipeline/stage_06_fixes.md` — fix-proposer diagnostic output

## stage_06.md Template

Write `{RUN_DIR}/pipeline/stage_06.md` with exactly these fields:

```
discriminatory_power:
  auc: [float]
  gini: [float]
  ks: [float]
  dp_test_pvalue: [float]
  dp_test_result: [PASS/FAIL]
  stability_auc_half1: [float]
  stability_auc_half2: [float]
  stability_result: [PASS/FAIL]
predictive_power:
  hosmer_lemeshow_pvalue: [float]
  hosmer_lemeshow_result: [PASS/FAIL]
  grade_results:
    - grade: [label]
      binomial_pvalue: [float]
      binomial_result: [PASS/FAIL]
      jeffreys_pvalue: [float]
      jeffreys_result: [PASS/FAIL]
homogeneity:
  overall_pvalue: [float]
  overall_result: [PASS/FAIL]
  grade_failures: [list or "None"]
heterogeneity:
  pvalue: [float]
  result: [PASS/FAIL]
psi: [float]
psi_result: [PASS/FAIL if applicable]
regulatory_flags: [list or "None"]
overall_assessment: [PASS / PASS WITH FLAGS / FAIL]
```

## Return Message

After writing all outputs, verify using the output-verifier skill checklist. Then run the fix-proposer diagnostic protocol and write `{RUN_DIR}/pipeline/stage_06_fixes.md`. Then return:

```
Stage 06 complete. Notebook: {RUN_DIR}/notebooks/06_validation.ipynb
Overall assessment: [PASS / PASS WITH FLAGS / FAIL]
DP: AUC [X] [PASS/FAIL]. Stability: [X] difference [PASS/FAIL].
PP: [summary]. Hosmer-Lemeshow [PASS/FAIL].
Homogeneity: [summary]. [any grade-level detail].
Heterogeneity: [PASS/FAIL].
PSI: [X] [PASS/FAIL].
Fix proposals: [N] issues ([N] critical) — see stage_06_fixes.md
Awaiting human review and sign-off.
```

**Human checkpoint note:** This is the final review gate. The human reviews the full validation notebook and stage_06.md. No further automated processing occurs after this point.
