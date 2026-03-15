---
description: Calibrate rating scale to portfolio target central tendency, assign grades and PDs
tools:
  - Read
  - Bash
  - Write
---

# Stage 05 — Calibrator

## Purpose

Calibrate the rating scale to the portfolio target central tendency. Assign rating grades and calibrated PDs. Perform stress testing.

## Skills to Load

Load these skills before proceeding:
- `pd-conventions` — for score scale convention, validation thresholds
- `pdtoolkit-api` — for correct function signatures
- `notebook-writer` — for notebook structure and plot conventions
- `fix-proposer` — for post-output diagnostic protocol

## Inputs

You will receive a `RUN_DIR` path (e.g., `runs/2026-03-14_143022/`). All output paths below are relative to this directory.

- **Binned dataset:** `{RUN_DIR}/data/loans_binned.csv` (from Stage 03 — variables replaced with optbinning bin labels)
- **Model parameters:** `{RUN_DIR}/pipeline/model_params.json`
- **Target central tendency:** provided by human at checkpoint, or defaulting to development sample default rate
- **Stage summary:** `{RUN_DIR}/pipeline/stage_04.md`

## Toolkit Import

```python
import sys; sys.path.insert(0, 'src')
import pdtoolkit as pdt
```

## Functions to Use

- `pdt.rs_calibration(rs, dr, w, ct, min_pd, method)` — calibrate PDs to target central tendency
- `pdt.score_to_prob(scores, score, odd, pdo)` — convert scores to probabilities
- `pdt.normal_test(pdc, odr)` — normal test for calibration validation
- `pdt.hhi(x)` — Herfindahl-Hirschman Index for grade concentration

## Critical: Model-Predicted PDs vs Observed Default Rates

The calibration input must be **model-predicted PDs per grade** (average of logistic regression predicted probabilities within each grade), **NOT** the raw observed default rates. Using observed DRs makes the calibration circular — the scaling factor collapses to ~1.0 when the target CT equals the in-sample default rate, producing calibrated PDs that are identical to observed DRs.

Correct approach:
1. Compute predicted probabilities for each observation from the logistic regression (intercept + sum of coef × WoE)
2. Assign observations to grades based on scores
3. Calculate the **mean predicted probability** per grade → this is the `dr` column for `rs_calibration()`
4. Calculate the **observed default rate** per grade separately → this is used only for comparison on the chart
5. Pass the model-predicted PDs (not observed DRs) to `pdt.rs_calibration()`

The rating scale chart must show **both** the calibrated PDs (bars) and observed DRs (line) as distinct series — they should NOT overlap perfectly. A visible gap between them is expected and healthy: it shows the calibration is smoothing the model's predictions rather than reproducing actuals.

## Analysis Steps

1. Load clean dataset and model parameters
2. Compute predicted probabilities and scores for all observations using model coefficients, WoE mappings, and `pdt.scaled_score()`
3. Define initial rating grade boundaries based on score distribution
4. For each grade, calculate **two** quantities separately:
   - **Model-predicted PD:** mean of logistic regression predicted probabilities within the grade
   - **Observed default rate:** actual default rate within the grade (for comparison only)
5. Run `pdt.rs_calibration(rs, dr='predicted_pd', w='n_obligors', ct=TARGET_CT)` using the model-predicted PDs
6. Run self-assessment checks (see below) — iterate if needed
7. Perform stress test: apply +1% and +2% PD shifts across all grades
8. Add TTC vs PIT commentary based on central tendency used
9. Generate all plots and write outputs

## Self-Assessment (5 checks)

### 1. Central tendency achieved vs target
- Tolerance: ±0.5%
- If breached: adjust calibration anchor and retry

### 2. Grade PD ordering
- Calibrated PD must be strictly increasing from best to worst grade
- If any grade out of order: adjust grade boundaries, re-calibrate, recheck

### 3. Grade population distribution
- No grade should contain < 2% or > 40% of the portfolio
- If any grade outside bounds: revise grade cut-offs, document

### 4. Worst grade PD
- Worst grade calibrated PD must be < 100%

### 5. Grade concentration
- Calculate `pdt.hhi()` on grade populations
- Flag if concentration is excessive (single grade dominates)

## Output Files

Write these files:
- `{RUN_DIR}/notebooks/05_calibration.ipynb` — following notebook-writer skill conventions
- `{RUN_DIR}/figures/05_rating_scale.png` — rating scale with calibrated PDs
- `{RUN_DIR}/figures/05_grade_distribution.png` — obligor distribution across grades
- `{RUN_DIR}/figures/05_stress_test.png` — stressed PD distributions
- `{RUN_DIR}/pipeline/stage_05.md`
- `{RUN_DIR}/pipeline/stage_05_fixes.md` — fix-proposer diagnostic output

## stage_05.md Template

Write `{RUN_DIR}/pipeline/stage_05.md` with exactly these fields:

```
model_params_path: {RUN_DIR}/pipeline/model_params.json
target_central_tendency: [float]
achieved_central_tendency: [float]
n_rating_grades: [int]
rating_scale:
  - grade: [label]
    score_range: [min, max]
    calibrated_pd: [float]
    n_obligors: [int]
    pct_portfolio: [float]
grade_pd_ordering_valid: [true/false]
stress_test:
  shift_1pct_ct: [float]
  shift_2pct_ct: [float]
calibration_flags: [list or "None"]
```

## Return Message

After writing all outputs, verify using the output-verifier skill checklist. Then run the fix-proposer diagnostic protocol and write `{RUN_DIR}/pipeline/stage_05_fixes.md`. Then return:

```
Stage 05 complete. Notebook: {RUN_DIR}/notebooks/05_calibration.ipynb
[N] rating grades. Target CT: [X]%, Achieved CT: [X]%. Grade PD ordering valid: [yes/no].
Stress test: +1% shift produces CT [X]%, +2% shift produces CT [X]%.
Flags: [list or "None"].
Fix proposals: [N] issues ([N] critical) — see stage_05_fixes.md
Awaiting human review before proceeding to Stage 06.
```
