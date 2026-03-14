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

## Inputs

You will receive a `RUN_DIR` path (e.g., `runs/2026-03-14_143022/`). All output paths below are relative to this directory.

- **Clean dataset:** `{RUN_DIR}/data/loans_clean.csv`
- **Model parameters:** `{RUN_DIR}/pipeline/model_params.json`
- **Target central tendency:** provided by human at checkpoint, or defaulting to development sample default rate
- **Stage summary:** `{RUN_DIR}/pipeline/stage_04.md`

## Toolkit Import

```python
import sys; sys.path.insert(0, 'src')
import pdtoolkit as pdt
```

## Functions to Use

- `pdt.rs_calibration(rs, dr, w, ct, min_pd, method)` — calibrate observed default rates
- `pdt.score_to_prob(scores, score, odd, pdo)` — convert scores to probabilities
- `pdt.normal_test(pdc, odr)` — normal test for calibration validation
- `pdt.hhi(x)` — Herfindahl-Hirschman Index for grade concentration

## Analysis Steps

1. Load clean dataset and model parameters
2. Compute scores for all observations using model coefficients and `pdt.scaled_score()`
3. Define initial rating grade boundaries based on score distribution
4. Calculate observed default rate per grade
5. Run `pdt.rs_calibration()` with target central tendency
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

After writing all outputs, verify using the output-verifier skill checklist. Then return:

```
Stage 05 complete. Notebook: {RUN_DIR}/notebooks/05_calibration.ipynb
[N] rating grades. Target CT: [X]%, Achieved CT: [X]%. Grade PD ordering valid: [yes/no].
Stress test: +1% shift produces CT [X]%, +2% shift produces CT [X]%.
Flags: [list or "None"].
Awaiting human review before proceeding to Stage 06.
```
