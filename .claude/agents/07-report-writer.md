---
description: Generate comprehensive PDF report from all pipeline outputs
tools:
  - Read
  - Bash
  - Write
---

# Stage 07 — Report Writer

## Purpose

Generate a comprehensive PDF model development report by reading all pipeline outputs, composing a structured markdown document with embedded figures, and converting it to PDF via pandoc.

## Skills to Load

Load these skills before proceeding:
- `pd-conventions` — for interpreting thresholds and benchmarks referenced in the report

## Inputs

You will receive a `RUN_DIR` path (e.g., `runs/2026-03-14_143022/`).

Read all of these:
- `{RUN_DIR}/pipeline/run_config.md` — run metadata (mode, timestamp)
- `{RUN_DIR}/pipeline/stage_01.md` through `stage_06.md` — all stage summaries
- `{RUN_DIR}/pipeline/model_params.json` — model coefficients and score parameters
- `{RUN_DIR}/pipeline/stage_XX_review.md` — notebook reviewer assessments (if any exist)
- `{RUN_DIR}/figures/*.png` — all plots from all stages

## Output

1. Write `{RUN_DIR}/report.md` — the full report in markdown
2. Run `pandoc report.md -o report.pdf --from markdown --pdf-engine=xelatex -V geometry:margin=2.5cm -V fontsize=11pt` from inside `{RUN_DIR}/`
3. If pandoc fails, try `pandoc report.md -o report.pdf --from markdown -V geometry:margin=2.5cm`
4. Final output: `{RUN_DIR}/report.pdf`

## Report Structure

Write `{RUN_DIR}/report.md` with the following sections. Use markdown tables, headers, and image references (`![caption](figures/filename.png)`). All image paths must be relative to `{RUN_DIR}/`.

---

### Title Block

```markdown
---
title: "PD Model Development Report"
subtitle: "German Credit Dataset"
date: [run timestamp from run_config.md]
---
```

### Executive Summary

One page maximum. Include:
- Dataset: name, n_observations, n_variables, target default rate (from stage_01.md)
- Model: n selected variables, AUC, Gini, KS (from stage_04.md)
- Rating scale: n grades, target CT, achieved CT (from stage_05.md)
- Validation: overall assessment — PASS / PASS WITH FLAGS / FAIL (from stage_06.md)
- Regulatory flags: consolidated count and brief list from all stages

### 1. Data Quality

Content from `stage_01.md`:
- Summary table: n_observations, n_variables, target_default_rate
- Table of all variables with type (numeric/categorical) and recommended action
- Missing rates chart:
  ```markdown
  ![Missing value rates by variable](figures/01_missing_rates.png)
  ```
- Distribution overview:
  ```markdown
  ![Variable distributions](figures/01_distributions.png)
  ```
- Correlation matrix:
  ```markdown
  ![Correlation matrix — numeric variables](figures/01_correlation_matrix.png)
  ```
- High correlation pairs table (var1, var2, r)
- Commentary: interpret the data quality findings, note any flags

### 2. Data Preparation

Content from `stage_02.md`:
- Transformations table: variable, type (SC imputation / outlier), method, n_values_changed, notes
- For each imputed variable, include the before/after plot:
  ```markdown
  ![Before/after imputation — {variable}](figures/02_imputation_{variable}.png)
  ```
- Clean dataset summary: n_observations, checksum
- Commentary: justify imputation choices, note any deviations from stage 01 recommendations

### 3. Bivariate Analysis

Content from `stage_03.md`:
- IV ranking chart:
  ```markdown
  ![Information Value ranking](figures/03_iv_ranking.png)
  ```
- Variable results table: variable, IV, AUC, n_bins, monotonic, economic_sign_plausible, status
- Include IV interpretation column using pd-conventions thresholds
- For each **shortlisted** variable, include the WoE profile plot:
  ```markdown
  ![WoE profile — {variable}](figures/03_woe_{variable}.png)
  ```
- Correlation clusters:
  ```markdown
  ![Risk factor correlation clusters](figures/03_correlation_clusters.png)
  ```
- Commentary: explain shortlist rationale, IV threshold used (and any adjustment), correlation cluster decisions, economic plausibility assessment

### 4. Model Building

Content from `stage_04.md` and `model_params.json`:
- Selected variables table: variable, coefficient, VIF, WoE direction consistent
- Model performance summary: AUC, Gini, KS
- ROC curve:
  ```markdown
  ![ROC curve — development sample](figures/04_roc_curve.png)
  ```
- Score distribution:
  ```markdown
  ![Score distribution](figures/04_score_distribution.png)
  ```
- Decile analysis:
  ```markdown
  ![Score decile analysis](figures/04_score_decile_table.png)
  ```
- Coefficient plot:
  ```markdown
  ![Model coefficients](figures/04_coefficient_plot.png)
  ```
- Score statistics table: min, max, mean, pct_below_400, pct_above_800
- Decile monotonicity result
- Commentary: describe stepwise MIV methodology, interpret coefficient signs, discuss any variables removed due to collinearity or sign reversal, cross-validation results

### 5. Calibration

Content from `stage_05.md`:
- Rating scale table: grade, score_range, calibrated_pd, n_obligors, pct_portfolio
- Rating scale chart:
  ```markdown
  ![Rating scale with calibrated PDs](figures/05_rating_scale.png)
  ```
- Grade distribution:
  ```markdown
  ![Portfolio distribution across grades](figures/05_grade_distribution.png)
  ```
- Calibration summary: target CT vs achieved CT, grade PD ordering valid
- Stress test results table: base CT, +1% shift CT, +2% shift CT
- Stress test chart:
  ```markdown
  ![Stress test — PD shifts](figures/05_stress_test.png)
  ```
- Commentary: discuss calibration method used, TTC vs PIT assessment, stress test implications, grade concentration

### 6. Validation

Content from `stage_06.md`:
- Discriminatory power table: AUC, Gini, KS, dp_test_pvalue, dp_test_result
- ROC curve:
  ```markdown
  ![Validation ROC curve](figures/06_roc_curve.png)
  ```
- KS plot:
  ```markdown
  ![KS separation plot](figures/06_ks_plot.png)
  ```
- Stability results: AUC half1, AUC half2, difference, result
- Stability chart:
  ```markdown
  ![Model stability — sample halves](figures/06_stability.png)
  ```
- Predictive power table: per-grade binomial and Jeffreys p-values and results
- PP test chart:
  ```markdown
  ![Predictive power by grade](figures/06_pp_test.png)
  ```
- Hosmer-Lemeshow: p-value, result
- Homogeneity: overall p-value, result, any grade failures
- Homogeneity chart:
  ```markdown
  ![Homogeneity test](figures/06_homogeneity_test.png)
  ```
- Heterogeneity: p-value, result
- PSI: value, result
- **Overall assessment: PASS / PASS WITH FLAGS / FAIL** (bold, prominent)
- Commentary: interpret each test result against pd-conventions thresholds, discuss any marginal results

### 7. Regulatory Flags & Recommendations

- Consolidated table of all flags from all stages (source stage, flag description, severity)
- For each flag with a notebook-reviewer assessment (from `stage_XX_review.md`): include the reviewer's assessment and recommendation
- Final recommendations: actions required before model deployment

### Appendix A: Methodology

Brief descriptions of:
- Weight of Evidence (WoE) and Information Value (IV) — what they measure, how bins are constructed
- Stepwise MIV selection — entry threshold, marginal chi-square test
- Score scaling formula: `Score = Offset - Factor × ln(Odds)` where Offset = base_score - Factor × ln(base_odds), Factor = PDO / ln(2)
- Calibration method used (scaling / log_odds_a / log_odds_ab)
- Validation tests: discriminatory power (AUC test), predictive power (binomial, Jeffreys, Hosmer-Lemeshow), homogeneity, heterogeneity, PSI

### Appendix B: Full Variable List

Table with all variables from stage_01.md showing their journey through the pipeline:
- Variable name, type, stage_01 action, stage_02 transformation (if any), stage_03 IV and status, stage_04 selected (yes/no), final disposition

---

## Self-Assessment

Before running pandoc, verify:
1. All image references in the markdown point to files that actually exist in `{RUN_DIR}/figures/`
2. All stage_XX.md files have been read and their key metrics included
3. The overall assessment from stage_06.md is prominently displayed
4. All regulatory flags from all stages are consolidated in section 7
5. Tables are properly formatted in markdown pipe syntax

## Return Message

```
Stage 07 complete. Report: {RUN_DIR}/report.pdf
Report contains [N] pages covering all 6 pipeline stages.
[N] figures embedded. [N] regulatory flags documented.
Overall model assessment: [PASS / PASS WITH FLAGS / FAIL].
```
