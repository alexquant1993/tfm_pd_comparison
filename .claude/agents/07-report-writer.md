---
description: Generate comprehensive DOCX report from all pipeline outputs
tools:
  - Read
  - Bash
  - Write
---

# Stage 07 — Report Writer

## Purpose

Generate a comprehensive Word (.docx) model development report by reading all pipeline outputs, composing a structured markdown document with embedded figures, and converting it to DOCX via pandoc with a styled reference template.

## Skills to Load

Load these skills before proceeding:
- `pd-conventions` — for interpreting thresholds and benchmarks referenced in the report

## Execution Logging

Append milestone entries to `{RUN_DIR}/pipeline/execution.log` using:
```bash
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] [stage-07] message" >> {RUN_DIR}/pipeline/execution.log
```

Log this milestone:
1. After report generation: `Report generated | path={RUN_DIR}/report.docx`

## Inputs

You will receive a `RUN_DIR` path (e.g., `runs/2026-03-14_143022/`).

Read all of these:
- `{RUN_DIR}/pipeline/run_config.md` — run metadata (mode, timestamp)
- `{RUN_DIR}/pipeline/stage_01.md` through `stage_03.md` — stages 01-03 summaries
- `{RUN_DIR}/pipeline/stage_04a.md`, `stage_04b.md`, `stage_04c.md` — three model builder summaries
- `{RUN_DIR}/pipeline/stage_04x.md` — model comparison and champion selection
- `{RUN_DIR}/pipeline/stage_05.md`, `stage_06.md` — calibration and validation summaries
- `{RUN_DIR}/pipeline/model_params.json` — champion model coefficients and score parameters
- `{RUN_DIR}/pipeline/model_params_miv.json`, `model_params_xgb.json`, `model_params_fwd.json` — per-method model parameters (for comparison detail)
- `{RUN_DIR}/pipeline/stage_XX_review.md` — notebook reviewer assessments (if any exist)
- `{RUN_DIR}/pipeline/fixes_summary.md` — consolidated fix proposals from all stages (if exists)
- `{RUN_DIR}/figures/*.png` — all plots from all stages

## Output

1. Write `{RUN_DIR}/report.md` — the full report in markdown
2. Run `pandoc report.md -o report.docx --from markdown --reference-doc=../../src/templates/doc_template.docx` from inside `{RUN_DIR}/`
3. If pandoc fails (e.g., reference doc not found), try `pandoc report.md -o report.docx --from markdown`
4. Run `/c/Python313/python.exe ../../src/scripts/format_report.py report.docx` to apply table formatting (borders, header shading, alternating rows)
5. Final output: `{RUN_DIR}/report.docx`

## Report Structure

Write `{RUN_DIR}/report.md` with the following sections. Use markdown tables, headers, and image references (`![caption](figures/filename.png)`). All image paths must be relative to `{RUN_DIR}/`.

**IMPORTANT — No manual numbering in headings.** The Word template applies automatic heading numbering. Do NOT include numbers in markdown headings. Write `# Data Quality` not `# 1. Data Quality`. Write `## Dataset Overview` not `## 1.1 Dataset Overview`. Any explicit numbers will duplicate the template's auto-numbering.

**Writing style:** This report will be read by model validators, risk committees, and regulators. Every table and figure MUST be followed by at least one interpretive paragraph explaining what it shows, why it matters, and what conclusions to draw. Do not simply list metrics — explain their significance in the context of PD model development. Use plain language where possible; define technical terms on first use. Aim for a report that a senior risk manager can read end-to-end and understand the model's strengths, weaknesses, and deployment readiness without needing to consult the notebooks.

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
- **Write a 1-paragraph narrative** summarizing the model development outcome: what was built, how well it performs, and whether it is ready for deployment. This should read as a standalone summary for executives who will not read the full report.

### Data Quality

Content from `stage_01.md`:
- Summary table: n_observations, n_variables, target_default_rate
- **After the summary table:** Write a paragraph discussing the dataset size adequacy for PD modelling, whether the default rate is within a reasonable range for model development, and any concerns about class imbalance.
- Table of all variables with type (numeric/categorical) and recommended action
- Missing rates chart:
  ```markdown
  ![Missing value rates by variable](figures/01_missing_rates.png)
  ```
- **After the missing rates chart:** Interpret the missing value pattern — are there any variables with concerning missing rates? If no missing values, note this as a data quality strength and discuss whether the absence of missingness is unusual (e.g., pre-cleaned data vs. raw origination data).
- Distribution overview:
  ```markdown
  ![Variable distributions](figures/01_distributions.png)
  ```
- **After the distributions chart:** Write 1-2 paragraphs interpreting the distributions — which numeric variables are skewed (and in which direction), which categorical variables are dominated by a single category (>70% in one level), and what this implies for WoE binning (e.g., skewed numerics may need fewer bins, dominant categories may produce small event counts in minority bins).
- Correlation matrix:
  ```markdown
  ![Correlation matrix — numeric variables](figures/01_correlation_matrix.png)
  ```
- High correlation pairs table (var1, var2, r)
- **After the correlation matrix:** Identify and discuss any pairs above r=0.5. Explain whether high correlations could cause multicollinearity in the logistic regression stage and whether variable exclusion may be warranted at that point. If no strong correlations exist, note this as positive for model building.
- **Outlier discussion:** After the outlier table, discuss the magnitude of outlier rates. For variables with outlier rates above 5%, explain why IQR capping is appropriate and what effect it will have on the distribution tail. Note whether outliers are one-sided or two-sided.

### Data Preparation

Content from `stage_02.md`:
- Transformations table: variable, type (SC imputation / outlier), method, n_values_changed, notes
- **After the transformations table:** Write a paragraph explaining the overall data preparation strategy — why IQR capping was chosen over alternatives (e.g., winsorization, log transformation), and whether the number of affected observations is material enough to change the variable's risk signal.
- For each imputed variable, include the before/after plot:
  ```markdown
  ![Before/after imputation — {variable}](figures/02_imputation_{variable}.png)
  ```
- **After each before/after plot:** For each treated variable, describe what the plot shows — where the distribution changed (left tail, right tail, or both), whether the treatment preserved the overall shape and central tendency, and any concerns about information loss from capping extreme values.
- Clean dataset summary: n_observations, checksum
- **After clean dataset summary:** Confirm that no observations were dropped during preparation and discuss whether the clean dataset is ready for bivariate analysis.

### Bivariate Analysis

Content from `stage_03.md`:
- IV ranking chart:
  ```markdown
  ![Information Value ranking](figures/03_iv_ranking.png)
  ```
- **After the IV ranking chart:** Interpret the IV distribution — how many variables fall into each IV category (strong >0.30, medium 0.10-0.30, weak 0.02-0.10, useless <0.02)? Is the predictive power concentrated in one or two dominant variables, or spread across many? Discuss what this means for model stability — a model relying heavily on one variable is fragile, while diversified predictive power is more robust.
- Variable results table: variable, IV, AUC, n_bins, monotonic, economic_sign_plausible, status
- Include IV interpretation column using pd-conventions thresholds
- For each **shortlisted** variable, include the WoE profile plot:
  ```markdown
  ![WoE profile — {variable}](figures/03_woe_{variable}.png)
  ```
- **After each WoE profile plot:** Write 2-3 sentences explaining the WoE pattern for this specific variable. Does it show a clear, economically intuitive relationship with default risk? (e.g., "Higher account balances are associated with lower default risk, as shown by the increasing WoE across bins — this is consistent with the economic expectation that borrowers with larger savings are more financially stable.") If non-monotonic, explain the specific bin where monotonicity breaks and whether the deviation is economically justifiable or a statistical artifact.
- Correlation clusters:
  ```markdown
  ![Risk factor correlation clusters](figures/03_correlation_clusters.png)
  ```
- **After correlation clusters:** Discuss which variables cluster together and the rationale for keeping both members of a correlated pair vs. dropping one. If no clusters required action, explain why (e.g., all pairwise correlations below the threshold).
- **Shortlist rationale:** Write a concluding paragraph for this section explaining the IV threshold used for shortlisting, any manual overrides or adjustments made, and why the final shortlist represents the best trade-off between predictive power and model parsimony.

### Model Building

Content from `stage_04a.md`, `stage_04b.md`, `stage_04c.md`, `stage_04x.md`, and `model_params.json`.

Three variable selection methods were evaluated in parallel. This section presents the comparison and the champion model in detail.

#### Variable Selection Approaches

Describe each of the three variable selection methods used:
1. **MIV Stepwise** (stage 04a) — Marginal Information Value as the entry criterion, with chi-square p-value confirmation. Variables enter the model one at a time based on their marginal contribution to model discrimination.
2. **XGBoost Feature Importance** (stage 04b) — An XGBoost classifier trained on WoE-encoded features to rank variables by gain-based importance. Top variables by cumulative importance are selected, then a logistic regression is fitted on the selected WoE features.
3. **Forward Stepwise** (stage 04c) — Classical forward stepwise selection using the likelihood ratio test p-value as the entry criterion. Variables are added one at a time, selecting the variable that produces the most significant improvement.

**After the descriptions:** Write a paragraph explaining why comparing multiple selection methods strengthens the model development process — it provides evidence of variable stability (variables selected by all methods are robust), reduces method-specific bias, and satisfies regulatory expectations for challenger model analysis.

#### Model Comparison

Content from `stage_04x.md`:
- **Comparison table** — copy the "Model Comparison" table from `stage_04x.md` verbatim (metrics for all three methods side by side)
- Comparison chart:
  ```markdown
  ![Model comparison — discrimination metrics](figures/04x_model_comparison.png)
  ```
- **After the comparison chart:** Interpret the differences between the three models. Are the AUC/Gini/KS values similar (suggesting stable variable importance) or divergent (suggesting method sensitivity)? Which method produced the most parsimonious model? Which had the best cross-validation stability?
- ROC overlay:
  ```markdown
  ![ROC curve overlay — three selection methods](figures/04x_roc_overlay.png)
  ```
- **After the ROC overlay:** Discuss how the ROC curves compare — do they overlap closely (similar discrimination) or diverge in specific FPR regions? Note any method that dominates across all operating points vs. methods that trade off sensitivity and specificity differently.
- Variable overlap:
  ```markdown
  ![Variable selection overlap](figures/04x_variable_overlap.png)
  ```
- **After the variable overlap chart:** Discuss which variables were selected by all three methods (core predictors), which were unique to one method, and what the overlap ratio means for model robustness. Variables selected by all methods form the most defensible feature set.

**Champion selection rationale:** Copy the champion rationale from `stage_04x.md`. Explain why this method was selected as champion and whether the selection was automatic (highest composite score) or human-overridden.

#### Champion Model Detail

Present the champion model (from the winning `stage_04{a,b,c}.md`) in full detail:
- **Model fit summary table** — copy the "Logistic Regression — Model Fit" table from the champion's stage summary verbatim
- **After the model fit table:** Interpret the Pseudo R-squared (McFadden's) — values of 0.2-0.4 are considered excellent fit for discrete choice models. Explain what the LLR p-value tells us about whether the model as a whole is statistically significant compared to a null (intercept-only) model.
- **Coefficient table** — copy the "Logistic Regression — Coefficients" table from the champion's stage summary verbatim
- **After the coefficients table:** Explain the relative importance of each variable based on coefficient magnitude. Discuss whether the coefficient signs are economically intuitive — in WoE-based models, all coefficients should be positive (higher WoE = lower risk, positive coefficient means higher WoE contributes to lower default probability via the log-odds). Note any variables with high p-values (>0.05) that may not be individually significant but contribute to overall model fit. Comment on confidence interval widths — narrow intervals indicate precise estimation, wide intervals suggest uncertainty. Also discuss VIF values and what they indicate about multicollinearity.
- Model performance summary: AUC, Gini, KS
- ROC curve (champion):
  ```markdown
  ![ROC curve — champion model (development sample)](figures/04{suffix}_roc_curve.png)
  ```
  Where `{suffix}` is `a`, `b`, or `c` depending on the champion method.
- **After the ROC curve:** Interpret the AUC in context — what does this level of discriminatory power mean for a retail PD model? How does it compare to typical benchmarks (AUC 0.70-0.85 is common for retail credit)? Is the Gini coefficient sufficient for regulatory purposes (EBA guidelines typically expect Gini > 0.35)? Discuss whether the model's discrimination is driven by a few strong variables or is well-balanced.
- Score distribution (champion):
  ```markdown
  ![Score distribution — champion model](figures/04{suffix}_score_distribution.png)
  ```
- **After the score distribution:** Describe the distribution shape — is it approximately normal, bimodal, or skewed? Are there problematic concentrations of borrowers at particular score ranges? What does the score range (min to max) imply about the model's ability to separate good borrowers from bad? Discuss the base score, PDO, and what these scaling parameters mean in practical terms.
- Decile analysis (champion):
  ```markdown
  ![Score decile analysis — champion model](figures/04{suffix}_score_decile_table.png)
  ```
- **After the decile analysis:** Discuss whether observed default rates decrease monotonically across score deciles (from highest-risk to lowest-risk). If monotonicity breaks, identify where and assess whether this is a sample size issue (small n per decile) or a genuine model weakness. Note the default rate spread between the worst and best deciles as a measure of practical discrimination.
- Coefficient plot (champion):
  ```markdown
  ![Model coefficients — champion model](figures/04{suffix}_coefficient_plot.png)
  ```
- Score statistics table: min, max, mean, pct_below_400, pct_above_800
- Decile monotonicity result
- **Cross-validation discussion:** Interpret the cross-validation and bootstrap results. What does the CV AUC difference tell us about overfitting risk? Is the model stable across different data splits?

#### Non-Champion Models (Summary)

For each of the two non-champion methods, include an abbreviated summary:
- Method name, number of variables selected, variable list
- AUC, Gini, KS, composite score
- Key differences from champion (e.g., "selected 2 fewer variables", "slightly lower AUC but better CV stability")
- 1-2 sentences explaining why it was not selected as champion

### Calibration

Content from `stage_05.md`:
- Rating scale table: grade, score_range, calibrated_pd, n_obligors, pct_portfolio
- **After the rating scale table:** Interpret the grade structure — is the PD spread across grades reasonable (typically spanning from <1% for the best grade to >50% for the worst)? Are any grades too narrow in score range (potentially unstable) or too wide (poor granularity)? Does the worst grade PD seem plausible given the portfolio characteristics? Comment on portfolio concentration — are borrowers evenly distributed across grades, or concentrated in a few grades? An HHI below 0.20 indicates good diversification.
- Rating scale chart:
  ```markdown
  ![Rating scale with calibrated PDs](figures/05_rating_scale.png)
  ```
- Grade distribution:
  ```markdown
  ![Portfolio distribution across grades](figures/05_grade_distribution.png)
  ```
- **After the grade distribution chart:** Discuss whether the grade distribution is well-balanced or shows concerning concentrations. In a well-calibrated model, borrowers should be spread across grades without excessive concentration in any single grade.
- Calibration summary: target CT vs achieved CT, grade PD ordering valid
- **After calibration summary:** Explain what the central tendency represents (portfolio-level long-run average PD) and whether the achieved CT matches the target. If a scaling factor was applied, explain why and what it means for the PD estimates.
- Stress test results table: base CT, +1% shift CT, +2% shift CT
- Stress test chart:
  ```markdown
  ![Stress test — PD shifts](figures/05_stress_test.png)
  ```
- **After stress test results:** Discuss stress test implications — how sensitive is the rating scale to PD shifts? Would a stress scenario (e.g., economic downturn increasing the CT by 1-2 percentage points) cause material grade migrations? Are the stressed PDs still within a plausible range? This is important for capital planning and ICAAP purposes.

### Validation

Content from `stage_06.md`:
- Discriminatory power table: AUC, Gini, KS, dp_test_pvalue, dp_test_result
- **After the discriminatory power table:** Explain what the AUC test p-value means — it tests whether the model's AUC is significantly greater than 0.50 (random). A low p-value confirms the model has genuine discriminatory power. Compare the validation AUC to the development AUC to check for overfitting.
- ROC curve:
  ```markdown
  ![Validation ROC curve](figures/06_roc_curve.png)
  ```
- KS plot:
  ```markdown
  ![KS separation plot](figures/06_ks_plot.png)
  ```
- **After the KS plot:** Explain what the KS statistic represents (maximum separation between cumulative default and non-default distributions) and what the observed value means in practical terms.
- Stability results: AUC half1, AUC half2, difference, result
- Stability chart:
  ```markdown
  ![Model stability — sample halves](figures/06_stability.png)
  ```
- **After the stability chart:** Interpret the stability result — a small AUC difference between sample halves indicates the model is not sensitive to the specific data sample used for development. Note the threshold used and how the observed difference compares.
- Predictive power table: per-grade binomial and Jeffreys p-values and results
- PP test chart:
  ```markdown
  ![Predictive power by grade](figures/06_pp_test.png)
  ```
- **After the predictive power table:** Explain what the binomial and Jeffreys tests measure — they test whether the observed default rate in each grade is consistent with the calibrated PD. A PASS means the model's PD estimates are accurate at the grade level. If any grades show marginal p-values, discuss whether this is a calibration concern or a sample size issue.
- Hosmer-Lemeshow: p-value, result
- **After Hosmer-Lemeshow:** Explain this test in plain language — it assesses overall calibration quality by comparing observed vs. expected defaults across score groups. A high p-value (>0.05) means the model's predictions are well-calibrated overall.
- Homogeneity: overall p-value, result, any grade failures
- Homogeneity chart:
  ```markdown
  ![Homogeneity test](figures/06_homogeneity_test.png)
  ```
- **After homogeneity results:** Explain what homogeneity tests — whether borrowers within the same grade have similar risk profiles. Discuss any power limitations (small sub-segment sizes) and what this means for the reliability of the test result.
- Heterogeneity: p-value, result
- **After heterogeneity results:** Explain what heterogeneity tests — whether adjacent grades are statistically distinguishable. If any adjacent pairs fail, discuss whether this is a genuine grade boundary problem or a statistical power issue, and whether grade merging should be considered.
- PSI: value, result
- **After PSI:** Explain what PSI measures (population stability between development and reference samples) and interpret the value against standard thresholds (<0.10 stable, 0.10-0.25 moderate shift, >0.25 significant shift).
- **Overall assessment: PASS / PASS WITH FLAGS / FAIL** (bold, prominent)
- **After overall assessment:** Write a concluding paragraph synthesizing all validation results. State the overall assessment clearly, list the specific flags that prevent a clean PASS (if any), and explain what these flags mean for model deployment readiness. Is the model suitable for production use? Are there monitoring requirements or conditions that must be met?

### Regulatory Flags & Recommendations

- Consolidated table of all flags from all stages (source stage, flag description, severity)
- For each flag with a notebook-reviewer assessment (from `stage_XX_review.md`): include the reviewer's assessment and recommendation
- **After the flag table:** Write a narrative summarizing the flag landscape — how many flags are low/medium/high severity, whether they form a pattern (e.g., multiple flags related to sample size), and what the overall regulatory risk posture is.
- Final recommendations: actions required before model deployment
- **After recommendations:** Write a closing paragraph stating whether the model is recommended for deployment, any conditions or monitoring requirements, and suggested timeline for first model review.

### Pipeline Diagnostics & Proposed Improvements

Content from `{RUN_DIR}/pipeline/fixes_summary.md` (if it exists). If no fixes_summary.md exists, write "No diagnostic issues were identified during this pipeline run."

If fixes exist:
- Summary table of all issues across stages: stage, issue title, category (A: Prompt / B: Utility / C: Approach), severity
- **After the summary table:** Write a paragraph explaining the diagnostic process — smoke tests run at every stage, with full diagnostics triggered on flags or failures. Note how many stages produced issues vs passed clean.
- For each **Critical** or **Warning** issue: include the full fix proposal (symptoms, root cause, proposed fix, verification)
- **After all fix proposals:** Write a concluding paragraph assessing the pipeline's health — are the issues concentrated in one stage or spread across many? Do they indicate systemic problems or isolated edge cases? What is the priority order for applying fixes before the next run?
- For **Info** issues: list in a compact table (stage, title, category) without full detail

### Appendix A: Methodology

Brief descriptions of:
- Weight of Evidence (WoE) and Information Value (IV) — what they measure, how bins are constructed
- Variable selection methods:
  - Stepwise MIV selection — entry threshold, marginal chi-square test, iterative variable addition based on marginal information value
  - XGBoost feature importance — gain-based importance from gradient boosted trees, cumulative importance threshold for variable selection, logistic regression refit
  - Forward stepwise selection — likelihood ratio test p-value as entry criterion, sequential variable addition based on statistical significance
- Model comparison methodology — weighted composite score across AUC, Gini, KS, CV stability, coefficient consistency, parsimony, and decile monotonicity
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
6. **Every table and figure is followed by at least one interpretive paragraph** — no orphaned charts or unexplained metrics

## Return Message

```
Stage 07 complete. Report: {RUN_DIR}/report.docx
Report contains [N] pages covering all 6 pipeline stages.
[N] figures embedded. [N] regulatory flags documented.
Overall model assessment: [PASS / PASS WITH FLAGS / FAIL].
```
