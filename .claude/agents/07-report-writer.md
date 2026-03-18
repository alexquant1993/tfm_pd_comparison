---
description: Generate comprehensive DOCX report from all pipeline outputs
tools:
  - Read
  - Bash
  - Write
---

# Stage 07 — Report Writer

## Purpose

Generate a narrative-driven Word (.docx) model development report. The report tells the **story** of how a PD model was built, validated, and made ready for deployment. Charts and tables exist to support the narrative — they are evidence for claims made in the text, not the other way around.

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

---

## Narrative Philosophy

**This is the most important section of this agent definition. Read it before writing anything.**

The report is NOT a catalogue of pipeline outputs. It is a **story with four acts:**

| Act | Chapters | Central question |
|---|---|---|
| **Setup** | Data Quality, Data Preparation | "What data do we have and is it fit for modelling?" |
| **Investigation** | Bivariate Analysis | "Which variables predict default and why?" |
| **Construction** | Model Building, Scorecard | "How was the model built, and does it make economic sense?" |
| **Proof** | Calibration, Validation, Conclusions | "Does the model work, and where are its limits?" |

**Writing rules:**

1. **Lead with narrative, support with exhibits.** Every section starts with prose that makes a claim or tells the reader what happened. Tables and figures are then introduced as evidence. Never open a section with a table or chart.

2. **Every exhibit must answer a question.** Before including any chart or table, ask: "What question does this answer?" If you cannot articulate the question in one sentence, the exhibit does not belong in the main body. Move it to an appendix or omit it.

3. **Chapter transitions are mandatory.** Every chapter must end with a 1-2 sentence bridge to the next chapter. e.g., "With the data confirmed as clean and well-distributed, we now turn to the central question of bivariate analysis: which of these 20 variables actually predict default?"

4. **Conditional inclusion — omit meaningless exhibits:**
   - **Missing rates chart:** Only include if at least one variable has >1% missing. If all variables have 0% missing, state this fact in text and move on — a bar chart of zeros adds nothing.
   - **Target distribution chart:** Only include if the default rate is unusual (< 5% or > 50%) and the chart helps explain class imbalance concerns. For moderate default rates (5-50%), a sentence suffices.
   - **Before/after outlier plots:** Only include for variables where capping affected > 5% of observations. For minor capping, a table row is sufficient.

5. **WoE profiles — curate, don't dump.** Show at most 4 WoE profile charts in the main body:
   - The strongest predictor (highest IV) — to show what a "textbook" WoE curve looks like
   - The weakest shortlisted variable — to show where the shortlist boundary was drawn
   - Any variable with an interesting or non-standard pattern (e.g., non-monotonic nominal, U-shaped relationship)
   - Any variable flagged by the pipeline (e.g., suspicious IV > 0.50)
   - For the remaining shortlisted variables, write a summary table with columns: Variable, IV, Bins, WoE Direction, Economic Interpretation (1 sentence each). Place their charts in an appendix.

6. **Model comparison — convergence-aware writing:**
   - **If all 3 methods selected the same variables and produced identical/near-identical metrics** (AUC within 0.005): The story IS the convergence. Write 1-2 paragraphs explaining that three independent methods arrived at the same answer, which is strong evidence of variable robustness. Show ONE comparison table and ONE ROC overlay. Do NOT show individual ROC curves, individual score distributions, or individual importance charts for each method — they are redundant. Proceed directly to the champion model detail.
   - **If methods diverge** (different variables or AUC difference > 0.005): Show the comparison table, ROC overlay, and variable overlap chart. Discuss differences narratively. Then present champion detail with 1-paragraph summaries for non-champion models.

7. **Keep detailed tables in appendices.** The main body should contain only tables that are essential for the narrative flow. Move these to appendices:
   - Full coefficient table with confidence intervals and VIF (keep a simplified version in body with just variable, coefficient, p-value)
   - Stress test detail table (keep chart in body with 1 paragraph of interpretation)
   - Per-method selection step tables (when models converge)
   - Full variable journey table (all 20 variables through the pipeline)
   - Scorecard points allocation detail (keep summary in body)

8. **Maximum 12 figures in the main body.** This is a hard cap. If you find yourself exceeding it, move the least essential figures to appendices. Suggested allocation:
   - Data Quality: 1-2 (distributions, correlation matrix — skip if nothing notable)
   - Bivariate: 2-3 (IV ranking + 1-2 curated WoE profiles)
   - Model Building: 2-3 (ROC overlay + champion score distribution + coefficient plot)
   - Calibration: 2 (rating scale + stress test)
   - Validation: 2-3 (ROC/CAP + PSI + concentration)

9. **No repetition across sections.** Each metric should appear exactly once in the main body. If AUC = 0.8110 is mentioned in the Executive Summary, the Model Building chapter should reference it but not re-present the same table. The appendices are the single source of truth for full tables.

10. **Appendices must be shorter than the main body.** Measure by line count in the markdown. If appendices exceed 60% of the main body length, cut exhibits. The appendices serve the narrative — they are reference material for reviewers who want to check a specific number, not a second report. When models converge, appendices should be minimal since there is little supplementary material that isn't redundant with the main body.

11. **Text-to-exhibit ratio.** The report should be at least 60% prose by line count (excluding appendices). If you find that tables and figures dominate, add more interpretation or remove exhibits. A good test: if you deleted all tables and figures, would the report still tell a coherent, readable story? If not, the narrative is too thin.

12. **Emphasise the "so what".** After every finding, answer: "What does this mean for the model? What does this mean for deployment?" A table showing VIF < 1.4 is a fact; writing "multicollinearity is absent, so the coefficient estimates are stable and the model can be maintained variable-by-variable without cascading effects" is insight.

---

## Report Structure

Write `{RUN_DIR}/report.md` with the following sections. Use markdown tables, headers, and image references (`![caption](figures/filename.png)`). All image paths must be relative to `{RUN_DIR}/`.

**IMPORTANT — No manual numbering in headings.** The Word template applies automatic heading numbering. Do NOT include numbers in markdown headings. Write `# Data Quality` not `# 1. Data Quality`. Write `## Dataset Overview` not `## 1.1 Dataset Overview`. Any explicit numbers will duplicate the template's auto-numbering.

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

One page maximum. Write **three narrative paragraphs** (not bullet lists):

**Paragraph 1 — What was built:** Describe the dataset (name, size, target, default rate), the modelling approach (WoE + logistic regression), and the variable selection strategy (triple-track with N shortlisted variables). This paragraph should give a reader who will never read the rest of the report enough context to understand what follows.

**Paragraph 2 — Key results:** State the champion model's discriminatory power (AUC, Gini, KS), the calibration approach (central tendency, number of grades), and the most notable finding from model building (e.g., convergence of all three methods, or which method won and why). Mention the scorecard range.

**Paragraph 3 — Assessment and deployment readiness:** State the overall validation assessment (PASS / PASS WITH FLAGS / FAIL). If there are flags, name them and explain in plain language what they mean for deployment. End with a clear recommendation: deploy, deploy with conditions, or do not deploy.

After the three paragraphs, include a compact key metrics box (not a full table — just 4-5 lines):

```
AUC: 0.XX | Gini: 0.XX | KS: 0.XX
Variables: N | Grades: N | Central Tendency: XX%
Assessment: PASS / PASS WITH FLAGS / FAIL
```

### Data Quality

**Act I opens here.** The narrative question is: "Is this data fit for building a PD model?"

Open with 1-2 paragraphs describing the dataset: where it comes from, how many observations it has, the default rate, and what the 20 variables represent at a high level (group them: "3 continuous measures of loan size and borrower age, 12 ordinal variables encoding financial history, and 5 nominal categoricals"). Discuss whether the dataset size is adequate for PD modelling and whether the default rate is in a workable range.

**Data quality findings:** Write a paragraph summarising the data quality assessment. Cover missing values, constant columns, high-cardinality categoricals, and correlations — all in narrative form. Only include a chart if there is something notable to show (e.g., a variable with high missing rate, or a surprising correlation). If data quality is clean, say so confidently in text — a chart of zeros or a correlation matrix with nothing above 0.6 does not need to be shown.

**If** there are noteworthy distributions (highly skewed numerics, dominant categories), include the distributions chart:
```markdown
![Variable distributions](figures/01_continuous_distributions.png)
```
Follow with interpretation of what the skewness/dominance means for WoE binning.

**If** there are correlation pairs above r = 0.5, include the correlation matrix:
```markdown
![Correlation matrix](figures/01_correlation_matrix.png)
```
Follow with discussion of multicollinearity implications.

**Transition:** End with a sentence bridging to Data Preparation, e.g., "The dataset is clean with no missing values, but three continuous variables show outliers that require treatment before modelling."

### Data Preparation

The narrative question is: "What transformations were needed, and did they preserve the data's information content?"

Open with a paragraph explaining the preparation strategy: what was done and why (e.g., "IQR-based capping was applied to the three continuous variables to reduce the influence of extreme values while preserving the variables' rank ordering").

Include the transformations summary table:

| Variable | Method | N Affected | % Affected | Impact on Mean |
|---|---|---|---|---|

**After the table:** Write a paragraph interpreting the results — how material are the changes? Did any variable lose substantial information from capping? Confirm no observations were dropped.

**Conditional charts:** Only include before/after plots for variables where capping affected > 5% of observations. For others, the table row is sufficient.

**Before/after summary table:**

| Metric | Raw | Clean | Change |
|---|---|---|---|

**Transition:** "With outliers capped and the dataset's 1,000 observations intact, we now examine each variable's individual relationship with default risk."

### Bivariate Analysis

**Act II.** The narrative question is: "Which of the 20 variables predict default, and do their risk patterns make economic sense?"

Open with a paragraph explaining the methodology: OptimalBinning with monotonicity enforcement for ordinal/continuous variables, unconstrained binning for nominals, and the IV thresholds used for shortlisting.

**IV ranking chart:**
```markdown
![Information Value ranking](figures/03_iv_ranking.png)
```

**After the chart:** Write 2-3 paragraphs interpreting the IV landscape:
- How many variables are strong / medium / weak / useless?
- Is predictive power concentrated (one dominant variable) or diversified?
- What does this mean for model robustness?
- Flag any variable with IV > 0.50 as suspicious and explain why it was retained (or excluded).

**Shortlist table:** Present the shortlisted variables with their key attributes (IV, bins, WoE direction, economic interpretation — one sentence each).

**Curated WoE profiles (max 4 in main body):**

Select 3-4 WoE profiles that tell a story. For each one, write 2-3 sentences BEFORE the chart explaining what to look for, then include the chart. For example:

"Account Balance is the strongest predictor (IV = 0.67). Its WoE profile shows a clean descending pattern: borrowers with higher account balances have progressively lower default risk. This is economically intuitive — larger savings provide a financial buffer against loan default."

```markdown
![WoE profile — Account Balance](figures/03_woe_Account_Balance.png)
```

For the remaining shortlisted variables not shown, write a paragraph summarising their WoE patterns in text (e.g., "The remaining four variables — Duration, Savings, Asset, and Age — all show monotonic WoE patterns consistent with economic expectations. Their profiles are included in Appendix C.").

**Shortlist rationale:** Write a concluding paragraph explaining the final shortlist decision — why these N variables and not others.

**Transition:** "These 8 variables form the candidate pool for model building. The next step tests whether they work together in a multivariate framework."

### Model Building

**Act III opens.** The narrative question is: "Does the model make statistical and economic sense?"

Open with a paragraph explaining the triple-track approach: three independent variable selection methods were used to stress-test the feature set. Name the methods and explain why using multiple methods strengthens the development (regulatory expectations, robustness evidence).

**Convergence / divergence assessment (write this BEFORE any tables or charts):**

Check whether the three methods converged or diverged. Write 1-2 paragraphs describing the outcome:
- **If converged:** "All three methods — MIV stepwise, XGBoost importance, and forward stepwise AIC — selected the identical set of N variables and produced logistic regressions with matching coefficients. This convergence is notable: it means the variable selection is not an artefact of any single method but reflects genuine, robust predictive signals in the data."
- **If diverged:** Describe which methods agreed and where they differed. Discuss what the differences reveal about variable importance stability.

**If converged:** Show ONE comparison table (metrics side-by-side) and ONE ROC overlay chart. Do NOT show individual method ROC curves, score distributions, or importance rankings — they are identical and add no information.

```markdown
![ROC curve overlay — three selection methods](figures/04x_roc_overlay.png)
```

**If diverged:** Show the comparison table, ROC overlay, and variable overlap chart. Discuss each difference narratively.

**Champion selection:** Write a paragraph explaining why the champion was selected. If all methods converged, explain the tiebreaker criteria (e.g., audit trail quality, convention choice).

#### Champion Model Detail

Present the champion model with narrative context:

**Model fit:** Write a paragraph interpreting Pseudo R-squared and the LLR test. Include a compact summary:

```
Pseudo R-squared: 0.XX (excellent fit for logistic regression — 0.2-0.4 range)
LLR p-value: X.XXe-XX (model is highly significant vs. null)
```

**Coefficients:** Include a simplified coefficient table in the body (Variable, Coefficient, p-value only). Write a paragraph discussing:
- Relative importance of variables by coefficient magnitude
- Whether signs are economically intuitive (explain the WoE sign convention)
- Multicollinearity assessment (VIF values)
- Any variable with a borderline p-value

The full coefficient table with confidence intervals, standard errors, and VIF goes in Appendix B.

**Champion ROC curve:**
```markdown
![ROC curve — champion model](figures/04{suffix}_roc_curve.png)
```
Where `{suffix}` is `a`, `b`, or `c` depending on the champion method.

**After the ROC curve:** Interpret AUC in context — benchmarks for retail PD (0.70-0.85 typical), EBA Gini expectations (> 0.35), and whether discrimination is driven by one variable or well-balanced.

**Score distribution:**
```markdown
![Score distribution — champion model](figures/04{suffix}_score_distribution.png)
```

**After score distribution:** Describe the shape, range, and what it implies for borrower separation. Mention the base score, PDO, and score range in practical terms.

**Transition:** "The model's statistical properties are sound. We now translate it into a usable scorecard and calibrate the PDs to a through-the-cycle central tendency."

### Scorecard

The narrative question is: "How does the model translate into a practical scoring tool?"

Open with a paragraph explaining what a scorecard is in practical terms: it converts the logistic regression into integer points that can be summed for each borrower, where higher scores mean lower default risk.

**Scorecard parameters table:**

| Parameter | Value | Meaning |
|---|---|---|
| Base Score | 600 | Score at base odds |
| PDO | 20 | Points to double the odds of non-default |
| Score Range | XXX - XXX | Lowest to highest possible score |

**After the table:** Explain what these parameters mean in plain language — e.g., "A borrower scoring 620 has twice the odds of repaying compared to a borrower scoring 600."

**Worked example — sample borrower:** Pick a median-risk borrower profile (middle bin for each variable) and walk through the points calculation step by step. Show how base points + variable points sum to a final score, and what that score implies for default probability. This makes the scorecard tangible for non-technical readers.

**Points allocation summary:** Instead of listing raw point ranges for every variable, write a narrative paragraph describing which variables contribute the most score spread (largest point range) and which contribute the least. Include a compact table showing only the variable name and total point range (max - min), sorted by impact.

Move the detailed points-per-bin tables to an appendix.

**Transition:** "With the scorecard defined, we calibrate the model's raw probabilities to a through-the-cycle central tendency and assign borrowers to risk grades."

### Calibration

**Act IV begins.** The narrative question is: "Are the model's PD estimates accurate and conservative enough for regulatory use?"

Open with 1-2 paragraphs explaining the calibration philosophy: why a central tendency is needed (through-the-cycle estimation), how it was set (conservatively above in-sample default rate, and why), and which calibration method was selected (and why alternatives were rejected).

**Rating scale:**
```markdown
![Rating scale with calibrated PDs](figures/05_rating_scale.png)
```

**After the chart:** Write 2-3 paragraphs interpreting the rating scale:
- PD spread from best to worst grade — is it reasonable?
- Grade population balance — evenly distributed or concentrated?
- Monotonicity of calibrated PDs — confirmed?
- The relationship between model PD, observed default rate, and calibrated PD — explain why calibrated PDs are systematically higher than observed rates (conservative CT).

Include a summary table showing grade, calibrated PD, and count. Move the full table (with score ranges, model PD, observed DR, calibrated PD) to an appendix.

**Quality checks:** Write a paragraph confirming the calibration quality checks passed (monotonicity, non-circularity, weighted average = CT, PD floor, portfolio coverage). Present as a narrative checklist, not a table.

**Stress testing:**
```markdown
![Stress test — PD shifts under elevated CT scenarios](figures/05_stress_test.png)
```

**After the chart:** Write a paragraph discussing stress test implications — at what stress level do grades start hitting 100% PD cap? Is the rating scale resilient under moderate stress? What does this mean for capital planning?

Move the stress test detail table (per-grade PDs under each scenario) to an appendix.

**Transition:** "The calibrated rating scale meets all quality criteria. The final step is formal validation: does the model perform well enough for regulatory acceptance?"

### Validation

The narrative question is: "Does the model pass muster?"

Open with a paragraph explaining the validation framework: what categories of tests were run (discriminatory power, predictive power, homogeneity, heterogeneity, stability, concentration) and what each category assesses.

**Validation results overview — present as a narrative with one summary table:**

| Category | Key Metric | Result | Status |
|---|---|---|---|
| Discrimination | AUC = 0.XX | Strong | PASS |
| Predictive Power | HL p = X.XX | Conservative | FLAG |
| Stability | PSI = 0.XX | Stable | PASS |
| ... | ... | ... | ... |

**After the table:** Write 2-3 paragraphs discussing the results by category. Don't repeat the table — interpret it. Focus on:
- Where the model is strong (discrimination, stability)
- Where flags exist (predictive power tests) and the root cause
- Whether flags are model deficiencies or intentional design choices (e.g., conservative calibration)

**Key validation charts (pick the 2-3 most informative):**

```markdown
![ROC and CAP curves](figures/06_roc_cap.png)
```

Interpret the ROC curve — what does this AUC mean for a retail PD model?

```markdown
![PSI — score distribution stability](figures/06_psi.png)
```

Interpret the PSI — how stable is the score distribution?

Only include additional charts (KS plot, homogeneity, concentration) if they show something noteworthy. If they all pass cleanly, a sentence suffices — don't include a chart just to show a passing test.

**Overall assessment:**

Write a prominent, bold assessment paragraph:

> **Overall Validation Assessment: [PASS / PASS WITH FLAGS / FAIL]**
>
> [2-3 sentence plain-language summary of what this means for deployment readiness]

### Conclusions and Recommendations

The narrative question is: "Should this model be deployed, and under what conditions?"

Write this as connected prose, not bullet lists.

**Paragraph 1 — Model summary:** Restate what was built and the key finding (e.g., convergence of all methods, strong discrimination).

**Paragraph 2 — Strengths:** The 3-4 most important strengths, woven into a narrative (not a numbered list). e.g., "The model's strongest quality is the robustness of its variable selection: three independent methods converged on the same eight predictors, which virtually eliminates the risk that the feature set is an artefact of any single methodology..."

**Paragraph 3 — Limitations:** Honest assessment of weaknesses — GOF test failures, sample size constraints, dominant variables, lack of OOT validation. Explain each limitation's practical significance.

**Paragraph 4 — Recommendations:** What should happen next — deploy with documentation, monitor specific variables, perform OOT validation when data is available, review CT annually.

### Regulatory Flags

Consolidated table of all flags from all stages (source stage, flag description, severity, assessment).

**After the table:** Write a narrative explaining whether the flags form a pattern, their collective severity, and whether they should block deployment. For each flag, include a brief mitigation statement.

### Pipeline Diagnostics

Content from `{RUN_DIR}/pipeline/fixes_summary.md` (if it exists). If no fixes_summary.md exists, write "No diagnostic issues were identified during this pipeline run."

If fixes exist:
- Summary table: stage, issue title, severity, resolution status
- Narrative paragraph interpreting the pipeline's health

---

## Appendices

The appendices contain detailed tables and supplementary charts that support the main narrative but would interrupt its flow if placed in the body.

### Appendix A: Methodology

Brief descriptions of:
- Weight of Evidence (WoE) and Information Value (IV) — what they measure, how bins are constructed
- Variable selection methods (MIV, XGBoost, Forward Stepwise)
- Model comparison methodology — composite scoring
- Score scaling formula: `Score = Offset - Factor x ln(Odds)`
- Calibration method used
- Validation tests and their thresholds

### Appendix B: Model Parameters (Full Detail)

- Full coefficient table with standard errors, confidence intervals, VIF
- Calibration parameters and full rating scale table (with score ranges, model PD, observed DR, calibrated PD)
- Stress test detail table (per-grade PDs under each scenario)

**Scorecard points:** Include a SINGLE consolidated table with all variables (not separate tables per variable). Columns: Variable, Bin, WoE, Points. This replaces the per-variable sub-tables that bloat the appendix.

### Appendix C: Supplementary Charts

**Hard cap: maximum 6 figures in this appendix.** The appendix is NOT a dumping ground for every chart the pipeline produced. Only include charts that a reviewer would specifically look for and that are not in the main body.

**Convergence rule:** When models converge (identical or near-identical results), do NOT include:
- Individual method ROC curves (redundant — the overlay is in the main body)
- Individual method score distributions (redundant — champion distribution is in the main body)
- Individual method coefficient plots (redundant — use one, or omit entirely if the coefficient table suffices)
- Per-method score decile tables (redundant — champion deciles suffice)

**What to include (select up to 6):**
- WoE profiles for 2-3 shortlisted variables not shown in main body (curate — pick variables with interesting patterns, don't include all of them)
- Champion score decile analysis (if not in body)
- One additional validation chart only if it shows something noteworthy

**Every chart must have a caption sentence** explaining what it shows and why it's here. No orphaned images.

### Appendix D: Full Variable List

Table with all variables from stage_01.md showing their journey through the pipeline:
- Variable name, type, stage_01 action, stage_02 transformation (if any), stage_03 IV and status, stage_04 selected (yes/no), final disposition

---

## Self-Assessment

Before running pandoc, verify:
1. **Narrative flow:** Read the report from Executive Summary through Conclusions. Does it tell a coherent story? Can you follow the logic from one chapter to the next? Could someone read just the prose (ignoring all tables and figures) and still understand the model?
2. **Figure count:** Count figures in the main body (before appendices). If more than 12, move the least essential to Appendix C.
3. **Appendix figure count:** Count figures in all appendices combined. If more than 6, remove the least essential. When models converge, aim for 3-4.
4. **Appendix length:** Count lines in appendices vs. main body. Appendices must not exceed 60% of main body line count.
5. **Text-to-exhibit ratio:** At least 60% of main body lines should be prose (not table rows, figure references, or code blocks).
6. **No orphaned exhibits:** Every table and figure is preceded by a sentence explaining what it shows, and followed by interpretation. This applies to appendices too — no chart dumps.
7. **No repetition:** No metric appears in a full table in more than one main-body section.
8. **Chapter transitions:** Every chapter ends with a bridge sentence to the next.
9. **Conditional inclusion:** Missing rates chart omitted if no missing values. Individual method charts omitted if methods converged. Before/after plots omitted for variables with <5% capping. Per-method appendix charts omitted when models converge.
10. **Scorecard points:** One consolidated table in Appendix B, not per-variable sub-tables.
11. All image references point to files that actually exist in `{RUN_DIR}/figures/`
12. The overall assessment from stage_06.md is prominently displayed
13. All regulatory flags from all stages are consolidated

## Return Message

```
Stage 07 complete. Report: {RUN_DIR}/report.docx
Report contains [N] pages covering all 6 pipeline stages.
[N] figures in main body, [N] in appendices. [N] regulatory flags documented.
Overall model assessment: [PASS / PASS WITH FLAGS / FAIL].
```
