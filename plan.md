# PD Modelling Agent Architecture
## Based on pdtoolkit_py

---

## Overview

This document proposes a Claude Code agent architecture for automated PD model development using the `pdtoolkit_py` library. The system produces one Jupyter notebook per pipeline stage, with a human checkpoint after every stage before proceeding.

### Design Principles

- Each subagent writes one notebook and one stage summary file to disk
- Subagents never return notebook content — only a short structured summary
- All inter-stage communication happens through `pipeline/*.md` files on disk
- Notebooks never embed base64 images — all plots saved to `figures/`
- Every stage requires human approval before the next stage is invoked
- Subagents self-assess their output and retry internally before returning
- Skills provide shared domain knowledge and verification checklists

---

## Repository Structure

```
CLAUDE.md
.claude/
  agents/
    01-data-explorer.md
    02-data-preparer.md
    03-bivariate-analyst.md
    04-model-builder.md
    05-calibrator.md
    06-validator.md
    notebook-reviewer.md
  skills/
    pd-conventions/
      SKILL.md
    pdtoolkit-api/
      SKILL.md
    notebook-writer/
      SKILL.md
    output-verifier/
      SKILL.md
data/
  loans.csv
  loans_clean.csv          ← written by stage 02
figures/
  01_*.png
  02_*.png
  03_*.png
  04_*.png
  05_*.png
  06_*.png
notebooks/
  01_data_quality.ipynb
  02_data_preparation.ipynb
  03_bivariate_analysis.ipynb
  04_model_building.ipynb
  05_calibration.ipynb
  06_validation.ipynb
pipeline/
  stage_01.md
  stage_02.md
  stage_03.md
  stage_04.md
  stage_05.md
  stage_06.md
  model_params.json        ← written by stage 04
```

---

## CLAUDE.md Contents

The project-level instruction file contains:

- Project description: PD model development on German Credit dataset using pdtoolkit_py
- Dataset location: `data/loans.csv`, target variable: `Creditability` (1 = default)
- The six-stage pipeline and which subagent handles each stage
- The rule: always load the `pd-conventions` skill before making any threshold decisions
- The rule: every subagent writes a notebook and a `pipeline/stage_XX.md` before returning
- The rule: notebooks are executed via `nbconvert` and stripped of base64 before saving
- The rule: **a human checkpoint occurs after every stage — never auto-proceed**
- The rule: on human approval, pass only the stage summary path to the next subagent
- The rule: on human rejection, re-invoke the same subagent with the reviewer's feedback

---

## Skills

### `pd-conventions` skill

Loaded by every subagent and by notebook-reviewer. Contains all domain thresholds and interpretation rules so decisions are consistent across the pipeline.

**IV thresholds**
| IV Range | Interpretation | Action |
|---|---|---|
| < 0.02 | Useless | Exclude |
| 0.02 – 0.10 | Weak | Exclude unless business justification |
| 0.10 – 0.30 | Medium | Include in shortlist |
| > 0.30 | Strong | Include in shortlist |

**WoE requirements**
- Strictly monotonic preferred
- Marginal violations (single bin out of order) — document and flag, do not auto-exclude
- Non-monotonic after 3 rebinning attempts — exclude and document reason

**Bin size minimums**
- Minimum 5% of total observations per bin
- Minimum 20 events (defaults) per bin
- Merge bins that fall below threshold before accepting result

**AUC benchmarks**
| AUC | Assessment |
|---|---|
| < 0.60 | Unacceptable — reject |
| 0.60 – 0.70 | Weak — flag for review |
| 0.70 – 0.80 | Acceptable |
| > 0.80 | Strong |

**Validation thresholds**
- Gini > 0.35: acceptable discriminatory power
- KS > 0.30: acceptable separation
- Homogeneity p > 0.05: fail (grade-level instability)
- Heterogeneity p < 0.05: fail (overlap between grades)
- PP testing: binomial test p < 0.05 at any grade = predictive power failure

**Score scale convention**
- Base score: 600 points at 50:1 odds
- PDO (Points to Double Odds): 20
- Score range target: 400–800

**Regulatory flags** (items to surface to human reviewer regardless of pass/fail)
- Any test result with p-value within 0.01 of threshold
- Any variable with non-monotonic WoE accepted under business justification
- Coefficient sign reversal in final model
- Stability AUC difference > 0.05 between sample halves

---

### `pdtoolkit-api` skill

A concise and accurate reference of every function used in the pipeline. Purpose: prevent hallucinated function calls and document known edge cases discovered during use.

**Key functions**

| Function | Key Parameters | Returns | Known Edge Cases |
|---|---|---|---|
| `univariate(db)` | db: DataFrame | DataFrame with stats per variable | Returns empty rows for binary variables |
| `imp_sc(x, method, sc)` | x: Series, method: 'mean'/'median'/'mode', sc: list of special case values | Imputed Series | Does not handle mixed-type columns |
| `imp_outliers(x, method, pct)` | method: 'iqr'/'percentile', pct: percentile threshold | Imputed Series | IQR method may over-impute in skewed distributions |
| `bivariate(db, target)` | db: DataFrame (all categorical), target: str | (summary_df, info_df) | Requires all variables to be string type |
| `woe_tbl(x, y)` | x: categorical Series, y: binary target | WoE table DataFrame | Returns None if a bin has zero events |
| `cat_bin(x, y, min_pct_obs, min_avg_rate, max_groups)` | | (summary, binned_x) | max_groups=5 is a safe default |
| `step_miv(db, target, reg_obj, miv_threshold, p_value)` | miv_threshold: MIV entry threshold | (model, selected_vars) | Returns empty if no variable clears threshold |
| `scaled_score(probs, score, odd, pdo)` | score=600, odd=50, pdo=20 | Score array | Input must be probability array not log-odds |
| `score_to_prob(scores, score, odd, pdo)` | | Probability array | Inverse of scaled_score |
| `rs_calibration(pd_estimate, ct)` | ct: target central tendency | Calibrated PDs | Sensitive to initial PD scale |
| `auc_model(y, p)` | y: actual, p: predicted | AUC float | |
| `dp_testing(y, p, alpha)` | alpha: significance level | Test results dict | |
| `pp_testing(rating, y, p, alpha)` | rating: grade assignments | Test results per grade | |
| `homogeneity(rating, y, alpha)` | | Chi-square results | |
| `heterogeneity(rating, p, alpha)` | | Overlap test results | |

---

### `notebook-writer` skill

Enforces consistent notebook structure and output conventions across all six notebooks.

**Cell order per notebook**
1. Markdown: stage title, description, inputs consumed
2. Imports and configuration
3. Data loading (read from disk, never re-derive from earlier stages)
4. Analysis cells
5. Results cells
6. Markdown: Summary table (key findings, flags, recommended actions)

**Plot conventions**
- Always use `plt.savefig('figures/XX_description.png', dpi=150, bbox_inches='tight')`
- Never use `plt.show()` — this embeds output in the notebook
- Standard figure size: `(10, 6)` for single plots, `(14, 6)` for side-by-side
- Colour palette: blue (`#2166AC`) for good/pass, red (`#D6604D`) for bad/fail, grey (`#999999`) for neutral

**Execution rule**
- After writing the notebook source, always execute: `jupyter nbconvert --to notebook --execute --inplace notebooks/XX_name.ipynb`
- If execution fails, read the error, fix the cell, and retry before returning
- After execution, clear all outputs: `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to notebook --inplace notebooks/XX_name.ipynb`
- The notebook on disk is always clean source — outputs are only in `figures/`

**Summary cell format** (always the last cell, always markdown)
```markdown
## Stage Summary

| Item | Value | Status |
|---|---|---|
| Key metric | value | ✓ / ⚠ / ✗ |

**Flags for human review:** [list or "None"]
**Recommended action for next stage:** [brief instruction]
```

---

### `output-verifier` skill

A checklist the main agent runs inline after each subagent completes, before presenting results to the human. Fast structural checks only — no domain reasoning.

**Checklist**
- [ ] Notebook file exists at expected path
- [ ] nbconvert execution completed without errors (check for `[NbConvertApp] Writing` in output)
- [ ] Stage `.md` file exists and contains required fields (see each stage definition)
- [ ] `figures/` contains at least one new `.png` for this stage
- [ ] No cell in the notebook has `"output_type": "error"` in its outputs
- [ ] Summary cell is present as the last cell

**On failure:** do not present to human. Re-invoke subagent with specific failure message. Maximum 2 retries before surfacing error to human for manual intervention.

---

## Subagents

### Stage 01 — `data-explorer`

**Purpose:** Analyse raw data quality and document every variable's status before any transformation.

**Input received in prompt**
- Path to raw dataset: `data/loans.csv`
- Target variable name: `Creditability`

**Tools allowed:** Read, Bash, Write

**Internal self-assessment loop**

Before writing the notebook, the agent checks:
1. Does every variable have a documented action (keep / impute-special-cases / impute-outliers / exclude)?
2. For variables flagged for imputation: is the missing pattern documented (random vs systematic)?
3. For variables flagged for exclusion: is the reason stated (high missingness / zero variance / wrong type)?
4. Are outlier thresholds justified (IQR vs percentile choice explained)?

If any check fails, the agent revises its analysis before writing.

**Functions used:** `pdt.univariate()`

**Proposed improvement over original toolkit:** Add a correlation matrix heatmap for numeric variables at this stage. Early identification of highly correlated pairs (|r| > 0.7) informs the bivariate stage and reduces wasted computation on redundant variables.

**Writes**
- `notebooks/01_data_quality.ipynb`
- `figures/01_missing_rates.png`
- `figures/01_distributions.png`
- `figures/01_correlation_matrix.png`
- `pipeline/stage_01.md`

**stage_01.md required fields**
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

**Returns to main agent**
```
Stage 01 complete. Notebook: notebooks/01_data_quality.ipynb
20 variables analysed. 3 need special case imputation. 2 need outlier imputation.
0 recommended for exclusion. 2 high-correlation pairs flagged.
Awaiting human review before proceeding to Stage 02.
```

**Human checkpoint:** Human reviews `notebooks/01_data_quality.ipynb` and `pipeline/stage_01.md`. Approves imputation decisions and exclusion list before Stage 02 is invoked.

---

### Stage 02 — `data-preparer`

**Purpose:** Execute approved imputation and outlier treatment. Produce clean dataset for all downstream stages.

**Input received in prompt**
- Path to raw dataset: `data/loans.csv`
- Path to stage summary: `pipeline/stage_01.md`
- Human-approved modifications (if reviewer changed any decisions)

**Tools allowed:** Read, Bash, Write

**Internal self-assessment loop**

After imputation, before writing:
1. Re-run distribution checks on all imputed variables — does the imputed distribution look plausible?
2. If imputed distribution looks suspicious (e.g. mean-imputation created an implausible spike): try alternative method (median or mode), re-check, document the switch
3. Check that no new missing values were introduced
4. Verify dataset dimensions match expectations (same rows as raw, same columns)
5. Write checksum of clean dataset to stage file so downstream agents can verify integrity

**Functions used:** `pdt.imp_sc()`, `pdt.imp_outliers()`

**Proposed improvement:** After imputation, add a before/after distribution overlay plot for each imputed variable. This makes the imputation effect visually reviewable rather than purely statistical.

**Writes**
- `notebooks/02_data_preparation.ipynb`
- `figures/02_imputation_[variable].png` (one per imputed variable)
- `data/loans_clean.csv`
- `pipeline/stage_02.md`

**stage_02.md required fields**
```
clean_dataset_path: data/loans_clean.csv
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

**Returns to main agent**
```
Stage 02 complete. Notebook: notebooks/02_data_preparation.ipynb
5 variables imputed. Clean dataset at data/loans_clean.csv (checksum: abc123).
All post-imputation distributions plausible. No flags.
Awaiting human review before proceeding to Stage 03.
```

**Human checkpoint:** Human reviews `notebooks/02_data_preparation.ipynb` — particularly the before/after distribution plots for each imputed variable. Approves clean dataset before Stage 03 is invoked.

---

### Stage 03 — `bivariate-analyst`

**Purpose:** Calculate WoE, IV, and AUC for all candidate variables. Produce a ranked shortlist for model building.

**Input received in prompt**
- Path to clean dataset: `data/loans_clean.csv`
- Target variable: `Creditability`
- List of candidate variables (all non-excluded variables from stage_01.md)
- Path to stage summary: `pipeline/stage_02.md` (to verify checksum)

**Tools allowed:** Read, Bash, Write

**Internal self-assessment loop**

Per variable, the agent iterates before accepting a result:

1. Calculate WoE with default binning
2. Check monotonicity — if fails:
   - Attempt 1: increase `max_groups` by 2
   - Attempt 2: decrease `max_groups` by 2
   - Attempt 3: force monotonic direction using `monobin`-style constraint
   - After 3 attempts still failing: mark as non-monotonic, document all attempts, do not auto-exclude (flag for human review)
3. Check bin size minimums — if any bin below threshold: merge with adjacent bin, re-check IV
4. After finalising binning: verify WoE signs are economically plausible given the variable's name and direction
5. Run sign plausibility check: does higher value of this variable logically correspond to higher or lower default risk? If WoE direction contradicts expectation: flag for human review

After processing all variables:

6. Build correlation clusters: for pairs with |WoE correlation| > 0.7, identify the cluster and recommend keeping only the highest-IV variable, marking the rest as substitutes
7. Check shortlist size: if fewer than 5 variables pass IV threshold, relax threshold to 0.05 and document; if more than 15 pass, tighten to 0.15 and document
8. Final check: does the shortlist contain at least one variable from each major risk category (behavioural, demographic, financial) if available in data?

**Functions used:** `pdt.bivariate()`, `pdt.woe_tbl()`, `pdt.cat_bin()`, `pdt.auc_model()`

**Proposed improvement:** Add a WoE profile plot per variable — a bar chart showing WoE value per bin with the bin default rate overlaid as a line. This is the single most useful visual for reviewing bivariate results and is not produced by the original toolkit.

**Parallelisation note:** For datasets with > 30 variables, the subagent should process in two batches of ~15, writing partial results to disk between batches. This prevents internal context from filling up on wide datasets.

**Writes**
- `notebooks/03_bivariate_analysis.ipynb`
- `figures/03_woe_[variable].png` (one per analysed variable)
- `figures/03_iv_ranking.png`
- `figures/03_correlation_clusters.png`
- `pipeline/stage_03.md`

**stage_03.md required fields**
```
clean_dataset_path: data/loans_clean.csv
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

**Returns to main agent**
```
Stage 03 complete. Notebook: notebooks/03_bivariate_analysis.ipynb
20 variables analysed. Shortlist: 8 variables. 12 excluded.
Exclusion reasons: 7 low IV, 3 non-monotonic after rebinning, 2 correlation duplicates.
Flags: Account Balance WoE direction flagged for economic plausibility review.
Awaiting human review before proceeding to Stage 04.
```

**Human checkpoint:** This is the most critical review gate. Human reviews the WoE profile plots, the IV ranking, the shortlist, and all flags before Stage 04 is invoked. Human can: approve shortlist as-is, add/remove variables, or adjust IV threshold. Any human modifications are passed to Stage 04 in the invocation prompt.

---

### Stage 04 — `model-builder`

**Purpose:** Build logistic regression model using stepwise MIV selection. Produce final variable set, coefficients, and score scale.

**Input received in prompt**
- Path to clean dataset: `data/loans_clean.csv`
- Approved variable shortlist from `pipeline/stage_03.md`
- Target variable: `Creditability`
- Any human modifications to shortlist from checkpoint

**Tools allowed:** Read, Bash, Write

**Internal self-assessment loop**

1. Run `step_miv()` with default thresholds
2. Check variable count in final model:
   - If < 4 variables: relax MIV entry threshold by 50%, re-run, document
   - If > 12 variables: tighten MIV entry threshold by 50%, re-run, document
3. Check coefficient signs against WoE directions: for each selected variable, positive WoE should correspond to positive logit direction. If sign reversal found:
   - Identify likely collinear pair
   - Remove the variable with lower IV from the pair
   - Re-run model and recheck — document the removal
4. Check VIF (Variance Inflation Factor) for multicollinearity: if VIF > 5 for any variable, flag and consider removal
5. Generate score distribution on development sample — check for reasonable spread (target range 400–800, no more than 5% of observations at either extreme)
6. Final model plausibility check: does the score ranking produce monotonically increasing default rates across deciles? If not, the model has a structural problem — flag for human review

**Functions used:** `pdt.step_miv()`, `pdt.scaled_score()`, `sklearn.linear_model.LogisticRegression`

**Proposed improvement:** Add a score decile analysis table showing observed default rate per score decile. This is the most intuitive way to assess whether the model is working correctly and is standard in credit risk model documentation.

**Proposed improvement:** Add a partial dependence plot per variable showing the contribution of each variable to the score. This supports model interpretability requirements.

**Writes**
- `notebooks/04_model_building.ipynb`
- `figures/04_roc_curve.png`
- `figures/04_score_distribution.png`
- `figures/04_score_decile_table.png`
- `figures/04_coefficient_plot.png`
- `pipeline/stage_04.md`
- `pipeline/model_params.json`

**stage_04.md required fields**
```
clean_dataset_path: data/loans_clean.csv
selected_variables: [list]
excluded_from_shortlist: [list with reason]
model_auc: [float]
model_gini: [float]
model_ks: [float]
coefficients:
  - variable: [name]
    coefficient: [float]
    woe_direction_consistent: [true/false]
    vif: [float]
miv_threshold_used: [float, note if adjusted]
score_statistics:
  min: [float]
  max: [float]
  mean: [float]
  pct_below_400: [float]
  pct_above_800: [float]
score_monotonicity_across_deciles: [true/false]
model_params_path: pipeline/model_params.json
model_flags: [list or "None"]
```

**Returns to main agent**
```
Stage 04 complete. Notebook: notebooks/04_model_building.ipynb
6 variables selected. AUC: 0.71. Gini: 0.42. KS: 0.38.
Score range: 480–730, mean 591. All coefficient signs consistent. Decile monotonicity: pass.
Flags: Duration coefficient required adjustment due to collinearity with Credit Amount.
Awaiting human review before proceeding to Stage 05.
```

**Human checkpoint:** Human reviews model variable selection, coefficients, and score distribution. Particularly reviews any coefficient sign adjustments or multicollinearity removals. Approves model before calibration.

---

### Stage 05 — `calibrator`

**Purpose:** Calibrate the rating scale to the portfolio target central tendency. Assign rating grades and calibrated PDs.

**Input received in prompt**
- Path to clean dataset: `data/loans_clean.csv`
- Path to model params: `pipeline/model_params.json`
- Target central tendency (portfolio default rate — provided by human at checkpoint or defaulting to development sample rate)
- Path to stage summary: `pipeline/stage_04.md`

**Tools allowed:** Read, Bash, Write

**Internal self-assessment loop**

1. Run `rs_calibration()` with initial anchor
2. Check central tendency achieved vs target — tolerance ±0.5%: if breached, adjust anchor and retry
3. Check grade PD ordering: calibrated PD must be strictly increasing from best to worst grade. If any grade out of order: adjust grade boundaries, re-calibrate, recheck
4. Check grade population distribution: no grade should contain < 2% or > 40% of the portfolio. If any grade outside bounds: revise grade cut-offs, document
5. Check that worst grade calibrated PD < 100%

**Functions used:** `pdt.rs_calibration()`, `pdt.score_to_prob()`

**Proposed improvement:** Add a stress test section. Apply two parallel PD shifts (+1 percentage point and +2 percentage points across all grades) and show the resulting grade distributions and central tendency under stress. This is standard regulatory documentation.

**Proposed improvement:** Add a through-the-cycle (TTC) vs point-in-time (PIT) commentary cell. Flag whether the calibration is likely TTC or PIT based on the central tendency used, and note implications for capital calculation.

**Writes**
- `notebooks/05_calibration.ipynb`
- `figures/05_rating_scale.png`
- `figures/05_grade_distribution.png`
- `figures/05_stress_test.png`
- `pipeline/stage_05.md`

**stage_05.md required fields**
```
model_params_path: pipeline/model_params.json
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

**Returns to main agent**
```
Stage 05 complete. Notebook: notebooks/05_calibration.ipynb
8 rating grades. Target CT: 30.0%, Achieved CT: 30.1%. Grade PD ordering valid.
Stress test: +1% shift produces CT 31.0%, +2% shift produces CT 32.1%.
No flags.
Awaiting human review before proceeding to Stage 06.
```

**Human checkpoint:** Human reviews rating scale, grade distribution, and calibrated PDs. Confirms target central tendency is appropriate for the portfolio. Approves calibration before validation.

---

### Stage 06 — `validator`

**Purpose:** Run full model validation suite. Produce validation report.

**Input received in prompt**
- Path to clean dataset: `data/loans_clean.csv`
- Path to model params: `pipeline/model_params.json`
- Path to stage summary: `pipeline/stage_05.md`

**Tools allowed:** Read, Bash, Write

**Internal self-assessment loop**

This stage does not iterate to improve the model — the model is fixed. Instead, the agent iterates on *completeness and precision of reporting*:

1. After running each test, check whether the result is marginal (within 0.01 of threshold). If marginal: run the test on a bootstrapped sample (n=1000) to get a confidence interval around the p-value. Report both point estimate and CI.
2. For any homogeneity failure at a specific grade: identify which grade and check whether it is a small grade (< 5% of portfolio). If small grade, note that the test may lack power.
3. Check stability: split development sample into two random halves, run AUC on each half. If |AUC_half1 - AUC_half2| > 0.05: flag as potential overfitting.
4. After all tests: produce an overall model assessment — a single clear statement of pass/fail with a list of all flags for the regulatory documentation.

**Functions used:** `pdt.dp_testing()`, `pdt.pp_testing()`, `pdt.homogeneity()`, `pdt.heterogeneity()`, `pdt.auc_model()`

**Proposed improvement:** Add a Gini stability index (PSI) calculation between a held-out 20% sample and the development 80% sample. PSI > 0.25 is a standard trigger for model review.

**Proposed improvement:** Add a backtesting section if a time variable is available in the data — train on earlier periods, test on later periods. This is the strongest evidence of out-of-time discriminatory power.

**Writes**
- `notebooks/06_validation.ipynb`
- `figures/06_roc_curve.png`
- `figures/06_ks_plot.png`
- `figures/06_homogeneity_test.png`
- `figures/06_pp_test.png`
- `figures/06_stability.png`
- `pipeline/stage_06.md`

**stage_06.md required fields**
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

**Returns to main agent**
```
Stage 06 complete. Notebook: notebooks/06_validation.ipynb
Overall assessment: PASS WITH FLAGS
DP: AUC 0.71 PASS. Stability: 0.02 difference PASS.
PP: All grade tests PASS. Hosmer-Lemeshow PASS.
Homogeneity: Grade 3 marginal pass p=0.049 (bootstrapped CI: 0.041–0.058) — FLAGGED.
Heterogeneity: PASS.
PSI: 0.12 PASS.
Awaiting human review and sign-off.
```

**Human checkpoint:** Final review gate. Human reviews full validation notebook and stage_06.md. Decides whether the model is acceptable for use given flags. This is the sign-off stage — no further automated processing occurs after this point.

---

### `notebook-reviewer` subagent

**Purpose:** Provide a second-pass domain review of a notebook when the stage summary contains flags or when the human requests it. Not in the critical path — invoked optionally by the main agent.

**Input received in prompt**
- Path to notebook to review
- Path to corresponding stage `.md`
- Specific flags to focus on (from the stage summary)

**Tools allowed:** Read, Bash (for notebook-to-script conversion only)

**Behaviour:**
1. Convert notebook to `.py` using `nbconvert --to script` to avoid base64 bloat
2. Read the script and stage `.md`
3. Load `pd-conventions` skill
4. Assess each flag in context — is it a genuine concern or a statistical artefact?
5. Produce a structured second opinion

**Returns to main agent:** A short structured assessment per flag:
```
Flag: Account Balance WoE direction
Assessment: Non-standard direction consistent with dataset characteristics. 
German Credit data encodes Account Balance inversely. Not a modelling error.
Recommendation: Accept, add explanatory note in notebook.
```

---

## Memory and Context Management

### Preventing context overflow

| Rule | Rationale |
|---|---|
| Subagents write to disk, return only summaries | Keeps main agent context clean |
| Notebooks never embed base64 — all plots to `figures/` | Single plot can be 50K+ characters |
| Convert notebooks to `.py` before reading | Removes outputs and metadata bloat |
| Inter-stage data via `pipeline/*.md` files, not prompt strings | Stage summaries are small structured text |
| `pdtoolkit-api` skill prevents hallucination retries | Wasted tool calls consume context fast |
| Stage 03 processes variables in batches for wide datasets | Prevents bivariate analyst context from filling |

### Context budget estimate per subagent
| Stage | Estimated context usage | Risk |
|---|---|---|
| 01 data-explorer | Low — reads one CSV, writes stats | Low |
| 02 data-preparer | Low — reads one CSV, applies transforms | Low |
| 03 bivariate-analyst | High — reads CSV + processes many variables | Medium — batch if > 30 variables |
| 04 model-builder | Medium — reads CSV + runs stepwise | Low |
| 05 calibrator | Low — reads params + calibrates | Low |
| 06 validator | Medium — reads CSV + runs multiple tests | Low |

---

## Pipeline Execution Flow

```
User: "Build PD model on loans.csv"
  │
  ├─► Invoke data-explorer (stage 01)
  │     └─► output-verifier skill (inline check)
  │     └─► Present summary + notebook path to human
  │     └─► [HUMAN CHECKPOINT 01] ◄── Approve / Reject / Modify
  │
  ├─► Invoke data-preparer (stage 02) with stage_01.md path
  │     └─► output-verifier skill
  │     └─► Present summary to human
  │     └─► [HUMAN CHECKPOINT 02] ◄── Approve clean dataset
  │
  ├─► Invoke bivariate-analyst (stage 03) with stage_02.md path
  │     └─► output-verifier skill
  │     └─► [optional] invoke notebook-reviewer if flags present
  │     └─► Present summary + shortlist to human
  │     └─► [HUMAN CHECKPOINT 03] ◄── Approve/modify shortlist (most critical gate)
  │
  ├─► Invoke model-builder (stage 04) with approved shortlist
  │     └─► output-verifier skill
  │     └─► Present summary to human
  │     └─► [HUMAN CHECKPOINT 04] ◄── Approve model
  │
  ├─► Invoke calibrator (stage 05) with model_params.json + CT target from human
  │     └─► output-verifier skill
  │     └─► Present summary to human
  │     └─► [HUMAN CHECKPOINT 05] ◄── Approve rating scale
  │
  └─► Invoke validator (stage 06) with model_params.json
        └─► output-verifier skill
        └─► [optional] invoke notebook-reviewer for any flagged tests
        └─► Present full validation summary to human
        └─► [HUMAN CHECKPOINT 06] ◄── Final sign-off
```

---

## Deliverables

After all six checkpoints are passed, the project contains:

| File | Contents |
|---|---|
| `notebooks/01_data_quality.ipynb` | Univariate analysis, missing rates, correlations |
| `notebooks/02_data_preparation.ipynb` | Imputation results, before/after distributions |
| `notebooks/03_bivariate_analysis.ipynb` | WoE profiles, IV ranking, shortlist |
| `notebooks/04_model_building.ipynb` | Stepwise selection, coefficients, score distribution |
| `notebooks/05_calibration.ipynb` | Rating scale, grade distribution, stress test |
| `notebooks/06_validation.ipynb` | Full validation suite, overall assessment |
| `pipeline/stage_01.md` through `stage_06.md` | Structured stage summaries — model audit trail |
| `pipeline/model_params.json` | Final model parameters |
| `data/loans_clean.csv` | Prepared dataset |
| `figures/*.png` | All plots referenced by notebooks |