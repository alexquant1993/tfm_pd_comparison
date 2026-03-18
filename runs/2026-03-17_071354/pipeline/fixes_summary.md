# Fixes Summary — Consolidated Fix Proposals

**Run:** 2026-03-17_071354
**Generated:** After Stage 06 completion, before Stage 07 (report-writer)
**Stages covered:** 01, 02, 03, 04a, 04b, 04c, 04x, 05, 06

---

## Totals

| Metric | Count |
|---|---|
| Total issues found | **4** |
| Critical issues (Error) | **0** |
| Warning issues | **3** |
| Info issues | **1** |
| Stages with no issues | 5 (01, 02, 04a, 04b, 05) |
| Stages with issues | 4 (03, 04c, 04x, 06) |

---

## Category A — Prompt Issues (Agent Instruction Bugs)

### A1: `pdt.step_fwd()` sign convention incompatible with target=1=default
- **Stage:** 04c
- **Severity:** Warning
- **Symptom:** `pdt.step_fwd()` selects zero variables because it enforces `coef > 0` for WoE features. When the target encodes default as 1, correct WoE coefficients are negative, causing all valid candidates to be rejected.
- **Root cause:** `step_fwd` has a hardcoded assumption that the non-default class is 1; this conflicts with the German Credit dataset encoding where `Creditability=1` means default (bad).
- **Proposed fix:** Update `.claude/agents/04c-model-builder-fwd.md` to note that when `pdt.step_fwd()` selects zero variables, fall back to a manual statsmodels forward stepwise loop accepting `coef < 0` as the correct sign direction for WoE features predicting default.
- **Verification:** Forward stepwise should select ≥4 variables with negative WoE coefficients and p-values below threshold.

---

## Category C — Approach Issues (Methodology / Design)

### C1: CP solver ascending monotonicity broken due to ortools version incompatibility
- **Stage:** 03
- **Severity:** Warning
- **Symptom:** `OptimalBinning(solver="cp", monotonic_trend="ascending")` raises a `TypeError` from ortools `cp_model` for all numerical variables. Descending with CP and both directions with MIP work correctly. The notebook fell back to `solver="mip"` for ascending monotonicity without issue.
- **Root cause:** A breaking change in the installed version of `ortools` in the CP-SAT solver's `__rsub__` method for `IntAffine` objects, incompatible with `optbinning`'s constraint builder.
- **Proposed fix:** Either (a) update `.claude/agents/03-bivariate-analyst.md` to prefer `solver="mip"` as the primary solver, or (b) pin `ortools==9.8.3296` (last known compatible version). MIP solver produces equivalent results for this dataset size.
- **Verification:** `OptimalBinning(solver="cp", monotonic_trend="ascending").fit(x, y)` should complete without TypeError for Duration of Credit (month).

### C2: Min-max normalization over-penalizes XGBoost with narrow metric spread
- **Stage:** 04x
- **Severity:** Info
- **Symptom:** XGBoost composite score (0.4298) is dramatically lower than MIV/Forward (0.9207) despite a raw AUC gap of only 0.0039. Min-max normalization assigns XGBoost 0.0 on all three normalized discrimination metrics because it is the sole minimum across all three.
- **Root cause:** Min-max normalization over-amplifies negligible differences when the metric spread is tiny (N=3 models, AUC range 0.0039). This is a known limitation of min-max with small sample sizes.
- **Proposed fix:** Update `.claude/agents/04x-model-comparator.md` to use either (a) raw-value scoring with reference benchmarks (AUC 0.60=0, AUC 0.90=1), or (b) min-max with a floor (worst model gets `max(0.3, normalized_value)`). Raw-value scoring is more defensible when metric spreads are <0.01.
- **Verification:** XGBoost composite score should be within ~0.1 of MIV/Forward when the raw AUC gap is <0.01.

### C3: Grade F predictive power failure — observed DR exceeds calibrated PD
- **Stage:** 06
- **Severity:** Warning
- **Symptom:** Grade F binomial test FAIL (p=0.0264): observed default rate 49.6% exceeds calibrated PD 40.6%. Jeffreys test also fails (p=0.0214). All other 7 grades pass. Hosmer-Lemeshow overall test passes (p=0.294) — calibration is adequate in aggregate.
- **Root cause:** Equal-frequency grading places Grade F in a narrow 13-point score band (491.6–504.9) where the model-predicted PDs locally underestimate observed risk. This is a local calibration issue in the mid-range of the score distribution.
- **Proposed fix:** Consider (a) adjusting grade boundaries to respect natural risk clustering rather than strict equal-frequency, or (b) testing with 7 or 9 grades to see if the concentration issue resolves. This is a human design decision. The failure is marginal (p=0.026) and isolated to one grade, so regulatory risk is low.
- **Verification:** After adjusting boundaries, Grade F binomial test p-value should be ≥0.05, or the grade should carry a documented regulatory note.

---

## Stages With No Issues

| Stage | Smoke Tests | Notes |
|---|---|---|
| 01 — Data Quality | 4/4 passed | No issues found |
| 02 — Data Preparation | 4/4 passed | No issues found |
| 04a — Model Building (MIV) | 5/5 passed | AUC=0.8109, VIF<2.0, score range [413.8, 618.3] |
| 04b — Model Building (XGB) | 6/6 passed | AUC=0.807, VIF<1.15, score range [424.9, 621.3] |
| 05 — Calibration | 5/5 passed | PD ordering correct, CT achieved (30.0%), non-circular |

---

## Action Priority

| Priority | Issue | Owner |
|---|---|---|
| High | A1: `step_fwd` sign convention → prompt fix in 04c agent | Prompt update |
| High | C1: CP solver ortools incompatibility → switch to MIP or pin ortools | Prompt update or env fix |
| Medium | C3: Grade F PP failure → grade boundary redesign | Human decision |
| Low | C2: Min-max normalization amplification → composite scoring methodology | Prompt update |
