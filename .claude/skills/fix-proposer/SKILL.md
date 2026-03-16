# Fix-Proposer Diagnostic Protocol

Structured diagnostic protocol run within each subagent (stages 01-06) after results are produced and output-verifier passes. Identifies domain-level issues and proposes concrete fixes for future runs.

**This protocol does NOT block the pipeline.** It produces diagnostic output alongside the stage results. Fixes are proposals only — they are not applied within the current run.

---

## When to Run

**Always.** After output-verifier passes, run Tier 1 smoke tests for the current stage. If any smoke test fails OR the stage summary contains flags (`_flags:` entries that are not "None"), escalate to Tier 2 full diagnostic.

---

## Tier 1 — Smoke Tests

Quick pass/fail checks. Run all tests for the current stage. Log results.

### Stage 01 — Data Explorer

| # | Check | Pass Condition |
|---|---|---|
| 1 | Target default rate plausible | Rate in [5%, 50%] |
| 2 | No extreme missingness | No variable with >50% missing |
| 3 | NZV output reviewed | `pdt.nzv()` results referenced in stage summary |
| 4 | Variable actions complete | Every variable has a documented action (keep/impute/exclude) |

### Stage 02 — Data Preparer

| # | Check | Pass Condition |
|---|---|---|
| 1 | Row count preserved | clean dataset rows == raw dataset rows |
| 2 | No new NaN | Zero NaN in clean dataset |
| 3 | Outlier cap rate reasonable | No variable with >20% values capped |
| 4 | Imputation methods documented | Every imputed variable has method and count in stage summary |

### Stage 03 — Bivariate Analyst

| # | Check | Pass Condition |
|---|---|---|
| 1 | Shortlist IV consistency | All shortlisted variables have IV >= threshold used |
| 2 | No duplicate variables | Shortlist contains no repeated variable names |
| 3 | OptimalBinning status | All shortlisted variables have OPTIMAL or FEASIBLE status |
| 4 | Bin size compliance | All bins have >= 5% observations (from binning table) |
| 5 | WoE dataset exists | `loans_binned.csv` written and non-empty |

### Stage 04a — Model Builder (MIV)

| # | Check | Pass Condition |
|---|---|---|
| 1 | Coefficient sign consistency | All WoE coefficients have expected sign direction |
| 2 | Multicollinearity | All VIF values < 10 |
| 3 | Minimum discrimination | AUC > 0.60 |
| 4 | Score range | Score range overlaps with target [400, 800] |
| 5 | Model params complete | `model_params_miv.json` contains all required keys |

### Stage 04b — Model Builder (XGBoost)

| # | Check | Pass Condition |
|---|---|---|
| 1 | Coefficient sign consistency | All WoE coefficients have expected sign direction |
| 2 | Multicollinearity | All VIF values < 10 |
| 3 | Minimum discrimination | AUC > 0.60 |
| 4 | Score range | Score range overlaps with target [400, 800] |
| 5 | Model params complete | `model_params_xgb.json` contains all required keys |
| 6 | XGBoost importances recorded | `model_params_xgb.json` contains `xgb_importances` dict |

### Stage 04c — Model Builder (Forward Stepwise)

| # | Check | Pass Condition |
|---|---|---|
| 1 | Coefficient sign consistency | All WoE coefficients have expected sign direction |
| 2 | Multicollinearity | All VIF values < 10 |
| 3 | Minimum discrimination | AUC > 0.60 |
| 4 | Score range | Score range overlaps with target [400, 800] |
| 5 | Model params complete | `model_params_fwd.json` contains all required keys |

### Stage 04x — Model Comparator

| # | Check | Pass Condition |
|---|---|---|
| 1 | Canonical params exist | `model_params.json` exists (champion copy) |
| 2 | Comparison table complete | `stage_04x.md` contains comparison table with all completed methods |
| 3 | Champion declared | `stage_04x.md` contains champion field with method name and rationale |
| 4 | Champion params match | `model_params.json` content matches the declared champion's `model_params_*.json` |

### Stage 05 — Calibrator

| # | Check | Pass Condition |
|---|---|---|
| 1 | Calibration not circular | Calibrated PD differs from observed DR by >0.01 for at least 50% of grades |
| 2 | Scaling factor reasonable | Scaling factor in [0.5, 2.0] (if method=scaling) |
| 3 | Central tendency achieved | \|achieved CT - target CT\| <= 0.005 |
| 4 | PD ordering | Calibrated PDs strictly monotonic across grades |
| 5 | Grade population | No grade with <1% of portfolio |

### Stage 06 — Validator

| # | Check | Pass Condition |
|---|---|---|
| 1 | Assessment consistency | Overall assessment is consistent with individual test PASS/FAIL counts |
| 2 | No contradictory results | No single test is reported as both PASS and FAIL |
| 3 | All required tests present | DP, PP, homogeneity, heterogeneity, PSI all have results |
| 4 | Stability reported | Half-split AUC values present and difference calculated |

---

## Tier 2 — Full Diagnostic

Run when any smoke test fails or stage flags exist. For each identified issue:

### Step 1: Classify Root Cause

| Category | Description | Diagnostic Signal | Fix Type |
|---|---|---|---|
| **A — Prompt Issue** | Agent followed its instructions correctly but instructions led to wrong output | Code ran without errors, output-verifier passed, but domain check fails | Proposed edit to `.claude/agents/XX.md` |
| **B — Missing Utility** | Agent needs a computation that neither pdtoolkit nor standard libraries provide cleanly | Notebook contains multi-step ad-hoc computation that could be encapsulated | New function in `src/pipeline_fixes/` |
| **C — Approach Issue** | Chosen method is inappropriate for the data characteristics | Results are technically correct but domain-inappropriate | Documented recommendation for human review |

### Step 2: Trace Root Cause

For each issue, trace the chain: **instruction → code generated → output produced → expected vs actual**.

Ask these diagnostic questions:
1. Did the agent follow its instructions? (If yes → Category A or C. If no → retry issue, not a fix-proposer concern.)
2. Is there a reusable computation the agent had to improvise? (If yes → Category B.)
3. Is the analytical method itself wrong for this data? (If yes → Category C.)

### Step 3: Write Fix Proposal

Use the format specified in the output section below.

---

## Output

Write `{RUN_DIR}/pipeline/stage_XX_fixes.md` with this structure:

```markdown
# Stage XX — Fix Proposals

## Diagnostic Summary
- Smoke tests: [N/N passed]
- Full diagnostic triggered: [yes/no]
- Issues found: [N]

---
```

If no issues found, stop here. The file still gets written (confirms diagnostics ran).

If issues found, add for each issue:

```markdown
## Issue [N]: [Short descriptive title]

### Classification
- Category: [A: Prompt Issue / B: Missing Utility / C: Approach Issue]
- Severity: [Critical / Warning / Info]

### Symptoms
[What was observed in the output that triggered this diagnosis. Reference specific values, metrics, or chart observations.]

### Root Cause
[Why this happened. Trace from instruction or code to the problematic output.]

### Proposed Fix

**[For Category A — Prompt Change]**
- **Target file:** `.claude/agents/[XX-agent-name].md`
- **Section:** [section name or line reference]
- **Current instruction:** `[quoted current text]`
- **Proposed instruction:** `[proposed replacement text]`
- **Rationale:** [why this fixes the issue]

**[For Category B — New Utility Function]**
- **Target file:** `src/pipeline_fixes/[module_name].py`
- **Function:** `[function_name](param1: type, param2: type) -> ReturnType`
- **Description:** [what it does, one sentence]
- **Implementation sketch:**
  ```python
  def function_name(param1, param2):
      """Docstring explaining purpose and parameters."""
      # implementation
      return result
  ```
- **Usage in notebook:**
  ```python
  sys.path.insert(0, 'lib')
  from pipeline_fixes.[module] import function_name
  result = function_name(data, params)
  ```

**[For Category C — Approach Change]**
- **Current approach:** [description of what was done]
- **Recommended approach:** [description of what should be done instead]
- **Requires:** [prompt change / new utility / both / human decision]
- **Trade-offs:** [risks of current approach vs recommended approach]

### Verification
[Specific check to confirm the fix works. E.g., "After applying: calibrated PD should differ from observed DR by >0.01 for all grades with >50 obligors."]
```

---

## Return Message Addition

After writing `stage_XX_fixes.md`, append this line to your return message:

```
Fix proposals: [N] issues ([N] critical) — see stage_XX_fixes.md
```

Or if no issues:

```
Fix proposals: None
```

---

## Rules

1. **Do NOT modify** agent `.md` files, skill files, or `src/pdtoolkit/` — only propose changes
2. **Do NOT block** the pipeline — diagnostics are advisory, not gating
3. **Always write** `stage_XX_fixes.md` even if empty (confirms diagnostics ran)
4. **Be specific** — quote exact text for prompt changes, provide runnable code for utility functions
5. **Severity guide:**
   - **Critical:** Produces incorrect model output (e.g., circular calibration)
   - **Warning:** May produce suboptimal results (e.g., non-monotonic WoE accepted without investigation)
   - **Info:** Improvement opportunity (e.g., could encapsulate a repeated 5-line computation)
