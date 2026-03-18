# Stage 04c -- Fix Proposals

## Diagnostic Summary
- Smoke tests: 5/5 passed
- Full diagnostic triggered: no
- Issues found: 1

---

## Issue 1: pdt.step_fwd() enforces positive WoE coefficients, incompatible with target=1=default

### Classification
- Category: A: Prompt Issue
- Severity: Warning

### Symptoms
`pdt.step_fwd()` selected zero variables because it requires `coef > 0` for WoE-coded variables (line 148 of `19_step_fwd.py`: `trend_ok = coef > 0`). However, when the target variable encodes default as 1, WoE coefficients in the logistic regression should be negative (higher WoE = lower risk = lower P(default)). This caused the function to reject all valid candidates.

### Root Cause
The `step_fwd` function assumes WoE coefficients should be positive, which is only correct when the target variable encodes non-default (good) as 1. The German Credit dataset uses `Creditability = 1` for default (bad), so the expected coefficient sign is negative.

### Proposed Fix

**Target file:** `.claude/agents/04c-model-builder-fwd.md`
**Section:** Step 3: Forward Stepwise Selection
**Current instruction:** Use `pdt.step_fwd()` directly
**Proposed instruction:** Add a note that if `pdt.step_fwd()` selects zero variables due to the WoE sign convention conflict (target=1=default), implement forward stepwise manually using statsmodels, accepting negative WoE coefficients as correct sign direction. The manual implementation should iterate through candidate variables, fitting logistic regression at each step and selecting the variable with the lowest p-value below the threshold, with sign check enforcing `coef < 0` for WoE features when predicting default.
**Rationale:** The pdtoolkit `step_fwd` function has a hardcoded assumption about coefficient sign direction that conflicts with the target encoding used in this dataset. A manual fallback ensures the forward stepwise method works regardless of target encoding.

### Verification
After applying: forward stepwise should select at least 4 variables with p-values below threshold and negative coefficients consistent with the WoE/default prediction direction.
