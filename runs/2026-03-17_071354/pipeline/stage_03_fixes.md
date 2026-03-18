# Stage 03 — Fix Proposals

## Diagnostic Summary
- Smoke tests: 5/5 passed
- Full diagnostic triggered: no
- Issues found: 1

---

## Issue 1: CP solver ascending monotonicity broken due to ortools version incompatibility

### Classification
- Category: C: Approach Issue
- Severity: Warning

### Symptoms
OptimalBinning with solver="cp" and monotonic_trend="ascending" raises a TypeError from ortools cp_model for all numerical variables. The error originates in `cp.py:399` where `add_constraint_monotonic_ascending` passes incompatible types to ortools `model.Add()`. Descending with CP works. Both directions work with solver="mip".

### Root Cause
The installed version of `ortools` has a breaking change in the CP-SAT solver's `__rsub__` method for `IntAffine` objects. The `optbinning` package's CP constraint builder uses arithmetic operations on CP-SAT variables that are no longer compatible. This affects only the ascending monotonicity constraint builder.

### Proposed Fix

**[Approach Change]**
- **Current approach:** Agent spec says to use `solver="cp"`. Due to the ortools incompatibility, the notebook falls back to `solver="mip"` for ascending monotonicity.
- **Recommended approach:** Update the agent instructions to prefer `solver="mip"` as the primary solver, with `solver="cp"` as a fallback. Alternatively, pin the ortools version to one compatible with the installed optbinning.
- **Requires:** prompt change to `.claude/agents/03-bivariate-analyst.md` OR `pip install ortools==9.8.3296` (last known compatible version)
- **Trade-offs:** MIP solver produces equivalent results for this dataset size. CP solver may be marginally faster for very large datasets but is non-functional for ascending constraints with current ortools.

### Verification
After fixing: `OptimalBinning(name="test", dtype="numerical", solver="cp", monotonic_trend="ascending").fit(x, y)` should complete without TypeError for Duration of Credit (month).
