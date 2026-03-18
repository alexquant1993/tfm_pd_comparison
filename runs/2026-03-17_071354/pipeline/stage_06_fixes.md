# Stage 06 — Fix Proposals

## Diagnostic Summary
- Smoke tests: 4/4 passed
- Full diagnostic triggered: yes (stage has flags: Grade F binomial test failure)
- Issues found: 1

---

## Issue 1: Grade F predictive power failure — observed DR exceeds calibrated PD

### Classification
- Category: C: Approach Issue
- Severity: Warning

### Symptoms
Grade F binomial test FAIL (p=0.0264): observed default rate 49.6% exceeds calibrated PD 40.6%. The Jeffreys test also fails (p=0.0214). This is the only grade with a PP failure out of 8 grades. The Hosmer-Lemeshow overall test passes (p=0.294), indicating the calibration is adequate in aggregate.

### Root Cause
The equal-frequency grading (125 obligors per grade) places the Grade F boundaries at score range [491.6, 504.9], which spans a narrow score interval where the observed default rate transitions sharply. The calibrated PD of 40.6% is derived from the model-predicted PDs (not observed DR, correctly avoiding circular calibration), but in this narrow score band the model's predicted PDs underestimate the true risk. This is a local calibration issue in the mid-range of the score distribution.

### Proposed Fix

**[Category C — Approach Change]**
- **Current approach:** Equal-frequency grading with 8 grades (125 obligors each). Calibrated PDs from model predictions.
- **Recommended approach:** Consider adjusting grade boundaries to use quantile-based breaks that respect natural risk clustering rather than strict equal-frequency. Alternatively, use 7 grades or 9 grades to see if the Grade F concentration issue resolves. A grade with only a 13-point score range (491.6 to 504.9) is narrow.
- **Requires:** Human decision — this is a calibration design choice
- **Trade-offs:** Changing grade count or boundaries affects all downstream reporting. The current failure is marginal (p=0.026 vs threshold 0.05) and isolated to one grade, so the regulatory risk is low.

### Verification
After adjusting grade boundaries: binomial test p-value for the affected grade should be >= 0.05, or the grade should be documented with a regulatory note explaining the deviation.
