# Stage 04x — Fix Proposals

## Diagnostic Summary
- Smoke tests: 4/4 passed
- Full diagnostic triggered: no
- Issues found: 1

---

## Issue 1: Min-max normalization over-penalizes XGBoost with narrow metric spread

### Classification
- Category: C: Approach Issue
- Severity: Info

### Symptoms
XGBoost composite score (0.4298) is dramatically lower than MIV/Forward (0.9207), despite raw discrimination metrics being only marginally lower (AUC 0.807 vs 0.8109, a difference of 0.0039). Min-max normalization assigns XGBoost a 0.0 on all three normalized discrimination metrics (AUC, Gini, KS) because it is the sole minimum across all three.

### Root Cause
The composite scoring methodology uses min-max normalization across the three models. When metrics are tightly clustered (AUC spread of only 0.0039), the model at the bottom receives 0.0 on all normalized metrics, which amplifies a negligible difference into a 50-point composite gap. This is a known limitation of min-max with small sample sizes (N=3 models).

### Proposed Fix

**Current approach:** Min-max normalization across models for AUC, Gini, KS.
**Recommended approach:** Consider either (a) raw-value scoring with reference benchmarks (e.g., AUC 0.60=0, AUC 0.90=1), or (b) min-max with a floor (e.g., worst model gets max(0.3, normalized_value)) to prevent extreme penalization when the spread is narrow.
**Requires:** Prompt change to `.claude/agents/04x-model-comparator.md` composite scoring section.
**Trade-offs:** Raw-value scoring better reflects absolute model quality; min-max better reflects relative ordering among candidates. With only 3 models and narrow spreads, raw-value scoring is more defensible.

### Verification
After applying: XGBoost composite score should be closer to MIV/Forward when the raw metric gap is <0.01 AUC points.
