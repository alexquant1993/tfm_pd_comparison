# Stage 04x Fix-Proposer Diagnostic

## Smoke Tests

### 1. Canonical params exist
- **Status:** PASS
- `model_params.json` exists at `runs/2026-03-15_201852/pipeline/model_params.json`

### 2. Comparison table complete
- **Status:** PASS
- `stage_04x.md` contains full comparison table with AUC, Gini, KS, CV AUC, N Variables for all 3 models

### 3. Champion declared
- **Status:** PASS
- Champion: MIV (miv), with rationale explaining tie with Forward Stepwise and selection by convention

### 4. Champion params match declared champion
- **Status:** PASS
- `model_params.json` contains `selection_method: miv`, matching the declared champion

## Issues Found
None

## Advisory Notes

1. **MIV-Forward tie (severity: info):** MIV and Forward Stepwise produced identical models (same 8 variables, same coefficients, same AUC/Gini/KS). This is expected when both methods converge on the same feature set from a well-structured shortlist. The tie-breaking is by convention only.

2. **XGBoost parsimony advantage unused (severity: info):** XGBoost's 7-variable model trades a small discrimination loss (AUC delta = -0.007) for one fewer variable. If parsimony were weighted higher, XGBoost would be more competitive. Current weighting appropriately favors discrimination.

## Summary
- **Total issues: 0**
- **Critical: 0**
- **Advisory notes: 2**
