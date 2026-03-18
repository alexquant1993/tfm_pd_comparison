# Stage 04a — Fix Proposals

## Diagnostic Summary
- Smoke tests: 5/5 passed
- Full diagnostic triggered: no
- Issues found: 0

### Smoke Test Results

| # | Check | Result |
|---|---|---|
| 1 | Coefficient sign consistency | PASS — all 8 WoE coefficients have expected sign (negative, consistent with WoE=ln(dist_good/dist_bad) and target=1 as default) |
| 2 | Multicollinearity | PASS — all VIF values < 2.0 (max VIF = 1.38 for Credit Amount) |
| 3 | Minimum discrimination | PASS — AUC = 0.8109 > 0.60 (Strong) |
| 4 | Score range | PASS — range [413.8, 618.3] overlaps target [400, 800], 0% at extremes |
| 5 | Model params complete | PASS — model_params_miv.json contains all required keys |
