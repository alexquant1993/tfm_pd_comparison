# Stage 05 — Fix Proposals

## Diagnostic Summary
- Smoke tests: 5/5 passed
- Full diagnostic triggered: no
- Issues found: 0

### Smoke Test Results

| # | Check | Result |
|---|---|---|
| 1 | Calibration not circular | PASS — 7/8 grades have \|calibrated_pd - observed_dr\| > 0.01. Calibration uses model-predicted PDs, not observed DRs. |
| 2 | Scaling factor reasonable | PASS — Factor = 1.0000, within [0.5, 2.0]. Factor is ~1.0 because the target CT (30.0%) equals the in-sample mean predicted probability — this is mathematically expected for a well-calibrated logistic regression, not a circularity issue. |
| 3 | Central tendency achieved | PASS — \|0.3000 - 0.3000\| = 0.0000 <= 0.005 |
| 4 | PD ordering | PASS — Calibrated PDs strictly monotonic across all 8 grades (A=3.56% to H=73.97%) |
| 5 | Grade population | PASS — All grades have 12.5% of portfolio (quantile-based boundaries ensure equal distribution) |
