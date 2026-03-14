# PD Modelling Conventions

Domain thresholds and interpretation rules for PD model development. Load this skill before making any threshold or classification decisions.

---

## IV Thresholds

| IV Range | Interpretation | Action |
|---|---|---|
| < 0.02 | Useless | Exclude |
| 0.02 - 0.10 | Weak | Exclude unless business justification |
| 0.10 - 0.30 | Medium | Include in shortlist |
| > 0.30 | Strong | Include in shortlist |

---

## WoE Requirements

- Strictly monotonic preferred
- Marginal violations (single bin out of order) -- document and flag, do not auto-exclude
- Non-monotonic after 3 rebinning attempts -- exclude and document reason
- Rebinning attempt protocol:
  1. Attempt 1: increase `max_groups` by 2
  2. Attempt 2: decrease `max_groups` by 2
  3. Attempt 3: force monotonic direction using constraint
  4. After 3 attempts still failing: mark as non-monotonic, document all attempts, flag for human review

---

## Bin Size Minimums

- Minimum 5% of total observations per bin
- Minimum 20 events (defaults) per bin
- Merge bins that fall below threshold before accepting result

---

## AUC Benchmarks

| AUC | Assessment |
|---|---|
| < 0.60 | Unacceptable -- reject |
| 0.60 - 0.70 | Weak -- flag for review |
| 0.70 - 0.80 | Acceptable |
| > 0.80 | Strong |

---

## Validation Thresholds

- Gini > 0.35: acceptable discriminatory power
- KS > 0.30: acceptable separation
- Homogeneity p > 0.05: fail (grade-level instability)
- Heterogeneity p < 0.05: fail (overlap between grades)
- PP testing: binomial test p < 0.05 at any grade = predictive power failure

---

## Score Scale Convention

- Base score: 600 points at 50:1 odds
- PDO (Points to Double Odds): 20
- Score range target: 400-800

---

## Regulatory Flags

Items to surface to human reviewer regardless of pass/fail:

- Any test result with p-value within 0.01 of threshold
- Any variable with non-monotonic WoE accepted under business justification
- Coefficient sign reversal in final model
- Stability AUC difference > 0.05 between sample halves
