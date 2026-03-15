models_compared: 3
models_available: [miv, xgboost_importance, forward_stepwise]

## Model Comparison

| Metric | MIV | XGBoost | Forward |
|---|---|---|---|
| AUC | 0.8242 | 0.8172 | 0.8242 |
| Gini | 0.6484 | 0.6344 | 0.6484 |
| KS | 0.5248 | 0.5076 | 0.5248 |
| CV AUC | 0.8201 | 0.8134 | 0.8201 |
| N Variables | 8 | 7 | 8 |
| All Signs Consistent | Yes | Yes | Yes |
| Decile Monotonic | Yes | Yes | Yes |
| Max VIF | 1.15 | 1.14 | 1.15 |

## Composite Score Breakdown

| Component (Weight) | MIV | XGBoost | Forward |
|---|---|---|---|
| AUC (25%) | 1.000 | 0.000 | 1.000 |
| Gini (15%) | 1.000 | 0.000 | 1.000 |
| KS (10%) | 1.000 | 0.000 | 1.000 |
| CV Stability (20%) | 0.863 | 0.873 | 0.863 |
| Sign Consistency (15%) | 1.000 | 1.000 | 1.000 |
| Parsimony (10%) | 0.500 | 0.625 | 0.500 |
| Monotonicity (5%) | 1.000 | 1.000 | 1.000 |
| **Composite** | **0.9226** | **0.4371** | **0.9226** |

## Variable Selection Overlap

- Variables selected by all 3 methods: 7
- Total unique variables: 8
- Overlap ratio: 0.88 (7/8)
- Variable unique to MIV/Forward only: Age (years)

| Variable | MIV | XGBoost | Forward |
|---|---|---|---|
| Account Balance | Yes | Yes | Yes |
| Credit Amount | Yes | Yes | Yes |
| Duration of Credit (month) | Yes | Yes | Yes |
| Most valuable available asset | Yes | Yes | Yes |
| Purpose | Yes | Yes | Yes |
| Payment Status of Previous Credit | Yes | Yes | Yes |
| Value Savings/Stocks | Yes | Yes | Yes |
| Age (years) | Yes | No | Yes |

## Champion Selection

- **Champion: MIV** (composite score: 0.9226)
- Runner-up: Forward Stepwise (composite score: 0.9226)
- Third: XGBoost Importance (composite score: 0.4371)

**Rationale:** MIV and Forward Stepwise selected identical variable sets (8 variables) and produced identical logistic regression models with matching coefficients and performance metrics (AUC=0.8242, Gini=0.6484, KS=0.5248). Both tie on composite score. MIV is selected as champion by convention (first in evaluation order). XGBoost Importance excluded Age (years) due to cumulative importance threshold, resulting in a 7-variable model with moderately lower discrimination (AUC=0.8172).

**Champion model params copied to:** `runs/2026-03-15_201852/pipeline/model_params.json`

## Flags
None
