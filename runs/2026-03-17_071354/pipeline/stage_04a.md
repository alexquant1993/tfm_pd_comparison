selection_method: miv
binned_dataset_path: runs/2026-03-17_071354/data/loans_binned.csv
selected_variables: [Account Balance, Payment Status of Previous Credit, Duration of Credit (month), Value Savings/Stocks, Purpose, Credit Amount, Most valuable available asset, Age (years)]
excluded_from_shortlist: []
model_auc: 0.8109
model_gini: 0.6218
model_ks: 0.5067
miv_threshold_used: 0.02

### Logistic Regression — Model Fit

| Metric | Value |
|---|---|
| Dep. Variable | Creditability |
| No. Observations | 1000 |
| Df Model | 8 |
| Pseudo R-squared | 0.2222 |
| Log-Likelihood | -475.102 |
| LL-Null | -610.864 |
| LLR p-value | 0.000000 |
| Converged | True |

### Logistic Regression — Coefficients

| Variable | Coef | Std Err | z | P>\|z\| | [0.025 | 0.975] |
|---|---|---|---|---|---|---|
| const | -0.8475 | 0.0822 | -10.3149 | 0.0000 | -1.0086 | -0.6865 |
| Account Balance | -0.7848 | 0.1033 | -7.5939 | 0.0000 | -0.9873 | -0.5822 |
| Payment Status of Previous Credit | -0.7257 | 0.1520 | -4.7759 | 0.0000 | -1.0235 | -0.4279 |
| Duration of Credit (month) | -0.7160 | 0.1783 | -4.0146 | 0.0001 | -1.0655 | -0.3664 |
| Value Savings/Stocks | -0.7488 | 0.1978 | -3.7855 | 0.0002 | -1.1365 | -0.3611 |
| Purpose | -1.0367 | 0.2024 | -5.1211 | 0.0000 | -1.4335 | -0.6400 |
| Credit Amount | -0.5164 | 0.2390 | -2.1608 | 0.0307 | -0.9848 | -0.0480 |
| Most valuable available asset | -0.5853 | 0.2575 | -2.2727 | 0.0230 | -1.0900 | -0.0805 |
| Age (years) | -0.8916 | 0.2554 | -3.4908 | 0.0005 | -1.3921 | -0.3910 |

Note: Coefficients are negative because pdtoolkit WoE = ln(dist_good/dist_bad). Positive WoE indicates lower risk, so negative coefficients on WoE correctly predict lower P(default=1). All coefficients are directionally consistent.

### Self-Assessment Results

```
coefficients:
  - variable: Account Balance
    coefficient: -0.784781
    woe_direction_consistent: true
    vif: 1.15
  - variable: Payment Status of Previous Credit
    coefficient: -0.725703
    woe_direction_consistent: true
    vif: 1.07
  - variable: Duration of Credit (month)
    coefficient: -0.715972
    woe_direction_consistent: true
    vif: 1.37
  - variable: Value Savings/Stocks
    coefficient: -0.748810
    woe_direction_consistent: true
    vif: 1.08
  - variable: Purpose
    coefficient: -1.036740
    woe_direction_consistent: true
    vif: 1.03
  - variable: Credit Amount
    coefficient: -0.516378
    woe_direction_consistent: true
    vif: 1.38
  - variable: Most valuable available asset
    coefficient: -0.585271
    woe_direction_consistent: true
    vif: 1.14
  - variable: Age (years)
    coefficient: -0.891556
    woe_direction_consistent: true
    vif: 1.06
score_statistics:
  min: 413.8
  max: 618.3
  mean: 520.7
  pct_below_400: 0.00
  pct_above_800: 0.00
score_monotonicity_across_deciles: true
cross_validation:
  cv_auc: 0.8065
  cv_auc_diff: 0.0044
  cv_stable: true
bootstrap_validation:
  boots_auc: 0.8026
  boots_auc_diff: 0.0083
model_params_path: runs/2026-03-17_071354/pipeline/model_params_miv.json
model_flags: None
```
