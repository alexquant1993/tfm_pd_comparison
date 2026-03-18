selection_method: forward_stepwise
binned_dataset_path: runs/2026-03-17_071354/data/loans_binned.csv
selected_variables: [Account Balance, Duration of Credit (month), Payment Status of Previous Credit, Purpose, Value Savings/Stocks, Age (years), Most valuable available asset, Credit Amount]
excluded_from_shortlist: []
model_auc: 0.8109
model_gini: 0.6218
model_ks: 0.5067
p_value_threshold_used: 0.05

### Forward Stepwise Selection Steps

| Step | Variable Added | LR p-value | Model AUC |
|---|---|---|---|
| 1 | Account Balance | 0.000000 | 0.7078 |
| 2 | Duration of Credit (month) | 0.000000 | 0.7567 |
| 3 | Payment Status of Previous Credit | 0.000000 | 0.7755 |
| 4 | Purpose | 0.000000 | 0.7924 |
| 5 | Value Savings/Stocks | 0.000114 | 0.7987 |
| 6 | Age (years) | 0.003016 | 0.8035 |
| 7 | Most valuable available asset | 0.006464 | 0.8070 |
| 8 | Credit Amount | 0.030454 | 0.8109 |

### Logistic Regression -- Model Fit

| Metric | Value |
|---|---|
| Dep. Variable | Creditability |
| No. Observations | 1000 |
| Df Model | 8 |
| Pseudo R-squared | 0.2222 |
| Log-Likelihood | -475.102 |
| LL-Null | -610.864 |
| LLR p-value | 4.665551e-54 |
| Converged | True |

### Logistic Regression -- Coefficients

| Variable | Coef | Std Err | z | P>\|z\| | [0.025 | 0.975] |
|---|---|---|---|---|---|---|
| const | -0.8475 | 0.0822 | -10.3149 | 0.000000 | -1.0086 | -0.6865 |
| Account Balance | -0.7848 | 0.1033 | -7.5939 | 0.000000 | -0.9873 | -0.5822 |
| Duration of Credit (month) | -0.7160 | 0.1783 | -4.0146 | 0.000060 | -1.0655 | -0.3664 |
| Payment Status of Previous Credit | -0.7257 | 0.1520 | -4.7759 | 0.000002 | -1.0235 | -0.4279 |
| Purpose | -1.0367 | 0.2024 | -5.1211 | 0.000000 | -1.4335 | -0.6400 |
| Value Savings/Stocks | -0.7488 | 0.1978 | -3.7855 | 0.000153 | -1.1365 | -0.3611 |
| Age (years) | -0.8916 | 0.2554 | -3.4908 | 0.000482 | -1.3921 | -0.3910 |
| Most valuable available asset | -0.5853 | 0.2575 | -2.2727 | 0.023044 | -1.0900 | -0.0805 |
| Credit Amount | -0.5164 | 0.2390 | -2.1608 | 0.030714 | -0.9848 | -0.0480 |

### Self-Assessment Results

```
coefficients:
  - variable: Account Balance
    coefficient: -0.7848
    woe_direction_consistent: true
    vif: 1.15
  - variable: Duration of Credit (month)
    coefficient: -0.7160
    woe_direction_consistent: true
    vif: 1.37
  - variable: Payment Status of Previous Credit
    coefficient: -0.7257
    woe_direction_consistent: true
    vif: 1.07
  - variable: Purpose
    coefficient: -1.0367
    woe_direction_consistent: true
    vif: 1.03
  - variable: Value Savings/Stocks
    coefficient: -0.7488
    woe_direction_consistent: true
    vif: 1.08
  - variable: Age (years)
    coefficient: -0.8916
    woe_direction_consistent: true
    vif: 1.06
  - variable: Most valuable available asset
    coefficient: -0.5853
    woe_direction_consistent: true
    vif: 1.14
  - variable: Credit Amount
    coefficient: -0.5164
    woe_direction_consistent: true
    vif: 1.38
score_statistics:
  min: 413.8
  max: 618.3
  mean: 520.7
  pct_below_400: 0.00
  pct_above_800: 0.00
score_monotonicity_across_deciles: true
model_params_path: runs/2026-03-17_071354/pipeline/model_params_fwd.json
model_flags: None
```
