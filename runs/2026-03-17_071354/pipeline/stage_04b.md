selection_method: xgboost_importance
binned_dataset_path: runs/2026-03-17_071354/data/loans_binned.csv
selected_variables: [Account Balance, Payment Status of Previous Credit, Value Savings/Stocks, Duration of Credit (month), Age (years), Purpose, Most valuable available asset]
excluded_from_shortlist: [Credit Amount — below cumulative importance threshold (0.90)]
model_auc: 0.807
model_gini: 0.614
model_ks: 0.4886
cumulative_importance_threshold: 0.90 (no adjustment needed)
xgb_hyperparameters: n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8

### XGBoost Feature Importances

| Variable | Importance (Gain) | Cumulative % | Selected |
|---|---|---|---|
| Account Balance | 0.2505 | 0.2505 | yes |
| Payment Status of Previous Credit | 0.1197 | 0.3701 | yes |
| Value Savings/Stocks | 0.1142 | 0.4843 | yes |
| Duration of Credit (month) | 0.1139 | 0.5982 | yes |
| Age (years) | 0.1054 | 0.7036 | yes |
| Purpose | 0.1010 | 0.8046 | yes |
| Most valuable available asset | 0.0986 | 0.9031 | yes |
| Credit Amount | 0.0969 | 1.0000 | no |

### Logistic Regression — Model Fit

| Metric | Value |
|---|---|
| Dep. Variable | Creditability |
| No. Observations | 1000 |
| Df Model | 7 |
| Pseudo R-squared | 0.2184 |
| Log-Likelihood | -477.437 |
| LL-Null | -610.864 |
| LLR p-value | 7.129918e-54 |
| Converged | True |

### Logistic Regression — Coefficients

| Variable | Coef | Std Err | z | P>\|z\| | [0.025 | 0.975] |
|---|---|---|---|---|---|---|
| const | -0.8467 | 0.0819 | -10.3392 | 0.000000 | -1.0072 | -0.6862 |
| Account Balance | -0.7784 | 0.1029 | -7.5634 | 0.000000 | -0.9801 | -0.5767 |
| Payment Status of Previous Credit | -0.7254 | 0.1516 | -4.7836 | 0.000002 | -1.0226 | -0.4282 |
| Value Savings/Stocks | -0.7366 | 0.1964 | -3.7504 | 0.000177 | -1.1216 | -0.3517 |
| Duration of Credit (month) | -0.8875 | 0.1625 | -5.4612 | 0.000000 | -1.2060 | -0.5690 |
| Age (years) | -0.8612 | 0.2542 | -3.3873 | 0.000706 | -1.3596 | -0.3629 |
| Purpose | -1.0202 | 0.2020 | -5.0516 | 0.000000 | -1.4161 | -0.6244 |
| Most valuable available asset | -0.6867 | 0.2521 | -2.7241 | 0.006448 | -1.1808 | -0.1926 |

### Self-Assessment Results

```
coefficients:
  - variable: Account Balance
    coefficient: -0.7784
    woe_direction_consistent: true
    vif: 1.15
  - variable: Payment Status of Previous Credit
    coefficient: -0.7254
    woe_direction_consistent: true
    vif: 1.07
  - variable: Value Savings/Stocks
    coefficient: -0.7366
    woe_direction_consistent: true
    vif: 1.08
  - variable: Duration of Credit (month)
    coefficient: -0.8875
    woe_direction_consistent: true
    vif: 1.10
  - variable: Age (years)
    coefficient: -0.8612
    woe_direction_consistent: true
    vif: 1.06
  - variable: Purpose
    coefficient: -1.0202
    woe_direction_consistent: true
    vif: 1.03
  - variable: Most valuable available asset
    coefficient: -0.6867
    woe_direction_consistent: true
    vif: 1.10
score_statistics:
  min: 424.9
  max: 621.3
  mean: 520.6
  pct_below_400: 0.00
  pct_above_800: 0.00
score_monotonicity_across_deciles: true
cross_validation:
  kfold_auc: 0.8021
  boots_auc: 0.7989
  dev_auc: 0.807
  kfold_diff: 0.0049
  boots_diff: 0.0081
  stability: PASS
model_params_path: runs/2026-03-17_071354/pipeline/model_params_xgb.json
model_flags: None
```
