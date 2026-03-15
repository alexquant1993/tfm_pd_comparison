selection_method: xgboost_importance
binned_dataset_path: runs/2026-03-15_201852/data/loans_binned.csv
selected_variables: ['Account Balance', 'Payment Status of Previous Credit', 'Duration of Credit (month)', 'Most valuable available asset', 'Credit Amount', 'Value Savings/Stocks', 'Purpose']
excluded_from_shortlist: ['Age (years): below cumulative importance threshold (100.0% cumulative)']
model_auc: 0.8172
model_gini: 0.6344
model_ks: 0.5076
cumulative_importance_threshold: 0.9
cv_auc: 0.8134
bootstrap_auc: 0.8109

## XGBoost Feature Importances

| Variable                          |   Importance (Gain) |   Cumulative % | Selected   |
|:----------------------------------|--------------------:|---------------:|:-----------|
| Account Balance                   |              0.2294 |           22.9 | Yes        |
| Payment Status of Previous Credit |              0.1352 |           36.5 | Yes        |
| Duration of Credit (month)        |              0.1162 |           48.1 | Yes        |
| Most valuable available asset     |              0.1114 |           59.2 | Yes        |
| Credit Amount                     |              0.1095 |           70.2 | Yes        |
| Value Savings/Stocks              |              0.1086 |           81   | Yes        |
| Purpose                           |              0.0977 |           90.8 | Yes        |
| Age (years)                       |              0.092  |          100   | No         |

## Logistic Regression Model Fit

| Metric | Value |
|---|---|
| Dep. Variable | Creditability |
| No. Observations | 1000 |
| Df Model | 7 |
| Pseudo R-squared | 0.2293 |
| Log-Likelihood | -470.78 |
| LL-Null | -610.86 |
| LLR p-value | 1.03e-56 |
| Converged | True |

## Logistic Regression Coefficients

| Variable                          |    Coef |   Std Err |        z |   P>|z| |   [0.025 |   0.975] |
|:----------------------------------|--------:|----------:|---------:|--------:|---------:|---------:|
| const                             | -0.8252 |    0.0822 | -10.037  |  0      |  -0.9863 |  -0.664  |
| Account Balance                   | -0.8173 |    0.1042 |  -7.8465 |  0      |  -1.0214 |  -0.6131 |
| Payment Status of Previous Credit | -0.7873 |    0.1525 |  -5.1627 |  0      |  -1.0862 |  -0.4884 |
| Duration of Credit (month)        | -0.7579 |    0.1619 |  -4.6809 |  0      |  -1.0753 |  -0.4406 |
| Most valuable available asset     | -0.3821 |    0.2497 |  -1.5302 |  0.126  |  -0.8715 |   0.1073 |
| Credit Amount                     | -0.8057 |    0.1694 |  -4.7571 |  0      |  -1.1376 |  -0.4737 |
| Value Savings/Stocks              | -0.7863 |    0.1989 |  -3.9524 |  0.0001 |  -1.1762 |  -0.3964 |
| Purpose                           | -0.9948 |    0.201  |  -4.9496 |  0      |  -1.3887 |  -0.6008 |

## Self-Assessment Results

### Check 1: Variable Count
- Variables selected: 7
- Status: PASS


### Check 2: Coefficient Signs
- All consistent: True
- Details: All coefficients negative (consistent with WoE coding where target=1=default)

### Check 3: VIF (Multicollinearity)
- All VIF < 5: True
- Max VIF: 1.14

### Check 4: Score Distribution
- Range: [401.1, 622.8]
- Mean: 520.7, Std: 40.8
- Below 400: 0.0%, Above 800: 0.0%

### Check 5: Decile Monotonicity
- Strictly monotonic: True
- Violations: 0

### Check 6: CV Stability
- Dev AUC: 0.8172, CV AUC: 0.8134, Gap: 0.0038
- Bootstrap AUC: 0.8109
- Stable (gap <= 0.03): True

## Flags
None
