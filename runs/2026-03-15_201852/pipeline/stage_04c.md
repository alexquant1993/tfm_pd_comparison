selection_method: forward_stepwise
binned_dataset_path: runs/2026-03-15_201852/data/loans_binned.csv
selected_variables: ['Account Balance', 'Duration of Credit (month)', 'Payment Status of Previous Credit', 'Purpose', 'Credit Amount', 'Value Savings/Stocks', 'Age (years)', 'Most valuable available asset']
excluded_from_shortlist: []
model_auc: 0.8242
model_gini: 0.6484
model_ks: 0.5248
p_value_threshold_used: 0.05
cv_auc: 0.8201
bootstrap_auc: 0.8163

## Forward Stepwise Selection Steps

| Step | Variable Added | LR p-value | AIC |
|------|---------------|------------|-----|
| 1 | Account Balance | 0.0000 | 1094.39 |
| 2 | Duration of Credit (month) | 0.0000 | 1048.43 |
| 3 | Payment Status of Previous Credit | 0.0000 | 1021.88 |
| 4 | Purpose | 0.0000 | 996.98 |
| 5 | Credit Amount | 0.0000 | 972.34 |
| 6 | Value Savings/Stocks | 0.0001 | 957.89 |
| 7 | Age (years) | 0.0004 | 946.73 |
| 8 | Most valuable available asset | 0.0443 | 944.67 |


## Logistic Regression Model Fit

| Metric | Value |
|---|---|
| Dep. Variable | Creditability |
| No. Observations | 1000 |
| Df Model | 8 |
| Pseudo R-squared | 0.2415 |
| Log-Likelihood | -463.34 |
| LL-Null | -610.86 |
| LLR p-value | 4.64e-59 |
| Converged | True |

## Coefficients

| Variable                          |    Coef |   Std Err |        z |   P>|z| |   [0.025 |   0.975] |
|:----------------------------------|--------:|----------:|---------:|--------:|---------:|---------:|
| const                             | -0.8363 |    0.0833 | -10.0396 |  0      |  -0.9996 |  -0.6731 |
| Account Balance                   | -0.7865 |    0.1049 |  -7.4954 |  0      |  -0.9921 |  -0.5808 |
| Duration of Credit (month)        | -0.7246 |    0.1642 |  -4.413  |  0      |  -1.0464 |  -0.4028 |
| Payment Status of Previous Credit | -0.7583 |    0.1544 |  -4.9129 |  0      |  -1.0608 |  -0.4558 |
| Purpose                           | -1.0519 |    0.2046 |  -5.141  |  0      |  -1.4529 |  -0.6509 |
| Credit Amount                     | -0.8309 |    0.1707 |  -4.8685 |  0      |  -1.1654 |  -0.4964 |
| Value Savings/Stocks              | -0.7424 |    0.201  |  -3.6941 |  0.0002 |  -1.1363 |  -0.3485 |
| Age (years)                       | -0.8787 |    0.2326 |  -3.7771 |  0.0002 |  -1.3346 |  -0.4227 |
| Most valuable available asset     | -0.5145 |    0.2557 |  -2.0121 |  0.0442 |  -1.0157 |  -0.0133 |

## Self-Assessment Results

### Check 1: Variable Count
- Variables selected: 8
- Status: PASS


### Check 2: Coefficient Signs
- All negative (consistent with WoE coding): True
- Details: All coefficients negative (consistent with WoE coding where target=1=default)

### Check 3: VIF (Multicollinearity)
- All VIF < 5: True
- Max VIF: 1.15

### Check 4: Score Distribution
- Range: [399.8, 625.5]
- Mean: 521.6, Std: 42.7
- Below 400: 0.1%, Above 800: 0.0%

### Check 5: Decile Monotonicity
- Strictly monotonic: True
- Violations: 0

### Check 6: CV Stability
- Dev AUC: 0.8242, CV AUC: 0.8201, Gap: 0.0041
- Bootstrap AUC: 0.8163
- Stable (gap <= 0.03): True

## Flags
None
