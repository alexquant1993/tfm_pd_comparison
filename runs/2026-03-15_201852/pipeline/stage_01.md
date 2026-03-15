dataset_path: data/loans.csv
n_observations: 1000
n_variables: 21
target_variable: Creditability
target_default_rate: 0.30
numeric_variables: ["Duration of Credit (month)", "Credit Amount", "Age (years)"]
categorical_variables: ["Account Balance", "Payment Status of Previous Credit", "Purpose", "Value Savings/Stocks", "Length of current employment", "Instalment per cent", "Sex & Marital Status", "Guarantors", "Duration in Current address", "Most valuable available asset", "Concurrent Credits", "Type of apartment", "No of Credits at this Bank", "Occupation", "No of dependents", "Telephone", "Foreign Worker"]
variables_to_impute_sc: []
variables_to_impute_outliers:
  - Duration of Credit (month): cap at 42.0 (IQR method, 70 outliers = 7.0%)
  - Credit Amount: cap at 7882.4 (IQR method, 72 outliers = 7.2%)
  - Age (years): cap at 64.5 (IQR method, 23 outliers = 2.3%)
variables_to_exclude: []
high_correlation_pairs:
  - (Duration of Credit (month), Credit Amount, 0.625)
  - (Payment Status of Previous Credit, No of Credits at this Bank, 0.4371)
  - (Occupation, Telephone, 0.383)
  - (Most valuable available asset, Type of apartment, 0.343)
  - (Credit Amount, Most valuable available asset, 0.3116)
  - (Duration of Credit (month), Most valuable available asset, 0.304)
  - (Age (years), Type of apartment, 0.3033)
data_quality_flags:
  - Foreign Worker has near-zero variance (frequency ratio 26:1, 96.3% in class 1). Retained for IV review in Stage 03.
  - No pairs exceed |r| > 0.7 threshold.
  - No missing values in the dataset — complete cases only.
  - All outliers are upper-tail (right-skewed distributions).

## Variable Actions

| Variable | Type | Action | Detail |
|---|---|---|---|
| Account Balance | Categorical | keep | 4 levels, no issues |
| Duration of Credit (month) | Numeric | impute-outliers | Cap at 42.0 (IQR). 70 upper-tail outliers (7.0%) |
| Payment Status of Previous Credit | Categorical | keep | 5 levels, no issues |
| Purpose | Categorical | keep | 10 levels, no issues |
| Credit Amount | Numeric | impute-outliers | Cap at 7882.4 (IQR). 72 upper-tail outliers (7.2%) |
| Value Savings/Stocks | Categorical | keep | 5 levels, no issues |
| Length of current employment | Categorical | keep | 5 levels, no issues |
| Instalment per cent | Categorical | keep | 4 levels, no issues |
| Sex & Marital Status | Categorical | keep | 4 levels, no issues |
| Guarantors | Categorical | keep | 3 levels, no issues |
| Duration in Current address | Categorical | keep | 4 levels, no issues |
| Most valuable available asset | Categorical | keep | 4 levels, no issues |
| Age (years) | Numeric | impute-outliers | Cap at 64.5 (IQR). 23 upper-tail outliers (2.3%) |
| Concurrent Credits | Categorical | keep | 3 levels, no issues |
| Type of apartment | Categorical | keep | 3 levels, no issues |
| No of Credits at this Bank | Categorical | keep | 4 levels, no issues |
| Occupation | Categorical | keep | 4 levels, no issues |
| No of dependents | Categorical | keep | 2 levels, no issues |
| Telephone | Categorical | keep | 2 levels, no issues |
| Foreign Worker | Categorical | keep (flagged) | NZV freq ratio 26:1. Review IV in Stage 03 |
