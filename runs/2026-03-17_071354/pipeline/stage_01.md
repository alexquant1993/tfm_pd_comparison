dataset_path: data/loans.csv
n_observations: 1000
n_variables: 21
target_variable: Creditability
target_default_rate: 0.30
numeric_variables: [Account Balance, Duration of Credit (month), Credit Amount, Value Savings/Stocks, Length of current employment, Instalment per cent, Duration in Current address, Most valuable available asset, Age (years), No of Credits at this Bank, Occupation, No of dependents]
categorical_variables: [Payment Status of Previous Credit, Purpose, Sex & Marital Status, Guarantors, Concurrent Credits, Type of apartment, Telephone, Foreign Worker]
variables_to_impute_sc: []
variables_to_impute_outliers: [Duration of Credit (month) (IQR cap, 70 outliers, 7.0%), Credit Amount (IQR cap, 72 outliers, 7.2%), Age (years) (IQR cap, 23 outliers, 2.3%)]
variables_to_exclude: []
high_correlation_pairs: []
data_quality_flags: [Foreign Worker has near-zero variance (freq ratio 26.0, 96.3% category 1)]
nzv_analysis: Foreign Worker flagged with frequency ratio 26.0 (963 vs 37 observations). All other variables have acceptable variance.
missing_values: No missing values in any variable (0% missing rate across all 21 variables).
outlier_method: IQR (1.5x multiplier) — selected for robustness to right-skewed distributions in Duration, Amount, and Age.
