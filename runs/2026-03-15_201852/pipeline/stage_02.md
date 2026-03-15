clean_dataset_path: runs/2026-03-15_201852/data/loans_clean.csv
clean_dataset_checksum: 0e5c2b38995e19bd7bedea863519d791
n_observations: 1000
transformations_applied:
  - variable: Duration of Credit (month)
    type: outlier_imputation
    method: iqr
    n_values_changed: 70
    cap_value: 42.0
    note: Upper-tail IQR capping (7.0% of observations)
  - variable: Credit Amount
    type: outlier_imputation
    method: iqr
    n_values_changed: 72
    cap_value: 7882.4
    note: Upper-tail IQR capping (7.2% of observations)
  - variable: Age (years)
    type: outlier_imputation
    method: iqr
    n_values_changed: 23
    cap_value: 64.5
    note: Upper-tail IQR capping (2.3% of observations)
data_preparation_flags: None
