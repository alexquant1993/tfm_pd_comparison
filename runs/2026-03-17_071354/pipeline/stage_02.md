clean_dataset_path: runs/2026-03-17_071354/data/loans_clean.csv
clean_dataset_checksum: 0e5c2b38995e19bd7bedea863519d791
n_observations: 1000
transformations_applied:
  - variable: Duration of Credit (month)
    type: outlier_imputation
    method: iqr
    n_values_changed: 70
    note: Upper cap at 42.0 (IQR 1.5x). No lower outliers. Max reduced from 72 to 42.
  - variable: Credit Amount
    type: outlier_imputation
    method: iqr
    n_values_changed: 72
    note: Upper cap at 7882.4 (IQR 1.5x). No lower outliers. Max reduced from 18424 to 7882.
  - variable: Age (years)
    type: outlier_imputation
    method: iqr
    n_values_changed: 23
    note: Upper cap at 64.5 (IQR 1.5x). No lower outliers. Max reduced from 75 to 64.5.
data_preparation_flags: None
