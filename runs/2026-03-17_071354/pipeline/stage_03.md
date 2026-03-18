clean_dataset_path: runs/2026-03-17_071354/data/loans_clean.csv
clean_dataset_checksum: 0e5c2b38995e19bd7bedea863519d791
binning_method: optbinning (OptimalBinning, MIP/CP solver with explicit ascending/descending monotonic_trend)
binned_dataset_path: runs/2026-03-17_071354/data/loans_binned.csv
binned_dataset_checksum: 718de3be9b55fc40c7ccd568b7ca586d
variables_analysed: 20
variable_results:
  - variable: Account Balance
    iv: 0.666
    auc: 0.6953
    n_bins: 3
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 2
    monotonic_trend: descending
    special_codes: [4]
    economic_sign_plausible: true
    status: shortlist
  - variable: Payment Status of Previous Credit
    iv: 0.2918
    auc: 0.6266
    n_bins: 4
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    monotonic_trend: auto(categorical)
    economic_sign_plausible: true
    status: shortlist
  - variable: Duration of Credit (month)
    iv: 0.2798
    auc: 0.6365
    n_bins: 7
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 2
    monotonic_trend: ascending
    economic_sign_plausible: true
    status: shortlist
  - variable: Value Savings/Stocks
    iv: 0.1925
    auc: 0.5988
    n_bins: 3
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 2
    monotonic_trend: descending
    special_codes: [5]
    economic_sign_plausible: true
    status: shortlist
  - variable: Purpose
    iv: 0.1676
    auc: 0.6104
    n_bins: 6
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    monotonic_trend: auto(categorical)
    economic_sign_plausible: true
    status: shortlist
  - variable: Credit Amount
    iv: 0.1493
    auc: 0.5855
    n_bins: 5
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 2
    monotonic_trend: ascending
    economic_sign_plausible: true
    status: shortlist
  - variable: Most valuable available asset
    iv: 0.1126
    auc: 0.5323
    n_bins: 3
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 2
    monotonic_trend: ascending
    special_codes: [4]
    economic_sign_plausible: true
    status: shortlist
  - variable: Age (years)
    iv: 0.1013
    auc: 0.5845
    n_bins: 4
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 2
    monotonic_trend: descending
    economic_sign_plausible: true
    status: shortlist
  - variable: Type of apartment
    iv: 0.0854
    auc: 0.5681
    n_bins: 3
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    monotonic_trend: auto(categorical)
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: low IV (0.0854 < 0.10)
  - variable: Length of current employment
    iv: 0.0829
    auc: 0.5769
    n_bins: 3
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 2
    monotonic_trend: descending
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: low IV (0.0829 < 0.10)
  - variable: Concurrent Credits
    iv: 0.0576
    auc: 0.5481
    n_bins: 2
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    monotonic_trend: auto(categorical)
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: low IV (0.0576 < 0.10)
  - variable: Sex & Marital Status
    iv: 0.0447
    auc: 0.5524
    n_bins: 4
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    monotonic_trend: auto(categorical)
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: low IV (0.0447 < 0.10)
  - variable: Instalment per cent
    iv: 0.0263
    auc: 0.5434
    n_bins: 4
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 2
    monotonic_trend: ascending
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: low IV (0.0263 < 0.10)
  - variable: Guarantors
    iv: 0.0164
    auc: 0.5133
    n_bins: 2
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    monotonic_trend: auto(categorical)
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: low IV (0.0164 < 0.10)
  - variable: No of Credits at this Bank
    iv: 0.0101
    auc: 0.524
    n_bins: 2
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 2
    monotonic_trend: descending
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: low IV (0.0101 < 0.10)
  - variable: Occupation
    iv: 0.0081
    auc: 0.5195
    n_bins: 3
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 2
    monotonic_trend: ascending
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: low IV (0.0081 < 0.10)
  - variable: Telephone
    iv: 0.0064
    auc: 0.5195
    n_bins: 2
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    monotonic_trend: auto(categorical)
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: low IV (0.0064 < 0.10)
  - variable: Duration in Current address
    iv: 0.0018
    auc: 0.5071
    n_bins: 2
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 2
    monotonic_trend: ascending
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: low IV (0.0018 < 0.10)
  - variable: No of dependents
    iv: 0.0
    auc: 0.5012
    n_bins: 2
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 2
    monotonic_trend: descending
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: low IV (0.0 < 0.10)
  - variable: Foreign Worker
    iv: 0.0
    auc: 0.5
    n_bins: 1
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    monotonic_trend: auto(categorical)
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: low IV (0.0 < 0.10); near-zero variance (flagged in stage 01)
shortlist: [Account Balance, Payment Status of Previous Credit, Duration of Credit (month), Value Savings/Stocks, Purpose, Credit Amount, Most valuable available asset, Age (years)]
correlation_clusters: None (no pairs with |WoE correlation| > 0.7)
substitute_variables: None
iv_threshold_used: 0.10 (default, no adjustment needed — 8 variables passed)
bivariate_flags: None
