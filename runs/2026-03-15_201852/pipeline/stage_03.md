clean_dataset_path: c:/projects/superagent\runs/2026-03-15_201852\data\loans_clean.csv
clean_dataset_checksum: 0e5c2b38995e19bd7bedea863519d791
binning_method: optbinning (OptimalBinning, monotonic_trend="auto")
binned_dataset_path: c:/projects/superagent\runs/2026-03-15_201852\data\loans_binned.csv
binned_dataset_checksum: 39ff8815afec3ac19e0f056dcba28eff
variables_analysed: 20
variable_results:
  - variable: Account Balance
    iv: 0.666
    auc: 0.7078
    n_bins: 5
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: shortlist
  - variable: Payment Status of Previous Credit
    iv: 0.2918
    auc: 0.6265
    n_bins: 4
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: shortlist
  - variable: Duration of Credit (month)
    iv: 0.2798
    auc: 0.6365
    n_bins: 8
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: shortlist
  - variable: Credit Amount
    iv: 0.2421
    auc: 0.6209
    n_bins: 7
    monotonic: false
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: shortlist
  - variable: Value Savings/Stocks
    iv: 0.1925
    auc: 0.5988
    n_bins: 5
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: shortlist
  - variable: Purpose
    iv: 0.1676
    auc: 0.6104
    n_bins: 7
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: shortlist
  - variable: Age (years)
    iv: 0.1332
    auc: 0.5957
    n_bins: 7
    monotonic: false
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: shortlist
  - variable: Most valuable available asset
    iv: 0.1126
    auc: 0.5851
    n_bins: 4
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: shortlist
  - variable: Length of current employment
    iv: 0.0864
    auc: 0.5808
    n_bins: 6
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: Low IV (0.0864 < 0.1)
  - variable: Type of apartment
    iv: 0.0854
    auc: 0.5681
    n_bins: 4
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: Low IV (0.0854 < 0.1)
  - variable: Concurrent Credits
    iv: 0.0576
    auc: 0.5481
    n_bins: 3
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: Low IV (0.0576 < 0.1)
  - variable: Sex & Marital Status
    iv: 0.0446
    auc: 0.5518
    n_bins: 4
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: Low IV (0.0446 < 0.1)
  - variable: Instalment per cent
    iv: 0.0263
    auc: 0.5434
    n_bins: 5
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: Low IV (0.0263 < 0.1)
  - variable: Guarantors
    iv: 0.0164
    auc: 0.5133
    n_bins: 3
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: Useless (IV < 0.02)
  - variable: No of Credits at this Bank
    iv: 0.0106
    auc: 0.5245
    n_bins: 3
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: Useless (IV < 0.02)
  - variable: Occupation
    iv: 0.0085
    auc: 0.5212
    n_bins: 4
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: Useless (IV < 0.02)
  - variable: Telephone
    iv: 0.0064
    auc: 0.5195
    n_bins: 3
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: Useless (IV < 0.02)
  - variable: Duration in Current address
    iv: 0.0036
    auc: 0.5162
    n_bins: 5
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: Useless (IV < 0.02)
  - variable: No of dependents
    iv: 0.0
    auc: 0.5
    n_bins: 2
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: Useless (IV < 0.02)
  - variable: Foreign Worker
    iv: 0.0
    auc: 0.5
    n_bins: 2
    monotonic: true
    optbinning_status: OPTIMAL
    monotonicity_attempts: 1
    economic_sign_plausible: true
    status: excluded
    exclusion_reason: Useless (IV < 0.02)
shortlist: ['Account Balance', 'Payment Status of Previous Credit', 'Duration of Credit (month)', 'Credit Amount', 'Value Savings/Stocks', 'Purpose', 'Age (years)', 'Most valuable available asset']
correlation_clusters: []
substitute_variables: ['Length of current employment', 'Type of apartment', 'Concurrent Credits']
iv_threshold_used: 0.1
bivariate_flags:
  - Foreign Worker confirmed useless (IV=0.0000) -- NZV flag from Stage 01 validated
  - Credit Amount shortlisted but non-monotonic WoE (status: OPTIMAL)
  - Age (years) shortlisted but non-monotonic WoE (status: OPTIMAL)
  - Account Balance has suspiciously high IV (0.6660) -- review for data leakage
