discriminatory_power:
  auc: 0.8109
  gini: 0.6218
  ks: 0.5067
  dp_test_pvalue: 2.5208e-11
  dp_test_result: PASS
  stability_auc_half1: 0.8236
  stability_auc_half2: 0.7950
  stability_result: PASS
predictive_power:
  hosmer_lemeshow_pvalue: 0.2940
  hosmer_lemeshow_result: PASS
  grade_results:
    - grade: A
      binomial_pvalue: 0.8258
      binomial_result: PASS
      jeffreys_pvalue: 0.7453
      jeffreys_result: PASS
    - grade: B
      binomial_pvalue: 0.6119
      binomial_result: PASS
      jeffreys_pvalue: 0.5445
      jeffreys_result: PASS
    - grade: C
      binomial_pvalue: 0.8537
      binomial_result: PASS
      jeffreys_pvalue: 0.8175
      jeffreys_result: PASS
    - grade: D
      binomial_pvalue: 0.1958
      binomial_result: PASS
      jeffreys_pvalue: 0.1674
      jeffreys_result: PASS
    - grade: E
      binomial_pvalue: 0.9315
      binomial_result: PASS
      jeffreys_pvalue: 0.9167
      jeffreys_result: PASS
    - grade: F
      binomial_pvalue: 0.0264
      binomial_result: FAIL
      jeffreys_pvalue: 0.0214
      jeffreys_result: FAIL
    - grade: G
      binomial_pvalue: 0.4400
      binomial_result: PASS
      jeffreys_pvalue: 0.4052
      jeffreys_result: PASS
    - grade: H
      binomial_pvalue: 0.8866
      binomial_result: PASS
      jeffreys_pvalue: 0.8666
      jeffreys_result: PASS
homogeneity:
  overall_pvalue: 0.3933
  overall_result: PASS
  grade_failures: None
heterogeneity:
  pvalue: 0.0000
  result: PASS
  dr_ordering_monotonic: true
psi: 0.0454
psi_result: PASS
concentration:
  hhi: 0.1250
cross_validation:
  cv_auc: 0.8065
  bootstrap_auc: 0.8026
regulatory_flags:
  - "Grade F binomial test FAIL (p=0.0264): observed DR 49.6% exceeds calibrated PD 40.6%. This grade shows the model underestimates risk for mid-range borrowers. The failure is isolated to a single grade and the Hosmer-Lemeshow overall calibration test passes (p=0.294)."
overall_assessment: PASS WITH FLAGS
