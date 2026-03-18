model_params_path: runs/2026-03-17_071354/pipeline/model_params.json
target_central_tendency: 0.3000
achieved_central_tendency: 0.3000
n_rating_grades: 8
rating_scale:
  - grade: A
    score_range: [567.6, 618.3]
    calibrated_pd: 0.0356
    n_obligors: 125
    pct_portfolio: 12.5
  - grade: B
    score_range: [551.5, 567.5]
    calibrated_pd: 0.0770
    n_obligors: 125
    pct_portfolio: 12.5
  - grade: C
    score_range: [536.8, 551.4]
    calibrated_pd: 0.1237
    n_obligors: 125
    pct_portfolio: 12.5
  - grade: D
    score_range: [520.3, 536.4]
    calibrated_pd: 0.1930
    n_obligors: 125
    pct_portfolio: 12.5
  - grade: E
    score_range: [505.3, 520.3]
    calibrated_pd: 0.2919
    n_obligors: 125
    pct_portfolio: 12.5
  - grade: F
    score_range: [491.6, 504.9]
    calibrated_pd: 0.4064
    n_obligors: 125
    pct_portfolio: 12.5
  - grade: G
    score_range: [473.0, 491.5]
    calibrated_pd: 0.5326
    n_obligors: 125
    pct_portfolio: 12.5
  - grade: H
    score_range: [413.8, 473.0]
    calibrated_pd: 0.7397
    n_obligors: 125
    pct_portfolio: 12.5
grade_pd_ordering_valid: true
stress_test:
  shift_1pct_ct: 0.3100
  shift_2pct_ct: 0.3200
calibration_method: scaling
scaling_factor: 1.0000
calibration_input: model_predicted_pd (NOT observed_dr)
normal_test_result: H0: ODR <= PD (p=0.5000)
hhi_concentration: 0.1250
score_range: [413.8, 618.3]
calibration_flags: None
