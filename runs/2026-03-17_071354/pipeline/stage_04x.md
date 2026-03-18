models_compared: 3
models_available: [miv, xgboost_importance, forward_stepwise]

### Model Comparison

| Metric | MIV | XGBoost | Forward Stepwise |
|---|---|---|---|
| Selected Variables | 8 | 7 | 8 |
| AUC | 0.8109 | 0.807 | 0.8109 |
| Gini | 0.6218 | 0.614 | 0.6218 |
| KS | 0.5067 | 0.4886 | 0.5067 |
| Pseudo R-squared | 0.2222 | 0.2184 | 0.2222 |
| CV AUC Gap | 0.0044 | 0.0049 | 0.0044 |
| Signs Consistent | yes | yes | yes |
| Decile Monotonic | yes | yes | yes |
| **Composite Score** | **0.9207** | **0.4298** | **0.9207** |

### Variable Selection Overlap

```
variables_selected_by_all_three: [Account Balance, Payment Status of Previous Credit, Duration of Credit (month), Value Savings/Stocks, Purpose, Most valuable available asset, Age (years)]
variables_unique_to_miv: []
variables_unique_to_xgboost: []
variables_unique_to_forward: []
variables_in_miv_and_forward_only: [Credit Amount]
overlap_ratio: 0.875
```

### Champion Selection

```
champion: miv
champion_composite_score: 0.9207
runner_up: forward_stepwise
runner_up_composite_score: 0.9207
champion_rationale: MIV and Forward Stepwise produced identical models (same 8 variables, identical coefficients). MIV is selected as champion by convention (first among equal-scoring methods). Both significantly outperform the XGBoost importance method, which excluded Credit Amount and scored lower on all discrimination metrics (AUC 0.807 vs 0.8109, Gini 0.614 vs 0.6218, KS 0.4886 vs 0.5067).
selection_mode: automatic
model_params_path: runs/2026-03-17_071354/pipeline/model_params.json
model_flags: MIV and Forward Stepwise produced identical models — this is expected when both selection thresholds admit the same variable set from the shortlist, but is noted for the audit trail.
```
