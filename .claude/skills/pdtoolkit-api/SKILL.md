# pdtoolkit API Reference

Use ONLY these functions from pdtoolkit. Do not hallucinate function names or parameters.

```python
import pdtoolkit as pdt
```

---

## Constants

| Constant | Value |
|---|---|
| `DEFAULT_SPECIAL_CASES` | `[None, np.nan, np.inf, -np.inf]` |
| `DEFAULT_SC_THRESHOLD` | `0.2` |
| `DEFAULT_MIN_PCT_OBS` | `0.05` |
| `DEFAULT_MIN_AVG_RATE` | `0.01` |
| `DEFAULT_P_VALUE` | `0.05` |
| `FLOAT_TOLERANCE` | `1e-8` |
| `DEFAULT_SC_REPLACEMENT` | `"SC"` |

---

## Univariate Analysis

### `univariate(db, sc=None, sc_method="together", sc_threshold=0.2) -> DataFrame`
Perform univariate analysis on risk factors. Generates distribution statistics for numeric variables and frequency metrics for categorical variables.
- `sc`: list of special case values (default: None)
- `sc_method`: how to handle special cases -- "together" or "separate"
- Returns empty rows for binary variables

### `imp_sc(db, sc_all=None, sc_replace=None, method_num="automatic", p_val=0.05) -> (DataFrame, DataFrame)`
Impute values for special cases (missing, inf). Replaces using mean, median, zero, or automatic selection based on normality test.
- `sc_all`: list of values considered special cases
- `sc_replace`: list of replacement values
- `method_num`: "automatic" (selects mean/median via normality test), "mean", "median", "zero"
- Returns: (imputed DataFrame, report DataFrame)
- Does not handle mixed-type columns

### `imp_outliers(db, sc=None, method="iqr", range_val=1.5, upper_pct=0.95, lower_pct=0.05) -> (DataFrame, DataFrame)`
Replace outliers with less extreme values (capping).
- `method`: "iqr" or "percentile"
- `range_val`: IQR multiplier (default 1.5)
- `upper_pct`/`lower_pct`: percentile bounds when method="percentile"
- Returns: (capped DataFrame, report DataFrame)
- IQR method may over-impute in skewed distributions

---

## Bivariate Analysis

### `bivariate(db, target) -> (DataFrame, DataFrame)`
Perform bivariate analysis on categorical risk factors. Calculates WoE and IV.
- `db`: DataFrame with all categorical columns (must be string type)
- `target`: name of binary target column
- Returns: (summary DataFrame, info DataFrame)
- **Requires all variables to be string type**

### `woe_tbl(tbl, x, y, y_check=True) -> DataFrame`
Calculate Weight of Evidence (WoE) table for a risk factor.
- `tbl`: DataFrame containing the data
- `x`: risk factor column name
- `y`: target column name
- Returns None if a bin has zero events

### `auc_model(predictions, observed) -> float`
Calculate Area Under the ROC Curve (AUC).
- `predictions`: predicted probabilities (ndarray)
- `observed`: actual binary outcomes (ndarray)

### `replace_woe(db, target) -> (DataFrame, DataFrame)`
Replace risk factor modalities with Weight of Evidence (WoE) values.
- Returns: (WoE-encoded DataFrame, mapping DataFrame)

---

## Binning

### `cat_bin(x, y, sc=None, sc_merge="none", min_pct_obs=0.05, min_avg_rate=0.01, max_groups=None, force_trend="modalities") -> (DataFrame, Series)`
Categorical risk factor binning with three-stage procedure (correction for min observations, min bad rate, and adjacent pooling).
- `sc`: special case values
- `sc_merge`: how to merge special cases -- "none", "closest", "first", "last"
- `min_pct_obs`: minimum percentage of observations per bin (default 5%)
- `min_avg_rate`: minimum average event rate per bin
- `max_groups`: maximum number of bins (5 is a safe default)
- `force_trend`: "modalities" or "monotonic"
- Returns: (summary DataFrame, binned Series)

### `rf_clustering(db, metric, k=None) -> DataFrame`
Risk factor clustering using correlation-based distance.
- `metric`: distance metric for clustering
- `k`: number of clusters (None = automatic)

### `nzv(db, sc=None) -> DataFrame`
Detect near-zero variance risk factors.
- `sc`: special case values to exclude from variance calculation

---

## Stepwise Selection

### `step_miv(start_model, miv_threshold, m_ch_p_val, coding, db, coding_start_model=False, offset_vals=None) -> StepMIVResult`
Stepwise logistic regression based on Marginal Information Value (MIV).
- `start_model`: formula string (e.g., "target ~ 1" or "target ~ rf1 + rf2")
- `miv_threshold`: MIV entrance threshold for candidate risk factors
- `m_ch_p_val`: significance level for marginal chi-square test
- `coding`: risk factor coding -- "WoE" or "dummy"
- `coding_start_model`: whether to WoE-code risk factors from starting model
- Returns: StepMIVResult with `.model`, `.steps`, `.iterations`, `.warnings`, `.dev_db`
- Returns empty if no variable clears threshold

### `step_fwd(start_model, p_value=0.05, coding="WoE", db=None, risk_factors=None, target=None, offset_vals=None) -> StepFWDResult`
Forward stepwise logistic regression.
- `risk_factors`: list of candidate variable names
- `target`: target variable name

### `step_rpc(start_model, risk_profile, p_value=0.05, coding="WoE", coding_start_model=True, check_start_model=True, db=None, offset_vals=None) -> StepRPCResult`
Stepwise regression with risk profile check.
- `risk_profile`: DataFrame defining expected risk factor directions

### `step_fwdr(start_model, db, p_value=0.05, check_start_model=True, offset_vals=None) -> StepFWDrResult`
Forward stepwise logistic regression with trend restrictions. Ensures coefficient signs match the observed correlation direction.

### `step_rpcr(start_model, risk_profile, db, p_value=0.05, check_start_model=True, offset_vals=None) -> StepRPCrResult`
Stepwise RPC logistic regression with trend restrictions.

---

## Block Methods

### `staged_blocks(method, target, db, coding="WoE", blocks=None, p_value=0.05, miv_threshold=0.02, m_ch_p_val=0.05) -> StagedBlocksResult`
Staged block regression. Processes blocks of risk factors sequentially.
- `method`: selection method within blocks ("step_miv" or "step_fwd")
- `blocks`: DataFrame mapping risk factors to block assignments

### `embedded_blocks(method, target, db, coding="WoE", blocks=None, p_value=0.05, miv_threshold=0.02, m_ch_p_val=0.05) -> EmbeddedBlocksResult`
Embedded block regression. Same parameters as staged_blocks.

### `ensemble_blocks(method, target, db, coding="WoE", blocks=None, p_value=0.05, miv_threshold=0.02, m_ch_p_val=0.05) -> EnsembleBlocksResult`
Ensemble block regression. Same parameters as staged_blocks.

---

## Constrained Regression

### `constrained_logit(db, x, y, lower, upper) -> ConstrainedLogitResult`
Fit constrained logistic regression using L-BFGS-B optimization.
- `x`: list of predictor column names
- `y`: target column name
- `lower`/`upper`: coefficient bounds (list or ndarray)

---

## Scoring

### `scaled_score(probs, score=600, odd=50, pdo=20) -> ndarray`
Scale probabilities to credit scores.
- `probs`: probability array (**must be probabilities, not log-odds**)
- `score`: base score (default 600)
- `odd`: base odds (default 50:1)
- `pdo`: points to double odds (default 20)

### `score_to_prob(scores, score=600, odd=50, pdo=20) -> ndarray`
Convert scaled scores back to probabilities. Inverse of `scaled_score`.

---

## Calibration

### `rs_calibration(rs, dr, w, ct, min_pd=0.0003, method="scaling") -> CalibrationResult`
Calibrate observed default rates across rating scale.
- `rs`: DataFrame with rating scale data
- `dr`: column name for observed default rate
- `w`: column name for weights (number of obligors)
- `ct`: target central tendency
- `min_pd`: minimum PD floor (default 0.03%)
- `method`: "scaling", "log_odds_a", or "log_odds_ab"
- Sensitive to initial PD scale

---

## Validation

### `kfold_vld(model, db, target, predictors, k=10, seed=1984) -> ValidationResult`
K-fold cross-validation for logistic regression model.

### `boots_vld(model, db, target, predictors, B=1000, seed=1122) -> ValidationResult`
Bootstrap model validation.

### `segment_vld(model, db, target, predictors, min_leaf=0.03, alpha=0.05) -> SegmentValidationResult`
Model segment validation based on residuals. Identifies segments where model underperforms.

---

## Testing

### `dp_testing(app_port, def_ind, pdc, auc_test, alternative="two.sided", alpha=0.05) -> DataFrame`
Test discriminatory power (AUC significance).
- `app_port`: application portfolio DataFrame
- `def_ind`: default indicator column name
- `pdc`: calibrated PD column name
- `auc_test`: reference AUC value to test against
- `alternative`: "less", "greater", or "two.sided"

### `pp_testing(rating_label, pdc, no, nb, alpha=0.05) -> DataFrame`
Test predictive power using binomial, Jeffreys, z-score, and Hosmer-Lemeshow tests.
- `rating_label`: rating grade labels (list/array)
- `pdc`: calibrated PDs per grade (list/array)
- `no`: number of obligors per grade (list/array)
- `nb`: number of defaults per grade (list/array)

### `power(rating_label, pdc, no, nb, alpha=0.05, sim_num=1000, seed=2211) -> PowerResult`
Monte Carlo simulation to estimate power of PP statistical tests.

### `homogeneity(app_port, def_ind, rating, segment, segment_num=4, alpha=0.05) -> DataFrame`
Test homogeneity of PD rating model. Tests whether default rates differ significantly between segment modalities within each rating.
- `segment`: segmentation variable column name
- `segment_num`: number of segments to create (default 4)

### `heterogeneity(app_port, def_ind, rating, alpha=0.05) -> DataFrame`
Test heterogeneity -- verifies that default rates follow expected order (higher risk ratings have higher default rates).

---

## Stability

### `psi(base, target, bins=10, alpha=0.05) -> PSIResult`
Calculate Population Stability Index (PSI). Measures distribution shift between base (development) and target (validation) samples.
- PSI > 0.25: significant shift, trigger model review

### `normal_test(pdc, odr, alpha=0.05) -> NormalTestResult`
Normal test for PD calibration validation.
- `pdc`: calibrated PDs
- `odr`: observed default rates

---

## Interactions

### `interaction_transformer(db, target, risk_factors, min_pct_obs=0.05, min_avg_rate=0.01, max_depth=3, seed=42) -> InteractionResult`
Extract interactions between risk factors using decision trees.

### `rf_interaction_transformer(db, rf, target, num_rf=None, num_tree=10, min_pct_obs=0.05, min_avg_rate=0.01, max_depth=2, create_interaction_rf=True, seed=991) -> RFInteractionResult`
Extract interactions using random forest of decision trees.
- `rf`: list of risk factor column names
- `num_rf`: number of features per tree (None = auto)

---

## Other Functions

### `create_partitions(db) -> PartitionsResult`
Create partitions (nested dummy variables). Useful for logistic regression to show log-odds differences between adjacent bins.

### `evrs(db, def_ind, pd_est, pd_bm, lgd=0.45, rf=0.02, elasticity=5.0, prob_threshold=0.5, sim_num=500, seed=991) -> EVRSResult`
Economic Value of Rating System. Simulates portfolio returns under measurement error scenarios.

### `hhi(x) -> float`
Calculate the Herfindahl-Hirschman Index (concentration measure).

### `confusion_matrix(predictions, observed, cutoff) -> ConfusionMatrixResult`
Compute confusion matrix and classification metrics at a given cutoff.

### `ush_test(x, y, p_value=0.05, g=20, sc=None) -> UShapeTestResult`
Test for U-shaped relationship between risk factor and target.

### `kfold_idx(target, k=10, type="random", seed=2191) -> dict[str, FoldIndices]`
Generate k-fold cross-validation indices.
- `type`: "random" or "stratified"

### `fairness_vld(db, sensitive, obs_outcome, mod_outcome, conditional=None, mod_outcome_type="disc", p_value=0.05) -> FairnessResult`
Validate model fairness across sensitive attributes.

### `decision_tree(db, target, risk_factors, min_pct_obs=0.05, min_avg_rate=0.01, p_value=0.05, max_depth=3, monotonicity=None) -> DecisionTreeResult`
Fit a customized decision tree for PD modeling.
- `monotonicity`: dict mapping variable names to "increasing" or "decreasing"

### `smote(db, target, minority_class, osr, ordinal_rf=None, num_rf_const=None, k=5, seed=81000) -> DataFrame`
SMOTE oversampling for imbalanced datasets.
- `osr`: oversampling ratio
- `ordinal_rf`: list of ordinal risk factor names
- `num_rf_const`: DataFrame with numeric RF constraints

---

## Helpers

### `num_slice(x, mapping, sc=None, sc_r="SC") -> ndarray`
Slice numeric variable into bins based on mapping from `cat_bin`.

### `cat_slice(x, mapping, sc=None, sc_r="SC") -> ndarray`
Slice categorical variable based on mapping from `cat_bin`.

### `encode_woe(x, mapping) -> ndarray`
Encode categorical variable with WoE values based on mapping.

---

## Data

### `load_loans(n=3000, seed=2191) -> DataFrame`
Generate a synthetic loans dataset for credit risk modeling.

### `get_loans_description() -> str`
Get description of the loans dataset variables.
