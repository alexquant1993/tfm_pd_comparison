"""Stepwise forward selection with trend restrictions.

This module provides stepwise logistic regression with additional
checks for coefficient trend consistency with observed correlations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


@dataclass
class StepFWDrResult:
    """Result from stepwise forward selection with restrictions.

    Attributes
    ----------
    model : statsmodels GLM result
        Final fitted logistic regression model.
    steps : pd.DataFrame
        Step-by-step summary of variable selection.
    warnings : pd.DataFrame
        Warnings about risk factors (missing values, many modalities, etc.).
    dev_db : pd.DataFrame
        Development database used for modeling.
    """
    model: object
    steps: pd.DataFrame
    warnings: pd.DataFrame
    dev_db: pd.DataFrame


def step_fwdr(
    start_model: str,
    db: pd.DataFrame,
    p_value: float = 0.05,
    check_start_model: bool = True,
    offset_vals: Optional[np.ndarray] = None
) -> StepFWDrResult:
    """
    Stepwise forward logistic regression with trend restrictions.

    This function performs forward stepwise selection of risk factors
    with additional checks that coefficient signs match the observed
    correlation direction with the target variable.

    Parameters
    ----------
    start_model : str
        Starting model formula (e.g., "target ~ 1" or "target ~ rf1 + rf2").
    db : pd.DataFrame
        Development database with target and risk factors.
    p_value : float, default 0.05
        Significance level for variable entry.
    check_start_model : bool, default True
        If True, also check trend consistency for starting model variables.
    offset_vals : array-like, optional
        Offset values for the model.

    Returns
    -------
    StepFWDrResult
        Object containing:
        - model: Final logistic regression model
        - steps: Step-by-step selection summary
        - warnings: Warnings about risk factors
        - dev_db: Development database

    Raises
    ------
    ValueError
        If db is not a DataFrame, p_value is invalid, or formula is incorrect.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> db = pd.DataFrame({
    ...     'target': np.random.binomial(1, 0.3, n),
    ...     'rf1': np.random.randn(n),
    ...     'rf2': np.random.choice(['A', 'B', 'C'], n)
    ... })
    >>> result = step_fwdr("target ~ 1", db)
    >>> print(result.steps)
    """
    # Validate arguments
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db is not a data frame.")

    if not isinstance(p_value, (int, float)) or not (0 < p_value < 1):
        raise ValueError("p_value has to be a numeric value between 0 and 1.")

    # Parse formula
    if '~' not in start_model:
        raise ValueError("Formula for start_model not specified correctly.")

    parts = start_model.split('~')
    target = parts[0].strip()

    if len(parts) > 1 and parts[1].strip() != '1':
        rf_start = [r.strip() for r in parts[1].split('+') if r.strip() != '1']
    else:
        rf_start = []

    # Validate target and starting variables
    if target not in db.columns:
        raise ValueError("Formula for start_model not specified correctly.")

    for rf in rf_start:
        if rf not in db.columns:
            raise ValueError("Formula for start_model not specified correctly.")

    # Get remaining risk factors
    rf_rest = [c for c in db.columns if c not in [target] + rf_start]

    if len(rf_rest) == 0:
        raise ValueError("Risk factors are missing. Check db argument.")

    # Make a copy of db
    db = db.copy()

    # Initialize warnings table
    warn_tbl = []

    # Check modalities for categorical variables
    num_type = {c: pd.api.types.is_numeric_dtype(db[c]) for c in rf_rest}
    num_rf = [c for c, is_num in num_type.items() if is_num]
    cat_rf = [c for c, is_num in num_type.items() if not is_num]

    for rf in cat_rf:
        n_unique = db[rf].nunique()
        if n_unique > 10:
            warn_tbl.append({'rf': rf, 'comment': 'More than 10 modalities.'})

    # Check for missing values
    for rf in cat_rf:
        if db[rf].isna().sum() > 0:
            warn_tbl.append({'rf': rf, 'comment': 'Contains missing values.'})

    for rf in num_rf:
        special_mask = db[rf].isna() | np.isinf(db[rf].replace([np.inf, -np.inf], np.nan))
        if special_mask.sum() > 0:
            warn_tbl.append({'rf': rf, 'comment': 'Contains special cases (NA, NaN, Inf, -Inf).'})

    # Generate summary tables for categorical risk factors
    rf_all = rf_start + rf_rest
    rf_cat_o = _generate_cat_summary(db, target, rf_all, num_rf)

    # Calculate observed correlations for numeric risk factors
    rf_num_o = _calculate_numeric_correlations(db, target, num_rf)

    # Check percentage per bin
    if len(rf_cat_o) > 0:
        check_pct = rf_cat_o[rf_cat_o['pct_check']]['rf'].unique()
        for rf in check_pct:
            warn_tbl.append({'rf': rf, 'comment': 'At least one pct per bin less than 5%.'})

    # Add mf column for categorical variables
    if len(rf_cat_o) > 0:
        rf_cat_o['mf'] = rf_cat_o['rf'].astype(str) + rf_cat_o['bin'].astype(str)

    # Add offset if provided
    if offset_vals is not None:
        db['offset_vals'] = offset_vals

    # Initialize stepwise selection
    rf_mod = []
    tbl_c = pd.DataFrame({'rf': rf_rest, 'checked': False})
    steps = []
    iter_num = 1

    while True:
        # Get iteration summary
        it_s = _iter_summary_r(
            target=target,
            rf_mod=rf_mod,
            rf_start=rf_start,
            tbl_c=tbl_c,
            p_value=p_value,
            rf_cat_o=rf_cat_o,
            rf_num_o=rf_num_o,
            db=db,
            offset_vals=offset_vals,
            check_start_model=check_start_model
        )

        # Find next variable
        step_result = _find_next(it_s, tbl_c)
        tbl_c = step_result['tbl_c']

        if step_result['rf_next'] is not None:
            rf_mod.append(step_result['rf_next'])

        if step_result['step'] is not None:
            steps.append(step_result['step'])

        # Check stopping condition
        if len(tbl_c) == 0 or tbl_c['checked'].all():
            break

        iter_num += 1

    # Build final model
    if len(rf_mod) == 0:
        formula_vars = '1'
    else:
        formula_vars = ' + '.join(rf_start + rf_mod)

    # Create design matrix
    y = db[target].values
    X_vars = rf_start + rf_mod

    if len(X_vars) == 0:
        X = np.ones((len(db), 1))
    else:
        X = _create_design_matrix(db, X_vars)

    # Fit final model
    if offset_vals is not None:
        model = sm.GLM(y, X, family=sm.families.Binomial(), offset=db['offset_vals']).fit()
    else:
        model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    # Create steps DataFrame
    if len(steps) > 0:
        steps_df = pd.DataFrame(steps)
    else:
        steps_df = pd.DataFrame(columns=['rf', 'aic', 'p_val', 'p_val_check', 'trend_check', 'selected'])

    # Create warnings DataFrame
    if len(warn_tbl) > 0:
        warn_df = pd.DataFrame(warn_tbl)
    else:
        warn_df = pd.DataFrame({'comment': ['There are no warnings.']})

    return StepFWDrResult(
        model=model,
        steps=steps_df,
        warnings=warn_df,
        dev_db=db
    )


def _generate_cat_summary(db: pd.DataFrame, target: str, rf_all: List[str], num_rf: List[str]) -> pd.DataFrame:
    """Generate summary table for categorical risk factors."""
    cat_rf = [rf for rf in rf_all if rf not in num_rf and rf in db.columns]

    if len(cat_rf) == 0:
        return pd.DataFrame()

    summaries = []
    for rf in cat_rf:
        summary = db.groupby(rf).agg(
            no=(target, 'count'),
            avg=(target, 'mean')
        ).reset_index()
        summary.columns = ['bin', 'no', 'avg']
        summary['pct_o'] = summary['no'] / summary['no'].sum()
        summary['rf'] = rf
        summary['bin'] = summary['bin'].astype(str)
        summary['pct_check'] = (summary['pct_o'] < 0.05).any()
        summaries.append(summary)

    return pd.concat(summaries, ignore_index=True)


def _calculate_numeric_correlations(db: pd.DataFrame, target: str, num_rf: List[str]) -> pd.DataFrame:
    """Calculate observed correlations for numeric risk factors."""
    if len(num_rf) == 0:
        return pd.DataFrame()

    correlations = []
    for rf in num_rf:
        # Exclude special values
        mask = ~(db[rf].isna() | np.isinf(db[rf].replace([np.inf, -np.inf], np.nan)))
        if mask.sum() > 2:
            cor_val = stats.spearmanr(db.loc[mask, rf], db.loc[mask, target])[0]
        else:
            cor_val = np.nan
        correlations.append({'rf': rf, 'cor': cor_val})

    return pd.DataFrame(correlations)


def _create_design_matrix(db: pd.DataFrame, variables: List[str]) -> np.ndarray:
    """Create design matrix with dummy encoding for categorical variables."""
    X_parts = [np.ones((len(db), 1))]  # Intercept

    for var in variables:
        if pd.api.types.is_numeric_dtype(db[var]):
            X_parts.append(db[var].values.reshape(-1, 1))
        else:
            # Dummy encoding (drop first level)
            dummies = pd.get_dummies(db[var], prefix=var, drop_first=True)
            X_parts.append(dummies.values)

    return np.hstack(X_parts)


def _iter_summary_r(
    target: str,
    rf_mod: List[str],
    rf_start: List[str],
    tbl_c: pd.DataFrame,
    p_value: float,
    rf_cat_o: pd.DataFrame,
    rf_num_o: pd.DataFrame,
    db: pd.DataFrame,
    offset_vals: Optional[np.ndarray],
    check_start_model: bool
) -> pd.DataFrame:
    """Get iteration summary for stepwise selection."""
    rf_current = rf_start + rf_mod
    results = []

    for _, row in tbl_c.iterrows():
        rf_l = row['rf']
        result = {'rf': rf_l, 'aic': np.nan, 'p_val': np.nan, 'p_val_check': False, 'trend_check': False}

        try:
            # Build model with this variable
            test_vars = rf_current + [rf_l]
            X = _create_design_matrix(db, test_vars)
            y = db[target].values

            if offset_vals is not None:
                model = sm.GLM(y, X, family=sm.families.Binomial(), offset=db['offset_vals']).fit()
            else:
                model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

            result['aic'] = model.aic

            # Check for NA coefficients
            if np.any(np.isnan(model.params)):
                result['p_val_check'] = False
                result['trend_check'] = False
            else:
                # Get coefficient table
                coef_tbl = _get_coef_table(model, db, test_vars)

                # Check p-values and trends
                checks = _ptc(
                    model=model,
                    coef_tbl=coef_tbl,
                    rf_mod=rf_current,
                    rf_start=rf_start,
                    check_start_model=check_start_model,
                    rf_l=rf_l,
                    rf_type='numeric' if pd.api.types.is_numeric_dtype(db[rf_l]) else 'categorical',
                    rf_cat_o=rf_cat_o,
                    rf_num_o=rf_num_o,
                    p_value=p_value
                )

                result['p_val'] = checks['p_val']
                result['p_val_check'] = checks['p_val_check']
                result['trend_check'] = checks['trend_check']

        except Exception:
            pass

        results.append(result)

    return pd.DataFrame(results)


def _get_coef_table(model, db: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """Extract coefficient table from model."""
    params = model.params
    pvalues = model.pvalues

    # Build variable names
    var_names = ['Intercept']
    for var in variables:
        if pd.api.types.is_numeric_dtype(db[var]):
            var_names.append(var)
        else:
            levels = db[var].unique()
            # Drop first level (reference)
            for level in sorted(levels)[1:]:
                var_names.append(f"{var}{level}")

    coef_tbl = pd.DataFrame({
        'rf': var_names[:len(params)],
        'Estimate': params,
        'Pr...z..': pvalues
    })

    return coef_tbl


def _ptc(
    model,
    coef_tbl: pd.DataFrame,
    rf_mod: List[str],
    rf_start: List[str],
    check_start_model: bool,
    rf_l: str,
    rf_type: str,
    rf_cat_o: pd.DataFrame,
    rf_num_o: pd.DataFrame,
    p_value: float
) -> Dict:
    """Check p-values and trend consistency."""
    rf_check = rf_mod + [rf_l]
    if not check_start_model:
        rf_check = [rf for rf in rf_check if rf not in rf_start]

    all_pvals = []
    all_trends = []
    rf_l_pval = np.nan

    # Check categorical variables
    if len(rf_cat_o) > 0:
        cat_vars = [rf for rf in rf_check if rf in rf_cat_o['rf'].values]

        for cat_var in cat_vars:
            var_data = rf_cat_o[rf_cat_o['rf'] == cat_var].copy()
            var_mfs = var_data['mf'].values

            # Get coefficients for this variable
            var_coefs = coef_tbl[coef_tbl['rf'].isin(var_mfs)].copy()

            if len(var_coefs) > 0:
                # Merge with observed averages
                var_data = var_data.merge(var_coefs, left_on='mf', right_on='rf', how='left')

                # Check trend consistency
                trend_ok = _cc_cat(var_data['avg'].values, var_data['Estimate'].values)
                all_trends.append(trend_ok)

                # Calculate Wald test p-value
                wald_pval = _wald_test(model, var_mfs, coef_tbl)
                all_pvals.append(wald_pval)

                if cat_var == rf_l:
                    rf_l_pval = wald_pval

    # Check numeric variables
    if len(rf_num_o) > 0:
        num_vars = [rf for rf in rf_check if rf in rf_num_o['rf'].values]

        for num_var in num_vars:
            var_cor = rf_num_o[rf_num_o['rf'] == num_var]['cor'].values[0]
            var_coef = coef_tbl[coef_tbl['rf'] == num_var]

            if len(var_coef) > 0:
                estimate = var_coef['Estimate'].values[0]
                pval = var_coef['Pr...z..'].values[0]

                # Check sign consistency
                if not np.isnan(var_cor) and not np.isnan(estimate):
                    trend_ok = np.sign(var_cor) == np.sign(estimate)
                else:
                    trend_ok = True

                all_trends.append(trend_ok)
                all_pvals.append(pval)

                if num_var == rf_l:
                    rf_l_pval = pval

    # Combine results
    p_val_check = all(p < p_value for p in all_pvals) if all_pvals else False
    trend_check = all(all_trends) if all_trends else True

    return {
        'p_val': rf_l_pval,
        'p_val_check': p_val_check,
        'trend_check': trend_check
    }


def _cc_cat(avg: np.ndarray, estimate: np.ndarray) -> bool:
    """Check coefficient consistency for categorical variables."""
    cc_cases = ~(np.isnan(avg) | np.isnan(estimate))

    # Find reference level (NA estimate)
    ref_idx = np.where(np.isnan(estimate))[0]
    if len(ref_idx) > 1:
        return False

    if len(ref_idx) == 0:
        # No reference found, assume first is reference
        return True

    ref_avg = avg[ref_idx[0]]
    ref_dir = avg - ref_avg

    # Check sign consistency
    if cc_cases.sum() > 0:
        check_1 = all(np.sign(ref_dir[cc_cases]) == np.sign(estimate[cc_cases]))
    else:
        check_1 = True

    # Check correlation
    if cc_cases.sum() > 1:
        cor_val = stats.spearmanr(estimate[cc_cases], ref_dir[cc_cases])[0]
        check_2 = round(cor_val, 5) == 1
    else:
        check_2 = True

    return check_1 and check_2


def _wald_test(model, coef_names: List[str], coef_tbl: pd.DataFrame) -> float:
    """Perform Wald test for a group of coefficients."""
    # Get indices of coefficients
    all_names = coef_tbl['rf'].tolist()
    indices = [all_names.index(name) for name in coef_names if name in all_names]

    if len(indices) == 0:
        return 1.0

    try:
        # Create contrast matrix
        r_matrix = np.zeros((len(indices), len(model.params)))
        for i, idx in enumerate(indices):
            r_matrix[i, idx] = 1

        wald_result = model.wald_test(r_matrix, scalar=True)
        return float(wald_result.pvalue)
    except Exception:
        return 1.0


def _find_next(it_s: pd.DataFrame, tbl_c: pd.DataFrame) -> Dict:
    """Find next variable to add based on iteration summary."""
    # Filter to variables that pass both checks
    candidates = it_s[(it_s['p_val_check']) & (it_s['trend_check'])].copy()

    if len(candidates) == 0:
        # Mark all as checked
        tbl_c = tbl_c.copy()
        tbl_c['checked'] = True
        return {'tbl_c': tbl_c, 'rf_next': None, 'step': None}

    # Select variable with lowest AIC
    best_idx = candidates['aic'].idxmin()
    best_rf = candidates.loc[best_idx, 'rf']

    # Create step record
    step = {
        'rf': best_rf,
        'aic': candidates.loc[best_idx, 'aic'],
        'p_val': candidates.loc[best_idx, 'p_val'],
        'p_val_check': candidates.loc[best_idx, 'p_val_check'],
        'trend_check': candidates.loc[best_idx, 'trend_check'],
        'selected': True
    }

    # Update tbl_c
    tbl_c = tbl_c[tbl_c['rf'] != best_rf].copy()

    return {'tbl_c': tbl_c, 'rf_next': best_rf, 'step': step}
