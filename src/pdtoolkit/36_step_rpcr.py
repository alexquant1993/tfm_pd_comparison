"""Stepwise RPC selection with trend restrictions.

This module provides stepwise logistic regression with risk profile
prioritization and additional trend consistency checks.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


@dataclass
class StepRPCrResult:
    """Result from stepwise RPC selection with restrictions.

    Attributes
    ----------
    model : statsmodels GLM result
        Final fitted logistic regression model.
    steps : pd.DataFrame
        Step-by-step summary of variable selection including group info.
    warnings : pd.DataFrame
        Warnings about risk factors (missing values, many modalities, etc.).
    dev_db : pd.DataFrame
        Development database used for modeling.
    """
    model: object
    steps: pd.DataFrame
    warnings: pd.DataFrame
    dev_db: pd.DataFrame


def step_rpcr(
    start_model: str,
    risk_profile: pd.DataFrame,
    db: pd.DataFrame,
    p_value: float = 0.05,
    check_start_model: bool = True,
    offset_vals: Optional[np.ndarray] = None
) -> StepRPCrResult:
    """
    Stepwise RPC logistic regression with trend restrictions.

    This function performs forward stepwise selection following risk profile
    priority groups, with additional checks that coefficient signs match
    observed correlation directions.

    Parameters
    ----------
    start_model : str
        Starting model formula (e.g., "target ~ 1" or "target ~ rf1 + rf2").
    risk_profile : pd.DataFrame
        DataFrame with columns 'rf' (risk factor names) and 'group' (priority group).
        Lower group numbers are processed first.
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
    StepRPCrResult
        Object containing:
        - model: Final logistic regression model
        - steps: Step-by-step selection summary with group info
        - warnings: Warnings about risk factors
        - dev_db: Development database

    Raises
    ------
    ValueError
        If inputs are invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> db = pd.DataFrame({
    ...     'target': np.random.binomial(1, 0.3, n),
    ...     'rf1': np.random.randn(n),
    ...     'rf2': np.random.choice(['A', 'B', 'C'], n),
    ...     'rf3': np.random.randn(n)
    ... })
    >>> risk_profile = pd.DataFrame({'rf': ['rf1', 'rf2', 'rf3'], 'group': [1, 1, 2]})
    >>> result = step_rpcr("target ~ 1", risk_profile, db)
    >>> print(result.steps)
    """
    # Validate arguments
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db is not a data frame.")

    if not isinstance(risk_profile, pd.DataFrame):
        raise ValueError("risk_profile is not a data frame.")

    if not all(col in risk_profile.columns for col in ['rf', 'group']):
        raise ValueError("risk_profile data frame has to contain columns: rf and group.")

    # Check risk factors exist in db
    missing_rf = [rf for rf in risk_profile['rf'] if rf not in db.columns]
    if missing_rf:
        raise ValueError(f"Following risk factors from risk_profile are missing in db: {', '.join(missing_rf)}.")

    if risk_profile['group'].isna().any():
        raise ValueError("Missing value(s) in risk_profile group.")

    if not isinstance(p_value, (int, float)) or not (0 < p_value < 1):
        raise ValueError("p_value has to be a numeric value between 0 and 1.")

    if not isinstance(check_start_model, bool):
        raise ValueError("check_start_model has to be logical (True or False).")

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

    # Get risk factors from risk profile
    rf_rest = list(risk_profile['rf'].unique())

    if len(rf_rest) == 0:
        raise ValueError("Risk factors are missing. Check risk_profile argument.")

    # Make a copy of db
    db = db.copy()

    # Initialize warnings table
    warn_tbl = []

    # Check modalities for categorical variables
    num_type = {c: pd.api.types.is_numeric_dtype(db[c]) for c in rf_rest if c in db.columns}
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

    # Process groups in order
    rf_mod = []
    groups = sorted(risk_profile['group'].unique())
    all_steps = []

    for g in groups:
        group_result = _group_summary_r(
            db=db,
            target=target,
            rp_tbl=risk_profile,
            g=g,
            rf_mod=rf_mod,
            rf_start=rf_start,
            rf_rest=rf_rest,
            p_value=p_value,
            rf_cat_o=rf_cat_o,
            rf_num_o=rf_num_o,
            check_start_model=check_start_model,
            offset_vals=offset_vals
        )

        all_steps.extend(group_result['steps'])
        rf_mod = group_result['rf_mod']

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
    if len(all_steps) > 0:
        steps_df = pd.DataFrame(all_steps)
    else:
        steps_df = pd.DataFrame(columns=['group', 'rf', 'aic', 'p_val', 'p_val_check', 'trend_check', 'selected'])

    # Create warnings DataFrame
    if len(warn_tbl) > 0:
        warn_df = pd.DataFrame(warn_tbl)
    else:
        warn_df = pd.DataFrame({'comment': ['There are no warnings.']})

    return StepRPCrResult(
        model=model,
        steps=steps_df,
        warnings=warn_df,
        dev_db=db
    )


def _group_summary_r(
    db: pd.DataFrame,
    target: str,
    rp_tbl: pd.DataFrame,
    g: int,
    rf_mod: List[str],
    rf_start: List[str],
    rf_rest: List[str],
    p_value: float,
    rf_cat_o: pd.DataFrame,
    rf_num_o: pd.DataFrame,
    check_start_model: bool,
    offset_vals: Optional[np.ndarray]
) -> Dict:
    """Process a single group in stepwise selection."""
    # Get risk factors for this group
    rf_g = rp_tbl[rp_tbl['group'] == g]['rf'].tolist()
    rf_g = [rf for rf in rf_g if rf in rf_rest]

    rf_current = rf_start + rf_mod
    steps = []

    if len(rf_g) == 0:
        return {'rf_mod': rf_current, 'steps': steps}

    # Initialize table for stepwise
    tbl_c = pd.DataFrame({'rf': rf_g, 'checked': False})
    iter_num = 1

    while True:
        # Get iteration summary
        it_s = _iter_summary_r(
            target=target,
            rf_mod=rf_current,
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
            rf_current.append(step_result['rf_next'])

        if step_result['step'] is not None:
            step_result['step']['group'] = g
            steps.append(step_result['step'])

        # Check stopping condition
        if len(tbl_c) == 0 or tbl_c['checked'].all():
            break

        iter_num += 1

    return {'rf_mod': rf_current, 'steps': steps}


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
        mask = ~(db[rf].isna() | np.isinf(db[rf].replace([np.inf, -np.inf], np.nan)))
        if mask.sum() > 2:
            cor_val = stats.spearmanr(db.loc[mask, rf], db.loc[mask, target])[0]
        else:
            cor_val = np.nan
        correlations.append({'rf': rf, 'cor': cor_val})

    return pd.DataFrame(correlations)


def _create_design_matrix(db: pd.DataFrame, variables: List[str]) -> np.ndarray:
    """Create design matrix with dummy encoding for categorical variables."""
    X_parts = [np.ones((len(db), 1))]

    for var in variables:
        if pd.api.types.is_numeric_dtype(db[var]):
            X_parts.append(db[var].values.reshape(-1, 1))
        else:
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
    results = []

    for _, row in tbl_c.iterrows():
        rf_l = row['rf']
        result = {'rf': rf_l, 'aic': np.nan, 'p_val': np.nan, 'p_val_check': False, 'trend_check': False}

        try:
            test_vars = rf_mod + [rf_l]
            X = _create_design_matrix(db, test_vars)
            y = db[target].values

            if offset_vals is not None:
                model = sm.GLM(y, X, family=sm.families.Binomial(), offset=db['offset_vals']).fit()
            else:
                model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

            result['aic'] = model.aic

            if np.any(np.isnan(model.params)):
                result['p_val_check'] = False
                result['trend_check'] = False
            else:
                coef_tbl = _get_coef_table(model, db, test_vars)

                checks = _ptc(
                    model=model,
                    coef_tbl=coef_tbl,
                    rf_mod=rf_mod,
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

    var_names = ['Intercept']
    for var in variables:
        if pd.api.types.is_numeric_dtype(db[var]):
            var_names.append(var)
        else:
            levels = db[var].unique()
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

            var_coefs = coef_tbl[coef_tbl['rf'].isin(var_mfs)].copy()

            if len(var_coefs) > 0:
                var_data = var_data.merge(var_coefs, left_on='mf', right_on='rf', how='left')

                trend_ok = _cc_cat(var_data['avg'].values, var_data['Estimate'].values)
                all_trends.append(trend_ok)

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

                if not np.isnan(var_cor) and not np.isnan(estimate):
                    trend_ok = np.sign(var_cor) == np.sign(estimate)
                else:
                    trend_ok = True

                all_trends.append(trend_ok)
                all_pvals.append(pval)

                if num_var == rf_l:
                    rf_l_pval = pval

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

    ref_idx = np.where(np.isnan(estimate))[0]
    if len(ref_idx) > 1:
        return False

    if len(ref_idx) == 0:
        return True

    ref_avg = avg[ref_idx[0]]
    ref_dir = avg - ref_avg

    if cc_cases.sum() > 0:
        check_1 = all(np.sign(ref_dir[cc_cases]) == np.sign(estimate[cc_cases]))
    else:
        check_1 = True

    if cc_cases.sum() > 1:
        cor_val = stats.spearmanr(estimate[cc_cases], ref_dir[cc_cases])[0]
        check_2 = round(cor_val, 5) == 1
    else:
        check_2 = True

    return check_1 and check_2


def _wald_test(model, coef_names: List[str], coef_tbl: pd.DataFrame) -> float:
    """Perform Wald test for a group of coefficients."""
    all_names = coef_tbl['rf'].tolist()
    indices = [all_names.index(name) for name in coef_names if name in all_names]

    if len(indices) == 0:
        return 1.0

    try:
        r_matrix = np.zeros((len(indices), len(model.params)))
        for i, idx in enumerate(indices):
            r_matrix[i, idx] = 1

        wald_result = model.wald_test(r_matrix, scalar=True)
        return float(wald_result.pvalue)
    except Exception:
        return 1.0


def _find_next(it_s: pd.DataFrame, tbl_c: pd.DataFrame) -> Dict:
    """Find next variable to add based on iteration summary."""
    candidates = it_s[(it_s['p_val_check']) & (it_s['trend_check'])].copy()

    if len(candidates) == 0:
        tbl_c = tbl_c.copy()
        tbl_c['checked'] = True
        return {'tbl_c': tbl_c, 'rf_next': None, 'step': None}

    # Check if all AICs are NaN
    if candidates['aic'].isna().all():
        tbl_c = tbl_c.copy()
        tbl_c['checked'] = True
        return {'tbl_c': tbl_c, 'rf_next': None, 'step': None}

    best_idx = candidates['aic'].idxmin()
    best_rf = candidates.loc[best_idx, 'rf']

    step = {
        'rf': best_rf,
        'aic': candidates.loc[best_idx, 'aic'],
        'p_val': candidates.loc[best_idx, 'p_val'],
        'p_val_check': candidates.loc[best_idx, 'p_val_check'],
        'trend_check': candidates.loc[best_idx, 'trend_check'],
        'selected': True
    }

    tbl_c = tbl_c[tbl_c['rf'] != best_rf].copy()

    return {'tbl_c': tbl_c, 'rf_next': best_rf, 'step': step}
