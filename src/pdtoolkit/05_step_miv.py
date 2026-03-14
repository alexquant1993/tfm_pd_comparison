"""
Stepwise logistic regression based on Marginal Information Value (MIV).

This module implements stepwise logistic regression using MIV as the selection
criterion for adding risk factors to the model.

Ported from: 05_STEP_MIV.R

References:
    Scallan, G. (2011). Class(ic) Scorecards: Selecting Characteristics and
    Attributes in Logistic Regression, Edinburgh Credit Scoring Conference.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats
from importlib import import_module

# Import from numbered modules
_bivariate = import_module('.02_bivariate_analysis', package='pdtoolkit')
woe_tbl = _bivariate.woe_tbl
replace_woe = _bivariate.replace_woe


@dataclass
class StepMIVResult:
    """Result container for stepMIV function."""
    model: Any  # statsmodels GLM result
    steps: pd.DataFrame
    miv_iter: pd.DataFrame
    warnings: pd.DataFrame
    dev_db: pd.DataFrame


def step_miv(
    start_model: str,
    miv_threshold: float,
    m_ch_p_val: float,
    coding: str,
    db: pd.DataFrame,
    coding_start_model: bool = False,
    offset_vals: Optional[np.ndarray] = None,
) -> StepMIVResult:
    """
    Perform stepwise logistic regression based on Marginal Information Value (MIV).

    Args:
        start_model: Formula string for starting model (e.g., "target ~ 1" or "target ~ rf1 + rf2").
        miv_threshold: MIV entrance threshold for candidate risk factors.
        m_ch_p_val: Significance level for marginal chi-square test.
        coding: Risk factor coding type - "WoE" or "dummy".
        db: Modeling DataFrame with risk factors and target variable.
        coding_start_model: Whether to WoE code risk factors from starting model.
        offset_vals: Optional offset values for the linear predictor.

    Returns:
        StepMIVResult with model, steps, iteration details, warnings, and dev database.

    Examples:
        >>> import pandas as pd
        >>> from pdtoolkit.05_step_miv import step_miv
        >>> # Assuming df has target 'default' and categorical risk factors
        >>> result = step_miv(
        ...     start_model="default ~ 1",
        ...     miv_threshold=0.02,
        ...     m_ch_p_val=0.05,
        ...     coding="WoE",
        ...     db=df
        ... )
        >>> print(result.steps)
    """
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db is not a DataFrame.")

    coding_opt = ["WoE", "dummy"]
    if coding not in coding_opt:
        raise ValueError(f"coding argument has to be one of: {', '.join(coding_opt)}.")

    if not isinstance(miv_threshold, (int, float)) or not isinstance(m_ch_p_val, (int, float)):
        raise ValueError("miv_threshold and m_ch_p_val have to be numeric.")

    if not isinstance(coding_start_model, bool):
        raise ValueError("coding_start_model has to be boolean.")

    # Parse formula
    start_vars = _parse_formula(start_model)
    target = start_vars['target']
    rf_start = start_vars.get('predictors', [])

    # Validate columns exist
    all_vars = [target] + rf_start
    missing = [v for v in all_vars if v not in db.columns]
    if missing or target is None:
        raise ValueError(
            "Formula for start_model not specified correctly. "
            "Check column names."
        )

    # Validate target is 0/1
    y = db[target]
    y_valid = y.dropna()
    if not set(y_valid.unique()).issubset({0, 1}):
        raise ValueError("Target is not 0/1 variable.")

    # Get remaining risk factors
    rf_rest = [col for col in db.columns if col not in [target] + rf_start]
    if len(rf_rest) == 0:
        raise ValueError("Risk factors are missing. Check db argument.")

    db = db.copy()
    warn_tbl = []

    # Check number of modalities per risk factor
    for rf in rf_rest:
        n_unique = db[rf].nunique()
        if n_unique > 10:
            warn_tbl.append({"rf": rf, "comment": "More than 10 modalities."})

    # Check for numeric risk factors
    numeric_rfs = []
    for rf in rf_rest:
        if pd.api.types.is_numeric_dtype(db[rf]) and not pd.api.types.is_bool_dtype(db[rf]):
            if not pd.api.types.is_object_dtype(db[rf]):
                numeric_rfs.append(rf)
                warn_tbl.append({
                    "rf": rf,
                    "comment": "Numeric type. Risk factor is excluded from further process."
                })

    rf_rest = [rf for rf in rf_rest if rf not in numeric_rfs]

    # Generate WoE tables for all risk factors
    rf_woe_o = {}
    pct_check_rfs = []
    woe_check_rfs = []

    for rf in rf_rest:
        woe_o = woe_tbl(db, x=rf, y=target, y_check=False)
        woe_o['rf'] = rf

        # Check percentage of observations per bin
        pct_check = (woe_o['pct_o'] < 0.05).any()
        woe_o['pct_check'] = pct_check
        if pct_check:
            pct_check_rfs.append(rf)

        # Check WoE calculation issues
        woe_check = woe_o['woe'].isna().any() | np.isinf(woe_o['woe']).any()
        woe_o['woe_check'] = woe_check
        if woe_check:
            woe_check_rfs.append(rf)

        rf_woe_o[rf] = woe_o

    # Add warnings for pct check
    for rf in set(pct_check_rfs):
        warn_tbl.append({"rf": rf, "comment": "At least one pct per bin less than 5%."})

    # Add warnings and exclude for WoE check
    for rf in set(woe_check_rfs):
        warn_tbl.append({
            "rf": rf,
            "comment": "Problem with WoE calculation (NA, NaN, Inf, -Inf). "
                       "Risk factor is excluded from further process."
        })
        rf_rest = [r for r in rf_rest if r != rf]

    # Apply WoE coding to starting model if requested
    if coding_start_model and coding == "WoE" and rf_start:
        woe_rep, woe_info = replace_woe(db[[target] + rf_start], target=target)
        if len(woe_info) > 0 and 'reason_code' in woe_info.columns:
            problem_rfs = woe_info[woe_info['reason_code'] == 3]['rf'].tolist()
            if problem_rfs:
                raise ValueError(
                    f"Problem with WoE calculations for starting model. "
                    f"Check: {', '.join(problem_rfs)}"
                )
        for rf in rf_start:
            if rf in woe_rep.columns:
                db[rf] = woe_rep[rf]

    # Add offset if provided
    if offset_vals is not None:
        db['offset_vals'] = offset_vals

    # MIV stepwise iteration
    steps = []
    miv_iter_tbl = []
    mod_vars = rf_start.copy()
    iter_num = 1

    while True:
        print(f"Running iteration: {iter_num}")

        if len(rf_rest) == 0:
            break

        miv_results = []
        miv_details = []

        for rf in rf_rest:
            woe_o = rf_woe_o.get(rf)
            miv_res, miv_detail = _miv(
                target=target,
                mod_vars=mod_vars,
                rf_new=rf,
                db=db,
                woe_o=woe_o,
                offset_vals=offset_vals
            )
            miv_results.append(miv_res)
            miv_detail['iter'] = iter_num
            miv_details.append(miv_detail)

        # Combine results
        miv_df = pd.DataFrame(miv_results)
        miv_df = miv_df.sort_values('miv', ascending=False)

        miv_detail_df = pd.concat(miv_details, ignore_index=True)
        miv_detail_df = miv_detail_df.sort_values('miv', ascending=False)
        miv_iter_tbl.append(miv_detail_df)

        # Find candidates meeting thresholds
        candidates = miv_df[
            (miv_df['miv'] > miv_threshold) &
            (miv_df['p_val'] < m_ch_p_val)
        ]

        if len(candidates) > 0:
            # Select best candidate
            best = candidates.iloc[0]
            rf_cand = best['rf_miv']

            # Remove from rest, add to model
            rf_rest = [r for r in rf_rest if r != rf_cand]
            steps.append(best.to_dict())
            mod_vars.append(rf_cand)

            # Apply WoE coding if selected
            if coding == "WoE":
                woe_o = rf_woe_o[rf_cand]
                db[rf_cand] = _replace_woe_aux(db[rf_cand], woe_o)
        else:
            break

        iter_num += 1

    # Fit final model
    if mod_vars:
        X = db[mod_vars].copy()
        # Handle dummy coding if needed
        if coding == "dummy":
            X = pd.get_dummies(X, drop_first=True, dtype=float)
        X = sm.add_constant(X)
    else:
        X = sm.add_constant(pd.DataFrame(index=db.index))

    y_final = db[target]

    if offset_vals is not None:
        model = sm.Logit(y_final, X, offset=offset_vals).fit(disp=0)
    else:
        model = sm.Logit(y_final, X).fit(disp=0)

    # Prepare output
    steps_df = pd.DataFrame(steps)
    if len(steps_df) > 0:
        steps_df.insert(0, 'target', target)

    miv_iter_df = pd.concat(miv_iter_tbl, ignore_index=True) if miv_iter_tbl else pd.DataFrame()

    warnings_df = pd.DataFrame(warn_tbl) if warn_tbl else pd.DataFrame({'comment': ['There are no warnings.']})

    return StepMIVResult(
        model=model,
        steps=steps_df,
        miv_iter=miv_iter_df,
        warnings=warnings_df,
        dev_db=db
    )


def _parse_formula(formula: str) -> Dict[str, Any]:
    """Parse a simple formula string like 'target ~ rf1 + rf2' or 'target ~ 1'."""
    parts = formula.replace(" ", "").split("~")
    if len(parts) != 2:
        raise ValueError(f"Invalid formula: {formula}")

    target = parts[0]
    rhs = parts[1]

    if rhs == "1":
        return {'target': target, 'predictors': []}

    predictors = [p.strip() for p in rhs.split("+") if p.strip() != "1"]
    return {'target': target, 'predictors': predictors}


def _miv(
    target: str,
    mod_vars: List[str],
    rf_new: str,
    db: pd.DataFrame,
    woe_o: Optional[pd.DataFrame],
    offset_vals: Optional[np.ndarray],
) -> Tuple[Dict, pd.DataFrame]:
    """
    Calculate Marginal Information Value for a candidate risk factor.

    Args:
        target: Target variable name.
        mod_vars: Current model variables.
        rf_new: New risk factor to evaluate.
        db: Data frame.
        woe_o: Pre-calculated WoE table for rf_new.
        offset_vals: Offset values if any.

    Returns:
        Tuple of (summary dict, detailed DataFrame).
    """
    # Fit current model
    if mod_vars:
        X = sm.add_constant(db[mod_vars])
    else:
        X = sm.add_constant(pd.DataFrame(index=db.index))

    y = db[target]

    try:
        if offset_vals is not None:
            model_c = sm.Logit(y, X, offset=offset_vals).fit(disp=0)
        else:
            model_c = sm.Logit(y, X).fit(disp=0)
        pred = model_c.predict(X)
    except Exception:
        pred = pd.Series(y.mean(), index=db.index)

    db_temp = db.copy()
    db_temp['pred'] = pred
    db_temp = db_temp[db_temp['pred'].notna()]

    # Get observed WoE table
    if woe_o is not None:
        observed = woe_o[['bin', 'no', 'ng', 'nb', 'woe']].copy()
    else:
        observed = woe_tbl(db_temp, x=rf_new, y=target)[['bin', 'no', 'ng', 'nb', 'woe']]

    # Get expected WoE table based on predictions
    expected = woe_tbl(db_temp, x=rf_new, y='pred', y_check=False)[['bin', 'no', 'ng', 'nb', 'woe']]

    # Merge observed and expected
    miv_tbl = observed.merge(
        expected,
        on='bin',
        how='outer',
        suffixes=('_o', '_e')
    )
    miv_tbl['rf'] = rf_new

    # Calculate MIV
    miv_tbl['delta'] = miv_tbl['woe_o'] - miv_tbl['woe_e']

    ng_o_sum = miv_tbl['ng_o'].sum()
    nb_o_sum = miv_tbl['nb_o'].sum()

    if ng_o_sum > 0:
        miv_val_g = (miv_tbl['ng_o'] * miv_tbl['delta']).sum() / ng_o_sum
    else:
        miv_val_g = 0

    if nb_o_sum > 0:
        miv_val_b = (miv_tbl['nb_o'] * miv_tbl['delta']).sum() / nb_o_sum
    else:
        miv_val_b = 0

    miv_val = miv_val_g - miv_val_b

    # Chi-square test
    eps = 1e-10
    m_chiq_g = miv_tbl['ng_o'] * np.log(np.maximum(miv_tbl['ng_o'], eps) / np.maximum(miv_tbl['ng_e'], eps))
    m_chiq_b = miv_tbl['nb_o'] * np.log(np.maximum(miv_tbl['nb_o'], eps) / np.maximum(miv_tbl['nb_e'], eps))
    m_chiq_gb = m_chiq_g + m_chiq_b
    m_chiq_stat = 2 * m_chiq_gb.sum()

    df = len(miv_tbl) - 1
    if df > 0:
        p_val = 1 - scipy_stats.chi2.cdf(m_chiq_stat, df)
    else:
        p_val = 1.0

    # Add details to table
    miv_tbl['miv_val_g'] = miv_val_g
    miv_tbl['miv_val_b'] = miv_val_b
    miv_tbl['miv'] = miv_val
    miv_tbl['m_chiq_gb'] = m_chiq_gb
    miv_tbl['m_chiq_stat'] = m_chiq_stat
    miv_tbl['p_val'] = p_val

    # Handle NaN/Inf values
    miv_val = 0 if np.isnan(miv_val) or np.isinf(miv_val) else miv_val
    m_chiq_stat = 0 if np.isnan(m_chiq_stat) or np.isinf(m_chiq_stat) else m_chiq_stat
    p_val = 1 if np.isnan(miv_val) or np.isinf(miv_val) else p_val

    summary = {
        'rf_miv': rf_new,
        'miv': miv_val,
        'm_chiq_stat': m_chiq_stat,
        'p_val': p_val
    }

    return summary, miv_tbl


def _replace_woe_aux(x: pd.Series, woe_tbl: pd.DataFrame) -> pd.Series:
    """Replace categorical values with WoE values."""
    woe_map = dict(zip(woe_tbl['bin'], woe_tbl['woe']))
    return x.map(woe_map)
