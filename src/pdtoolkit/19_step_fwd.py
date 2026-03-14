"""
Forward stepwise regression module for PDtoolkit.

This module performs forward stepwise logistic regression with
p-value and trend validation.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class StepFWDResult:
    """Result of forward stepwise regression.

    Attributes:
        model: Final fitted model summary
        steps: DataFrame with iteration steps
        warnings: List of warning messages
        db: Development database used
    """
    model: pd.DataFrame
    steps: pd.DataFrame
    warnings: List[str]
    db: pd.DataFrame


def step_fwd(
    start_model: str,
    p_value: float = 0.05,
    coding: str = 'WoE',
    db: pd.DataFrame = None,
    risk_factors: Optional[List[str]] = None,
    target: str = None,
    offset_vals: Optional[np.ndarray] = None
) -> StepFWDResult:
    """
    Perform forward stepwise logistic regression.

    Parameters
    ----------
    start_model : str
        Initial formula (e.g., "target ~ 1" for intercept-only).
    p_value : float, default=0.05
        Significance threshold for coefficient p-values.
    coding : str, default='WoE'
        Variable coding: 'WoE' or 'dummy'.
    db : pd.DataFrame
        Data frame containing risk factors and target.
    risk_factors : list of str, optional
        Names of candidate risk factors. If None, uses all non-target columns.
    target : str
        Name of target variable.
    offset_vals : np.ndarray, optional
        Optional offset for linear predictor.

    Returns
    -------
    StepFWDResult
        Object containing model, steps, warnings, and database.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> db = pd.DataFrame({
    ...     'target': np.random.binomial(1, 0.1, 500),
    ...     'woe_rf1': np.random.randn(500) * 0.5,
    ...     'woe_rf2': np.random.randn(500) * 0.3
    ... })
    >>> result = step_fwd('target ~ 1', db=db, target='target')
    """
    if db is None or not isinstance(db, pd.DataFrame):
        raise ValueError("db must be a data frame.")

    if target is None or target not in db.columns:
        raise ValueError("target must be specified and exist in db.")

    coding_opts = ['WoE', 'dummy']
    if coding not in coding_opts:
        raise ValueError(f"coding must be one of: {coding_opts}")

    if p_value <= 0 or p_value >= 1:
        raise ValueError("p_value must be between 0 and 1.")

    # Get risk factors
    if risk_factors is None:
        risk_factors = [c for c in db.columns if c != target]

    if len(risk_factors) == 0:
        raise ValueError("No risk factors available.")

    # Initialize
    selected_rfs = []
    remaining_rfs = risk_factors.copy()
    steps = []
    warning_msgs = []

    # Import statsmodels
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels is required for step_fwd.")

    # Prepare data
    db_clean = db[[target] + risk_factors].dropna()
    y = db_clean[target].values

    iteration = 0

    while remaining_rfs:
        iteration += 1

        # Evaluate each candidate
        candidates = []

        for rf in remaining_rfs:
            try:
                # Build model with current selected + candidate
                current_rfs = selected_rfs + [rf]
                X = sm.add_constant(db_clean[current_rfs].values)

                if offset_vals is not None:
                    model = sm.Logit(y, X, offset=offset_vals[:len(y)])
                else:
                    model = sm.Logit(y, X)

                result = model.fit(disp=0, method='bfgs', maxiter=100)

                # Get p-value for the new variable
                # Last coefficient is the new variable
                p_val = result.pvalues[-1]
                coef = result.params[-1]

                # Check trend for WoE coding
                trend_ok = True
                if coding == 'WoE':
                    # WoE should have positive coefficient
                    trend_ok = coef > 0

                candidates.append({
                    'rf': rf,
                    'p_value': p_val,
                    'coefficient': coef,
                    'trend_ok': trend_ok,
                    'aic': result.aic
                })

            except Exception as e:
                warning_msgs.append(f"Error evaluating {rf}: {str(e)}")
                continue

        if not candidates:
            break

        # Find best candidate
        valid_candidates = [c for c in candidates
                          if c['p_value'] < p_value and c['trend_ok']]

        if not valid_candidates:
            # No more valid candidates
            break

        # Select by lowest p-value
        best = min(valid_candidates, key=lambda x: x['p_value'])

        # Add to model
        selected_rfs.append(best['rf'])
        remaining_rfs.remove(best['rf'])

        steps.append({
            'iteration': iteration,
            'rf_added': best['rf'],
            'p_value': best['p_value'],
            'coefficient': best['coefficient'],
            'aic': best['aic']
        })

    # Fit final model
    if selected_rfs:
        X_final = sm.add_constant(db_clean[selected_rfs].values)
        if offset_vals is not None:
            final_model = sm.Logit(y, X_final, offset=offset_vals[:len(y)])
        else:
            final_model = sm.Logit(y, X_final)

        final_result = final_model.fit(disp=0, method='bfgs', maxiter=100)

        # Create model summary
        model_summary = pd.DataFrame({
            'variable': ['const'] + selected_rfs,
            'coefficient': final_result.params,
            'std_err': final_result.bse,
            'z_value': final_result.tvalues,
            'p_value': final_result.pvalues
        })
    else:
        # Intercept-only model
        X_final = sm.add_constant(np.ones(len(y)))
        final_model = sm.Logit(y, X_final)
        final_result = final_model.fit(disp=0)

        model_summary = pd.DataFrame({
            'variable': ['const'],
            'coefficient': final_result.params,
            'std_err': final_result.bse,
            'z_value': final_result.tvalues,
            'p_value': final_result.pvalues
        })
        warning_msgs.append("No risk factors met selection criteria.")

    # Steps DataFrame
    steps_df = pd.DataFrame(steps) if steps else pd.DataFrame(
        columns=['iteration', 'rf_added', 'p_value', 'coefficient', 'aic']
    )

    return StepFWDResult(
        model=model_summary,
        steps=steps_df,
        warnings=warning_msgs,
        db=db_clean
    )
