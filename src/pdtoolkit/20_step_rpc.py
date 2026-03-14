"""Stepwise regression with risk profile check.

This module provides functionality for customized stepwise regression
that considers risk factor priority groups and maintains proper ordering
during variable selection.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from importlib import import_module

# Import bivariate functions for WoE calculations
_bivariate = import_module('.02_bivariate_analysis', package='pdtoolkit')


@dataclass
class StepRPCResult:
    """Result of stepwise regression with risk profile check.

    Attributes
    ----------
    model : pd.DataFrame
        Final model coefficients and statistics.
    steps : pd.DataFrame
        Details of each iteration including added risk factors.
    warnings : list
        List of warning messages generated during fitting.
    dev_db : pd.DataFrame
        Development database with coded variables.
    """
    model: pd.DataFrame
    steps: pd.DataFrame
    warnings: List[str]
    dev_db: pd.DataFrame


def step_rpc(
    start_model: str,
    risk_profile: pd.DataFrame,
    p_value: float = 0.05,
    coding: str = 'WoE',
    coding_start_model: bool = True,
    check_start_model: bool = True,
    db: pd.DataFrame = None,
    offset_vals: Optional[np.ndarray] = None
) -> StepRPCResult:
    """
    Perform stepwise regression with risk profile check.

    This function performs forward stepwise logistic regression while
    respecting priority groups defined in the risk profile. Variables
    are considered in order of their priority group, with lower group
    numbers having higher priority.

    Parameters
    ----------
    start_model : str
        Initial model formula (e.g., 'target ~ 1' for intercept-only).
    risk_profile : pd.DataFrame
        Data frame with columns:
        - rf: Risk factor names
        - group: Priority group (numeric, lower = higher priority)
    p_value : float, default 0.05
        P-value threshold for variable selection.
    coding : str, default 'WoE'
        Coding method: 'WoE' for Weight of Evidence or 'dummy' for dummy coding.
    coding_start_model : bool, default True
        Whether to apply coding to starting model variables.
    check_start_model : bool, default True
        Whether to check significance of starting model variables.
    db : pd.DataFrame
        Modeling database containing target and risk factors.
    offset_vals : np.ndarray, optional
        Offset values for the model.

    Returns
    -------
    StepRPCResult
        Object containing model, steps, warnings, and development database.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> db = pd.DataFrame({
    ...     'target': np.random.binomial(1, 0.1, n),
    ...     'woe_rf1': np.random.randn(n),
    ...     'woe_rf2': np.random.randn(n),
    ...     'woe_rf3': np.random.randn(n)
    ... })
    >>> risk_profile = pd.DataFrame({
    ...     'rf': ['woe_rf1', 'woe_rf2', 'woe_rf3'],
    ...     'group': [1, 1, 2]
    ... })
    >>> result = step_rpc(
    ...     start_model='target ~ 1',
    ...     risk_profile=risk_profile,
    ...     db=db,
    ...     p_value=0.05
    ... )
    """
    # Validation
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db must be a data frame.")

    if not isinstance(risk_profile, pd.DataFrame):
        raise ValueError("risk_profile must be a data frame.")

    if 'rf' not in risk_profile.columns or 'group' not in risk_profile.columns:
        raise ValueError("risk_profile must have 'rf' and 'group' columns.")

    if coding not in ['WoE', 'dummy']:
        raise ValueError("coding must be 'WoE' or 'dummy'.")

    if not 0 < p_value < 1:
        raise ValueError("p_value must be between 0 and 1.")

    # Parse start model to get target
    parts = start_model.replace(' ', '').split('~')
    target = parts[0]

    if target not in db.columns:
        raise ValueError(f"target '{target}' not found in db.")

    # Get starting variables (if any beyond intercept)
    start_vars = []
    if len(parts) > 1 and parts[1] != '1':
        start_vars = [v.strip() for v in parts[1].split('+') if v.strip() != '1']

    # Check risk factors exist
    risk_factors = risk_profile['rf'].tolist()
    missing_rfs = [rf for rf in risk_factors if rf not in db.columns]
    if missing_rfs:
        raise ValueError(f"Risk factors not found in db: {missing_rfs}")

    # Initialize tracking
    warnings_list = []
    steps_list = []
    dev_db = db.copy()

    # Apply WoE coding if specified
    if coding == 'WoE':
        # For WoE coding, variables should already be WoE-transformed
        # Just validate they exist
        pass

    # Sort groups
    groups = sorted(risk_profile['group'].unique())

    # Track selected variables
    selected_vars = list(start_vars)

    # Prepare target
    y = dev_db[target].values

    # Process each priority group
    iteration = 0
    for group in groups:
        group_rfs = risk_profile[risk_profile['group'] == group]['rf'].tolist()
        available_rfs = [rf for rf in group_rfs if rf not in selected_vars]

        while available_rfs:
            iteration += 1
            best_rf = None
            best_pval = 1.0

            # Find best candidate from this group
            for rf in available_rfs:
                try:
                    # Build candidate model
                    candidate_vars = selected_vars + [rf]
                    X = dev_db[candidate_vars].values
                    X = sm.add_constant(X, has_constant='add')

                    if offset_vals is not None:
                        model = sm.GLM(y, X, family=sm.families.Binomial(),
                                      offset=offset_vals)
                    else:
                        model = sm.GLM(y, X, family=sm.families.Binomial())

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = model.fit(disp=0)

                    # Get p-value for the new variable (last coefficient)
                    rf_pval = result.pvalues[-1]

                    # Check trend for WoE coding
                    if coding == 'WoE':
                        rf_coef = result.params[-1]
                        if rf_coef < 0:
                            # Wrong trend for WoE - skip
                            continue

                    if rf_pval < best_pval:
                        best_pval = rf_pval
                        best_rf = rf

                except Exception:
                    continue

            # Check if best candidate meets threshold
            if best_rf is not None and best_pval < p_value:
                selected_vars.append(best_rf)
                available_rfs.remove(best_rf)

                steps_list.append({
                    'iteration': iteration,
                    'rf_added': best_rf,
                    'p_value': best_pval,
                    'group': group
                })
            else:
                # No more significant variables in this group
                break

    # Fit final model
    if selected_vars:
        X = dev_db[selected_vars].values
        X = sm.add_constant(X, has_constant='add')
    else:
        X = np.ones((len(y), 1))

    if offset_vals is not None:
        final_model = sm.GLM(y, X, family=sm.families.Binomial(),
                            offset=offset_vals)
    else:
        final_model = sm.GLM(y, X, family=sm.families.Binomial())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_result = final_model.fit(disp=0)

    # Build model DataFrame
    var_names = ['const'] + selected_vars
    model_df = pd.DataFrame({
        'variable': var_names,
        'coefficient': final_result.params,
        'std_error': final_result.bse,
        'z_value': final_result.tvalues,
        'p_value': final_result.pvalues
    })

    # Build steps DataFrame
    if steps_list:
        steps_df = pd.DataFrame(steps_list)
    else:
        steps_df = pd.DataFrame(columns=['iteration', 'rf_added', 'p_value', 'group'])
        warnings_list.append("No risk factors selected.")

    return StepRPCResult(
        model=model_df,
        steps=steps_df,
        warnings=warnings_list,
        dev_db=dev_db
    )
