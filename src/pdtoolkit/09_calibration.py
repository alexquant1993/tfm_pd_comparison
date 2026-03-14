"""
Calibration module for PDtoolkit.

This module provides functionality for rating scale calibration,
adjusting observed default rates to align with central tendency values.
"""

from dataclasses import dataclass
from typing import Union, Literal, Optional

import numpy as np
import pandas as pd
from scipy import optimize


@dataclass
class CalibrationResult:
    """Result of rating scale calibration.

    Attributes:
        pd_calib: Calibrated probabilities of default
        method: Calibration method used
        params: Optimization parameters (method-specific)
    """
    pd_calib: np.ndarray
    method: str
    params: dict


def rs_calibration(
    rs: pd.DataFrame,
    dr: str,
    w: str,
    ct: float,
    min_pd: float = 0.0003,
    method: Literal['scaling', 'log_odds_a', 'log_odds_ab'] = 'scaling'
) -> CalibrationResult:
    """
    Perform calibration of observed default rates.

    Calibrate observed default rates across rating scales using three methods:
    "scaling", "log_odds_a", and "log_odds_ab".

    Parameters
    ----------
    rs : pd.DataFrame
        Rating scale data frame containing default rates and weights.
    dr : str
        Column name for default rates.
    w : str
        Column name for weights (number of observations per rating).
    ct : float
        Central tendency (target portfolio default rate).
    min_pd : float, default=0.0003
        Minimum probability of default constraint (3 basis points).
    method : str, default='scaling'
        Calibration method. One of 'scaling', 'log_odds_a', 'log_odds_ab'.

    Returns
    -------
    CalibrationResult
        Object containing calibrated PDs, method used, and optimization parameters.

    Raises
    ------
    ValueError
        If input validation fails.

    Examples
    --------
    >>> rs = pd.DataFrame({
    ...     'rating': ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'],
    ...     'dr': [0.001, 0.002, 0.005, 0.01, 0.03, 0.08, 0.20],
    ...     'n': [100, 200, 300, 400, 300, 200, 100]
    ... })
    >>> result = rs_calibration(rs, dr='dr', w='n', ct=0.05, method='scaling')
    """
    if not isinstance(rs, pd.DataFrame):
        raise ValueError("rs (rating scale) is not a data frame.")

    if dr not in rs.columns or w not in rs.columns:
        raise ValueError(
            "arguments dr (default rate) and w (weights) cannot be found in rs (rating scale)."
        )

    method_opts = ['scaling', 'log_odds_a', 'log_odds_ab']
    if method not in method_opts:
        raise ValueError(
            f"method argument has to be one of: {', '.join(method_opts)}."
        )

    dr_vals = rs[dr].values.astype(float)
    w_vals = rs[w].values.astype(float)

    if np.any((dr_vals > 1) | (dr_vals < 0)):
        raise ValueError("Default rate has to be within 0-1 range.")

    if ct <= 0 or ct >= 1:
        raise ValueError("Central tendency (ct) has to be within 0-1 range.")

    if min_pd < 0 or min_pd >= 1:
        raise ValueError("min_pd has to be within 0-1 range.")

    if method == 'scaling':
        pd_calib, params = _calib_scaling(dr_vals, w_vals, ct, min_pd)
    elif method == 'log_odds_a':
        pd_calib, params = _calib_log_odds_a(dr_vals, w_vals, ct, min_pd)
    else:  # log_odds_ab
        pd_calib, params = _calib_log_odds_ab(dr_vals, w_vals, ct, min_pd)

    return CalibrationResult(
        pd_calib=pd_calib,
        method=method,
        params=params
    )


def _calib_scaling(
    dr: np.ndarray,
    w: np.ndarray,
    ct: float,
    min_pd: float
) -> tuple:
    """
    Calibration using linear scaling method.

    Calculates a scaling factor as the ratio between the central tendency
    and portfolio default rate.
    """
    # Calculate portfolio default rate
    pdr = np.sum(dr * w) / np.sum(w)

    # Calculate scaling factor
    factor = ct / pdr if pdr > 0 else 1.0

    # Apply scaling
    pd_calib = dr * factor

    # Apply constraints iteratively if needed
    iterations = 0
    max_iter = 100

    while np.any(pd_calib < min_pd) and iterations < max_iter:
        # Set minimum PD for violating ratings
        mask = pd_calib < min_pd
        pd_calib[mask] = min_pd

        # Recalculate factor for remaining ratings
        remaining_mask = ~mask
        if np.sum(remaining_mask) == 0:
            break

        remaining_w = w[remaining_mask]
        remaining_dr = dr[remaining_mask]

        # Calculate remaining target
        fixed_contribution = np.sum(min_pd * w[mask])
        remaining_target = ct * np.sum(w) - fixed_contribution
        remaining_pdr = np.sum(remaining_dr * remaining_w)

        if remaining_pdr > 0 and remaining_target > 0:
            new_factor = remaining_target / remaining_pdr
            pd_calib[remaining_mask] = remaining_dr * new_factor

        iterations += 1

    # Cap at 1.0
    pd_calib = np.minimum(pd_calib, 1.0)

    return pd_calib, {'factor': factor, 'iterations': iterations}


def _calib_log_odds_a(
    dr: np.ndarray,
    w: np.ndarray,
    ct: float,
    min_pd: float
) -> tuple:
    """
    Calibration using log-odds transformation with intercept optimization.

    Optimizes the intercept parameter of logit transformation to align
    portfolio default rate with the specified central tendency.
    """
    # Convert to log odds
    dr_safe = np.clip(dr, 1e-10, 1 - 1e-10)
    log_odds = np.log(dr_safe / (1 - dr_safe))

    def objective(a):
        # Transform with intercept shift
        new_log_odds = log_odds + a
        new_pd = 1 / (1 + np.exp(-new_log_odds))

        # Apply minimum PD constraint
        new_pd = np.maximum(new_pd, min_pd)
        new_pd = np.minimum(new_pd, 1.0)

        # Calculate weighted average
        weighted_pd = np.sum(new_pd * w) / np.sum(w)

        return (weighted_pd - ct) ** 2

    # Optimize
    result = optimize.minimize_scalar(objective, bounds=(-10, 10), method='bounded')
    a_opt = result.x

    # Calculate final calibrated PDs
    new_log_odds = log_odds + a_opt
    pd_calib = 1 / (1 + np.exp(-new_log_odds))
    pd_calib = np.maximum(pd_calib, min_pd)
    pd_calib = np.minimum(pd_calib, 1.0)

    return pd_calib, {'a': a_opt}


def _calib_log_odds_ab(
    dr: np.ndarray,
    w: np.ndarray,
    ct: float,
    min_pd: float
) -> tuple:
    """
    Calibration using log-odds transformation with intercept and slope optimization.

    Optimizes both intercept and slope parameters of logit transformation.
    """
    # Convert to log odds
    dr_safe = np.clip(dr, 1e-10, 1 - 1e-10)
    log_odds = np.log(dr_safe / (1 - dr_safe))

    def objective(params):
        a, b = params

        # Transform with intercept and slope
        new_log_odds = a + b * log_odds
        new_pd = 1 / (1 + np.exp(-new_log_odds))

        # Apply minimum PD constraint
        new_pd = np.maximum(new_pd, min_pd)
        new_pd = np.minimum(new_pd, 1.0)

        # Calculate weighted average
        weighted_pd = np.sum(new_pd * w) / np.sum(w)

        # Primary objective: match central tendency
        ct_loss = (weighted_pd - ct) ** 2

        # Secondary: prefer b close to 1 (preserve relative ordering)
        reg_loss = 0.001 * (b - 1) ** 2

        return ct_loss + reg_loss

    # Optimize
    result = optimize.minimize(
        objective,
        x0=[0.0, 1.0],
        method='BFGS',
        options={'maxiter': 1000}
    )
    a_opt, b_opt = result.x

    # Calculate final calibrated PDs
    new_log_odds = a_opt + b_opt * log_odds
    pd_calib = 1 / (1 + np.exp(-new_log_odds))
    pd_calib = np.maximum(pd_calib, min_pd)
    pd_calib = np.minimum(pd_calib, 1.0)

    return pd_calib, {'a': a_opt, 'b': b_opt}
