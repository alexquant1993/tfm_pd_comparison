"""U-shape testing for risk factors.

This module provides functionality to test whether a numeric variable
exhibits a U-shaped relationship with a binary target.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class UShapeTestResult:
    """Result of U-shape test.

    Attributes
    ----------
    candidates : pd.DataFrame
        All tested candidate knots with statistics.
    optimal_knot : float
        The optimal knot value (if U-shape detected).
    detected : bool
        Whether U-shape was detected.
    basis : pd.DataFrame
        B-spline basis functions at optimal knot.
    """
    candidates: pd.DataFrame
    optimal_knot: Optional[float]
    detected: bool
    basis: Optional[pd.DataFrame]


def ush_test(
    x: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series],
    p_value: float = 0.05,
    g: int = 20,
    sc: Optional[List] = None
) -> UShapeTestResult:
    """
    Test for U-shaped relationship between x and y.

    This function tests whether a numeric variable exhibits a U-shaped
    relationship with a binary target using B-spline basis functions
    and logistic regression.

    Parameters
    ----------
    x : array-like
        Numeric variable to test.
    y : array-like
        Binary target variable (0/1).
    p_value : float, default 0.05
        Significance threshold for the test.
    g : int, default 20
        Number of candidate knot points (2-50).
    sc : list, optional
        Special case values to exclude.
        Default is [np.nan, np.inf, -np.inf].

    Returns
    -------
    UShapeTestResult
        Object containing test results.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> x = np.random.randn(500)
    >>> # U-shaped relationship
    >>> prob = 1 / (1 + np.exp(-((x - 0)**2 - 1)))
    >>> y = (np.random.rand(500) < prob).astype(int)
    >>> result = ush_test(x, y)
    >>> result.detected
    True
    """
    # Default special cases
    if sc is None:
        sc = [np.nan, np.inf, -np.inf]

    # Convert to numpy
    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Validate
    if not 0 < p_value < 1:
        raise ValueError("p_value must be between 0 and 1.")

    if not 2 <= g <= 50:
        raise ValueError("g must be between 2 and 50.")

    # Remove special cases
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x)
    x_clean = x[mask]
    y_clean = y[mask]

    if not np.all(np.isin(y_clean, [0, 1])):
        raise ValueError("y must be 0/1 variable.")

    if len(x_clean) < 10:
        raise ValueError("Not enough observations after removing special cases.")

    # Generate candidate knots using quantiles
    quantiles = np.linspace(0.05, 0.95, g)
    knots = np.quantile(x_clean, quantiles)
    knots = np.unique(knots)  # Remove duplicates

    candidates = []
    best_knot = None
    best_deviance = np.inf

    for knot in knots:
        try:
            result = _lr_summary(x_clean, y_clean, knot, p_value)
            candidates.append(result)

            if result['detected'] and result['deviance'] < best_deviance:
                best_deviance = result['deviance']
                best_knot = knot
        except Exception:
            continue

    # Build candidates DataFrame
    if candidates:
        candidates_df = pd.DataFrame(candidates)
    else:
        candidates_df = pd.DataFrame(columns=[
            'knot', 'coef_b1', 'coef_b2', 'pval_b1', 'pval_b2',
            'direction_test', 'significance_test', 'detected', 'deviance'
        ])

    # Get basis at optimal knot
    if best_knot is not None:
        b1, b2 = _bs_cp(x_clean, best_knot)
        basis_df = pd.DataFrame({'b1': b1, 'b2': b2})
        detected = True
    else:
        basis_df = None
        detected = False

    return UShapeTestResult(
        candidates=candidates_df,
        optimal_knot=best_knot,
        detected=detected,
        basis=basis_df
    )


def _bs_cp(x: np.ndarray, knot: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate B-spline basis functions for a single knot."""
    x_min = np.min(x)
    x_max = np.max(x)

    # Left basis (from min to knot)
    b1 = np.where(x <= knot, (knot - x) / (knot - x_min), 0)

    # Right basis (from knot to max)
    b2 = np.where(x >= knot, (x - knot) / (x_max - knot), 0)

    return b1, b2


def _lr_summary(
    x: np.ndarray,
    y: np.ndarray,
    knot: float,
    p_value: float
) -> dict:
    """Fit logistic regression with basis functions and summarize."""
    b1, b2 = _bs_cp(x, knot)

    # Fit logistic regression
    X = np.column_stack([b1, b2])
    X = sm.add_constant(X, has_constant='add')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(y, X, family=sm.families.Binomial())
        result = model.fit(disp=0)

    # Extract coefficients and p-values
    coef_b1 = result.params[1]
    coef_b2 = result.params[2]
    pval_b1 = result.pvalues[1]
    pval_b2 = result.pvalues[2]

    # Direction test: coefficients have opposite signs
    # For U-shape: b1 should be positive, b2 should be positive (both arms go up)
    # or both should be same sign
    direction_test = (coef_b1 > 0) and (coef_b2 > 0)

    # Significance test: both p-values below threshold
    significance_test = (pval_b1 < p_value) and (pval_b2 < p_value)

    # U-shape detected if both tests pass
    detected = direction_test and significance_test

    return {
        'knot': knot,
        'coef_b1': coef_b1,
        'coef_b2': coef_b2,
        'pval_b1': pval_b1,
        'pval_b2': pval_b2,
        'direction_test': direction_test,
        'significance_test': significance_test,
        'detected': detected,
        'deviance': result.deviance
    }
