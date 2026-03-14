"""Normal test for calibration validation.

This module provides functionality to test whether observed default
rates significantly exceed calibrated probabilities of default.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Union


@dataclass
class NormalTestResult:
    """Result of normal test for calibration.

    Attributes
    ----------
    estimate : float
        Sum of differences (ODR - PD).
    test_stat : float
        Test statistic value.
    se : float
        Standard error estimate.
    alpha : float
        Significance level used.
    p_value : float
        P-value from the test.
    result : str
        Test conclusion (H0 or H1).
    """
    estimate: float
    test_stat: float
    se: float
    alpha: float
    p_value: float
    result: str


def normal_test(
    pdc: Union[np.ndarray, pd.Series, list],
    odr: Union[np.ndarray, pd.Series, list],
    alpha: float = 0.05
) -> NormalTestResult:
    """
    Perform normal test for PD calibration validation.

    This test evaluates whether observed default rates (ODR) are
    significantly higher than calibrated probabilities of default (PD).
    It is a one-sided test with H0: ODR <= PD vs H1: ODR > PD.

    Parameters
    ----------
    pdc : array-like
        Calibrated probability of default values (between 0 and 1).
    odr : array-like
        Observed default rates (between 0 and 1).
    alpha : float, default 0.05
        Significance level for the test.

    Returns
    -------
    NormalTestResult
        Object containing test results including:
        - estimate: Sum of (ODR - PD)
        - test_stat: Test statistic
        - se: Standard error
        - alpha: Significance level
        - p_value: P-value
        - result: Test conclusion

    Raises
    ------
    ValueError
        If inputs are not numeric, have different lengths,
        or values are outside [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> pdc = np.array([0.05, 0.08, 0.06, 0.07])
    >>> odr = np.array([0.06, 0.09, 0.05, 0.08])
    >>> result = normal_test(pdc, odr)
    >>> result.result
    'H0: ODR <= PD'
    """
    # Convert inputs
    if isinstance(pdc, (pd.Series, list)):
        pdc = np.array(pdc, dtype=float)
    if isinstance(odr, (pd.Series, list)):
        odr = np.array(odr, dtype=float)

    # Validation
    if not (np.issubdtype(pdc.dtype, np.number) and
            np.issubdtype(odr.dtype, np.number)):
        raise ValueError("All arguments have to be numeric.")

    if len(pdc) != len(odr):
        raise ValueError("pdc and odr have to be of the same length.")

    if np.any((pdc > 1) | (pdc < 0) | (odr > 1) | (odr < 0)):
        raise ValueError("pdc and odr have to be between 0 and 1.")

    if alpha > 1 or alpha < 0:
        raise ValueError("significance level (alpha) has to be between 0 and 1.")

    # Calculate test statistics
    T = len(odr)
    diff = odr - pdc

    # Standard error estimate
    se = (np.sum(diff ** 2) - (np.sum(diff) ** 2 / T)) / (T - 1)

    # Test statistic
    if se <= 0:
        # Handle edge case where variance is zero
        test_stat = 0.0 if np.sum(diff) == 0 else np.inf
    else:
        test_stat = np.sum(diff) / np.sqrt(T * se)

    # P-value (one-sided, upper tail)
    p_value = 1 - stats.norm.cdf(test_stat)

    # Estimate
    estimate = np.sum(diff)

    # Conclusion
    if p_value >= alpha:
        result = "H0: ODR <= PD"
    else:
        result = "H1: ODR > PD"

    return NormalTestResult(
        estimate=float(estimate),
        test_stat=float(test_stat),
        se=float(se),
        alpha=float(alpha),
        p_value=float(p_value),
        result=result
    )
