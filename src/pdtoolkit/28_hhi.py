"""Herfindahl-Hirschman Index calculation.

This module provides functionality to calculate the HHI,
a measure of concentration commonly used in economics.
"""

import numpy as np
import pandas as pd
from typing import Union


def hhi(x: Union[np.ndarray, pd.Series, list]) -> float:
    """
    Calculate the Herfindahl-Hirschman Index.

    The HHI is a measure of concentration calculated as the sum of
    squared market shares. Values range from 0 (perfect competition)
    to 1 (monopoly). In credit risk, it can be used to measure
    portfolio concentration.

    Parameters
    ----------
    x : array-like
        Numeric vector of values (e.g., exposures, market shares).
        Missing values (NA/NaN) are excluded from calculation.

    Returns
    -------
    float
        The HHI value between 0 and 1.

    Raises
    ------
    ValueError
        If x is not numeric or contains infinite values.

    Examples
    --------
    >>> import numpy as np
    >>> # Perfect concentration (one entity)
    >>> hhi([100])
    1.0

    >>> # Equal distribution
    >>> hhi([25, 25, 25, 25])
    0.25

    >>> # Unequal distribution
    >>> hhi([50, 30, 20])
    0.38
    """
    # Convert to numpy array
    if isinstance(x, pd.Series):
        x = x.values
    elif isinstance(x, list):
        x = np.array(x)

    # Validation
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("x has to be numeric vector.")

    if np.any(np.isinf(x)):
        raise ValueError("x contains Inf or -Inf values.")

    # Remove missing values
    x = x[~np.isnan(x)]

    if len(x) == 0:
        raise ValueError("x contains only missing values.")

    # Calculate HHI
    total = np.sum(x)
    if total == 0:
        raise ValueError("Sum of x is zero.")

    shares = x / total
    hhi_value = np.sum(shares ** 2)

    return float(hhi_value)
