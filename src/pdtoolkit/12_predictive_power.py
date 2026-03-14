"""
Predictive power testing module for PDtoolkit.

This module tests the predictive power of PD rating models using
binomial, Jeffreys, z-score and Hosmer-Lemeshow tests.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, List


def pp_testing(
    rating_label: Union[List[str], np.ndarray],
    pdc: Union[List[float], np.ndarray],
    no: Union[List[int], np.ndarray],
    nb: Union[List[int], np.ndarray],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Test the predictive power of PD rating model.

    Performs testing using four tests: binomial, Jeffreys, z-score
    and Hosmer-Lemeshow.

    Parameters
    ----------
    rating_label : array-like
        Vector of rating labels.
    pdc : array-like
        Vector of calibrated probabilities of default (PD).
    no : array-like
        Number of observations per rating grade.
    nb : array-like
        Number of defaults (bad cases) per rating grade.
    alpha : float, default=0.05
        Significance level of p-value.

    Returns
    -------
    pd.DataFrame
        DataFrame with input parameters, p-values for each test,
        and accepted hypotheses.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Examples
    --------
    >>> rating_label = ['AAA', 'AA', 'A', 'BBB']
    >>> pdc = [0.001, 0.005, 0.01, 0.03]
    >>> no = [100, 200, 300, 200]
    >>> nb = [0, 1, 4, 8]
    >>> result = pp_testing(rating_label, pdc, no, nb)
    """
    # Convert to arrays
    rating_label = np.asarray(rating_label)
    pdc = np.asarray(pdc, dtype=float)
    no = np.asarray(no, dtype=float)
    nb = np.asarray(nb, dtype=float)

    # Validate lengths
    lengths = [len(rating_label), len(pdc), len(no), len(nb)]
    if len(set(lengths)) != 1:
        raise ValueError(
            "arguments rating_label, pdc, no and nb have to be of the same length."
        )

    # Validate types
    if not (np.issubdtype(pdc.dtype, np.number) and
            np.issubdtype(no.dtype, np.number) and
            np.issubdtype(nb.dtype, np.number)):
        raise ValueError("Arguments pdc, no, and nb have to be of numeric type.")

    # Validate ranges
    if np.any((pdc > 1) | (pdc < 0)):
        raise ValueError("pdc has to be between 0 and 1.")

    if np.any(nb > no):
        raise ValueError("Number of defaults cannot be greater than number of observations.")

    if np.any((nb < 0) | (no < 0)):
        raise ValueError("no and nb arguments cannot be negative.")

    if alpha > 1 or alpha < 0:
        raise ValueError("significance level (alpha) has to be between 0 and 1.")

    # Calculate observed default rate
    odr = nb / no

    # Binomial test
    bino_test = 1 - stats.binom.cdf(nb - 1, no, pdc)
    bino_test_res = np.where(
        bino_test >= alpha,
        "H0: ODR <= PDC",
        "H1: ODR > PDC"
    )

    # Jeffreys test (beta distribution)
    jeff_test = stats.beta.cdf(pdc, nb + 0.5, no - nb + 0.5)
    jeff_test_res = np.where(
        jeff_test >= alpha,
        "H0: ODR <= PDC",
        "H1: ODR > PDC"
    )

    # Z-score test
    zsco_test = _zscore_test(pdc, odr, no)
    zsco_test_res = np.where(
        zsco_test >= alpha,
        "H0: ODR <= PDC",
        "H1: ODR > PDC"
    )

    # Hosmer-Lemeshow test
    if len(pdc) == 1:
        hosm_test = np.array([np.nan])
        hosm_test_res = np.array(["Only one rating grade."])
    else:
        hosm_test_val = _hl_test(pdc, no, nb)
        hosm_test = np.full(len(pdc), hosm_test_val)
        hosm_test_res = np.where(
            hosm_test >= alpha,
            "H0: PDC is TRUE",
            "H1: PDC is not TRUE"
        )

    result = pd.DataFrame({
        'rating': rating_label,
        'no': no.astype(int),
        'nb': nb.astype(int),
        'odr': odr,
        'pdc': pdc,
        'alpha': alpha,
        'binomial': bino_test,
        'binomial_res': bino_test_res,
        'jeffreys': jeff_test,
        'jeffreys_res': jeff_test_res,
        'zscore': zsco_test,
        'zscore_res': zsco_test_res,
        'hosmer_lemeshow': hosm_test,
        'hosmer_lemeshow_res': hosm_test_res
    })

    return result


def _zscore_test(
    pdc: np.ndarray,
    odr: np.ndarray,
    no: np.ndarray
) -> np.ndarray:
    """
    Z-score test for comparing observed vs calibrated default rates.

    Parameters
    ----------
    pdc : np.ndarray
        Calibrated PD.
    odr : np.ndarray
        Observed default rate.
    no : np.ndarray
        Number of observations.

    Returns
    -------
    np.ndarray
        P-values for each rating.
    """
    se = np.sqrt((pdc * (1 - pdc)) / no)
    with np.errstate(divide='ignore', invalid='ignore'):
        test_stat = (odr - pdc) / se
        p_val = 1 - stats.norm.cdf(test_stat)
    p_val = np.where(np.isfinite(p_val), p_val, 1.0)
    return p_val


def _hl_test(
    pdc: np.ndarray,
    no: np.ndarray,
    nb: np.ndarray
) -> float:
    """
    Hosmer-Lemeshow test.

    Parameters
    ----------
    pdc : np.ndarray
        Calibrated PD.
    no : np.ndarray
        Number of observations.
    nb : np.ndarray
        Number of defaults.

    Returns
    -------
    float
        P-value for Hosmer-Lemeshow test.
    """
    k = len(no)

    # Calculate HL statistic
    with np.errstate(divide='ignore', invalid='ignore'):
        hl = np.nansum((nb - no * pdc) ** 2 / (no * pdc * (1 - pdc)))

    # P-value from chi-squared distribution
    p_val = 1 - stats.chi2.cdf(hl, df=k)

    return p_val
