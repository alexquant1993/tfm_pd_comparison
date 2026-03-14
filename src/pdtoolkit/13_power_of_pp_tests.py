"""
Power of predictive power tests module for PDtoolkit.

This module performs Monte Carlo simulation to estimate the power
of statistical tests used for predictive ability testing.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, List, Dict
from dataclasses import dataclass


@dataclass
class PowerResult:
    """Result of power analysis.

    Attributes:
        interval_estimator: DataFrame with power for binomial, Jeffreys, z-score tests
        hosmer_lemeshow: DataFrame with power for Hosmer-Lemeshow test
    """
    interval_estimator: pd.DataFrame
    hosmer_lemeshow: pd.DataFrame


def power(
    rating_label: Union[List[str], np.ndarray],
    pdc: Union[List[float], np.ndarray],
    no: Union[List[int], np.ndarray],
    nb: Union[List[int], np.ndarray],
    alpha: float = 0.05,
    sim_num: int = 1000,
    seed: int = 2211
) -> PowerResult:
    """
    Calculate power of statistical tests for predictive ability testing.

    Performs Monte Carlo simulation to estimate the power of the binomial,
    Jeffreys, z-score and Hosmer-Lemeshow tests.

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
        Significance level for tests.
    sim_num : int, default=1000
        Number of Monte Carlo simulations.
    seed : int, default=2211
        Random seed for reproducibility.

    Returns
    -------
    PowerResult
        Object containing power calculations for individual rating tests
        and Hosmer-Lemeshow test.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Examples
    --------
    >>> rating_label = ['AAA', 'AA', 'A']
    >>> pdc = [0.005, 0.01, 0.02]
    >>> no = [100, 200, 150]
    >>> nb = [1, 3, 4]
    >>> result = power(rating_label, pdc, no, nb, sim_num=500)
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
        raise ValueError("All arguments have to be of numeric type.")

    # Validate ranges
    if np.any((pdc > 1) | (pdc < 0)):
        raise ValueError("pdc has to be between 0 and 1.")

    if np.any(nb > no):
        raise ValueError("Number of defaults cannot be greater than number of observations.")

    if np.any((nb < 0) | (no < 0)):
        raise ValueError("no and nb arguments cannot be negative.")

    # Calculate observed default rate
    odr = nb / no

    # Individual rating tests
    ie_results = []
    for i in range(len(rating_label)):
        res_l = _mc_sim_binom(
            pdc_r=pdc[i],
            odr_r=odr[i],
            no_r=int(no[i]),
            alpha=alpha,
            sim_num=sim_num,
            seed=seed
        )
        res_l.update({
            'rating': rating_label[i],
            'no': int(no[i]),
            'nb': int(nb[i]),
            'odr': odr[i],
            'pdc': pdc[i]
        })
        ie_results.append(res_l)

    ie_df = pd.DataFrame(ie_results)
    # Reorder columns
    cols = ['rating', 'no', 'nb', 'odr', 'pdc', 'binomial', 'jeffreys', 'zscore']
    ie_df = ie_df[cols]

    # Hosmer-Lemeshow test
    if len(pdc) == 1:
        hl_df = pd.DataFrame({
            'rating': [rating_label[0]],
            'hosmer_lemeshow': [np.nan],
            'comment': ['Only one rating grade.']
        })
    else:
        hl_result = _mc_sim_hl(
            pdc=pdc,
            odr=odr,
            no=no,
            nb=nb,
            alpha=alpha,
            sim_num=sim_num,
            seed=seed
        )
        hl_df = pd.DataFrame({
            'rating': [' + '.join(rating_label)],
            **hl_result
        })

    return PowerResult(
        interval_estimator=ie_df,
        hosmer_lemeshow=hl_df
    )


def _mc_sim_binom(
    pdc_r: float,
    odr_r: float,
    no_r: int,
    alpha: float,
    sim_num: int,
    seed: int
) -> Dict[str, float]:
    """
    Monte Carlo simulation for binomial-based tests.

    Parameters
    ----------
    pdc_r : float
        Calibrated PD for this rating.
    odr_r : float
        Observed default rate for this rating.
    no_r : int
        Number of observations.
    alpha : float
        Significance level.
    sim_num : int
        Number of simulations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Power estimates for binomial, Jeffreys, and z-score tests.
    """
    rng = np.random.default_rng(seed)

    res_eb = np.zeros(sim_num)
    res_jb = np.zeros(sim_num)
    res_zs = np.zeros(sim_num)

    for i in range(sim_num):
        # Simulate defaults under observed default rate
        def_sim = rng.binomial(n=1, p=odr_r, size=no_r).sum()

        # Binomial test
        res_eb[i] = 1 - stats.binom.cdf(def_sim - 1, no_r, pdc_r)

        # Jeffreys test
        res_jb[i] = stats.beta.cdf(pdc_r, def_sim + 0.5, no_r - def_sim + 0.5)

        # Z-score test
        se = np.sqrt((pdc_r * (1 - pdc_r)) / no_r) if pdc_r > 0 and pdc_r < 1 else 0
        if se > 0:
            test_stat = ((def_sim / no_r) - pdc_r) / se
            res_zs[i] = 1 - stats.norm.cdf(test_stat)
        else:
            res_zs[i] = 1.0

    return {
        'binomial': np.mean(res_eb < alpha),
        'jeffreys': np.mean(res_jb < alpha),
        'zscore': np.mean(res_zs < alpha)
    }


def _mc_sim_hl(
    pdc: np.ndarray,
    odr: np.ndarray,
    no: np.ndarray,
    nb: np.ndarray,
    alpha: float,
    sim_num: int,
    seed: int
) -> Dict[str, Union[float, str]]:
    """
    Monte Carlo simulation for Hosmer-Lemeshow test.

    Parameters
    ----------
    pdc : np.ndarray
        Calibrated PDs.
    odr : np.ndarray
        Observed default rates.
    no : np.ndarray
        Number of observations per rating.
    nb : np.ndarray
        Number of defaults per rating.
    alpha : float
        Significance level.
    sim_num : int
        Number of simulations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Power estimate for Hosmer-Lemeshow test or comment if not applicable.
    """
    # Check portfolio-level rates
    port_odr = np.sum(odr * no) / np.sum(no)
    port_pdc = np.sum(pdc * no) / np.sum(no)

    if port_odr <= port_pdc:
        return {'hosmer_lemeshow': np.nan, 'comment': 'ODR <= PDC'}

    res_hl = np.zeros(sim_num)
    k = len(no)

    for i in range(sim_num):
        # Simulate defaults under observed rates
        nb_sim = _rbin_aux(no, nb, seed + i)

        # Calculate HL statistic
        with np.errstate(divide='ignore', invalid='ignore'):
            hl = np.nansum((nb_sim - no * pdc) ** 2 / (no * pdc * (1 - pdc)))

        # P-value
        res_hl[i] = 1 - stats.chi2.cdf(hl, df=k)

    return {'hosmer_lemeshow': np.mean(res_hl < alpha)}


def _rbin_aux(
    no: np.ndarray,
    nb: np.ndarray,
    seed: int
) -> np.ndarray:
    """
    Generate simulated defaults for each rating.

    Parameters
    ----------
    no : np.ndarray
        Number of observations per rating.
    nb : np.ndarray
        Number of defaults per rating.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Simulated number of defaults per rating.
    """
    rng = np.random.default_rng(seed)
    def_sim = np.zeros(len(no))

    for i in range(len(no)):
        prob = nb[i] / no[i] if no[i] > 0 else 0
        def_sim[i] = rng.binomial(n=1, p=prob, size=int(no[i])).sum()

    return def_sim
