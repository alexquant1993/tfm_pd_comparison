"""
Population Stability Index (PSI) module for PDtoolkit.

This module calculates PSI for comparing distributions between
base and target samples.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Tuple, List
from dataclasses import dataclass


@dataclass
class PSIResult:
    """Result of PSI calculation.

    Attributes:
        summary: DataFrame with PSI value and critical values
        table: DataFrame with bin-level details
    """
    summary: pd.DataFrame
    table: pd.DataFrame


def psi(
    base: Union[np.ndarray, pd.Series, list],
    target: Union[np.ndarray, pd.Series, list],
    bins: int = 10,
    alpha: float = 0.05
) -> PSIResult:
    """
    Calculate Population Stability Index (PSI).

    PSI measures the shift in distribution between a base (development)
    sample and a target (validation/monitoring) sample.

    Parameters
    ----------
    base : array-like
        Vector of values from base sample.
    target : array-like
        Vector of values from target sample.
    bins : int, default=10
        Number of bins for numeric variables.
    alpha : float, default=0.05
        Significance level for critical value calculation.

    Returns
    -------
    PSIResult
        Object containing PSI summary and bin-level table.

    Raises
    ------
    ValueError
        If inputs are invalid.

    References
    ----------
    Yurdakul, B. (2018). Statistical Properties of Population Stability Index.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> base = np.random.normal(0, 1, 1000)
    >>> target = np.random.normal(0.2, 1, 1000)
    >>> result = psi(base, target, bins=10)
    >>> print(f"PSI: {result.summary['psi'].iloc[0]:.4f}")
    """
    # Convert to numpy arrays
    base = np.asarray(base)
    target = np.asarray(target)

    # Check types match
    base_numeric = np.issubdtype(base.dtype, np.number)
    target_numeric = np.issubdtype(target.dtype, np.number)

    if base_numeric != target_numeric:
        raise ValueError("base and target are of different class.")

    # Remove NaN values
    base = base[~pd.isna(base)]
    target = target[~pd.isna(target)]

    if len(base) == 0 or len(target) == 0:
        raise ValueError("base or target of 0 length.")

    if not isinstance(bins, (int, np.integer)) or bins <= 1 or bins > 50:
        raise ValueError("bins has to be numeric value between 2 and 50.")

    if not isinstance(alpha, (int, float)) or alpha < 0 or alpha > 1:
        raise ValueError("alpha has to be numeric value between 0 and 1.")

    # Calculate bin table
    if base_numeric:
        tbl = _num_bt(base, target, bins)
    else:
        tbl = _cat_bt(base, target)

    # Fill NaN with 0
    for col in ['n_base', 'pct_base', 'n_target', 'pct_target']:
        if col in tbl.columns:
            tbl[col] = tbl[col].fillna(0)

    # Calculate PSI
    result = _psi_aux(tbl, alpha)

    return result


def _num_bt(
    base: np.ndarray,
    target: np.ndarray,
    bins: int
) -> pd.DataFrame:
    """
    Create bin table for numeric variables.

    Parameters
    ----------
    base : np.ndarray
        Base sample values.
    target : np.ndarray
        Target sample values.
    bins : int
        Number of bins.

    Returns
    -------
    pd.DataFrame
        Bin table with base and target distributions.
    """
    unique_base = np.unique(base)

    if len(unique_base) <= bins:
        # Use unique values as bins
        base_cut = base
        cut_points = np.sort(unique_base)
    else:
        # Create quantile-based bins
        percentiles = np.linspace(0, 100, bins + 1)
        cut_points = np.percentile(base, percentiles)
        cut_points = np.unique(cut_points)  # Remove duplicates

        # Assign bins
        base_cut = np.digitize(base, cut_points[1:-1])

    # Base table
    base_df = pd.DataFrame({'value': base, 'bin': base_cut if len(unique_base) > bins else base})

    if len(unique_base) <= bins:
        base_tbl = base_df.groupby('bin').agg(
            min_base=('value', 'min'),
            max_base=('value', 'max'),
            n_base=('value', 'count')
        ).reset_index()
    else:
        base_tbl = base_df.groupby('bin').agg(
            min_base=('value', 'min'),
            max_base=('value', 'max'),
            n_base=('value', 'count')
        ).reset_index()

    base_tbl['pct_base'] = base_tbl['n_base'] / base_tbl['n_base'].sum()

    # Create cut points for target
    if len(unique_base) <= bins:
        cut_points_target = np.concatenate([[-np.inf], np.sort(unique_base)[1:], [np.inf]])
    else:
        cut_points_target = np.concatenate([[-np.inf], cut_points[1:-1], [np.inf]])

    # Target binning
    target_cut = np.digitize(target, cut_points_target[1:-1])

    target_df = pd.DataFrame({'value': target, 'bin': target_cut})
    target_tbl = target_df.groupby('bin').agg(
        min_target=('value', 'min'),
        max_target=('value', 'max'),
        n_target=('value', 'count')
    ).reset_index()
    target_tbl['pct_target'] = target_tbl['n_target'] / target_tbl['n_target'].sum()

    # Merge
    bt = pd.merge(base_tbl, target_tbl, on='bin', how='left')

    # Set boundaries - convert to float first to avoid dtype warning
    if 'min_base' in bt.columns:
        bt['min_base'] = bt['min_base'].astype(float)
        bt.loc[bt.index[0], 'min_base'] = -np.inf
    if 'max_base' in bt.columns:
        bt['max_base'] = bt['max_base'].astype(float)
        bt.loc[bt.index[-1], 'max_base'] = np.inf

    return bt


def _cat_bt(
    base: np.ndarray,
    target: np.ndarray
) -> pd.DataFrame:
    """
    Create bin table for categorical variables.

    Parameters
    ----------
    base : np.ndarray
        Base sample values.
    target : np.ndarray
        Target sample values.

    Returns
    -------
    pd.DataFrame
        Bin table with base and target distributions.
    """
    # Base table
    base_counts = pd.Series(base).value_counts().reset_index()
    base_counts.columns = ['bin', 'n_base']
    base_counts['pct_base'] = base_counts['n_base'] / base_counts['n_base'].sum()

    # Target table
    target_counts = pd.Series(target).value_counts().reset_index()
    target_counts.columns = ['bin', 'n_target']
    target_counts['pct_target'] = target_counts['n_target'] / target_counts['n_target'].sum()

    # Merge
    bt = pd.merge(base_counts, target_counts, on='bin', how='outer')

    return bt


def _psi_aux(
    tbl: pd.DataFrame,
    alpha: float
) -> PSIResult:
    """
    Calculate PSI and critical values.

    Parameters
    ----------
    tbl : pd.DataFrame
        Bin table with distributions.
    alpha : float
        Significance level.

    Returns
    -------
    PSIResult
        PSI result with summary and table.
    """
    # Avoid log(0) by adding small value
    eps = 1e-10
    pct_base = np.maximum(tbl['pct_base'].values, eps)
    pct_target = np.maximum(tbl['pct_target'].values, eps)

    # Calculate PSI per bin
    tbl = tbl.copy()
    tbl['psi_b'] = (pct_base - pct_target) * np.log(pct_base / pct_target)

    # Calculate critical values
    ci = 1 - alpha
    n = tbl['n_base'].sum()
    m = tbl['n_target'].sum()
    b = len(tbl)

    # Z-score critical value
    cv_zscore = ((1/n + 1/m) * (b - 1) +
                 stats.norm.ppf(ci) * (1/n + 1/m) * np.sqrt(2 * (b - 1)))

    # Chi-square critical value
    cv_chisq = stats.chi2.ppf(ci, df=b - 1) * (1/n + 1/m)

    # PSI** (alternative formulation)
    psi_star = np.sum((pct_base - pct_target) ** 2 / pct_base)

    # Summary table
    summary = pd.DataFrame({
        'psi': [tbl['psi_b'].sum()],
        'cv_zscore': [cv_zscore],
        'psi_star': [psi_star],
        'cv_chisq': [cv_chisq],
        'ci': [ci]
    })

    return PSIResult(summary=summary, table=tbl)
