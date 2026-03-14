"""
Heterogeneity testing module for PDtoolkit.

This module tests whether adjacent rating grades have properly
ordered default rates (monotonicity of default rates).
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional
import warnings


def heterogeneity(
    app_port: pd.DataFrame,
    def_ind: str,
    rating: str,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Test heterogeneity of PD rating model.

    Performs testing of adjacent rating grades to verify that default rates
    follow the expected order (higher risk ratings have higher default rates).

    Parameters
    ----------
    app_port : pd.DataFrame
        Application portfolio containing default indicator and ratings.
    def_ind : str
        Name of column representing observed default indicator (0/1).
    rating : str
        Name of column representing rating grades.
    alpha : float, default=0.05
        Significance level for two proportion test.

    Returns
    -------
    pd.DataFrame
        DataFrame with rating summary, p-values and test results for each
        adjacent rating pair comparison.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> app_port = pd.DataFrame({
    ...     'default': [0, 0, 1, 0, 1, 1],
    ...     'rating': ['A', 'A', 'B', 'B', 'C', 'C']
    ... })
    >>> result = heterogeneity(app_port, 'default', 'rating')
    """
    # Validate inputs
    if not isinstance(app_port, pd.DataFrame):
        raise ValueError("app_port is not a data frame.")

    if def_ind not in app_port.columns or rating not in app_port.columns:
        raise ValueError(
            "def_ind and/or rating do not exist in supplied app_port data frame."
        )

    if alpha < 0 or alpha > 1:
        raise ValueError("alpha has to be between 0 and 1.")

    # Check default indicator
    y = app_port[def_ind]
    valid_y = y[y.notna()]
    if not set(valid_y.unique()).issubset({0, 1}):
        raise ValueError("def_ind has to be 0/1 variable.")

    # Handle missing values
    cc = app_port[[def_ind, rating]].notna().all(axis=1)
    app_port_clean = app_port[cc].copy()

    if len(app_port_clean) == 0:
        raise ValueError("No complete cases for app_port.")

    if not cc.all():
        warnings.warn("There are some incomplete cases. Check def_ind and rating columns.")

    # Calculate rating summary
    rs = app_port_clean.groupby(rating).agg(
        no=(def_ind, 'count'),
        nb=(def_ind, 'sum')
    ).reset_index()
    rs.columns = ['rating', 'no', 'nb']
    rs['dr'] = rs['nb'] / rs['no']

    # Sort by rating
    rs = rs.sort_values('rating').reset_index(drop=True)

    # Determine correlation sign for test direction
    rating_numeric = pd.Categorical(app_port_clean[rating]).codes
    cor_s = np.corrcoef(rating_numeric, app_port_clean[def_ind])[0, 1]
    if np.isnan(cor_s):
        cor_s = 0

    sts = "less" if cor_s <= 0 else "greater"

    # Perform neighbor tests
    result = _t2p_neighbors(rs, sts, alpha)

    return result


def _t2p_neighbors(
    rs: pd.DataFrame,
    sts: str,
    alpha: float
) -> pd.DataFrame:
    """
    Two-proportion test for adjacent rating neighbors.

    Parameters
    ----------
    rs : pd.DataFrame
        Rating summary with columns: rating, no, nb, dr
    sts : str
        Test direction: "less" or "greater"
    alpha : float
        Significance level

    Returns
    -------
    pd.DataFrame
        Rating summary with p-values and test results
    """
    h0 = ">=" if sts == "less" else "<="
    h1 = "<" if sts == "less" else ">"

    rs = rs.copy()
    rs['p_val'] = np.nan
    rs['alpha'] = alpha
    rs['res'] = None

    for i in range(1, len(rs)):
        nb1 = int(rs.iloc[i - 1]['nb'])
        nb2 = int(rs.iloc[i]['nb'])
        no1 = int(rs.iloc[i - 1]['no'])
        no2 = int(rs.iloc[i]['no'])
        rc1 = rs.iloc[i - 1]['rating']
        rc2 = rs.iloc[i]['rating']

        # Perform two-proportion z-test
        try:
            # Calculate proportions
            p1 = nb1 / no1 if no1 > 0 else 0
            p2 = nb2 / no2 if no2 > 0 else 0

            # Pooled proportion
            p_pool = (nb1 + nb2) / (no1 + no2) if (no1 + no2) > 0 else 0

            # Standard error
            se = np.sqrt(p_pool * (1 - p_pool) * (1/no1 + 1/no2)) if p_pool > 0 and p_pool < 1 else 0

            if se > 0:
                z = (p2 - p1) / se
                if sts == "less":
                    p_val = stats.norm.cdf(z)
                else:
                    p_val = 1 - stats.norm.cdf(z)
            else:
                p_val = 1.0

        except Exception:
            p_val = np.nan

        rs.iloc[i, rs.columns.get_loc('p_val')] = p_val

        if pd.notna(p_val):
            if p_val >= alpha:
                res_str = f"H0: DR({rc2}) {h0} DR({rc1})"
            else:
                res_str = f"H1: DR({rc2}) {h1} DR({rc1})"
        else:
            res_str = None

        rs.iloc[i, rs.columns.get_loc('res')] = res_str

    return rs
