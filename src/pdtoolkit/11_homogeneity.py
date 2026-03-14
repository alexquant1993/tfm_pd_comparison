"""
Homogeneity testing module for PDtoolkit.

This module tests whether default rates are homogeneous across
segments within each rating grade.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Union
import warnings


def homogeneity(
    app_port: pd.DataFrame,
    def_ind: str,
    rating: str,
    segment: str,
    segment_num: int = 4,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Test homogeneity of PD rating model.

    Performs testing of homogeneity based on rating grades and selected segment.
    Tests whether default rates differ significantly between segment modalities
    within each rating.

    Parameters
    ----------
    app_port : pd.DataFrame
        Application portfolio containing default indicator, ratings, and segment.
    def_ind : str
        Name of column representing observed default indicator (0/1).
    rating : str
        Name of column representing rating grades.
    segment : str
        Name of column for testing segments.
    segment_num : int, default=4
        Number of groups for numeric segment variables.
    alpha : float, default=0.05
        Significance level for two proportion test.

    Returns
    -------
    pd.DataFrame
        DataFrame with segment analysis results including p-values and
        hypothesis test outcomes.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> app_port = pd.DataFrame({
    ...     'default': [0, 0, 1, 0, 1, 1, 0, 1],
    ...     'rating': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
    ...     'region': ['N', 'N', 'S', 'S', 'N', 'N', 'S', 'S']
    ... })
    >>> result = homogeneity(app_port, 'default', 'rating', 'region')
    """
    # Validate inputs
    if not isinstance(app_port, pd.DataFrame):
        raise ValueError("app_port is not a data frame.")

    required_cols = [def_ind, rating, segment]
    missing = [c for c in required_cols if c not in app_port.columns]
    if missing:
        raise ValueError(
            f"def_ind and/or rating and/or segment do not exist in supplied app_port data frame."
        )

    if alpha < 0 or alpha > 1:
        raise ValueError("alpha has to be between 0 and 1.")

    # Check default indicator
    y = app_port[def_ind]
    valid_y = y[y.notna()]
    if not set(valid_y.unique()).issubset({0, 1}):
        raise ValueError("def_ind has to be 0/1 variable.")

    # Handle missing values
    cc = app_port[[def_ind, rating, segment]].notna().all(axis=1)
    app_port_clean = app_port[cc].copy()

    if len(app_port_clean) == 0:
        raise ValueError("No complete cases for app_port.")

    if not cc.all():
        warnings.warn(
            "There are some incomplete cases. Check def_ind, rating and segment columns."
        )

    # Process segment variable
    seg = app_port_clean[segment].copy()

    if pd.api.types.is_numeric_dtype(seg):
        if seg.nunique() > 4:
            seg = pd.cut(seg, bins=segment_num, include_lowest=True)

    # Get unique values
    ratings_data = app_port_clean[rating]
    defaults_data = app_port_clean[def_ind]
    rat_unique = sorted(ratings_data.unique())

    results = []

    for rat_g in rat_unique:
        mask = ratings_data == rat_g
        rat_defaults = defaults_data[mask]
        rat_segment = seg[mask]

        rat_result = _t2p(
            rat_g=rat_g,
            def_vals=rat_defaults,
            rat_s=rat_segment,
            alpha=alpha
        )

        results.append(rat_result)

    result = pd.concat(results, ignore_index=True)
    result.insert(0, 'segment_var', segment)

    return result


def _t2p(
    rat_g: str,
    def_vals: pd.Series,
    rat_s: pd.Series,
    alpha: float
) -> pd.DataFrame:
    """
    Two-proportion test for segment comparisons within a rating.

    Parameters
    ----------
    rat_g : str
        Rating grade label.
    def_vals : pd.Series
        Default indicator values for this rating.
    rat_s : pd.Series
        Segment values for this rating.
    alpha : float
        Significance level.

    Returns
    -------
    pd.DataFrame
        Test results for each segment modality.
    """
    su = sorted(rat_s.unique())
    results = []

    for su_l in su:
        mask = rat_s == su_l

        nb1 = int(def_vals[mask].sum())
        nb2 = int(def_vals[~mask].sum())
        no1 = int(mask.sum())
        no2 = int((~mask).sum())

        if no1 < 30 or no2 < 30:
            p_val = np.nan
            com = "Less than 30 observations."
        else:
            # Two-proportion z-test (two-sided)
            try:
                p1 = nb1 / no1 if no1 > 0 else 0
                p2 = nb2 / no2 if no2 > 0 else 0
                p_pool = (nb1 + nb2) / (no1 + no2)
                se = np.sqrt(p_pool * (1 - p_pool) * (1/no1 + 1/no2)) if 0 < p_pool < 1 else 0

                if se > 0:
                    z = (p1 - p2) / se
                    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                else:
                    p_val = 1.0
            except Exception:
                p_val = np.nan

            if pd.notna(p_val):
                if p_val >= alpha:
                    com = f"H0: DR({su_l}) == DR(rest)"
                else:
                    com = f"H1: DR({su_l}) != DR(rest)"
            else:
                com = "Test could not be performed."

        res_l = {
            'rating': rat_g,
            'segment_mod': su_l,
            'no': len(def_vals),
            'nb': int(def_vals.sum()),
            'no_segment': no1,
            'no_rest': no2,
            'nb_segment': nb1,
            'nb_rest': nb2,
            'p_val': p_val,
            'alpha': alpha,
            'res': com
        }
        results.append(res_l)

    return pd.DataFrame(results)
