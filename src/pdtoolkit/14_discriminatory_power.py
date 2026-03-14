"""
Discriminatory power testing module for PDtoolkit.

This module tests whether the discriminatory power (AUC) of a model
has changed compared to a reference value (e.g., development sample).
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Literal
import warnings


def dp_testing(
    app_port: pd.DataFrame,
    def_ind: str,
    pdc: str,
    auc_test: float,
    alternative: Literal['less', 'greater', 'two.sided'] = 'two.sided',
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Test the discriminatory power of PD rating model.

    Tests whether the discriminatory power (AUC) of the model applied to
    an application portfolio differs from a reference AUC value.

    Parameters
    ----------
    app_port : pd.DataFrame
        Application portfolio containing default indicator and PD predictions.
    def_ind : str
        Name of column representing observed default indicator (0/1).
    pdc : str
        Name of column representing calibrated PD predictions.
    auc_test : float
        Reference AUC value to test against (usually from development sample).
    alternative : str, default='two.sided'
        Alternative hypothesis: 'less', 'greater', or 'two.sided'.
    alpha : float, default=0.05
        Significance level for hypothesis testing.

    Returns
    -------
    pd.DataFrame
        DataFrame with AUC values, test statistics, p-value and hypothesis result.

    Raises
    ------
    ValueError
        If inputs are invalid.

    References
    ----------
    Hanley J. and McNeil B. (1982). "The meaning and use of the area under a
    receiver operating characteristic (ROC) curve." Radiology 43(1): 29-36.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> app_port = pd.DataFrame({
    ...     'default': np.random.binomial(1, 0.1, 1000),
    ...     'pd': np.random.uniform(0, 0.3, 1000)
    ... })
    >>> result = dp_testing(app_port, 'default', 'pd', auc_test=0.70)
    """
    # Validate inputs
    if not isinstance(app_port, pd.DataFrame):
        raise ValueError("app_port (application portfolio) is not a data frame.")

    if def_ind not in app_port.columns or pdc not in app_port.columns:
        raise ValueError("def_ind and/or pdc columns cannot be found in app_port.")

    if alpha < 0 or alpha > 1:
        raise ValueError("alpha has to be between 0 and 1.")

    if auc_test < 0 or auc_test > 1:
        raise ValueError("auc_test has to be between 0 and 1.")

    alt_opts = ['less', 'greater', 'two.sided']
    if alternative not in alt_opts:
        raise ValueError(
            f"alternative argument must be: {', '.join(alt_opts)}."
        )

    # Check default indicator
    y = app_port[def_ind]
    valid_y = y[y.notna()]
    if not set(valid_y.unique()).issubset({0, 1}):
        raise ValueError("def_ind has to be 0/1 variable.")

    # Handle missing values
    cc = app_port[[def_ind, pdc]].notna().all(axis=1)
    app_port_clean = app_port[cc].copy()

    if len(app_port_clean) == 0:
        raise ValueError("No complete cases for app_port.")

    if not cc.all():
        warnings.warn("Incomplete cases found. Check def_ind and pdc columns.")

    y = app_port_clean[def_ind].values
    pd_vals = app_port_clean[pdc].values

    # Calculate AUC
    auc = _auc_model(pd_vals, y)

    # Calculate AUC standard error using Hanley-McNeil formula
    n1 = np.sum(y)  # Number of defaults
    n0 = np.sum(1 - y)  # Number of non-defaults

    if n1 == 0 or n0 == 0:
        raise ValueError("Need both default and non-default observations.")

    q1 = auc / (2 - auc)
    q2 = (2 * auc ** 2) / (1 + auc)

    auc_var = (auc * (1 - auc) +
               (n1 - 1) * (q1 - auc ** 2) +
               (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0)
    auc_se = np.sqrt(auc_var)

    # Calculate test statistic
    test_stat = (auc - auc_test) / auc_se if auc_se > 0 else 0

    # Calculate p-value
    if alternative == 'less':
        p_val = stats.norm.cdf(test_stat)
        h_sign = (' >= ', ' < ')
    elif alternative == 'greater':
        p_val = 1 - stats.norm.cdf(test_stat)
        h_sign = (' <= ', ' > ')
    else:  # two.sided
        p_val = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        h_sign = (' == ', ' != ')

    # Determine result
    if p_val >= alpha:
        res = f"H0: AUC{h_sign[0]}AUC test"
    else:
        res = f"H1: AUC{h_sign[1]}AUC test"

    result = pd.DataFrame({
        'auc': [auc],
        'auc_test': [auc_test],
        'estimate': [auc - auc_test],
        'auc_se': [auc_se],
        'test_stat': [test_stat],
        'p_val': [p_val],
        'alpha': [alpha],
        'res': [res]
    })

    return result


def _auc_model(predictions: np.ndarray, observed: np.ndarray) -> float:
    """
    Calculate AUC (Area Under the ROC Curve).

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions (probabilities).
    observed : np.ndarray
        Observed binary outcomes (0/1).

    Returns
    -------
    float
        AUC value.
    """
    # Handle missing values
    mask = ~(np.isnan(predictions) | np.isnan(observed))
    predictions = predictions[mask]
    observed = observed[mask]

    if len(predictions) == 0:
        return np.nan

    # Check for single class
    if len(np.unique(observed)) < 2:
        return np.nan

    # Rank-based AUC calculation (Mann-Whitney U statistic)
    pos = predictions[observed == 1]
    neg = predictions[observed == 0]

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    # Calculate AUC using Mann-Whitney
    n_pos = len(pos)
    n_neg = len(neg)

    # Count concordant pairs
    u = 0.0
    for p in pos:
        u += np.sum(neg < p) + 0.5 * np.sum(neg == p)

    auc = u / (n_pos * n_neg)

    return auc
