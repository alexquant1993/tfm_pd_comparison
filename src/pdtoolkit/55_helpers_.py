"""Helper functions for variable encoding and slicing.

This module provides utility functions for encoding and transforming
variables during scorecard development.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def num_slice(
    x: Union[np.ndarray, pd.Series, list],
    mapping: pd.DataFrame,
    sc: Optional[List] = None,
    sc_r: Union[str, List[str]] = "SC"
) -> np.ndarray:
    """
    Slice numeric variable into bins based on mapping.

    This function transforms a numeric variable into binned categories
    based on a provided mapping table with bin boundaries.

    Parameters
    ----------
    x : array-like
        Numeric variable to transform.
    mapping : pd.DataFrame
        DataFrame with columns 'x_min' and 'x_max' defining bin boundaries.
    sc : list, optional
        Special case values to handle separately (default: [NA, NaN, Inf, -Inf]).
    sc_r : str or list, default "SC"
        Replacement value(s) for special cases.

    Returns
    -------
    np.ndarray
        Transformed variable with bin labels.

    Raises
    ------
    ValueError
        If x is not numeric or mapping is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> x = np.array([1, 5, 10, 15, np.nan])
    >>> mapping = pd.DataFrame({'x_min': [0, 5, 10], 'x_max': [5, 10, 20]})
    >>> num_slice(x, mapping)
    array(['01 (-inf,5)', '02 [5,10)', '03 [10,inf)', 'SC'], dtype=object)
    """
    # Set default special cases
    if sc is None:
        sc = [np.nan, np.inf, -np.inf]

    # Convert to numpy array
    if isinstance(x, (pd.Series, list)):
        x = np.array(x)

    # Validate inputs
    if not np.issubdtype(x.dtype, np.number):
        # Try to convert
        try:
            x = x.astype(float)
        except (ValueError, TypeError):
            raise ValueError("x has to be numeric vector.")

    if not isinstance(mapping, pd.DataFrame):
        raise ValueError("mapping is not a data frame.")

    # Check for required columns (support both underscore and dot notation)
    if 'x_min' in mapping.columns and 'x_max' in mapping.columns:
        x_min_col, x_max_col = 'x_min', 'x_max'
    elif 'x.min' in mapping.columns and 'x.max' in mapping.columns:
        x_min_col, x_max_col = 'x.min', 'x.max'
    else:
        raise ValueError("mapping data frame has to contain columns: x_min, x_max.")

    # Check for missing or infinite values in mapping
    if mapping[[x_min_col, x_max_col]].isna().any().any():
        raise ValueError("mapping data frame contains missing values.")

    # Handle sc_r recycling
    scl = len(sc)
    if isinstance(sc_r, str):
        sc_r = [sc_r] * scl
    elif len(sc_r) != scl:
        raise ValueError("sc and sc_r has to be of the same length unless sc_r is of length one.")

    # Initialize result
    x_trans = np.empty(len(x), dtype=object)

    # Handle special cases first
    for i, sc_val in enumerate(sc):
        if pd.isna(sc_val):
            mask = pd.isna(x)
        elif np.isinf(sc_val):
            mask = x == sc_val
        else:
            mask = x == sc_val
        x_trans[mask] = sc_r[i]

    # Get bin boundaries (convert to float to allow -inf)
    x_lb = mapping[x_min_col].values.astype(float).copy()
    x_ub = mapping[x_max_col].values.astype(float).copy()

    # First lower bound is -Inf
    x_lb[0] = -np.inf

    # Create lagged lower bounds for upper limits
    x_lb_lag = np.concatenate([x_lb[1:], [np.inf]])

    # Create bin labels
    lg = len(mapping)
    for i in range(lg):
        lb = x_lb[i]
        lb_lag = x_lb_lag[i]
        ub = x_ub[i]

        bin_n = f"{i+1:02d}"

        # Format bin label
        if abs(lb - ub) < 1e-8:
            bin_f = f"{bin_n} [{round(lb, 4)}]"
        elif lb == -np.inf:
            bin_f = f"{bin_n} (-inf,{round(lb_lag, 4)})"
        else:
            bin_f = f"{bin_n} [{round(lb, 4)},{round(lb_lag, 4)})"

        # Find indices not in special cases and within bounds
        not_sc = np.array([not any(
            (pd.isna(x[j]) and pd.isna(sc_val)) or x[j] == sc_val
            for sc_val in sc
        ) for j in range(len(x))])

        in_bounds = (x >= lb) & (x <= lb_lag)
        rep_idx = not_sc & in_bounds

        x_trans[rep_idx] = bin_f

    return x_trans


def cat_slice(
    x: Union[np.ndarray, pd.Series, list],
    mapping: pd.DataFrame,
    sc: Optional[List] = None,
    sc_r: Union[str, List[str]] = "SC"
) -> np.ndarray:
    """
    Slice categorical variable based on mapping.

    This function transforms a categorical variable into new categories
    based on a provided mapping table.

    Parameters
    ----------
    x : array-like
        Categorical variable to transform (character, factor, or logical).
    mapping : pd.DataFrame
        DataFrame with columns 'x_orig' (original values) and 'x_mapp' (mapped values).
    sc : list, optional
        Special case values to handle separately (default: [NA]).
    sc_r : str or list, default "SC"
        Replacement value(s) for special cases.

    Returns
    -------
    np.ndarray
        Transformed variable with mapped labels.

    Raises
    ------
    ValueError
        If x class is inappropriate or mapping is invalid.

    Examples
    --------
    >>> x = np.array(['A', 'B', 'C', 'A'])
    >>> mapping = pd.DataFrame({'x_orig': ['A', 'B', 'C'], 'x_mapp': ['01 Low', '02 Med', '03 High']})
    >>> cat_slice(x, mapping)
    array(['01 Low', '02 Med', '03 High', '01 Low'], dtype=object)
    """
    # Set default special cases
    if sc is None:
        sc = [np.nan]

    # Convert to numpy array
    if isinstance(x, (pd.Series, list)):
        x = np.array(x)

    # Validate x type
    if x.dtype == np.float64 or x.dtype == np.int64:
        # Allow numeric for categorical encoding
        x = x.astype(str)

    if not isinstance(mapping, pd.DataFrame):
        raise ValueError("mapping is not a data frame.")

    # Check for required columns (support both underscore and dot notation)
    if 'x_orig' in mapping.columns and 'x_mapp' in mapping.columns:
        orig_col, mapp_col = 'x_orig', 'x_mapp'
    elif 'x.orig' in mapping.columns and 'x.mapp' in mapping.columns:
        orig_col, mapp_col = 'x.orig', 'x.mapp'
    else:
        raise ValueError("mapping data frame has to contain columns: x_orig, x_mapp.")

    # Check that all unique values are in mapping or sc
    ux = np.unique(x[~pd.isna(x)])
    mapping_vals = mapping[orig_col].values
    missing_vals = [v for v in ux if v not in mapping_vals and v not in sc]
    if missing_vals:
        raise ValueError(f"x contains values not reported in mapping (x_orig): {', '.join(map(str, missing_vals))}.")

    # Handle sc_r recycling
    scl = len(sc)
    if isinstance(sc_r, str):
        sc_r = [sc_r] * scl
    elif len(sc_r) != scl:
        raise ValueError("sc and sc_r has to be of the same length unless sc_r is of length one.")

    # Create mapping dictionary
    nv = dict(zip(mapping[orig_col].astype(str), mapping[mapp_col]))

    # Apply mapping
    x_trans = np.array([nv.get(str(v), None) if not pd.isna(v) else None for v in x], dtype=object)

    # Handle special cases
    for i, sc_val in enumerate(sc):
        if pd.isna(sc_val):
            mask = pd.isna(x) | pd.isna(x_trans)
        else:
            mask = x == sc_val
        x_trans[mask] = sc_r[i]

    return x_trans


def encode_woe(
    x: Union[np.ndarray, pd.Series, list],
    mapping: pd.DataFrame
) -> np.ndarray:
    """
    Encode categorical variable with WoE values.

    This function replaces categorical values with their corresponding
    Weight of Evidence (WoE) values based on a mapping table.

    Parameters
    ----------
    x : array-like
        Categorical variable to transform (character, factor, or logical).
    mapping : pd.DataFrame
        DataFrame with columns 'x_mod' (modality) and 'x_woe' (WoE value).

    Returns
    -------
    np.ndarray
        Numeric array with WoE encoded values.

    Raises
    ------
    ValueError
        If x class is inappropriate or mapping is invalid.

    Examples
    --------
    >>> x = np.array(['A', 'B', 'C', 'A'])
    >>> mapping = pd.DataFrame({'x_mod': ['A', 'B', 'C'], 'x_woe': [0.5, -0.2, 0.1]})
    >>> encode_woe(x, mapping)
    array([ 0.5, -0.2,  0.1,  0.5])
    """
    # Convert to numpy array
    if isinstance(x, (pd.Series, list)):
        x = np.array(x)

    if not isinstance(mapping, pd.DataFrame):
        raise ValueError("mapping is not a data frame.")

    # Check for required columns (support both underscore and dot notation)
    if 'x_mod' in mapping.columns and 'x_woe' in mapping.columns:
        mod_col, woe_col = 'x_mod', 'x_woe'
    elif 'x.mod' in mapping.columns and 'x.woe' in mapping.columns:
        mod_col, woe_col = 'x.mod', 'x.woe'
    else:
        raise ValueError("mapping data frame has to contain columns: x_mod, x_woe.")

    # Check that all unique values are in mapping
    ux = np.unique(x[~pd.isna(x)])
    mapping_vals = mapping[mod_col].values
    # Handle NA in mapping
    mapping_vals_str = [str(v) if not pd.isna(v) else None for v in mapping_vals]

    missing_vals = [v for v in ux if str(v) not in mapping_vals_str and not pd.isna(v)]
    if missing_vals and not any(pd.isna(mapping_vals)):
        raise ValueError(f"x contains values not reported in mapping (x_mod): {', '.join(map(str, missing_vals))}.")

    # Create mapping dictionary
    nv = {}
    na_woe = None
    for mod, woe in zip(mapping[mod_col], mapping[woe_col]):
        if pd.isna(mod):
            na_woe = woe
        else:
            nv[str(mod)] = woe

    # Apply mapping
    x_trans = np.array([nv.get(str(v), np.nan) if not pd.isna(v) else np.nan for v in x], dtype=float)

    # Handle NA values if NA is in mapping
    if na_woe is not None:
        na_idx = pd.isna(x_trans)
        x_trans[na_idx] = na_woe

    return x_trans
