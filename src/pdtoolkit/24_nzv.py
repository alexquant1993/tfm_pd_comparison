"""Near-zero variance detection for risk factors.

This module provides functionality to detect risk factors with
near-zero variance, which may be problematic for modeling.
"""

from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import pandas as pd


def nzv(
    db: pd.DataFrame,
    sc: Optional[List[Union[float, str]]] = None
) -> pd.DataFrame:
    """
    Detect near-zero variance risk factors.

    This function analyzes each column in the database to identify
    risk factors with near-zero variance. Variables with very low
    variance or highly imbalanced distributions may cause problems
    in modeling and should be reviewed before inclusion.

    Parameters
    ----------
    db : pd.DataFrame
        Data frame containing risk factors to analyze.
    sc : list, optional
        Special case values to exclude from variance calculation.
        Default is [np.nan, np.inf, -np.inf].

    Returns
    -------
    pd.DataFrame
        Data frame with the following columns:
        - rf: Risk factor name
        - type: 'numeric' or 'categorical'
        - sc_num: Number of special case observations
        - sc_pct: Percentage of special cases
        - cc_num: Number of complete cases
        - cc_pct: Percentage of complete cases
        - cc_unv: Number of unique values in complete cases
        - cc_unv_pct: Percentage of unique values
        - cc_lbl_1: Most frequent value label
        - cc_frq_1: Frequency of most frequent value
        - cc_lbl_2: Second most frequent value label
        - cc_frq_2: Frequency of second most frequent value
        - cc_fqr: Frequency ratio (most frequent / second most frequent)
        - ind: Near-zero variance indicator (1 if cc_pct < 0.1 and cc_fqr > 19)

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> db = pd.DataFrame({
    ...     'rf1': [1, 2, 3, 4, 5],
    ...     'rf2': [1, 1, 1, 1, 2],  # Near-zero variance
    ...     'rf3': ['A', 'A', 'A', 'B', 'B']
    ... })
    >>> result = nzv(db)
    >>> result['rf'].tolist()
    ['rf1', 'rf2', 'rf3']
    """
    # Validation
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db is not a data frame.")

    # Default special cases
    if sc is None:
        sc = [np.nan, np.inf, -np.inf]

    # Convert special cases for numeric and categorical comparisons
    sc_num = set()
    sc_cat = set()
    for val in sc:
        if val is np.nan or (isinstance(val, float) and np.isnan(val)):
            sc_num.add(np.nan)
            sc_cat.add('nan')
            sc_cat.add('NaN')
        elif val is None:
            sc_cat.add(None)
            sc_cat.add('None')
        elif isinstance(val, (int, float)):
            sc_num.add(float(val))
            sc_cat.add(str(val))
        else:
            sc_cat.add(str(val))

    results = []

    for rf_name in db.columns:
        x = db[rf_name]
        is_numeric = pd.api.types.is_numeric_dtype(x)

        if is_numeric:
            # For numeric columns, identify special cases
            is_sc = x.isna()
            for val in sc_num:
                if not np.isnan(val):
                    is_sc = is_sc | (x == val)
                else:
                    is_sc = is_sc | x.isna()

            x_sc = x[is_sc]
            x_cc = x[~is_sc]
        else:
            # For categorical columns
            x_str = x.astype(str)
            is_sc = x.isna() | x_str.isin(sc_cat)

            x_sc = x[is_sc]
            x_cc = x[~is_sc]

        # Calculate statistics
        n_total = len(x)
        x_sc_num = len(x_sc)
        x_sc_pct = x_sc_num / n_total if n_total > 0 else 0
        x_cc_num = len(x_cc)
        x_cc_pct = 1 - x_sc_pct

        # Unique values in complete cases
        x_cc_unv = x_cc.nunique()
        x_cc_upc = x_cc_unv / x_cc_num if x_cc_num > 0 else 0

        # Value frequency table
        if x_cc_num > 0:
            value_counts = x_cc.value_counts().sort_values(ascending=False)

            if len(value_counts) > 1:
                x_cc_lb1 = str(value_counts.index[0])
                x_cc_lb2 = str(value_counts.index[1])
                x_cc_fq1 = value_counts.iloc[0]
                x_cc_fq2 = value_counts.iloc[1]
                x_cc_fqr = x_cc_fq1 / x_cc_fq2 if x_cc_fq2 > 0 else np.inf
            elif len(value_counts) == 1:
                x_cc_lb1 = str(value_counts.index[0])
                x_cc_lb2 = None
                x_cc_fq1 = value_counts.iloc[0]
                x_cc_fq2 = None
                x_cc_fqr = 1.0
            else:
                x_cc_lb1 = None
                x_cc_lb2 = None
                x_cc_fq1 = None
                x_cc_fq2 = None
                x_cc_fqr = 1.0
        else:
            x_cc_lb1 = None
            x_cc_lb2 = None
            x_cc_fq1 = None
            x_cc_fq2 = None
            x_cc_fqr = 1.0

        # Near-zero variance indicator
        # Flagged if complete cases < 10% AND frequency ratio > 19
        nzv_ind = 1 if (x_cc_pct < 0.1 and x_cc_fqr > 19) else 0

        results.append({
            'rf': rf_name,
            'type': 'numeric' if is_numeric else 'categorical',
            'sc_num': x_sc_num,
            'sc_pct': x_sc_pct,
            'cc_num': x_cc_num,
            'cc_pct': x_cc_pct,
            'cc_unv': x_cc_unv,
            'cc_unv_pct': x_cc_upc,
            'cc_lbl_1': x_cc_lb1,
            'cc_frq_1': x_cc_fq1,
            'cc_lbl_2': x_cc_lb2,
            'cc_frq_2': x_cc_fq2,
            'cc_fqr': x_cc_fqr,
            'ind': nzv_ind
        })

    return pd.DataFrame(results)
