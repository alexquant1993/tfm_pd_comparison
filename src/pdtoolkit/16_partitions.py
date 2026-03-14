"""
Partitions module for PDtoolkit.

This module creates partitions (nested dummy variables) from risk factors.
Partitions provide insight into the difference of log-odds between adjacent
risk factor bins.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PartitionsResult:
    """Result of partition creation.

    Attributes:
        partitions: DataFrame with nested dummy variables
        info: DataFrame with partition process details
    """
    partitions: pd.DataFrame
    info: pd.DataFrame


def create_partitions(db: pd.DataFrame) -> PartitionsResult:
    """
    Create partitions (nested dummy variables).

    Partitions are useful for logistic regression as they provide insight
    into the difference of log-odds of adjacent risk factor bins.

    Parameters
    ----------
    db : pd.DataFrame
        Data set of risk factors to be converted into partitions.

    Returns
    -------
    PartitionsResult
        Object containing partitions DataFrame and info DataFrame.

    Raises
    ------
    ValueError
        If db is not a DataFrame.

    References
    ----------
    Scallan, G. (2011). Class(ic) Scorecards: Selecting Characteristics
    and Attributes in Logistic Regression, Edinburgh Credit Scoring Conference.

    Examples
    --------
    >>> import pandas as pd
    >>> db = pd.DataFrame({
    ...     'grade': ['A', 'B', 'C', 'A', 'B', 'C'],
    ...     'region': ['N', 'N', 'S', 'S', 'N', 'S']
    ... })
    >>> result = create_partitions(db)
    """
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db is not a data frame.")

    rf_names = db.columns.tolist()
    partitions_list = []
    info_list = []

    for rf in rf_names:
        x = db[rf]
        result = _ndv(x, rf)

        if result['success']:
            partitions_list.append(result['db'])
            info_entry = result['info'].copy()
            info_entry.insert(0, 'rf', rf)
            info_list.append(info_entry)
        else:
            info_entry = pd.DataFrame({
                'rf': [rf],
                'info': [result['info']],
                'code': [result['code']]
            })
            info_list.append(info_entry)

    # Combine partitions
    if partitions_list:
        partitions = pd.concat(partitions_list, axis=1)
    else:
        partitions = pd.DataFrame()

    # Combine info
    info = pd.concat(info_list, ignore_index=True)

    return PartitionsResult(partitions=partitions, info=info)


def _ndv(x: pd.Series, rf_n: str) -> dict:
    """
    Create nested dummy variables for a single risk factor.

    Parameters
    ----------
    x : pd.Series
        Risk factor values.
    rf_n : str
        Risk factor name.

    Returns
    -------
    dict
        Dictionary with success flag, info, and optionally the dummy DataFrame.
    """
    # Get unique values (excluding NA)
    x_counts = x.value_counts(dropna=True)
    x_categories = sorted(x_counts.index.tolist())

    # Check minimum categories
    if len(x_categories) < 2:
        return {
            'success': False,
            'info': 'Less than 2 non-NA groups.',
            'code': 'terminal'
        }

    # Check maximum categories
    if len(x_categories) > 10:
        return {
            'success': False,
            'info': 'More than 10 groups.',
            'code': 'terminal'
        }

    # Check for small groups
    total = len(x.dropna())
    pct = x_counts / total
    small_groups = pct[pct < 0.05].index.tolist()

    if small_groups:
        info_str = f"Group(s) with less than 5% of obs.: {', '.join(map(str, small_groups))}"
        info = pd.DataFrame({'info': [info_str], 'code': ['warning']})
    else:
        info = pd.DataFrame({'info': ['Partitions created.'], 'code': ['success']})

    # Create nested dummies
    nd_list = []
    for i in range(1, len(x_categories)):
        level_l = x_categories[:i]
        dummy = x.apply(lambda v: np.nan if pd.isna(v) else (0 if v in level_l else 1))
        col_name = f"{rf_n}{x_categories[i]}"
        nd_list.append(pd.DataFrame({col_name: dummy}))

    nd_db = pd.concat(nd_list, axis=1)

    return {
        'success': True,
        'info': info,
        'db': nd_db
    }
