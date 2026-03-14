"""Customized decision tree for credit risk modeling.

This module provides a decision tree implementation optimized for
PD model development with monotonicity constraints.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DecisionTreeResult:
    """Result of decision tree fitting.

    Attributes
    ----------
    tree_info : pd.DataFrame
        Information about each node in the tree.
    leaf_assignments : np.ndarray
        Leaf assignment for each observation.
    leaf_stats : pd.DataFrame
        Statistics for each leaf node.
    """
    tree_info: pd.DataFrame
    leaf_assignments: np.ndarray
    leaf_stats: pd.DataFrame


def decision_tree(
    db: pd.DataFrame,
    target: str,
    risk_factors: List[str],
    min_pct_obs: float = 0.05,
    min_avg_rate: float = 0.01,
    p_value: float = 0.05,
    max_depth: int = 3,
    monotonicity: Optional[Dict[str, Literal['increasing', 'decreasing']]] = None
) -> DecisionTreeResult:
    """
    Fit a customized decision tree for PD modeling.

    This function builds a decision tree using statistical tests
    for splitting and optionally enforcing monotonicity constraints.

    Parameters
    ----------
    db : pd.DataFrame
        Data frame containing target and risk factors.
    target : str
        Name of the target variable (binary 0/1).
    risk_factors : list of str
        Names of risk factors to use for splitting.
    min_pct_obs : float, default 0.05
        Minimum percentage of observations in a node.
    min_avg_rate : float, default 0.01
        Minimum average default rate threshold.
    p_value : float, default 0.05
        Significance level for split tests.
    max_depth : int, default 3
        Maximum depth of the tree.
    monotonicity : dict, optional
        Dictionary mapping risk factors to monotonicity direction
        ('increasing' or 'decreasing').

    Returns
    -------
    DecisionTreeResult
        Object containing tree information and predictions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> db = pd.DataFrame({
    ...     'target': np.random.binomial(1, 0.1, n),
    ...     'rf1': np.random.randn(n),
    ...     'rf2': np.random.choice(['A', 'B', 'C'], n)
    ... })
    >>> result = decision_tree(
    ...     db=db,
    ...     target='target',
    ...     risk_factors=['rf1', 'rf2'],
    ...     max_depth=2
    ... )
    """
    # Validation
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db is not a data frame.")

    if target not in db.columns:
        raise ValueError(f"target '{target}' not found in db.")

    missing_rf = [rf for rf in risk_factors if rf not in db.columns]
    if missing_rf:
        raise ValueError(f"Risk factors not found: {missing_rf}")

    target_vals = db[target].dropna().unique()
    if not set(target_vals).issubset({0, 1}):
        raise ValueError("target must be binary 0/1.")

    if monotonicity is None:
        monotonicity = {}

    # Initialize
    n_obs = len(db)
    min_obs = int(min_pct_obs * n_obs)

    # Determine correlation direction for monotonicity
    rf_directions = {}
    for rf in risk_factors:
        if pd.api.types.is_numeric_dtype(db[rf]):
            corr = db[rf].corr(db[target])
            rf_directions[rf] = 'increasing' if corr > 0 else 'decreasing'

    # Build tree
    tree_nodes = []
    leaf_assignments = np.zeros(n_obs, dtype=int)

    # Start with root node
    root_node = {
        'node_id': 0,
        'parent_id': None,
        'depth': 0,
        'indices': np.arange(n_obs),
        'is_leaf': True,
        'split_var': None,
        'split_value': None,
        'n_obs': n_obs,
        'n_bad': int(db[target].sum()),
        'dr': db[target].mean()
    }

    nodes_to_process = [root_node]
    next_node_id = 1

    while nodes_to_process:
        node = nodes_to_process.pop(0)

        # Check if we should split
        if (node['depth'] >= max_depth or
            node['n_obs'] < 2 * min_obs or
            node['dr'] < min_avg_rate):
            node['is_leaf'] = True
            tree_nodes.append(node)
            continue

        # Find best split
        best_split = _find_best_split(
            db=db,
            target=target,
            risk_factors=risk_factors,
            indices=node['indices'],
            min_obs=min_obs,
            p_value=p_value,
            monotonicity=monotonicity,
            rf_directions=rf_directions
        )

        if best_split is None:
            node['is_leaf'] = True
            tree_nodes.append(node)
            continue

        # Create child nodes
        node['is_leaf'] = False
        node['split_var'] = best_split['var']
        node['split_value'] = best_split['value']

        left_indices = best_split['left_indices']
        right_indices = best_split['right_indices']

        left_node = {
            'node_id': next_node_id,
            'parent_id': node['node_id'],
            'depth': node['depth'] + 1,
            'indices': left_indices,
            'is_leaf': True,
            'split_var': None,
            'split_value': None,
            'n_obs': len(left_indices),
            'n_bad': int(db.iloc[left_indices][target].sum()),
            'dr': db.iloc[left_indices][target].mean()
        }
        next_node_id += 1

        right_node = {
            'node_id': next_node_id,
            'parent_id': node['node_id'],
            'depth': node['depth'] + 1,
            'indices': right_indices,
            'is_leaf': True,
            'split_var': None,
            'split_value': None,
            'n_obs': len(right_indices),
            'n_bad': int(db.iloc[right_indices][target].sum()),
            'dr': db.iloc[right_indices][target].mean()
        }
        next_node_id += 1

        tree_nodes.append(node)
        nodes_to_process.extend([left_node, right_node])

    # Process remaining nodes
    while nodes_to_process:
        node = nodes_to_process.pop(0)
        node['is_leaf'] = True
        tree_nodes.append(node)

    # Assign leaf IDs
    leaf_id = 0
    for node in tree_nodes:
        if node['is_leaf']:
            leaf_id += 1
            node['leaf_id'] = f'L{leaf_id}'
            leaf_assignments[node['indices']] = leaf_id
        else:
            node['leaf_id'] = None

    # Build tree info DataFrame
    tree_info = pd.DataFrame([
        {
            'node_id': n['node_id'],
            'parent_id': n['parent_id'],
            'depth': n['depth'],
            'is_leaf': n['is_leaf'],
            'leaf_id': n['leaf_id'],
            'split_var': n['split_var'],
            'split_value': n['split_value'],
            'n_obs': n['n_obs'],
            'n_bad': n['n_bad'],
            'dr': n['dr']
        }
        for n in tree_nodes
    ])

    # Build leaf stats
    leaf_nodes = [n for n in tree_nodes if n['is_leaf']]
    leaf_stats = pd.DataFrame([
        {
            'leaf_id': n['leaf_id'],
            'n_obs': n['n_obs'],
            'n_bad': n['n_bad'],
            'dr': n['dr']
        }
        for n in leaf_nodes
    ])

    return DecisionTreeResult(
        tree_info=tree_info,
        leaf_assignments=leaf_assignments,
        leaf_stats=leaf_stats
    )


def _find_best_split(
    db: pd.DataFrame,
    target: str,
    risk_factors: List[str],
    indices: np.ndarray,
    min_obs: int,
    p_value: float,
    monotonicity: Dict[str, str],
    rf_directions: Dict[str, str]
) -> Optional[dict]:
    """Find the best split for a node."""
    best_split = None
    best_stat = 0

    db_node = db.iloc[indices]
    y = db_node[target].values

    for rf in risk_factors:
        x = db_node[rf].values

        if pd.api.types.is_numeric_dtype(db[rf]):
            split = _find_best_split_numeric(
                x, y, indices, min_obs, p_value,
                monotonicity.get(rf), rf_directions.get(rf)
            )
        else:
            split = _find_best_split_categorical(
                x, y, indices, min_obs, p_value
            )

        if split is not None and split['statistic'] > best_stat:
            best_stat = split['statistic']
            best_split = {
                'var': rf,
                'value': split['value'],
                'left_indices': split['left_indices'],
                'right_indices': split['right_indices'],
                'statistic': split['statistic']
            }

    return best_split


def _find_best_split_numeric(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    min_obs: int,
    p_value: float,
    monotonicity: Optional[str],
    direction: Optional[str]
) -> Optional[dict]:
    """Find best split for numeric variable."""
    # Remove missing values
    mask = ~np.isnan(x)
    x_clean = x[mask]
    y_clean = y[mask]
    indices_clean = indices[mask]

    if len(x_clean) < 2 * min_obs:
        return None

    # Get candidate split points
    percentiles = np.percentile(x_clean, np.linspace(10, 90, 9))
    percentiles = np.unique(percentiles)

    best_split = None
    best_stat = 0

    for threshold in percentiles:
        left_mask = x_clean <= threshold
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        if n_left < min_obs or n_right < min_obs:
            continue

        # Test of two proportions
        y_left = y_clean[left_mask]
        y_right = y_clean[right_mask]

        p_left = y_left.mean() if len(y_left) > 0 else 0
        p_right = y_right.mean() if len(y_right) > 0 else 0

        # Check monotonicity
        if monotonicity is not None:
            if monotonicity == 'increasing' and p_left > p_right:
                continue
            if monotonicity == 'decreasing' and p_left < p_right:
                continue

        # Z-test for difference in proportions
        stat, pval = _two_proportion_test(y_left, y_right)

        if pval < p_value and stat > best_stat:
            best_stat = stat
            best_split = {
                'value': threshold,
                'left_indices': indices_clean[left_mask],
                'right_indices': indices_clean[right_mask],
                'statistic': stat
            }

    return best_split


def _find_best_split_categorical(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    min_obs: int,
    p_value: float
) -> Optional[dict]:
    """Find best split for categorical variable."""
    # Get categories
    categories = np.unique(x[~pd.isna(x)])

    if len(categories) < 2:
        return None

    best_split = None
    best_stat = 0

    # Try binary splits
    for cat in categories:
        left_mask = x == cat
        right_mask = ~left_mask & ~pd.isna(x)

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        if n_left < min_obs or n_right < min_obs:
            continue

        y_left = y[left_mask]
        y_right = y[right_mask]

        stat, pval = _two_proportion_test(y_left, y_right)

        if pval < p_value and stat > best_stat:
            best_stat = stat
            best_split = {
                'value': cat,
                'left_indices': indices[left_mask],
                'right_indices': indices[right_mask],
                'statistic': stat
            }

    return best_split


def _two_proportion_test(y1: np.ndarray, y2: np.ndarray) -> Tuple[float, float]:
    """Perform z-test for difference in two proportions."""
    n1, n2 = len(y1), len(y2)
    if n1 == 0 or n2 == 0:
        return 0, 1.0

    p1 = y1.mean()
    p2 = y2.mean()

    # Pooled proportion
    p_pooled = (y1.sum() + y2.sum()) / (n1 + n2)

    if p_pooled == 0 or p_pooled == 1:
        return 0, 1.0

    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

    if se == 0:
        return 0, 1.0

    # Z-statistic
    z = abs(p1 - p2) / se

    # P-value (two-sided)
    p_value = 2 * (1 - stats.norm.cdf(z))

    return z, p_value
