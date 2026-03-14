"""
Interaction transformer module for PDtoolkit.

This module extracts interactions between risk factors using
a decision tree algorithm.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier


@dataclass
class InteractionResult:
    """Result of interaction extraction.

    Attributes:
        tree_info: DataFrame with tree summary information
        interaction: DataFrame with interaction variable
    """
    tree_info: pd.DataFrame
    interaction: pd.DataFrame


def interaction_transformer(
    db: pd.DataFrame,
    target: str,
    risk_factors: List[str],
    min_pct_obs: float = 0.05,
    min_avg_rate: float = 0.01,
    max_depth: int = 3,
    seed: int = 42
) -> InteractionResult:
    """
    Extract interactions between risk factors using decision trees.

    This function builds a decision tree on the provided risk factors
    and extracts the leaf assignments as a new interaction variable.

    Parameters
    ----------
    db : pd.DataFrame
        Database containing target and risk factors.
    target : str
        Name of target variable (0/1).
    risk_factors : list of str
        Names of risk factors to use for interaction extraction.
    min_pct_obs : float, default=0.05
        Minimum percentage of observations per leaf.
    min_avg_rate : float, default=0.01
        Minimum default rate per leaf.
    max_depth : int, default=3
        Maximum depth of decision tree.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    InteractionResult
        Object containing tree info and interaction variable.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> db = pd.DataFrame({
    ...     'target': np.random.binomial(1, 0.1, 500),
    ...     'rf1': np.random.randn(500),
    ...     'rf2': np.random.randn(500)
    ... })
    >>> result = interaction_transformer(db, 'target', ['rf1', 'rf2'])
    """
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db is not a data frame.")

    if target not in db.columns:
        raise ValueError(f"Target '{target}' not found in db.")

    missing_rfs = [rf for rf in risk_factors if rf not in db.columns]
    if missing_rfs:
        raise ValueError(f"Risk factors not found: {missing_rfs}")

    if len(risk_factors) < 2:
        raise ValueError("At least 2 risk factors required for interaction.")

    # Check target is binary
    y = db[target].dropna()
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target must be 0/1 variable.")

    # Prepare data
    df_clean = db[[target] + risk_factors].dropna()
    X = df_clean[risk_factors]
    y = df_clean[target]

    # Handle categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Calculate min_samples_leaf
    min_samples_leaf = max(int(min_pct_obs * len(df_clean)), 30)

    # Build decision tree
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        criterion='gini'
    )

    tree.fit(X_encoded, y)

    # Get leaf assignments
    leaf_ids = tree.apply(X_encoded)

    # Create interaction variable
    interaction = pd.DataFrame(index=db.index)
    interaction['leaf'] = np.nan
    interaction.loc[df_clean.index, 'leaf'] = leaf_ids

    # Create leaf labels
    unique_leaves = np.unique(leaf_ids)
    leaf_map = {leaf: f"L{i+1}" for i, leaf in enumerate(unique_leaves)}
    interaction['interaction'] = interaction['leaf'].map(leaf_map)

    # Create tree info
    tree_info = _extract_tree_info(tree, X_encoded.columns.tolist(), y, leaf_ids)

    return InteractionResult(
        tree_info=tree_info,
        interaction=interaction[['interaction']]
    )


def _extract_tree_info(
    tree: DecisionTreeClassifier,
    feature_names: List[str],
    y: pd.Series,
    leaf_ids: np.ndarray
) -> pd.DataFrame:
    """
    Extract information about tree structure.

    Parameters
    ----------
    tree : DecisionTreeClassifier
        Fitted decision tree.
    feature_names : list of str
        Names of features.
    y : pd.Series
        Target variable.
    leaf_ids : np.ndarray
        Leaf assignments.

    Returns
    -------
    pd.DataFrame
        Tree information.
    """
    tree_ = tree.tree_
    unique_leaves = np.unique(leaf_ids)

    info_list = []

    for leaf_id in unique_leaves:
        mask = leaf_ids == leaf_id
        n_obs = mask.sum()
        n_bad = y[mask].sum()
        dr = n_bad / n_obs if n_obs > 0 else 0

        # Get path to leaf
        path_str = _get_leaf_path(tree_, leaf_id, feature_names)

        info_list.append({
            'leaf_id': leaf_id,
            'n_obs': n_obs,
            'n_bad': int(n_bad),
            'dr': dr,
            'path': path_str
        })

    return pd.DataFrame(info_list)


def _get_leaf_path(
    tree_,
    leaf_id: int,
    feature_names: List[str]
) -> str:
    """
    Get the decision path to a leaf node.

    Parameters
    ----------
    tree_ : Tree
        Tree structure.
    leaf_id : int
        Target leaf node ID.
    feature_names : list of str
        Feature names.

    Returns
    -------
    str
        String representation of the path.
    """
    path = []

    def recurse(node_id):
        if node_id == leaf_id:
            return True

        left_child = tree_.children_left[node_id]
        right_child = tree_.children_right[node_id]

        if left_child == -1:  # Leaf node
            return False

        feature_idx = tree_.feature[node_id]
        threshold = tree_.threshold[node_id]
        feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"f{feature_idx}"

        if recurse(left_child):
            path.insert(0, f"{feature_name} <= {threshold:.4f}")
            return True
        if recurse(right_child):
            path.insert(0, f"{feature_name} > {threshold:.4f}")
            return True

        return False

    recurse(0)
    return " & ".join(path) if path else "root"
