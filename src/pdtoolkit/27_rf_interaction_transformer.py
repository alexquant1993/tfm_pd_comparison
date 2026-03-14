"""Random Forest Interaction Transformer.

This module provides functionality to extract interactions using
a random forest of decision trees.
"""

from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from importlib import import_module


@dataclass
class RFInteractionResult:
    """Result of random forest interaction transformer.

    Attributes
    ----------
    tree_info : pd.DataFrame
        Information about all trees including splits and leaf statistics.
    interaction : pd.DataFrame
        Interaction variables from each tree.
    """
    tree_info: pd.DataFrame
    interaction: pd.DataFrame


def rf_interaction_transformer(
    db: pd.DataFrame,
    rf: List[str],
    target: str,
    num_rf: Optional[int] = None,
    num_tree: int = 10,
    min_pct_obs: float = 0.05,
    min_avg_rate: float = 0.01,
    max_depth: int = 2,
    create_interaction_rf: bool = True,
    seed: int = 991
) -> RFInteractionResult:
    """
    Extract interactions using random forest of decision trees.

    This function builds multiple decision trees on bootstrap samples
    with random feature subsets, then extracts interaction patterns
    from the leaf assignments.

    Parameters
    ----------
    db : pd.DataFrame
        Data frame containing risk factors and target.
    rf : list of str
        Names of risk factors to use.
    target : str
        Name of the target variable (binary 0/1).
    num_rf : int, optional
        Number of features to consider at each tree.
        Default is sqrt(len(rf)).
    num_tree : int, default 10
        Number of trees to build.
    min_pct_obs : float, default 0.05
        Minimum percentage of observations in a leaf.
    min_avg_rate : float, default 0.01
        Minimum average rate threshold.
    max_depth : int, default 2
        Maximum depth of each tree.
    create_interaction_rf : bool, default True
        Whether to create interaction risk factor columns.
    seed : int, default 991
        Random seed for reproducibility.

    Returns
    -------
    RFInteractionResult
        Object containing tree_info and interaction DataFrames.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> db = pd.DataFrame({
    ...     'target': np.random.binomial(1, 0.1, n),
    ...     'rf1': np.random.randn(n),
    ...     'rf2': np.random.randn(n),
    ...     'rf3': np.random.randn(n)
    ... })
    >>> result = rf_interaction_transformer(
    ...     db=db,
    ...     rf=['rf1', 'rf2', 'rf3'],
    ...     target='target',
    ...     num_tree=5,
    ...     max_depth=2
    ... )
    >>> len(result.interaction.columns)
    5
    """
    # Validation
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db is not a data frame.")

    if target not in db.columns:
        raise ValueError(f"target '{target}' not found in db.")

    missing_rf = [r for r in rf if r not in db.columns]
    if missing_rf:
        raise ValueError(f"Risk factors not found in db: {missing_rf}")

    if num_tree < 1:
        raise ValueError("num_tree argument has to be single integer greater than 0.")

    num_tree = int(round(num_tree))
    rfl = len(rf)

    # Default num_rf
    if num_rf is None:
        num_rf = int(round(np.sqrt(rfl)))
    else:
        num_rf = min(num_rf, rfl)

    # Validate target is binary
    target_vals = db[target].dropna().unique()
    if not set(target_vals).issubset({0, 1}):
        raise ValueError("target must be binary 0/1.")

    nr_db = len(db)
    obs_ss = int(round(2 / 3 * nr_db))

    tree_info_list = []
    interaction_list = []

    # Calculate min_samples_leaf from min_pct_obs
    min_samples_leaf = max(1, int(round(min_pct_obs * obs_ss)))

    for i in range(num_tree):
        np.random.seed(seed + i)

        # Bootstrap sample
        indx = np.random.choice(nr_db, obs_ss, replace=True)

        # Random feature subset
        rf_subset = np.random.choice(rf, num_rf, replace=False).tolist()

        # Build tree on bootstrap sample
        db_sample = db.iloc[indx]

        X = db_sample[rf_subset].values
        y = db_sample[target].values

        # Handle missing values
        mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < min_samples_leaf * 2:
            continue

        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed + i
        )

        try:
            tree.fit(X_clean, y_clean)
        except Exception:
            continue

        # Get tree info
        tree_info_i = _extract_tree_info(tree, rf_subset, db_sample, target)
        tree_info_i['tree'] = i + 1
        tree_info_list.append(tree_info_i)

        # Create interaction if requested
        if create_interaction_rf:
            # Apply tree to full dataset
            X_full = db[rf_subset].values
            # Handle missing values for prediction
            mask_full = ~np.any(np.isnan(X_full), axis=1)

            interaction_col = pd.Series([None] * len(db), dtype=object)

            if np.any(mask_full):
                leaves = tree.apply(X_full[mask_full])
                unique_leaves = np.unique(leaves)
                leaf_map = {leaf: f'L{j+1}' for j, leaf in enumerate(unique_leaves)}
                interaction_col.iloc[np.where(mask_full)[0]] = [leaf_map[l] for l in leaves]

            interaction_list.append(
                pd.DataFrame({f'tree_{i+1}': interaction_col})
            )

    # Combine results
    if tree_info_list:
        tree_info = pd.concat(tree_info_list, ignore_index=True)
        # Reorder columns
        cols = ['tree'] + [c for c in tree_info.columns if c != 'tree']
        tree_info = tree_info[cols]
    else:
        tree_info = pd.DataFrame()

    if interaction_list and create_interaction_rf:
        interaction = pd.concat(interaction_list, axis=1)
    else:
        interaction = pd.DataFrame()

    return RFInteractionResult(
        tree_info=tree_info,
        interaction=interaction
    )


def _extract_tree_info(
    tree: DecisionTreeClassifier,
    rf_subset: List[str],
    db_sample: pd.DataFrame,
    target: str
) -> pd.DataFrame:
    """Extract information about tree leaves."""
    X = db_sample[rf_subset].values
    y = db_sample[target].values

    # Handle missing values
    mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
    X_clean = X[mask]
    y_clean = y[mask]

    if len(X_clean) == 0:
        return pd.DataFrame()

    # Get leaf assignments
    leaves = tree.apply(X_clean)
    unique_leaves = np.unique(leaves)

    results = []
    for j, leaf in enumerate(unique_leaves):
        leaf_mask = leaves == leaf
        n_obs = np.sum(leaf_mask)
        n_bad = np.sum(y_clean[leaf_mask])
        dr = n_bad / n_obs if n_obs > 0 else 0

        results.append({
            'leaf_id': f'L{j+1}',
            'n_obs': n_obs,
            'n_bad': n_bad,
            'dr': dr
        })

    return pd.DataFrame(results)
