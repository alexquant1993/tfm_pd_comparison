"""
Segment validation module for PDtoolkit.

This module provides functionality for model segment validation based on residuals.
The main goal is to identify segments where model overestimates or underestimates
the observed default rate.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeRegressor


@dataclass
class SegmentValidationResult:
    """Result of segment validation.

    Attributes:
        segment_model: The fitted decision tree model
        segment_testing: DataFrame with segment overview and test results
        segment_rules: DataFrame with identification rules for each segment
    """
    segment_model: DecisionTreeRegressor
    segment_testing: pd.DataFrame
    segment_rules: Optional[pd.DataFrame]


def segment_vld(
    model,
    db: pd.DataFrame,
    target: str,
    predictors: List[str],
    min_leaf: float = 0.03,
    alpha: float = 0.05
) -> SegmentValidationResult:
    """
    Perform model segment validation based on residuals.

    This procedure identifies segments where the model in use overestimates
    or underestimates the observed default rate.

    Parameters
    ----------
    model : object or None
        Fitted model with predict_proba method, or None to use internal logistic regression.
    db : pd.DataFrame
        Data with risk factors and target variable.
    target : str
        Name of the target variable.
    predictors : list of str
        Names of predictor variables used in the model.
    min_leaf : float, default=0.03
        Minimum percentage of observations per leaf.
    alpha : float, default=0.05
        Significance level for one proportion test.

    Returns
    -------
    SegmentValidationResult
        Object containing:
        - segment_model: The fitted decision tree model
        - segment_testing: DataFrame with segment overview
        - segment_rules: DataFrame with identification rules

    Raises
    ------
    ValueError
        If no additional risk factors are available for analysis.
    """
    if not isinstance(min_leaf, (int, float)) or min_leaf <= 0 or min_leaf >= 1:
        raise ValueError("min_leaf has to be a numeric value between 0 and 1.")

    if not isinstance(alpha, (int, float)) or alpha <= 0 or alpha >= 1:
        raise ValueError("alpha has to be a numeric value between 0 and 1.")

    # Get non-model columns for tree building
    model_vars = [target] + list(predictors)
    tree_vars = [col for col in db.columns if col not in model_vars]

    if len(tree_vars) == 0:
        raise ValueError("No additional risk factors for analysis.")

    # Get predictions
    if model is None:
        # Use internal logistic regression
        from importlib import import_module
        sm = import_module('statsmodels.api')

        X = sm.add_constant(db[predictors].values)
        y = db[target].values

        try:
            logit = sm.Logit(y, X).fit(disp=0)
            mpred = logit.predict(X)
        except Exception:
            # Fallback to simple proportion
            mpred = np.full(len(db), y.mean())
    else:
        if hasattr(model, 'predict_proba'):
            mpred = model.predict_proba(db[predictors])[:, 1]
        elif hasattr(model, 'predict'):
            mpred = model.predict(db[predictors])
        else:
            raise ValueError("Model must have predict_proba or predict method.")

    # Calculate residuals
    db_work = db.copy()
    db_work['mpred'] = mpred
    db_work['error'] = db_work[target] - db_work['mpred']

    # Calculate minimum leaf size
    min_leaf_n = int(round(min_leaf * len(db)))
    min_leaf_n = max(min_leaf_n, 30)

    # Prepare tree features - convert categorical to numeric
    tree_features = db_work[tree_vars].copy()
    for col in tree_features.columns:
        if tree_features[col].dtype == 'object' or isinstance(tree_features[col].dtype, pd.CategoricalDtype):
            tree_features[col] = pd.Categorical(tree_features[col]).codes

    # Fit regression tree on residuals
    reg_tree = DecisionTreeRegressor(
        min_samples_split=min_leaf_n,
        min_samples_leaf=min_leaf_n,
        random_state=42
    )

    try:
        reg_tree.fit(tree_features, db_work['error'])
    except Exception as e:
        # No valid split found
        seg_overview = pd.DataFrame({'info': ['No significant split of residuals.']})
        return SegmentValidationResult(
            segment_model=reg_tree,
            segment_testing=seg_overview,
            segment_rules=None
        )

    # Check if tree has splits
    if reg_tree.tree_.node_count == 1:
        seg_overview = pd.DataFrame({'info': ['No significant split of residuals.']})
        return SegmentValidationResult(
            segment_model=reg_tree,
            segment_testing=seg_overview,
            segment_rules=None
        )

    # Extract rules and segment assignments
    tree_rules = _extract_rules(reg_tree, tree_vars)

    if tree_rules is None or len(tree_rules) <= 1:
        seg_overview = pd.DataFrame({'info': ['No significant split of residuals.']})
        return SegmentValidationResult(
            segment_model=reg_tree,
            segment_testing=seg_overview,
            segment_rules=None
        )

    # Assign segments
    db_work['segment'] = reg_tree.apply(tree_features)

    # Calculate segment overview
    seg_overview = db_work.groupby('segment').agg(
        no=('error', 'count'),
        ng_obs=(target, lambda x: (1 - x).sum()),
        ng_mod=('mpred', lambda x: (1 - x).sum()),
        nb_obs=(target, 'sum'),
        nb_mod=('mpred', 'sum')
    ).reset_index()

    seg_overview['dr_obs'] = seg_overview['nb_obs'] / seg_overview['no']
    seg_overview['dr_mod'] = seg_overview['nb_mod'] / seg_overview['no']
    seg_overview['dr_diff'] = seg_overview['dr_mod'] - seg_overview['dr_obs']

    # Adjust dr_obs for edge cases (avoid 0 or 1 in prop test)
    seg_overview['dr_obs_adj'] = seg_overview['dr_obs'].apply(
        lambda x: 0.00001 if round(x, 5) == 0 else (0.99999 if round(x, 5) == 1 else x)
    )

    # Perform proportion tests
    p_values = []
    test_results = []

    for _, row in seg_overview.iterrows():
        n = int(row['no'])
        nb_mod = row['nb_mod']
        nb_obs = row['nb_obs']
        dr_obs_adj = row['dr_obs_adj']

        # Determine alternative hypothesis
        if nb_mod <= nb_obs:
            alternative = 'less'
        else:
            alternative = 'greater'

        try:
            # Use binomial test as approximation
            # prop.test in R with correct=FALSE is chi-square based
            # For simplicity, use z-test approximation
            p_hat = nb_mod / n
            p0 = dr_obs_adj
            se = np.sqrt(p0 * (1 - p0) / n)

            if se > 0:
                z = (p_hat - p0) / se
                if alternative == 'less':
                    p_val = stats.norm.cdf(z)
                else:
                    p_val = 1 - stats.norm.cdf(z)
            else:
                p_val = 1.0
        except Exception:
            p_val = 1.0

        p_values.append(p_val)

        # Determine test result
        if p_val < alpha:
            if nb_mod > nb_obs:
                test_results.append('overestimate')
            else:
                test_results.append('underestimate')
        else:
            test_results.append('equal')

    seg_overview['p_val'] = p_values
    seg_overview['alpha'] = alpha
    seg_overview['test_res'] = test_results

    # Drop adjusted column
    seg_overview = seg_overview.drop(columns=['dr_obs_adj'])

    return SegmentValidationResult(
        segment_model=reg_tree,
        segment_testing=seg_overview,
        segment_rules=tree_rules
    )


def _extract_rules(tree: DecisionTreeRegressor, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """
    Extract decision rules from a fitted decision tree.

    Parameters
    ----------
    tree : DecisionTreeRegressor
        Fitted decision tree model.
    feature_names : list of str
        Names of features used in the tree.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with rules for each leaf node, or None if only root node.
    """
    tree_ = tree.tree_

    # Get leaf nodes
    leaf_ids = np.where(tree_.feature == -2)[0]

    if len(leaf_ids) <= 1:
        return None

    rules = []
    avgs = []

    for leaf_id in leaf_ids:
        # Get path to leaf
        path = _get_path_to_node(tree_, leaf_id)
        rule_parts = []

        for node_id, direction in path:
            if node_id == leaf_id:
                continue

            feature_idx = tree_.feature[node_id]
            threshold = tree_.threshold[node_id]
            feature_name = feature_names[feature_idx]

            if direction == 'left':
                rule_parts.append(f"{feature_name} <= {threshold:.4f}")
            else:
                rule_parts.append(f"{feature_name} > {threshold:.4f}")

        rule = " & ".join(rule_parts) if rule_parts else "root"
        avg = tree_.value[leaf_id][0][0]

        rules.append(rule)
        avgs.append(avg)

    return pd.DataFrame({
        'avg': avgs,
        'rule': rules
    })


def _get_path_to_node(tree_, target_node: int) -> List[tuple]:
    """
    Get the path from root to a target node.

    Parameters
    ----------
    tree_ : Tree object
        The tree structure from sklearn.
    target_node : int
        Index of the target node.

    Returns
    -------
    list of tuples
        List of (node_id, direction) tuples representing the path.
    """
    path = []

    def recurse(node_id):
        if node_id == target_node:
            path.append((node_id, None))
            return True

        left_child = tree_.children_left[node_id]
        right_child = tree_.children_right[node_id]

        if left_child != -1:
            if recurse(left_child):
                path.insert(0, (node_id, 'left'))
                return True

        if right_child != -1:
            if recurse(right_child):
                path.insert(0, (node_id, 'right'))
                return True

        return False

    recurse(0)
    return path
