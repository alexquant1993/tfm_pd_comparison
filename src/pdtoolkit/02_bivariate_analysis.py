"""
Bivariate analysis module for PDtoolkit.

This module provides functions for bivariate statistical analysis of risk factors,
including WoE (Weight of Evidence) calculations, Information Value, and AUC.

Ported from: 02_BIVARIATE_ANALYSIS.R
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score


def bivariate(
    db: pd.DataFrame,
    target: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform bivariate analysis on categorical risk factors.

    This function expects all risk factors to be categorical. Numeric risk factors
    should be categorized first. Maximum number of groups per risk factor is 10.

    Bivariate report includes:
    - rf: Risk factor name
    - bin: Risk factor group (bin)
    - no: Number of observations per bin
    - ng: Number of good cases (target=0) per bin
    - nb: Number of bad cases (target=1) per bin
    - pct_o: Percentage of observations per bin
    - pct_g: Percentage of good cases per bin
    - pct_b: Percentage of bad cases per bin
    - dr: Default rate per bin
    - so: Total number of observations
    - sg: Total number of good cases
    - sb: Total number of bad cases
    - dist_g: Distribution of good cases per bin
    - dist_b: Distribution of bad cases per bin
    - woe: Weight of Evidence value
    - iv_b: Information value per bin
    - iv_s: Total Information value of risk factor
    - auc: Area under ROC curve

    Args:
        db: DataFrame with risk factors and target variable.
        target: Name of target variable (must be 0/1 binary).

    Returns:
        Tuple of (results DataFrame, info DataFrame with validation messages).

    Examples:
        >>> import pandas as pd
        >>> from pdtoolkit.bivariate_analysis import bivariate
        >>> df = pd.DataFrame({
        ...     'target': [0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
        ...     'category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C']
        ... })
        >>> results, info = bivariate(df, target='target')
    """
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db is not a DataFrame.")

    if target not in db.columns:
        raise ValueError("Target variable does not exist in supplied db.")

    y = db[target]
    y_valid = y.dropna()
    if not set(y_valid.unique()).issubset({0, 1}):
        raise ValueError("Target is not 0/1 variable.")

    rf = [col for col in db.columns if col != target]
    if len(rf) == 0:
        raise ValueError("There are no risk factors in supplied db.")

    results = []
    info = []

    for col in rf:
        x = db[col]

        # Check if categorical type
        is_categorical = (
            pd.api.types.is_string_dtype(x)
            or isinstance(x.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(x)
            or pd.api.types.is_object_dtype(x)
        )

        if not is_categorical:
            info.append({
                "rf": col,
                "reason_code": 1,
                "comment": "Inappropriate class. It has to be one of: character, factor or logical."
            })
            continue

        # Check number of categories
        if x.nunique() > 10:
            info.append({
                "rf": col,
                "reason_code": 2,
                "comment": "More than 10 categories."
            })
            continue

        # Calculate WoE table
        woe_result = woe_tbl(db, x=col, y=target)

        # Calculate AUC using logistic regression
        try:
            # Prepare data for logistic regression
            valid_mask = db[[col, target]].notna().all(axis=1)
            db_valid = db.loc[valid_mask].copy()

            if len(db_valid) < 10 or db_valid[target].nunique() < 2:
                auc = np.nan
            else:
                # Create dummy variables for the categorical predictor
                X = pd.get_dummies(db_valid[col], drop_first=True, dtype=float)
                X = sm.add_constant(X)
                y_model = db_valid[target].values

                # Fit logistic regression
                model = sm.Logit(y_model, X)
                result = model.fit(disp=0)
                predictions = result.predict(X)

                auc = auc_model(predictions=predictions.values, observed=y_model)
        except Exception:
            auc = np.nan

        woe_result["auc"] = auc
        woe_result.insert(0, "rf", col)
        results.append(woe_result)

    results_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    info_df = pd.DataFrame(info) if info else pd.DataFrame()

    return results_df, info_df


def woe_tbl(
    tbl: pd.DataFrame,
    x: str,
    y: str,
    y_check: bool = True,
) -> pd.DataFrame:
    """
    Calculate Weight of Evidence (WoE) table for a risk factor.

    WoE table includes:
    - bin: Risk factor group (bin)
    - no: Number of observations per bin
    - ng: Number of good cases (target=0) per bin
    - nb: Number of bad cases (target=1) per bin
    - pct_o: Percentage of observations per bin
    - pct_g: Percentage of good cases per bin
    - pct_b: Percentage of bad cases per bin
    - dr: Default rate per bin
    - so: Total number of observations
    - sg: Total number of good cases
    - sb: Total number of bad cases
    - dist_g: Distribution of good cases per bin
    - dist_b: Distribution of bad cases per bin
    - woe: Weight of Evidence value
    - iv_b: Information value per bin
    - iv_s: Total Information value

    Args:
        tbl: DataFrame containing target and risk factor.
        x: Name of risk factor column.
        y: Name of target variable column.
        y_check: Whether to validate target is 0/1. Default True.

    Returns:
        DataFrame with WoE calculations.

    Examples:
        >>> import pandas as pd
        >>> from pdtoolkit.bivariate_analysis import woe_tbl
        >>> df = pd.DataFrame({
        ...     'target': [0, 0, 1, 1, 0, 1],
        ...     'bin': ['A', 'A', 'A', 'B', 'B', 'B']
        ... })
        >>> woe = woe_tbl(df, x='bin', y='target')
    """
    if not isinstance(tbl, pd.DataFrame):
        raise ValueError("tbl is not a DataFrame.")

    if y not in tbl.columns:
        raise ValueError("y (target variable) does not exist in supplied tbl.")

    if x not in tbl.columns:
        raise ValueError("x (risk factor) does not exist in supplied tbl.")

    if not isinstance(y_check, bool):
        raise ValueError("y_check has to be of logical type.")

    target = tbl[y]

    if y_check:
        target_valid = target.dropna()
        if not set(target_valid.unique()).issubset({0, 1}):
            raise ValueError("y (target variable) is not 0/1 variable.")

    # Group by bin and calculate statistics
    grouped = tbl.groupby(x, dropna=False).agg(
        no=(y, 'count'),
        ng=(y, lambda vals: (vals == 0).sum()),
        nb=(y, lambda vals: (vals == 1).sum())
    ).reset_index()
    grouped.columns = ['bin', 'no', 'ng', 'nb']

    # Calculate totals
    so = grouped['no'].sum()
    sg = grouped['ng'].sum()
    sb = grouped['nb'].sum()

    # Calculate percentages and metrics
    grouped['pct_o'] = grouped['no'] / so if so > 0 else 0
    grouped['pct_g'] = grouped['ng'] / sg if sg > 0 else 0
    grouped['pct_b'] = grouped['nb'] / sb if sb > 0 else 0
    grouped['dr'] = grouped['nb'] / grouped['no']

    grouped['so'] = so
    grouped['sg'] = sg
    grouped['sb'] = sb

    grouped['dist_g'] = grouped['ng'] / sg if sg > 0 else 0
    grouped['dist_b'] = grouped['nb'] / sb if sb > 0 else 0

    # Calculate WoE - handle zeros by using small epsilon
    eps = 1e-10
    dist_g = np.maximum(grouped['dist_g'].values, eps)
    dist_b = np.maximum(grouped['dist_b'].values, eps)
    grouped['woe'] = np.log(dist_g / dist_b)

    # Information value per bin
    grouped['iv_b'] = (grouped['dist_g'] - grouped['dist_b']) * grouped['woe']

    # Total information value
    grouped['iv_s'] = grouped['iv_b'].sum()

    return grouped


def auc_model(
    predictions: np.ndarray,
    observed: np.ndarray,
) -> float:
    """
    Calculate Area Under the ROC Curve (AUC).

    Args:
        predictions: Model predicted probabilities.
        observed: Observed target values (0/1).

    Returns:
        AUC value.

    Examples:
        >>> import numpy as np
        >>> from pdtoolkit.bivariate_analysis import auc_model
        >>> predictions = np.array([0.1, 0.4, 0.35, 0.8])
        >>> observed = np.array([0, 0, 1, 1])
        >>> auc = auc_model(predictions, observed)
    """
    predictions = np.asarray(predictions)
    observed = np.asarray(observed)

    # Remove missing values
    valid_mask = ~(np.isnan(predictions) | np.isnan(observed))
    predictions = predictions[valid_mask]
    observed = observed[valid_mask]

    if len(predictions) == 0 or len(np.unique(observed)) < 2:
        return np.nan

    # Use sklearn's roc_auc_score for efficiency and accuracy
    return roc_auc_score(observed, predictions)


def replace_woe(
    db: pd.DataFrame,
    target: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replace risk factor modalities with Weight of Evidence (WoE) values.

    This function processes only categorical risk factors. Numeric risk factors
    should be categorized first.

    Info report includes:
    - rf: Risk factor name
    - reason_code: 1 = inappropriate class, 2 = >10 categories, 3 = WoE calculation problem
    - comment: Reason description

    Args:
        db: DataFrame of categorical risk factors and target variable.
        target: Name of target variable.

    Returns:
        Tuple of (WoE-encoded DataFrame, info DataFrame).

    Examples:
        >>> import pandas as pd
        >>> from pdtoolkit.bivariate_analysis import replace_woe
        >>> df = pd.DataFrame({
        ...     'target': [0, 0, 1, 1, 0, 1],
        ...     'category': ['A', 'A', 'A', 'B', 'B', 'B']
        ... })
        >>> encoded, info = replace_woe(df, target='target')
    """
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db is not a DataFrame.")

    if target not in db.columns:
        raise ValueError("Target variable does not exist in supplied db.")

    y = db[target]
    y_valid = y.dropna()
    if not set(y_valid.unique()).issubset({0, 1}):
        raise ValueError("Target is not 0/1 variable.")

    rf = [col for col in db.columns if col != target]
    if len(rf) == 0:
        raise ValueError("There are no risk factors in supplied db.")

    result_cols = {target: db[target].copy()}
    info = []

    for col in rf:
        x = db[col]

        # Check if categorical type
        is_categorical = (
            pd.api.types.is_string_dtype(x)
            or isinstance(x.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(x)
            or pd.api.types.is_object_dtype(x)
        )

        if not is_categorical:
            info.append({
                "rf": col,
                "reason_code": 1,
                "comment": "Inappropriate class. It has to be one of: character, factor or logical."
            })
            result_cols[col] = x.copy()
            continue

        # Check number of categories
        if x.nunique() > 10:
            info.append({
                "rf": col,
                "reason_code": 2,
                "comment": "More than 10 categories."
            })

        # Calculate WoE table
        woe_result = woe_tbl(db, x=col, y=target)

        # Check for problematic WoE values
        woe_vals = woe_result['woe'].values
        if any(pd.isna(woe_vals)) or any(np.isinf(woe_vals)):
            info.append({
                "rf": col,
                "reason_code": 3,
                "comment": "Problem with WoE calculation (NA, NaN, Inf, -Inf)"
            })
            result_cols[col] = x.copy()
            continue

        # Create mapping from bin to WoE
        woe_map = dict(zip(woe_result['bin'], woe_result['woe']))

        # Replace values with WoE
        result_cols[col] = x.map(woe_map)

    results_df = pd.DataFrame(result_cols)
    info_df = pd.DataFrame(info) if info else pd.DataFrame()

    return results_df, info_df
