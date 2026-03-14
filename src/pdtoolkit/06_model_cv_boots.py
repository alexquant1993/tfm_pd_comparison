"""
K-fold cross-validation and bootstrap model validation.

This module provides functions for model validation using k-fold cross-validation
and bootstrap resampling methods.

Ported from: 06_MODEL_CV_BOOTS.R
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from importlib import import_module

_bivariate = import_module('.02_bivariate_analysis', package='pdtoolkit')
auc_model = _bivariate.auc_model


@dataclass
class ValidationResult:
    """Result container for validation functions."""
    iter: pd.DataFrame
    summary: pd.DataFrame


def kfold_vld(
    model: Any,
    db: pd.DataFrame,
    target: str,
    predictors: List[str],
    k: int = 10,
    seed: int = 1984,
) -> ValidationResult:
    """
    Perform k-fold cross-validation for a logistic regression model.

    Args:
        model: Fitted statsmodels logistic regression model (or None to fit from scratch).
        db: DataFrame with target and predictors.
        target: Name of target variable.
        predictors: List of predictor column names.
        k: Number of folds. Default 10.
        seed: Random seed for reproducibility. Default 1984.

    Returns:
        ValidationResult with iteration metrics and summary.

    Examples:
        >>> from pdtoolkit.06_model_cv_boots import kfold_vld
        >>> result = kfold_vld(model=None, db=df, target='default',
        ...                    predictors=['rf1', 'rf2'], k=5)
        >>> print(result.summary)
    """
    n = len(db)

    if k < 1:
        raise ValueError("k cannot be negative or zero.")

    if k > n:
        k = n
        print("Warning: k corrected to LOOCV method.")

    np.random.seed(seed)
    indices = np.random.permutation(n)
    db = db.iloc[indices].reset_index(drop=True)

    # Create fold assignments
    fold_size = n // k
    folds = np.zeros(n, dtype=int)
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else n
        folds[start:end] = i + 1

    if k == n:
        print("Warning: LOOCV method. AUC cannot be calculated reliably.")

    results = []

    for fold in range(1, k + 1):
        # Split data
        train_mask = folds != fold
        test_mask = folds == fold

        db_train = db[train_mask]
        db_test = db[test_mask]

        vld_no = len(db_test)

        # Fit model on training data
        X_train = sm.add_constant(db_train[predictors])
        y_train = db_train[target]

        try:
            lr_model = sm.Logit(y_train, X_train).fit(disp=0)

            # Predict on test data
            X_test = sm.add_constant(db_test[predictors])
            predictions = lr_model.predict(X_test)

            y_test = db_test[target].values

            # Calculate metrics
            amse = np.mean((predictions - y_test) ** 2)
            rmse = np.sqrt(amse)

            # AUC calculation
            if k != n and vld_no >= 2 and len(np.unique(y_test)) > 1:
                auc = auc_model(predictions.values, y_test)
            else:
                auc = np.nan

        except Exception:
            amse = np.nan
            rmse = np.nan
            auc = np.nan

        results.append({
            'k': fold,
            'no': vld_no,
            'amse': amse,
            'rmse': rmse,
            'auc': auc
        })

    iter_df = pd.DataFrame(results)
    summary_df = pd.DataFrame({
        'amse': [iter_df['amse'].mean()],
        'rmse': [iter_df['rmse'].mean()],
        'auc': [iter_df['auc'].mean()]
    })

    return ValidationResult(iter=iter_df, summary=summary_df)


def boots_vld(
    model: Any,
    db: pd.DataFrame,
    target: str,
    predictors: List[str],
    B: int = 1000,
    seed: int = 1122,
) -> ValidationResult:
    """
    Perform bootstrap model validation.

    Args:
        model: Fitted statsmodels logistic regression model (or None).
        db: DataFrame with target and predictors.
        target: Name of target variable.
        predictors: List of predictor column names.
        B: Number of bootstrap samples. Default 1000.
        seed: Random seed for reproducibility. Default 1122.

    Returns:
        ValidationResult with iteration metrics and summary.

    Examples:
        >>> from pdtoolkit.06_model_cv_boots import boots_vld
        >>> result = boots_vld(model=None, db=df, target='default',
        ...                    predictors=['rf1', 'rf2'], B=100)
        >>> print(result.summary)
    """
    n = len(db)

    if B < 1:
        raise ValueError("B must be positive.")

    np.random.seed(seed)

    results = []

    for b in range(1, B + 1):
        # Bootstrap sample (with replacement)
        boot_indices = np.random.choice(n, size=n, replace=True)
        oob_indices = np.setdiff1d(np.arange(n), boot_indices)

        if len(oob_indices) == 0:
            continue

        db_boot = db.iloc[boot_indices]
        db_oob = db.iloc[oob_indices]

        oob_no = len(db_oob)

        try:
            # Fit model on bootstrap sample
            X_boot = sm.add_constant(db_boot[predictors])
            y_boot = db_boot[target]
            lr_model = sm.Logit(y_boot, X_boot).fit(disp=0)

            # Predict on out-of-bag sample
            X_oob = sm.add_constant(db_oob[predictors])
            predictions = lr_model.predict(X_oob)

            y_oob = db_oob[target].values

            # Calculate metrics
            amse = np.mean((predictions - y_oob) ** 2)
            rmse = np.sqrt(amse)

            # AUC calculation
            if oob_no >= 2 and len(np.unique(y_oob)) > 1:
                auc = auc_model(predictions.values, y_oob)
            else:
                auc = np.nan

        except Exception:
            amse = np.nan
            rmse = np.nan
            auc = np.nan

        results.append({
            'b': b,
            'no': oob_no,
            'amse': amse,
            'rmse': rmse,
            'auc': auc
        })

    iter_df = pd.DataFrame(results)
    summary_df = pd.DataFrame({
        'amse': [iter_df['amse'].mean()],
        'rmse': [iter_df['rmse'].mean()],
        'auc': [iter_df['auc'].mean()]
    })

    return ValidationResult(iter=iter_df, summary=summary_df)
