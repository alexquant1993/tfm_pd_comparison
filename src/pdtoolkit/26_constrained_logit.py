"""Constrained logistic regression.

This module provides functionality for logistic regression with
coefficient constraints using L-BFGS-B optimization.
"""

from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class ConstrainedLogitResult:
    """Result of constrained logistic regression.

    Attributes
    ----------
    beta : np.ndarray
        Optimized coefficient values.
    beta_names : list
        Names of coefficients (including intercept).
    prediction : np.ndarray
        Linear predictor values (X @ beta).
    """
    beta: np.ndarray
    beta_names: List[str]
    prediction: np.ndarray


def constrained_logit(
    db: pd.DataFrame,
    x: List[str],
    y: str,
    lower: Union[List[float], np.ndarray],
    upper: Union[List[float], np.ndarray]
) -> ConstrainedLogitResult:
    """
    Fit constrained logistic regression using L-BFGS-B optimization.

    This function fits a logistic regression model with box constraints
    on the coefficients. It uses scipy's L-BFGS-B optimizer to find
    coefficients that minimize negative log-likelihood while respecting
    the specified bounds.

    Parameters
    ----------
    db : pd.DataFrame
        Data frame containing predictors and target.
    x : list of str
        Names of predictor variables.
    y : str
        Name of the target variable (binary 0/1).
    lower : array-like
        Lower bounds for coefficients. Use -np.inf for no lower bound.
        Must include bound for intercept as first element.
    upper : array-like
        Upper bounds for coefficients. Use np.inf for no upper bound.
        Must include bound for intercept as first element.

    Returns
    -------
    ConstrainedLogitResult
        Object containing optimized coefficients and predictions.

    Raises
    ------
    ValueError
        If inputs are invalid or optimization fails.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 200
    >>> db = pd.DataFrame({
    ...     'target': np.random.binomial(1, 0.3, n),
    ...     'x1': np.random.randn(n),
    ...     'x2': np.random.randn(n)
    ... })
    >>> # Constrain coefficients to be positive
    >>> result = constrained_logit(
    ...     db=db,
    ...     x=['x1', 'x2'],
    ...     y='target',
    ...     lower=[-np.inf, 0, 0],  # intercept, x1, x2
    ...     upper=[np.inf, np.inf, np.inf]
    ... )
    >>> all(result.beta[1:] >= 0)  # All non-intercept coefficients >= 0
    True
    """
    # Validation
    lower = np.array(lower, dtype=float)
    upper = np.array(upper, dtype=float)

    if not (np.all(np.isreal(lower)) and np.all(np.isreal(upper))):
        raise ValueError("arguments lower and upper have to be numeric vectors.")

    if not all(col in db.columns for col in [y] + x):
        raise ValueError("x or y cannot be found in db data frame.")

    n_params = len(x) + 1  # +1 for intercept

    if len(lower) != n_params or len(upper) != n_params:
        raise ValueError(
            "lower and upper bounds have to be equal to number of model "
            "parameters (including intercept)."
        )

    # Check that upper > lower where bounds are finite
    for i in range(len(lower)):
        if not (np.isinf(lower[i]) and np.isinf(upper[i])):
            if np.isfinite(lower[i]) and np.isfinite(upper[i]):
                if upper[i] <= lower[i]:
                    raise ValueError(
                        "Upper bound element(s) less or equal to lower bound element(s)."
                    )

    # Prepare data
    X = np.column_stack([np.ones(len(db)), db[x].values])
    y_vec = db[y].values.astype(float)

    # Initial coefficients
    b_start = np.zeros(n_params)

    # Bounds for L-BFGS-B
    bounds = list(zip(lower, upper))

    # Optimize
    try:
        result = minimize(
            fun=_neg_log_likelihood,
            x0=b_start,
            args=(X, y_vec),
            method='L-BFGS-B',
            bounds=bounds
        )

        if not result.success:
            raise ValueError(f"Optimization did not converge: {result.message}")

        b_opt = result.x

    except Exception as e:
        raise ValueError(
            f"Error in optimization procedure. Check supplied arguments "
            f"and try manual estimation. Details: {str(e)}"
        )

    # Calculate predictions (linear predictor)
    pred_opt = X @ b_opt

    # Coefficient names
    beta_names = ['const'] + x

    return ConstrainedLogitResult(
        beta=b_opt,
        beta_names=beta_names,
        prediction=pred_opt
    )


def _neg_log_likelihood(b: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Calculate negative log-likelihood for logistic regression."""
    linear_pred = X @ b

    # Use log-sum-exp trick for numerical stability
    # log(1 + exp(x)) = x + log(1 + exp(-x)) for large x
    # log(1 + exp(x)) = log(1 + exp(x)) for small x

    log1pexp = np.where(
        linear_pred > 0,
        linear_pred + np.log1p(np.exp(-linear_pred)),
        np.log1p(np.exp(linear_pred))
    )

    # Negative log-likelihood
    # -sum(y * (xb - log(1+exp(xb))) + (1-y) * (-log(1+exp(xb))))
    # = -sum(y * xb - log(1+exp(xb)))
    nll = -np.sum(y * linear_pred - log1pexp)

    return nll
