"""Confusion matrix and classification metrics.

This module provides functionality to compute confusion matrix
and various classification performance metrics.
"""

from dataclasses import dataclass
from typing import Union
import numpy as np
import pandas as pd


@dataclass
class ConfusionMatrixResult:
    """Result of confusion matrix calculation.

    Attributes
    ----------
    confusion_matrix : pd.DataFrame
        The confusion matrix as a DataFrame.
    metrics : pd.DataFrame
        Performance metrics including accuracy, sensitivity, etc.
    """
    confusion_matrix: pd.DataFrame
    metrics: pd.DataFrame


def confusion_matrix(
    predictions: Union[np.ndarray, pd.Series, list],
    observed: Union[np.ndarray, pd.Series, list],
    cutoff: float
) -> ConfusionMatrixResult:
    """
    Compute confusion matrix and classification metrics.

    This function creates a confusion matrix from probability predictions
    and observed binary outcomes, then calculates various performance metrics.

    Parameters
    ----------
    predictions : array-like
        Model probability predictions (values between 0 and 1).
    observed : array-like
        Binary observed outcomes (0 or 1).
    cutoff : float
        Classification threshold. Predictions > cutoff are classified as 1.

    Returns
    -------
    ConfusionMatrixResult
        Object containing:
        - confusion_matrix: 2x2 matrix (rows=observed, cols=predicted)
        - metrics: DataFrame with accuracy, sensitivity, specificity, etc.

    Raises
    ------
    ValueError
        If predictions are not in [0,1], observed is not 0/1, or cutoff invalid.

    Examples
    --------
    >>> import numpy as np
    >>> predictions = np.array([0.1, 0.4, 0.6, 0.9])
    >>> observed = np.array([0, 0, 1, 1])
    >>> result = confusion_matrix(predictions, observed, cutoff=0.5)
    >>> result.metrics[result.metrics['metric'] == 'accuracy']['value'].values[0]
    1.0
    """
    # Convert inputs
    if isinstance(predictions, (pd.Series, list)):
        predictions = np.array(predictions, dtype=float)
    if isinstance(observed, (pd.Series, list)):
        observed = np.array(observed, dtype=float)

    # Remove missing values
    mask = ~(np.isnan(predictions) | np.isnan(observed))
    predictions = predictions[mask]
    observed = observed[mask]

    # Validation
    if not np.all((predictions >= 0) & (predictions <= 1)):
        raise ValueError("predictions should be between 0 and 1")

    if not np.all(np.isin(observed, [0, 1])):
        raise ValueError("observed is not 0/1 variable.")

    if not (0 < cutoff < 1):
        raise ValueError("cutoff should be between 0 and 1")

    # Create predicted classes
    predicted = (predictions > cutoff).astype(int)

    # Build confusion matrix
    # Rows: observed, Columns: predicted
    cm = np.zeros((2, 2), dtype=int)
    cm[0, 0] = np.sum((observed == 0) & (predicted == 0))  # TN
    cm[0, 1] = np.sum((observed == 0) & (predicted == 1))  # FP
    cm[1, 0] = np.sum((observed == 1) & (predicted == 0))  # FN
    cm[1, 1] = np.sum((observed == 1) & (predicted == 1))  # TP

    cm_df = pd.DataFrame(
        cm,
        index=['observed_0', 'observed_1'],
        columns=['predicted_0', 'predicted_1']
    )

    # Calculate metrics
    metrics = _cm_metrics(cm)

    return ConfusionMatrixResult(
        confusion_matrix=cm_df,
        metrics=metrics
    )


def _cm_metrics(conf_mat: np.ndarray) -> pd.DataFrame:
    """Calculate classification metrics from confusion matrix."""
    tn, fp, fn, tp = conf_mat[0, 0], conf_mat[0, 1], conf_mat[1, 0], conf_mat[1, 1]
    total = tn + fp + fn + tp

    # Accuracy
    accuracy = (tp + tn) / total if total > 0 else 0

    # Error rate
    error_rate = 1 - accuracy

    # Sensitivity (Recall, True Positive Rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Precision (Positive Predictive Value)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # F1 Score
    if precision + sensitivity > 0:
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        f1_score = 0

    # False Positive Rate
    false_positive = 1 - specificity

    # False Discovery Rate
    false_discovery = 1 - precision

    metrics = pd.DataFrame({
        'metric': [
            'accuracy', 'error_rate', 'sensitivity', 'specificity',
            'precision', 'f1_score', 'false_positive', 'false_discovery'
        ],
        'value': [
            accuracy, error_rate, sensitivity, specificity,
            precision, f1_score, false_positive, false_discovery
        ]
    })

    return metrics
