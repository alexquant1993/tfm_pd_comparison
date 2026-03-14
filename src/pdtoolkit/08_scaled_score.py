"""
Scaled score module for PDtoolkit.

This module provides functionality for scaling probabilities to credit scores.
"""

import numpy as np
from typing import Union
import pandas as pd


def scaled_score(
    probs: Union[np.ndarray, pd.Series, list],
    score: float = 600,
    odd: float = 50,
    pdo: float = 20
) -> np.ndarray:
    """
    Scale probabilities to credit scores.

    Performs scaling of the probabilities for a certain set up.
    User has to select three parameters (score, odd, pdo), while the
    probabilities (probs) are usually predictions of the final model.

    Parameters
    ----------
    probs : array-like
        Model predicted probabilities of default.
    score : float, default=600
        Specific score for selected odd.
    odd : float, default=50
        Odd (good/bad) at specific score. Default is 50 (50:1).
    pdo : float, default=20
        Points for double the odds.

    Returns
    -------
    np.ndarray
        Vector of scaled scores.

    Raises
    ------
    ValueError
        If arguments are not numeric or have invalid values.

    References
    ----------
    Siddiqi, N. (2012). Credit Risk Scorecards: Developing and Implementing
    Intelligent Credit Scoring, John Wiley & Sons, Inc.

    Examples
    --------
    >>> import numpy as np
    >>> probs = np.array([0.1, 0.2, 0.3, 0.5])
    >>> scores = scaled_score(probs, score=600, odd=50, pdo=20)
    """
    # Convert to numpy array
    probs = np.asarray(probs, dtype=float)

    # Validate inputs
    if not isinstance(score, (int, float)):
        raise ValueError("All arguments have to be of numeric type.")
    if not isinstance(odd, (int, float)):
        raise ValueError("All arguments have to be of numeric type.")
    if not isinstance(pdo, (int, float)):
        raise ValueError("All arguments have to be of numeric type.")

    if odd <= 0:
        raise ValueError("odd has to be greater than 0.")
    if pdo <= 0:
        raise ValueError("pdo has to be greater than 0.")

    # Handle edge cases in probabilities
    probs = np.clip(probs, 1e-10, 1 - 1e-10)

    # Calculate odds
    odds = (1 - probs) / probs

    # Calculate scaling parameters
    factor = pdo / np.log(2)
    offset = score - factor * np.log(odd)

    # Calculate scaled scores
    ss = offset + factor * np.log(odds)

    return ss


def score_to_prob(
    scores: Union[np.ndarray, pd.Series, list],
    score: float = 600,
    odd: float = 50,
    pdo: float = 20
) -> np.ndarray:
    """
    Convert scaled scores back to probabilities.

    This is the inverse function of scaled_score.

    Parameters
    ----------
    scores : array-like
        Scaled credit scores.
    score : float, default=600
        Specific score for selected odd.
    odd : float, default=50
        Odd (good/bad) at specific score.
    pdo : float, default=20
        Points for double the odds.

    Returns
    -------
    np.ndarray
        Vector of probabilities.

    Examples
    --------
    >>> scores = np.array([600, 620, 580])
    >>> probs = score_to_prob(scores, score=600, odd=50, pdo=20)
    """
    scores = np.asarray(scores, dtype=float)

    if not isinstance(score, (int, float)):
        raise ValueError("All arguments have to be of numeric type.")
    if not isinstance(odd, (int, float)):
        raise ValueError("All arguments have to be of numeric type.")
    if not isinstance(pdo, (int, float)):
        raise ValueError("All arguments have to be of numeric type.")

    if odd <= 0:
        raise ValueError("odd has to be greater than 0.")
    if pdo <= 0:
        raise ValueError("pdo has to be greater than 0.")

    # Calculate scaling parameters
    factor = pdo / np.log(2)
    offset = score - factor * np.log(odd)

    # Calculate odds from scores
    odds = np.exp((scores - offset) / factor)

    # Convert odds to probabilities
    probs = 1 / (1 + odds)

    return probs
