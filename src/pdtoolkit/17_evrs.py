"""
Economic Value of Rating System (EVRS) module for PDtoolkit.

This module calculates the economic benefit of improved PD models
by simulating portfolio returns under measurement error conditions.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
from dataclasses import dataclass


@dataclass
class EVRSResult:
    """Result of EVRS calculation.

    Attributes:
        summary: DataFrame with mean returns for each model
        returns: DataFrame with simulated return vectors
    """
    summary: pd.DataFrame
    returns: pd.DataFrame


def evrs(
    db: pd.DataFrame,
    def_ind: str,
    pd_est: str,
    pd_bm: str,
    lgd: Union[str, float] = 0.45,
    rf: float = 0.02,
    elasticity: float = 5.0,
    prob_threshold: float = 0.5,
    sim_num: int = 500,
    seed: int = 991
) -> EVRSResult:
    """
    Calculate Economic Value of Rating System.

    Simulates portfolio returns under measurement error conditions to
    estimate the economic benefit of improved PD models.

    Parameters
    ----------
    db : pd.DataFrame
        Database with required columns.
    def_ind : str
        Name of column for default indicator (0/1).
    pd_est : str
        Name of column for estimated PD from the model being evaluated.
    pd_bm : str
        Name of column for benchmark PD (e.g., true PD or competitor model).
    lgd : str or float, default=0.45
        Loss Given Default - column name or constant value.
    rf : float, default=0.02
        Risk-free rate.
    elasticity : float, default=5.0
        Elasticity parameter for customer churn probability.
    prob_threshold : float, default=0.5
        Probability threshold for customer leaving.
    sim_num : int, default=500
        Number of simulations.
    seed : int, default=991
        Random seed for reproducibility.

    Returns
    -------
    EVRSResult
        Object containing summary and simulated returns.

    References
    ----------
    Jankowitsch, R., Pichler, S., & Schwaiger, W. (2007).
    "Modelling the economic value of credit rating systems."
    Journal of Banking & Finance.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> db = pd.DataFrame({
    ...     'default': np.random.binomial(1, 0.05, 1000),
    ...     'pd_model': np.random.uniform(0.01, 0.15, 1000),
    ...     'pd_benchmark': np.random.uniform(0.02, 0.12, 1000)
    ... })
    >>> result = evrs(db, 'default', 'pd_model', 'pd_benchmark', sim_num=100)
    """
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db is not a data frame.")

    required_cols = [def_ind, pd_est, pd_bm]
    for col in required_cols:
        if col not in db.columns:
            raise ValueError(f"Column '{col}' not found in db.")

    # Handle LGD
    if isinstance(lgd, str):
        if lgd not in db.columns:
            raise ValueError(f"LGD column '{lgd}' not found in db.")
        lgd_vals = db[lgd].values
    else:
        lgd_vals = np.full(len(db), lgd)

    # Extract values
    defaults = db[def_ind].values
    pd_estimated = db[pd_est].values
    pd_benchmark = db[pd_bm].values

    # Set random seed
    rng = np.random.default_rng(seed)

    # Run simulations
    returns_est = np.zeros(sim_num)
    returns_bm = np.zeros(sim_num)

    for i in range(sim_num):
        # Generate observed PDs with measurement error
        pd_obs_est = _pd_observed(pd_estimated, rng)
        pd_obs_bm = _pd_observed(pd_benchmark, rng)

        # Calculate portfolio returns
        returns_est[i] = _portfolio_return(
            defaults, pd_obs_est, pd_benchmark, lgd_vals,
            rf, elasticity, prob_threshold
        )
        returns_bm[i] = _portfolio_return(
            defaults, pd_obs_bm, pd_benchmark, lgd_vals,
            rf, elasticity, prob_threshold
        )

    # Summary statistics
    summary = pd.DataFrame({
        'model': [pd_est, pd_bm],
        'mean_return': [np.mean(returns_est), np.mean(returns_bm)],
        'std_return': [np.std(returns_est), np.std(returns_bm)],
        'min_return': [np.min(returns_est), np.min(returns_bm)],
        'max_return': [np.max(returns_est), np.max(returns_bm)]
    })

    # Returns DataFrame
    returns_df = pd.DataFrame({
        f'return_{pd_est}': returns_est,
        f'return_{pd_bm}': returns_bm
    })

    return EVRSResult(summary=summary, returns=returns_df)


def _score_trans(pd_vals: np.ndarray, to_score: bool = True) -> np.ndarray:
    """
    Convert between PD and log-odds score.

    Parameters
    ----------
    pd_vals : np.ndarray
        PD values (if to_score=True) or scores (if to_score=False).
    to_score : bool, default=True
        If True, convert PD to score. If False, convert score to PD.

    Returns
    -------
    np.ndarray
        Converted values.
    """
    if to_score:
        # PD to log-odds score
        pd_safe = np.clip(pd_vals, 1e-10, 1 - 1e-10)
        return np.log(pd_safe / (1 - pd_safe))
    else:
        # Log-odds score to PD
        return 1 / (1 + np.exp(-pd_vals))


def _pd_observed(pd_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Generate observed PDs with measurement error.

    Parameters
    ----------
    pd_true : np.ndarray
        True PD values.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Observed PDs with noise.
    """
    # Convert to score space
    scores = _score_trans(pd_true, to_score=True)

    # Add noise in score space
    noise_std = 0.5  # Standard deviation of measurement error
    scores_noisy = scores + rng.normal(0, noise_std, len(scores))

    # Convert back to PD space
    return _score_trans(scores_noisy, to_score=False)


def _spread(pd_vals: np.ndarray, lgd_vals: np.ndarray, rf: float) -> np.ndarray:
    """
    Calculate loan spreads.

    Parameters
    ----------
    pd_vals : np.ndarray
        Probability of default.
    lgd_vals : np.ndarray
        Loss given default.
    rf : float
        Risk-free rate.

    Returns
    -------
    np.ndarray
        Loan spreads.
    """
    expected_loss = pd_vals * lgd_vals
    expected_loss = np.clip(expected_loss, 0, 0.999)  # Avoid division issues
    return (1 + rf) * expected_loss / (1 - expected_loss)


def _prob_to_leave(
    spread_model: np.ndarray,
    spread_benchmark: np.ndarray,
    elasticity: float
) -> np.ndarray:
    """
    Calculate probability of customer leaving due to overpricing.

    Parameters
    ----------
    spread_model : np.ndarray
        Spreads from evaluated model.
    spread_benchmark : np.ndarray
        Spreads from benchmark model.
    elasticity : float
        Elasticity parameter.

    Returns
    -------
    np.ndarray
        Probability of leaving.
    """
    overprice = spread_model - spread_benchmark
    overprice = np.maximum(overprice, 0)  # Only overpricing causes churn
    return 1 - np.exp(-elasticity * overprice)


def _portfolio_return(
    defaults: np.ndarray,
    pd_observed: np.ndarray,
    pd_benchmark: np.ndarray,
    lgd_vals: np.ndarray,
    rf: float,
    elasticity: float,
    prob_threshold: float
) -> float:
    """
    Calculate portfolio return for one simulation.

    Parameters
    ----------
    defaults : np.ndarray
        Observed default indicators.
    pd_observed : np.ndarray
        Observed PDs from model.
    pd_benchmark : np.ndarray
        Benchmark PDs.
    lgd_vals : np.ndarray
        Loss given default.
    rf : float
        Risk-free rate.
    elasticity : float
        Elasticity for churn.
    prob_threshold : float
        Threshold for customer leaving.

    Returns
    -------
    float
        Average portfolio return.
    """
    # Calculate spreads
    spread_model = _spread(pd_observed, lgd_vals, rf)
    spread_benchmark = _spread(pd_benchmark, lgd_vals, rf)

    # Calculate churn probability
    prob_leave = _prob_to_leave(spread_model, spread_benchmark, elasticity)

    # Determine which customers stay
    stays = prob_leave < prob_threshold

    if stays.sum() == 0:
        return 0.0

    # Calculate returns for staying customers
    returns = np.where(
        defaults[stays] == 1,
        -lgd_vals[stays],  # Loss from default
        spread_model[stays]  # Gain from spread
    )

    return np.mean(returns)
