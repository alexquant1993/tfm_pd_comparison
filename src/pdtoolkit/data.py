"""Sample datasets for PDtoolkit.

This module provides sample datasets for testing and demonstrating
the functionality of the PDtoolkit package.
"""

import numpy as np
import pandas as pd


def load_loans(n: int = 3000, seed: int = 2191) -> pd.DataFrame:
    """
    Generate a synthetic loans dataset for credit risk modeling.

    This function creates a realistic synthetic dataset with loan
    characteristics and default indicators suitable for PD model development.

    Parameters
    ----------
    n : int, default 3000
        Number of observations to generate.
    seed : int, default 2191
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:
        - default: Binary target (0/1) indicating loan default
        - age: Borrower age (numeric)
        - income: Annual income in thousands (numeric)
        - emp_length: Employment length in years (numeric)
        - loan_amount: Loan amount in thousands (numeric)
        - interest_rate: Interest rate percentage (numeric)
        - dti: Debt-to-income ratio (numeric)
        - credit_score: Credit score (numeric)
        - home_ownership: Home ownership status (categorical)
        - purpose: Loan purpose (categorical)
        - grade: Loan grade (categorical)

    Examples
    --------
    >>> from pdtoolkit.data import load_loans
    >>> loans = load_loans()
    >>> loans.shape
    (3000, 11)
    >>> loans['default'].mean()  # Default rate around 15-25%
    """
    np.random.seed(seed)

    # Generate base features
    age = np.random.normal(40, 12, n).clip(18, 80).astype(int)
    income = np.random.lognormal(3.5, 0.8, n).clip(10, 500)
    emp_length = np.random.exponential(5, n).clip(0, 40).astype(int)
    loan_amount = np.random.lognormal(2.5, 0.6, n).clip(1, 100)
    credit_score = np.random.normal(680, 80, n).clip(300, 850).astype(int)

    # Calculate derived features
    dti = (loan_amount * 12) / (income * 10) * 100
    dti = dti.clip(0, 60)

    # Interest rate inversely related to credit score
    interest_rate = 25 - (credit_score - 300) / 550 * 20 + np.random.normal(0, 2, n)
    interest_rate = interest_rate.clip(3, 30)

    # Categorical features
    home_ownership = np.random.choice(
        ['RENT', 'OWN', 'MORTGAGE'],
        n,
        p=[0.35, 0.15, 0.50]
    )

    purpose = np.random.choice(
        ['debt_consolidation', 'credit_card', 'home_improvement',
         'major_purchase', 'medical', 'other'],
        n,
        p=[0.40, 0.20, 0.15, 0.10, 0.05, 0.10]
    )

    grade = np.random.choice(
        ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        n,
        p=[0.15, 0.25, 0.25, 0.15, 0.10, 0.06, 0.04]
    )

    # Generate default probability based on features
    # Higher probability for: low income, high dti, low credit score, high interest
    log_odds = (
        -3.0
        + 0.02 * (40 - age)  # Younger = higher risk
        - 0.01 * income  # Lower income = higher risk
        - 0.01 * emp_length  # Less employment = higher risk
        + 0.02 * loan_amount  # Higher loan = higher risk
        + 0.05 * interest_rate  # Higher rate = higher risk
        + 0.03 * dti  # Higher DTI = higher risk
        - 0.01 * (credit_score - 650)  # Lower score = higher risk
    )

    # Add grade effect
    grade_effect = {'A': -1.0, 'B': -0.5, 'C': 0.0, 'D': 0.5, 'E': 1.0, 'F': 1.5, 'G': 2.0}
    log_odds += np.array([grade_effect[g] for g in grade])

    # Add home ownership effect
    home_effect = {'RENT': 0.3, 'OWN': -0.3, 'MORTGAGE': 0.0}
    log_odds += np.array([home_effect[h] for h in home_ownership])

    # Convert to probability
    prob = 1 / (1 + np.exp(-log_odds))

    # Generate defaults
    default = (np.random.rand(n) < prob).astype(int)

    # Create DataFrame
    loans = pd.DataFrame({
        'default': default,
        'age': age,
        'income': np.round(income, 2),
        'emp_length': emp_length,
        'loan_amount': np.round(loan_amount, 2),
        'interest_rate': np.round(interest_rate, 2),
        'dti': np.round(dti, 2),
        'credit_score': credit_score,
        'home_ownership': home_ownership,
        'purpose': purpose,
        'grade': grade
    })

    return loans


def get_loans_description() -> str:
    """
    Get description of the loans dataset.

    Returns
    -------
    str
        Description of the loans dataset and its columns.
    """
    return """
Synthetic Loans Dataset
=======================

A synthetic dataset simulating loan applications for credit risk modeling.
The dataset is designed to have realistic relationships between features
and the default outcome.

Variables:
----------
- default: Binary target variable (1 = defaulted, 0 = not defaulted)
- age: Borrower's age in years (18-80)
- income: Annual income in thousands of dollars
- emp_length: Employment length in years
- loan_amount: Loan amount in thousands of dollars
- interest_rate: Interest rate as percentage
- dti: Debt-to-income ratio as percentage
- credit_score: Credit score (300-850)
- home_ownership: Home ownership status (RENT, OWN, MORTGAGE)
- purpose: Loan purpose category
- grade: Loan grade assigned by lender (A-G, A being best)

Default Rate:
-------------
The dataset is generated with an overall default rate of approximately 15-25%,
with higher default rates for:
- Lower credit scores
- Higher debt-to-income ratios
- Higher interest rates
- Lower income
- Grades D-G

Usage:
------
>>> from pdtoolkit.data import load_loans
>>> loans = load_loans(n=1000, seed=42)
>>> print(f"Default rate: {loans['default'].mean():.2%}")
"""


# Alias for backwards compatibility
loans = load_loans
