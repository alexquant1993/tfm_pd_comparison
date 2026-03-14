"""Fairness validation for model outcomes.

This module provides functionality to test for fairness across
sensitive attributes using statistical parity and equal opportunity tests.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


@dataclass
class FairnessResult:
    """Result of fairness validation.

    Attributes
    ----------
    statistical_parity : pd.DataFrame
        Results of statistical parity test.
    conditional_statistical_parity : pd.DataFrame
        Results of conditional statistical parity test (if applicable).
    equal_opportunity : pd.DataFrame
        Results of equal opportunity/predictive equality tests.
    """
    statistical_parity: pd.DataFrame
    conditional_statistical_parity: Optional[pd.DataFrame]
    equal_opportunity: pd.DataFrame


def fairness_vld(
    db: pd.DataFrame,
    sensitive: str,
    obs_outcome: str,
    mod_outcome: str,
    conditional: Optional[str] = None,
    mod_outcome_type: Literal['disc', 'cont'] = 'disc',
    p_value: float = 0.05
) -> FairnessResult:
    """
    Validate model fairness across sensitive attributes.

    This function tests whether model outcomes are fair with respect
    to sensitive attributes using statistical parity, conditional
    statistical parity, and equal opportunity tests.

    Parameters
    ----------
    db : pd.DataFrame
        Data frame containing all required columns.
    sensitive : str
        Name of the sensitive attribute column (categorical).
    obs_outcome : str
        Name of the observed outcome column (binary 0/1).
    mod_outcome : str
        Name of the model outcome column.
    conditional : str, optional
        Name of conditional variable for CSP test.
    mod_outcome_type : {'disc', 'cont'}, default 'disc'
        Type of model outcome:
        - 'disc': Discrete (binary)
        - 'cont': Continuous (probabilities)
    p_value : float, default 0.05
        Significance level for tests.

    Returns
    -------
    FairnessResult
        Object containing test results.

    Raises
    ------
    ValueError
        If inputs are invalid or columns not found.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> db = pd.DataFrame({
    ...     'sensitive': np.random.choice(['A', 'B'], 200),
    ...     'obs_outcome': np.random.binomial(1, 0.3, 200),
    ...     'mod_outcome': np.random.binomial(1, 0.3, 200)
    ... })
    >>> result = fairness_vld(
    ...     db=db,
    ...     sensitive='sensitive',
    ...     obs_outcome='obs_outcome',
    ...     mod_outcome='mod_outcome'
    ... )
    """
    # Validation
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db is not a data frame.")

    required_cols = [sensitive, obs_outcome, mod_outcome]
    if conditional is not None:
        required_cols.append(conditional)

    missing_cols = [c for c in required_cols if c not in db.columns]
    if missing_cols:
        raise ValueError(f"columns cannot be found in db data frame: {missing_cols}")

    # Validate observed outcome
    y = db[obs_outcome].dropna()
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("obs_outcome has to be 0/1 variable.")

    if mod_outcome_type not in ['disc', 'cont']:
        raise ValueError("mod_outcome_type must be: disc, cont")

    if not (0 < p_value < 1):
        raise ValueError("p_value must be numeric between 0 and 1.")

    # Remove incomplete cases
    cols = [sensitive, obs_outcome, mod_outcome]
    if conditional is not None:
        cols.append(conditional)

    db_clean = db[cols].dropna()

    if len(db_clean) == 0:
        raise ValueError("No complete cases for db.")

    # Route to appropriate tests
    if mod_outcome_type == 'disc':
        return _disc_tests(db_clean, sensitive, obs_outcome, mod_outcome,
                          conditional, p_value)
    else:
        return _cont_tests(db_clean, sensitive, obs_outcome, mod_outcome,
                          conditional, p_value)


def _disc_tests(
    db: pd.DataFrame,
    sensitive: str,
    obs_outcome: str,
    mod_outcome: str,
    conditional: Optional[str],
    p_value: float
) -> FairnessResult:
    """Perform fairness tests for discrete model outcomes."""

    # Statistical Parity: P(Y_hat | A) should be equal across A
    sp_result = _disc_test_aux(db, mod_outcome, sensitive)
    sp_df = pd.DataFrame([{
        'test': 'statistical_parity',
        'statistic': sp_result['statistic'],
        'df': sp_result['df'],
        'p_value': sp_result['p_value'],
        'result': 'fair' if sp_result['p_value'] >= p_value else 'unfair'
    }])

    # Conditional Statistical Parity
    if conditional is not None:
        csp_result = _disc_test_cond(db, mod_outcome, sensitive, conditional)
        csp_df = pd.DataFrame([{
            'test': 'conditional_statistical_parity',
            'statistic': csp_result['statistic'],
            'df': csp_result['df'],
            'p_value': csp_result['p_value'],
            'result': 'fair' if csp_result['p_value'] >= p_value else 'unfair'
        }])
    else:
        csp_df = None

    # Equal Opportunity: P(Y_hat=1 | Y=1, A) should be equal across A
    db_positive = db[db[obs_outcome] == 1]
    if len(db_positive) > 0:
        eo_result = _disc_test_aux(db_positive, mod_outcome, sensitive)
        eo_df = pd.DataFrame([{
            'test': 'equal_opportunity',
            'statistic': eo_result['statistic'],
            'df': eo_result['df'],
            'p_value': eo_result['p_value'],
            'result': 'fair' if eo_result['p_value'] >= p_value else 'unfair'
        }])
    else:
        eo_df = pd.DataFrame([{
            'test': 'equal_opportunity',
            'statistic': np.nan,
            'df': np.nan,
            'p_value': np.nan,
            'result': 'insufficient_data'
        }])

    return FairnessResult(
        statistical_parity=sp_df,
        conditional_statistical_parity=csp_df,
        equal_opportunity=eo_df
    )


def _cont_tests(
    db: pd.DataFrame,
    sensitive: str,
    obs_outcome: str,
    mod_outcome: str,
    conditional: Optional[str],
    p_value: float
) -> FairnessResult:
    """Perform fairness tests for continuous model outcomes."""

    # Statistical Parity: E[Y_hat | A] should be equal across A
    sp_result = _cont_test_aux(db, mod_outcome, sensitive)
    sp_df = pd.DataFrame([{
        'test': 'statistical_parity',
        'statistic': sp_result['statistic'],
        'df': sp_result['df'],
        'p_value': sp_result['p_value'],
        'result': 'fair' if sp_result['p_value'] >= p_value else 'unfair'
    }])

    # Conditional Statistical Parity
    if conditional is not None:
        # Test with conditional as additional covariate
        csp_result = _cont_test_cond(db, mod_outcome, sensitive, conditional)
        csp_df = pd.DataFrame([{
            'test': 'conditional_statistical_parity',
            'statistic': csp_result['statistic'],
            'df': csp_result['df'],
            'p_value': csp_result['p_value'],
            'result': 'fair' if csp_result['p_value'] >= p_value else 'unfair'
        }])
    else:
        csp_df = None

    # Equal Opportunity
    db_positive = db[db[obs_outcome] == 1]
    if len(db_positive) > 0:
        eo_result = _cont_test_aux(db_positive, mod_outcome, sensitive)
        eo_df = pd.DataFrame([{
            'test': 'equal_opportunity',
            'statistic': eo_result['statistic'],
            'df': eo_result['df'],
            'p_value': eo_result['p_value'],
            'result': 'fair' if eo_result['p_value'] >= p_value else 'unfair'
        }])
    else:
        eo_df = pd.DataFrame([{
            'test': 'equal_opportunity',
            'statistic': np.nan,
            'df': np.nan,
            'p_value': np.nan,
            'result': 'insufficient_data'
        }])

    return FairnessResult(
        statistical_parity=sp_df,
        conditional_statistical_parity=csp_df,
        equal_opportunity=eo_df
    )


def _disc_test_aux(db: pd.DataFrame, outcome: str, sensitive: str) -> dict:
    """Perform chi-square test for discrete outcomes."""
    contingency = pd.crosstab(db[outcome], db[sensitive])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {'statistic': np.nan, 'df': np.nan, 'p_value': 1.0}

    chi2, p_value, df, expected = stats.chi2_contingency(contingency)

    return {'statistic': chi2, 'df': df, 'p_value': p_value}


def _disc_test_cond(
    db: pd.DataFrame,
    outcome: str,
    sensitive: str,
    conditional: str
) -> dict:
    """Perform stratified chi-square test."""
    total_chi2 = 0
    total_df = 0

    for level in db[conditional].unique():
        db_strat = db[db[conditional] == level]
        if len(db_strat) > 0:
            result = _disc_test_aux(db_strat, outcome, sensitive)
            if not np.isnan(result['statistic']):
                total_chi2 += result['statistic']
                total_df += result['df']

    if total_df > 0:
        p_value = 1 - stats.chi2.cdf(total_chi2, total_df)
    else:
        p_value = 1.0

    return {'statistic': total_chi2, 'df': total_df, 'p_value': p_value}


def _cont_test_aux(db: pd.DataFrame, outcome: str, sensitive: str) -> dict:
    """Perform Wald test for continuous outcomes using linear regression."""
    # Create dummy variables for sensitive attribute
    dummies = pd.get_dummies(db[sensitive], drop_first=True, dtype=float)

    if dummies.shape[1] == 0:
        return {'statistic': np.nan, 'df': np.nan, 'p_value': 1.0}

    X = sm.add_constant(dummies)
    y = db[outcome].values

    try:
        model = sm.OLS(y, X).fit()

        # Wald test on coefficients (excluding intercept)
        coef_indices = list(range(1, len(model.params)))
        if len(coef_indices) == 0:
            return {'statistic': np.nan, 'df': np.nan, 'p_value': 1.0}

        # Joint test of all sensitive attribute coefficients = 0
        r_matrix = np.zeros((len(coef_indices), len(model.params)))
        for i, idx in enumerate(coef_indices):
            r_matrix[i, idx] = 1

        wald_result = model.wald_test(r_matrix)
        statistic = float(wald_result.statistic)
        df = int(wald_result.df_num)
        p_value = float(wald_result.pvalue)

        return {'statistic': statistic, 'df': df, 'p_value': p_value}

    except Exception:
        return {'statistic': np.nan, 'df': np.nan, 'p_value': 1.0}


def _cont_test_cond(
    db: pd.DataFrame,
    outcome: str,
    sensitive: str,
    conditional: str
) -> dict:
    """Perform Wald test with conditional variable as covariate."""
    # Create dummy variables
    sens_dummies = pd.get_dummies(db[sensitive], drop_first=True, prefix='sens', dtype=float)
    cond_dummies = pd.get_dummies(db[conditional], drop_first=True, prefix='cond', dtype=float)

    X = pd.concat([sens_dummies, cond_dummies], axis=1)
    X = sm.add_constant(X)
    y = db[outcome].values

    try:
        model = sm.OLS(y, X).fit()

        # Test only the sensitive attribute coefficients
        n_sens = sens_dummies.shape[1]
        if n_sens == 0:
            return {'statistic': np.nan, 'df': np.nan, 'p_value': 1.0}

        coef_indices = list(range(1, 1 + n_sens))
        r_matrix = np.zeros((len(coef_indices), len(model.params)))
        for i, idx in enumerate(coef_indices):
            r_matrix[i, idx] = 1

        wald_result = model.wald_test(r_matrix)
        statistic = float(wald_result.statistic)
        df = int(wald_result.df_num)
        p_value = float(wald_result.pvalue)

        return {'statistic': statistic, 'df': df, 'p_value': p_value}

    except Exception:
        return {'statistic': np.nan, 'df': np.nan, 'p_value': 1.0}
