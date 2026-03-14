"""Ensemble block regression.

This module provides functionality for block-wise regression where
all block predictions are combined in a final ensemble model.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class EnsembleBlocksResult:
    """Result of ensemble block regression.

    Attributes
    ----------
    block_models : dict
        Dictionary mapping block names to model DataFrames.
    ensemble_model : pd.DataFrame
        Final ensemble model combining all block predictions.
    steps : pd.DataFrame
        Details of selected risk factors per block.
    dev_db : pd.DataFrame
        Development database with block predictions.
    """
    block_models: Dict[str, pd.DataFrame]
    ensemble_model: pd.DataFrame
    steps: pd.DataFrame
    dev_db: pd.DataFrame


def ensemble_blocks(
    method: str,
    target: str,
    db: pd.DataFrame,
    coding: str = 'WoE',
    blocks: pd.DataFrame = None,
    p_value: float = 0.05,
    miv_threshold: float = 0.02,
    m_ch_p_val: float = 0.05
) -> EnsembleBlocksResult:
    """
    Perform ensemble block regression.

    This function performs block-wise logistic regression where each
    block is fitted independently, then all block predictions are
    combined in a final ensemble model.

    Parameters
    ----------
    method : str
        Regression method to use:
        - 'stepMIV': Stepwise based on Marginal Information Value
        - 'stepFWD': Forward stepwise regression
        - 'stepRPC': Stepwise with risk profile check
    target : str
        Name of the target variable.
    db : pd.DataFrame
        Modeling database containing target and risk factors.
    coding : str, default 'WoE'
        Coding method: 'WoE' for Weight of Evidence or 'dummy' for dummy coding.
    blocks : pd.DataFrame
        Data frame with columns:
        - rf: Risk factor names
        - block: Block assignment (processed in sorted order)
    p_value : float, default 0.05
        P-value threshold for variable selection.
    miv_threshold : float, default 0.02
        Minimum Information Value threshold (for stepMIV method).
    m_ch_p_val : float, default 0.05
        Chi-square p-value threshold (for stepMIV method).

    Returns
    -------
    EnsembleBlocksResult
        Object containing block models, ensemble model, steps, and dev database.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 500
    >>> db = pd.DataFrame({
    ...     'target': np.random.binomial(1, 0.1, n),
    ...     'woe_rf1': np.random.randn(n),
    ...     'woe_rf2': np.random.randn(n),
    ...     'woe_rf3': np.random.randn(n)
    ... })
    >>> blocks = pd.DataFrame({
    ...     'rf': ['woe_rf1', 'woe_rf2', 'woe_rf3'],
    ...     'block': ['A', 'A', 'B']
    ... })
    >>> result = ensemble_blocks(
    ...     method='stepFWD',
    ...     target='target',
    ...     db=db,
    ...     blocks=blocks
    ... )
    """
    # Validation
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db must be a data frame.")

    if not isinstance(blocks, pd.DataFrame):
        raise ValueError("blocks must be a data frame.")

    if 'rf' not in blocks.columns or 'block' not in blocks.columns:
        raise ValueError("blocks must have 'rf' and 'block' columns.")

    valid_methods = ['stepMIV', 'stepFWD', 'stepRPC']
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}.")

    if coding not in ['WoE', 'dummy']:
        raise ValueError("coding must be 'WoE' or 'dummy'.")

    if target not in db.columns:
        raise ValueError(f"target '{target}' not found in db.")

    if not 0 < p_value < 1:
        raise ValueError("p_value must be between 0 and 1.")

    # Check risk factors exist
    risk_factors = blocks['rf'].tolist()
    missing_rfs = [rf for rf in risk_factors if rf not in db.columns]
    if missing_rfs:
        raise ValueError(f"Risk factors not found in db: {missing_rfs}")

    # Initialize
    dev_db = db.copy()
    block_models = {}
    steps_list = []
    prediction_cols = []

    # Sort blocks
    block_names = sorted(blocks['block'].unique())

    # Process each block independently
    for block_name in block_names:
        block_rfs = blocks[blocks['block'] == block_name]['rf'].tolist()

        if not block_rfs:
            continue

        # Fit model for this block
        block_result = _fit_ensemble_block(
            method=method,
            target=target,
            db=dev_db,
            risk_factors=block_rfs,
            coding=coding,
            p_value=p_value,
            miv_threshold=miv_threshold,
            m_ch_p_val=m_ch_p_val
        )

        block_models[block_name] = block_result['model']

        # Record steps
        for step in block_result['steps']:
            step['block'] = block_name
            steps_list.append(step)

        # Store predictions
        pred_col = f'pred_{block_name}'
        dev_db[pred_col] = block_result['predictions']
        prediction_cols.append(pred_col)

    # Build ensemble model combining all block predictions
    y = dev_db[target].values

    if prediction_cols:
        X_ensemble = dev_db[prediction_cols].values
        X_ensemble = sm.add_constant(X_ensemble, has_constant='add')

        ensemble_glm = sm.GLM(y, X_ensemble, family=sm.families.Binomial())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ensemble_result = ensemble_glm.fit(disp=0)

        var_names = ['const'] + prediction_cols
        ensemble_df = pd.DataFrame({
            'variable': var_names,
            'coefficient': ensemble_result.params,
            'std_error': ensemble_result.bse,
            'z_value': ensemble_result.tvalues,
            'p_value': ensemble_result.pvalues
        })
    else:
        # Intercept-only model
        X_ensemble = np.ones((len(y), 1))
        ensemble_glm = sm.GLM(y, X_ensemble, family=sm.families.Binomial())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ensemble_result = ensemble_glm.fit(disp=0)

        ensemble_df = pd.DataFrame({
            'variable': ['const'],
            'coefficient': ensemble_result.params,
            'std_error': ensemble_result.bse,
            'z_value': ensemble_result.tvalues,
            'p_value': ensemble_result.pvalues
        })

    # Build steps DataFrame
    if steps_list:
        steps_df = pd.DataFrame(steps_list)
    else:
        steps_df = pd.DataFrame(columns=['iteration', 'rf_added', 'p_value', 'block'])

    return EnsembleBlocksResult(
        block_models=block_models,
        ensemble_model=ensemble_df,
        steps=steps_df,
        dev_db=dev_db
    )


def _fit_ensemble_block(
    method: str,
    target: str,
    db: pd.DataFrame,
    risk_factors: List[str],
    coding: str,
    p_value: float,
    miv_threshold: float,
    m_ch_p_val: float
) -> dict:
    """Fit a single block for ensemble using the specified method."""

    y = db[target].values
    selected_vars = []
    steps = []
    iteration = 0

    available_rfs = list(risk_factors)

    while available_rfs:
        iteration += 1
        best_rf = None
        best_pval = 1.0
        best_miv = 0.0

        for rf in available_rfs:
            try:
                candidate_vars = selected_vars + [rf]
                X = db[candidate_vars].values
                X = sm.add_constant(X, has_constant='add')

                model = sm.GLM(y, X, family=sm.families.Binomial())

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = model.fit(disp=0)

                rf_pval = result.pvalues[-1]
                rf_coef = result.params[-1]

                # Check trend for WoE coding
                if coding == 'WoE' and rf_coef < 0:
                    continue

                if method == 'stepMIV':
                    if rf_pval < best_pval and abs(rf_coef) > best_miv:
                        best_pval = rf_pval
                        best_miv = abs(rf_coef)
                        best_rf = rf
                else:
                    if rf_pval < best_pval:
                        best_pval = rf_pval
                        best_rf = rf

            except Exception:
                continue

        if best_rf is not None and best_pval < p_value:
            selected_vars.append(best_rf)
            available_rfs.remove(best_rf)

            steps.append({
                'iteration': iteration,
                'rf_added': best_rf,
                'p_value': best_pval
            })
        else:
            break

    # Fit final model
    if selected_vars:
        X = db[selected_vars].values
        X = sm.add_constant(X, has_constant='add')
    else:
        X = np.ones((len(y), 1))

    final_model = sm.GLM(y, X, family=sm.families.Binomial())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_result = final_model.fit(disp=0)

    # Build model DataFrame
    var_names = ['const'] + selected_vars
    model_df = pd.DataFrame({
        'variable': var_names,
        'coefficient': final_result.params,
        'std_error': final_result.bse,
        'z_value': final_result.tvalues,
        'p_value': final_result.pvalues
    })

    # Calculate predictions for ensemble
    predictions = final_result.predict(X)

    return {
        'model': model_df,
        'steps': steps,
        'predictions': predictions
    }
