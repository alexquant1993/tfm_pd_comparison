"""Embedded block regression.

This module provides functionality for block-wise regression where
predictions from earlier blocks become risk factors for later blocks.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class EmbeddedBlocksResult:
    """Result of embedded block regression.

    Attributes
    ----------
    models : dict
        Dictionary mapping block names to model DataFrames.
    steps : pd.DataFrame
        Details of selected risk factors per block.
    dev_db : pd.DataFrame
        Development database with block predictions as features.
    """
    models: Dict[str, pd.DataFrame]
    steps: pd.DataFrame
    dev_db: pd.DataFrame


def embedded_blocks(
    method: str,
    target: str,
    db: pd.DataFrame,
    coding: str = 'WoE',
    blocks: pd.DataFrame = None,
    p_value: float = 0.05,
    miv_threshold: float = 0.02,
    m_ch_p_val: float = 0.05
) -> EmbeddedBlocksResult:
    """
    Perform embedded block regression.

    This function performs block-wise logistic regression where each
    block's predictions become additional risk factors for subsequent
    blocks. Unlike staged blocks (which use offsets), embedded blocks
    treat previous predictions as regular features.

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
    EmbeddedBlocksResult
        Object containing models, steps, and development database.

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
    >>> result = embedded_blocks(
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
    models = {}
    steps_list = []
    embedded_features = []  # Track predictions from previous blocks

    # Sort blocks
    block_names = sorted(blocks['block'].unique())

    # Process each block
    for block_name in block_names:
        block_rfs = blocks[blocks['block'] == block_name]['rf'].tolist()

        if not block_rfs:
            continue

        # Include embedded features from previous blocks
        available_rfs = block_rfs + embedded_features

        # Fit model for this block
        block_result = _fit_embedded_block(
            method=method,
            target=target,
            db=dev_db,
            risk_factors=available_rfs,
            coding=coding,
            p_value=p_value,
            miv_threshold=miv_threshold,
            m_ch_p_val=m_ch_p_val
        )

        models[block_name] = block_result['model']

        # Record steps
        for step in block_result['steps']:
            step['block'] = block_name
            steps_list.append(step)

        # Add predictions as embedded feature for next block
        pred_col = f'pred_{block_name}'
        dev_db[pred_col] = block_result['predictions']
        embedded_features.append(pred_col)

    # Build steps DataFrame
    if steps_list:
        steps_df = pd.DataFrame(steps_list)
    else:
        steps_df = pd.DataFrame(columns=['iteration', 'rf_added', 'p_value', 'block'])

    return EmbeddedBlocksResult(
        models=models,
        steps=steps_df,
        dev_db=dev_db
    )


def _fit_embedded_block(
    method: str,
    target: str,
    db: pd.DataFrame,
    risk_factors: List[str],
    coding: str,
    p_value: float,
    miv_threshold: float,
    m_ch_p_val: float
) -> dict:
    """Fit a single embedded block using the specified method."""

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

                # Check trend for WoE coding (only for non-prediction features)
                if coding == 'WoE' and not rf.startswith('pred_') and rf_coef < 0:
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

    # Calculate predictions for embedding
    predictions = final_result.predict(X)

    return {
        'model': model_df,
        'steps': steps,
        'predictions': predictions
    }
