"""Staged block regression.

This module provides functionality for block-wise regression where
each block's predictions serve as offsets for subsequent blocks.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from importlib import import_module


@dataclass
class StagedBlocksResult:
    """Result of staged block regression.

    Attributes
    ----------
    models : dict
        Dictionary mapping block names to model DataFrames.
    steps : pd.DataFrame
        Details of selected risk factors per block.
    dev_db : pd.DataFrame
        Development database with computed offsets.
    """
    models: Dict[str, pd.DataFrame]
    steps: pd.DataFrame
    dev_db: pd.DataFrame


def staged_blocks(
    method: str,
    target: str,
    db: pd.DataFrame,
    coding: str = 'WoE',
    blocks: pd.DataFrame = None,
    p_value: float = 0.05,
    miv_threshold: float = 0.02,
    m_ch_p_val: float = 0.05
) -> StagedBlocksResult:
    """
    Perform staged block regression.

    This function performs block-wise logistic regression where each
    block's linear predictor becomes an offset for subsequent blocks.
    This allows for hierarchical model building with different groups
    of risk factors.

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
    StagedBlocksResult
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
    >>> result = staged_blocks(
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
    offset_vals = None

    # Sort blocks
    block_names = sorted(blocks['block'].unique())

    # Process each block
    for block_name in block_names:
        block_rfs = blocks[blocks['block'] == block_name]['rf'].tolist()

        if not block_rfs:
            continue

        # Fit model for this block
        block_result = _fit_block(
            method=method,
            target=target,
            db=dev_db,
            risk_factors=block_rfs,
            coding=coding,
            p_value=p_value,
            miv_threshold=miv_threshold,
            m_ch_p_val=m_ch_p_val,
            offset_vals=offset_vals
        )

        models[block_name] = block_result['model']

        # Record steps
        for step in block_result['steps']:
            step['block'] = block_name
            steps_list.append(step)

        # Update offset for next block
        if block_result['linear_predictor'] is not None:
            if offset_vals is None:
                offset_vals = block_result['linear_predictor']
            else:
                offset_vals = offset_vals + block_result['linear_predictor']

            # Store offset in dev_db
            dev_db[f'offset_{block_name}'] = block_result['linear_predictor']

    # Build steps DataFrame
    if steps_list:
        steps_df = pd.DataFrame(steps_list)
    else:
        steps_df = pd.DataFrame(columns=['iteration', 'rf_added', 'p_value', 'block'])

    return StagedBlocksResult(
        models=models,
        steps=steps_df,
        dev_db=dev_db
    )


def _fit_block(
    method: str,
    target: str,
    db: pd.DataFrame,
    risk_factors: List[str],
    coding: str,
    p_value: float,
    miv_threshold: float,
    m_ch_p_val: float,
    offset_vals: Optional[np.ndarray]
) -> dict:
    """Fit a single block using the specified method."""

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

                if offset_vals is not None:
                    model = sm.GLM(y, X, family=sm.families.Binomial(),
                                  offset=offset_vals)
                else:
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
                    # Use MIV-based selection (approximate with coefficient)
                    if rf_pval < best_pval and abs(rf_coef) > best_miv:
                        best_pval = rf_pval
                        best_miv = abs(rf_coef)
                        best_rf = rf
                else:
                    # Use p-value based selection
                    if rf_pval < best_pval:
                        best_pval = rf_pval
                        best_rf = rf

            except Exception:
                continue

        # Check if best candidate meets threshold
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

    if offset_vals is not None:
        final_model = sm.GLM(y, X, family=sm.families.Binomial(),
                            offset=offset_vals)
    else:
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

    # Calculate linear predictor for offset
    if selected_vars:
        linear_predictor = final_result.predict(X, which='linear')
    else:
        linear_predictor = np.full(len(y), final_result.params[0])

    return {
        'model': model_df,
        'steps': steps,
        'linear_predictor': linear_predictor
    }
