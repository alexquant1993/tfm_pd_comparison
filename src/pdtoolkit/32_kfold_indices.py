"""K-fold cross-validation indices.

This module provides functionality to generate indices for
k-fold cross-validation with random or stratified sampling.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Union
import numpy as np
import pandas as pd


@dataclass
class FoldIndices:
    """Indices for a single fold.

    Attributes
    ----------
    estimation : np.ndarray
        Indices for the training/estimation set.
    validation : np.ndarray
        Indices for the validation/test set.
    """
    estimation: np.ndarray
    validation: np.ndarray


def kfold_idx(
    target: Union[np.ndarray, pd.Series, list],
    k: int = 10,
    type: Literal['random', 'stratified'] = 'random',
    seed: int = 2191
) -> Dict[str, FoldIndices]:
    """
    Generate k-fold cross-validation indices.

    This function creates indices for k-fold cross-validation,
    supporting both random and stratified sampling methods.

    Parameters
    ----------
    target : array-like
        Binary target variable (0/1). Used for stratification.
    k : int, default 10
        Number of folds.
    type : {'random', 'stratified'}, default 'random'
        Sampling type:
        - 'random': Simple random assignment to folds
        - 'stratified': Preserves class proportions in each fold
    seed : int, default 2191
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys 'k_1', 'k_2', ..., 'k_k', each containing
        a FoldIndices object with estimation and validation indices.

    Raises
    ------
    ValueError
        If type is invalid, k is negative, or target is not 0/1.

    Examples
    --------
    >>> import numpy as np
    >>> target = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> folds = kfold_idx(target, k=5, type='stratified', seed=42)
    >>> len(folds)
    5
    >>> len(folds['k_1'].validation)  # 2 observations per fold
    2
    """
    # Validation
    type_options = ['random', 'stratified']
    if type not in type_options:
        raise ValueError(f"type argument has to be one of: {', '.join(type_options)}.")

    if k < 0:
        raise ValueError("k cannot be negative.")

    # Convert to numpy array
    if isinstance(target, (pd.Series, list)):
        target = np.array(target)

    # Remove NA and validate
    target_clean = target[~np.isnan(target)]
    if not np.all(np.isin(target_clean, [0, 1])):
        raise ValueError("target is not 0/1 variable")

    # Get indices of non-NA values
    valid_indices = np.where(~np.isnan(target))[0]
    target_valid = target[valid_indices]
    tl = len(target_valid)

    # Check class frequencies
    n_zeros = np.sum(target_valid == 0)
    n_ones = np.sum(target_valid == 1)
    min_class_count = min(n_zeros, n_ones)

    # Adjust k if necessary
    original_k = k
    if min_class_count < k and min_class_count > 0:
        k = min_class_count
        # Warning would be issued in R

    if k > tl:
        k = tl
        if type == 'stratified':
            type = 'random'
        # Warning would be issued in R (LOOCV)

    np.random.seed(seed)

    if type == 'random':
        # Random shuffling
        idx = np.random.permutation(tl)
        # Assign to folds
        cv_folds = np.zeros(tl, dtype=int)
        fold_sizes = np.diff(np.linspace(0, tl, k + 1, dtype=int))
        fold_idx = 0
        pos = 0
        for size in fold_sizes:
            cv_folds[pos:pos + size] = fold_idx
            pos += size
            fold_idx += 1
    else:
        # Stratified sampling
        idx_0 = np.where(target_valid == 0)[0]
        idx_1 = np.where(target_valid == 1)[0]

        # Shuffle within each class
        np.random.shuffle(idx_0)
        np.random.shuffle(idx_1)

        # Assign folds within each class
        folds_0 = np.zeros(len(idx_0), dtype=int)
        folds_1 = np.zeros(len(idx_1), dtype=int)

        fold_sizes_0 = np.diff(np.linspace(0, len(idx_0), k + 1, dtype=int))
        fold_sizes_1 = np.diff(np.linspace(0, len(idx_1), k + 1, dtype=int))

        pos = 0
        for i, size in enumerate(fold_sizes_0):
            folds_0[pos:pos + size] = i
            pos += size

        pos = 0
        for i, size in enumerate(fold_sizes_1):
            folds_1[pos:pos + size] = i
            pos += size

        # Combine
        idx = np.concatenate([idx_0, idx_1])
        cv_folds = np.concatenate([folds_0, folds_1])

    # Create result
    result = {}
    for fold_num in range(k):
        est_mask = cv_folds != fold_num
        vld_mask = cv_folds == fold_num

        # Map back to original indices
        est_indices = valid_indices[idx[est_mask]]
        vld_indices = valid_indices[idx[vld_mask]]

        result[f'k_{fold_num + 1}'] = FoldIndices(
            estimation=est_indices,
            validation=vld_indices
        )

    return result
