"""SMOTE - Synthetic Minority Over-sampling Technique.

This module provides functionality for oversampling minority class
using SMOTE with HEOM (Heterogeneous Euclidean-Overlap Metric) distance.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def smote(
    db: pd.DataFrame,
    target: str,
    minority_class: Union[int, str],
    osr: float,
    ordinal_rf: Optional[List[str]] = None,
    num_rf_const: Optional[pd.DataFrame] = None,
    k: int = 5,
    seed: int = 81000
) -> pd.DataFrame:
    """
    Apply SMOTE oversampling to imbalanced dataset.

    This function creates synthetic samples for the minority class
    using the SMOTE algorithm with HEOM distance metric, which handles
    both numeric and categorical features.

    Parameters
    ----------
    db : pd.DataFrame
        Input data frame containing features and target.
    target : str
        Name of the target variable column.
    minority_class : int or str
        Value representing the minority class in target.
    osr : float
        Oversampling rate (e.g., 1.0 = 100% more minority samples).
    ordinal_rf : list of str, optional
        Names of ordinal categorical features to convert to intervals.
    num_rf_const : pd.DataFrame, optional
        Constraints for numeric features with columns:
        - rf: feature name
        - lower: lower bound
        - upper: upper bound
        - type: 'integer' or 'numeric'
    k : int, default 5
        Number of nearest neighbors to use.
    seed : int, default 81000
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Original data plus synthetic samples with 'smote' indicator column
        (0 = original, 1 = synthetic).

    Raises
    ------
    ValueError
        If inputs are invalid or constraints are violated.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> db = pd.DataFrame({
    ...     'target': [0]*90 + [1]*10,
    ...     'x1': np.random.randn(100),
    ...     'x2': np.random.choice(['A', 'B', 'C'], 100)
    ... })
    >>> result = smote(db, target='target', minority_class=1, osr=1.0)
    >>> result['smote'].sum()  # Number of synthetic samples
    10
    """
    # Input validation
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db is not a data frame.")

    rf_names = [c for c in db.columns if c != target]
    if len(rf_names) == 0:
        raise ValueError("only target supplied in db.")

    if target not in db.columns:
        raise ValueError("target does not exist in db.")

    if ordinal_rf is not None:
        missing_ord = [rf for rf in ordinal_rf if rf not in db.columns]
        if missing_ord:
            raise ValueError(f"ordinal_rf do not exist in db: {missing_ord}")

    if num_rf_const is not None:
        if not isinstance(num_rf_const, pd.DataFrame):
            raise ValueError("num_rf_const is not a data frame.")
        required_cols = ['rf', 'lower', 'upper', 'type']
        if not all(c in num_rf_const.columns for c in required_cols):
            raise ValueError(
                f"column names of num_rf_const data frame have to be: "
                f"{', '.join(required_cols)}."
            )
        valid_types = ['integer', 'numeric']
        if not all(t in valid_types for t in num_rf_const['type'].unique()):
            raise ValueError("column type of num_rf_const must be integer or numeric.")
        missing_rf = [rf for rf in num_rf_const['rf'] if rf not in db.columns]
        if missing_rf:
            raise ValueError(f"supplied risk factor(s) in num_rf_const do not exist in db.")

    # Check minority class exists
    target_vals = db[target].dropna().unique()
    if minority_class not in target_vals:
        raise ValueError("minority_class does not exist in target indicator.")

    if not isinstance(osr, (int, float)) or not isinstance(k, int):
        raise ValueError("osr and k have to be of numeric type.")

    if k <= 0 or k > 50:
        raise ValueError("k has to be between 1 and 50.")

    if osr <= 0:
        raise ValueError("oversampling rate has to be greater than 0.")

    # Work with copy
    db_c = db.copy()

    # Convert ordinal to interval
    if ordinal_rf is not None:
        for rf in ordinal_rf:
            db_c[rf] = _ord_to_int(db_c[rf])

    # Identify numeric features
    rf_features = [c for c in db_c.columns if c != target]
    rf_num_mask = {rf: pd.api.types.is_numeric_dtype(db_c[rf]) for rf in rf_features}
    rf_num = [rf for rf, is_num in rf_num_mask.items() if is_num]

    # Normalize numeric features for distance calculation
    for rf in rf_num:
        col = db_c[rf]
        col_range = col.max() - col.min()
        if col_range > 0:
            db_c[rf] = col / col_range

    # Get minority class samples
    db_mc = db[db[target] == minority_class].copy()
    db_c_mc = db_c[db_c[target] == minority_class].copy()
    mc_num = len(db_c_mc)

    # Number of synthetic samples
    os_num = round(mc_num * osr)

    # Sample indices
    np.random.seed(seed)
    replace = os_num > mc_num
    indx = np.random.choice(mc_num, os_num, replace=replace)

    # Adjust k if necessary
    if k > mc_num:
        k = mc_num

    # Find k nearest neighbors
    rf_num_indices = [i for i, rf in enumerate(rf_features) if rf_num_mask[rf]]
    res_knn = _heom(
        db=db_c_mc[rf_features].values,
        indx=indx,
        k=k,
        rfn=rf_num_indices,
        nc=len(rf_features)
    )

    # Create synthetic data
    db_s = _create_synthetic_data(
        db=db_mc[rf_features],
        res_knn=res_knn,
        rf_num=[rf_features.index(rf) for rf in rf_num],
        seed=seed
    )

    # Apply constraints
    if num_rf_const is not None:
        db_s = _synthetic_data_corr(db_s, num_rf_const)

    # Add target and smote indicator
    db_s[target] = minority_class
    db_s['smote'] = 1

    # Combine with original data
    db_orig = db.copy()
    db_orig['smote'] = 0

    # Reorder columns to match
    cols_order = list(db_orig.columns)
    db_s = db_s[cols_order]

    result = pd.concat([db_orig, db_s], ignore_index=True)

    return result


def _ord_to_int(x: pd.Series) -> pd.Series:
    """Convert ordinal values to integers."""
    unique_vals = sorted(x.dropna().unique())
    mapping = {v: i + 1 for i, v in enumerate(unique_vals)}
    return x.map(mapping)


def _heom(
    db: np.ndarray,
    indx: np.ndarray,
    k: int,
    rfn: List[int],
    nc: int
) -> pd.DataFrame:
    """
    Calculate HEOM distance and find k nearest neighbors.

    HEOM = Heterogeneous Euclidean-Overlap Metric
    """
    indx_unique = np.unique(indx)
    results = []

    for idx in indx_unique:
        count = np.sum(indx == idx)
        row = db[idx]

        # Calculate distances to all other points
        distances = np.zeros(len(db))
        for j in range(len(db)):
            distances[j] = _heom_distance(row, db[j], rfn, nc)

        # Find k nearest (excluding self)
        sorted_indices = np.argsort(distances)
        k_nearest = [i for i in sorted_indices if i != idx][:k]

        results.append({
            'indx': idx,
            'ss': count,
            'neighbors': k_nearest
        })

    return pd.DataFrame(results)


def _heom_distance(x: np.ndarray, y: np.ndarray, rfn: List[int], nc: int) -> float:
    """Calculate HEOM distance between two samples."""
    dist_components = np.zeros(nc)

    for i in range(nc):
        x_val = x[i]
        y_val = y[i]

        # Handle missing/infinite values
        if pd.isna(x_val) or pd.isna(y_val):
            dist_components[i] = 1.0
        elif i in rfn:
            # Numeric: normalized difference
            dist_components[i] = abs(x_val - y_val)
        else:
            # Categorical: overlap (0 if same, 1 if different)
            dist_components[i] = 0.0 if x_val == y_val else 1.0

    return np.sqrt(np.sum(dist_components ** 2))


def _create_synthetic_data(
    db: pd.DataFrame,
    res_knn: pd.DataFrame,
    rf_num: List[int],
    seed: int
) -> pd.DataFrame:
    """Create synthetic samples using SMOTE interpolation."""
    synthetic_rows = []
    columns = list(db.columns)

    for _, knn_row in res_knn.iterrows():
        idx = knn_row['indx']
        count = knn_row['ss']
        neighbors = knn_row['neighbors']

        original_row = db.iloc[idx].values

        for c in range(count):
            # Randomly select a neighbor
            np.random.seed(seed + idx + c)
            nn_idx = np.random.choice(neighbors)
            neighbor_row = db.iloc[nn_idx].values

            # Random interpolation factor
            rand_factor = np.random.rand()

            # Create synthetic sample
            synthetic = np.zeros(len(columns), dtype=object)

            for j in range(len(columns)):
                if j in rf_num:
                    # Numeric: interpolate
                    synthetic[j] = original_row[j] + rand_factor * (
                        neighbor_row[j] - original_row[j]
                    )
                else:
                    # Categorical: take neighbor's value
                    synthetic[j] = neighbor_row[j]

            synthetic_rows.append(synthetic)

    return pd.DataFrame(synthetic_rows, columns=columns)


def _synthetic_data_corr(db: pd.DataFrame, num_rf_const: pd.DataFrame) -> pd.DataFrame:
    """Apply constraints to synthetic data."""
    db = db.copy()

    for _, row in num_rf_const.iterrows():
        rf_name = row['rf']
        lower = row['lower']
        upper = row['upper']
        rf_type = row['type']

        if rf_name in db.columns:
            # Clip to bounds
            db[rf_name] = db[rf_name].clip(lower=lower, upper=upper)

            # Round if integer type
            if rf_type == 'integer':
                db[rf_name] = db[rf_name].round()

    return db
