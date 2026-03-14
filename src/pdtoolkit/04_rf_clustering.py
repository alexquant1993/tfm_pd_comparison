"""
Risk factor clustering module for PDtoolkit.

This module implements correlation-based clustering of risk factors using
hierarchical clustering (scipy.cluster.hierarchy).

Ported from: 04_RF_CLUSTERING.R
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from itertools import combinations


def rf_clustering(
    db: pd.DataFrame,
    metric: str,
    k: Optional[int] = None,
) -> pd.DataFrame:
    """
    Risk factor clustering using correlation-based distance.

    Clustering procedure is based on hierarchical clustering with centroid linkage.

    Args:
        db: DataFrame of risk factors for clustering (without target variable).
        metric: Correlation metric for distance calculation:
            - "raw pearson": dist = 1 - cor(pearson)
            - "raw spearman": dist = 1 - cor(spearman)
            - "common pearson": dist = (1 - cor(pearson)) / 2
            - "common spearman": dist = (1 - cor(spearman)) / 2
            - "absolute pearson": dist = 1 - |cor(pearson)|
            - "absolute spearman": dist = 1 - |cor(spearman)|
            - "sqrt pearson": dist = sqrt(1 - cor(pearson))
            - "sqrt spearman": dist = sqrt(1 - cor(spearman))
            - "x2y": dist = 1 - x2y metric (handles categorical and nonlinear)
        k: Number of clusters. If None, automatic elbow method is used.

    Returns:
        DataFrame with columns: rf, clusters, dist_to_centroid
        Ordered by cluster and distance to centroid.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pdtoolkit.rf_clustering import rf_clustering
        >>> df = pd.DataFrame({
        ...     'rf1': np.random.randn(100),
        ...     'rf2': np.random.randn(100),
        ...     'rf3': np.random.randn(100),
        ...     'rf4': np.random.randn(100),
        ... })
        >>> result = rf_clustering(df, metric='common spearman', k=2)
    """
    db_ncol = db.shape[1]
    if db_ncol < 4:
        raise ValueError("At least 4 risk factors have to be available in db argument.")

    metric_opt = [
        "raw pearson", "raw spearman", "common pearson", "common spearman",
        "absolute pearson", "absolute spearman", "sqrt pearson", "sqrt spearman", "x2y"
    ]
    if metric not in metric_opt:
        raise ValueError(f"metric argument has to be one of: {', '.join(metric_opt)}.")

    if metric != "x2y":
        # Check if all columns are numeric
        if not all(pd.api.types.is_numeric_dtype(db[col]) for col in db.columns):
            raise ValueError(f"For metric {metric} all risk factors have to be of numeric type.")

    # Calculate distance matrix
    if metric == "x2y":
        x2y_result = _dx2y(db)
        corr_matrix = x2y_result[1].values  # averaged x2y
        distance_matrix = 1 - corr_matrix
    else:
        method = "pearson" if "pearson" in metric else "spearman"
        corr_matrix = db.corr(method=method).values

        if "raw" in metric:
            distance_matrix = 1 - corr_matrix
        elif "common" in metric:
            distance_matrix = (1 - corr_matrix) / 2
        elif "absolute" in metric:
            distance_matrix = 1 - np.abs(corr_matrix)
        elif "sqrt" in metric:
            distance_matrix = np.sqrt(np.maximum(1 - corr_matrix, 0))
        else:
            distance_matrix = 1 - corr_matrix

    # Ensure distance matrix is symmetric and has zero diagonal
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # Convert to condensed form for hierarchical clustering
    # Handle any NaN values
    distance_matrix = np.nan_to_num(distance_matrix, nan=1.0)
    condensed_dist = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering with centroid linkage
    Z = linkage(condensed_dist, method='centroid')

    # Determine number of clusters
    if k is None:
        min_g = 3
        max_g = min(30, db_ncol)
        clusters_range = {}
        for ki in range(min_g, max_g + 1):
            clusters_range[ki] = fcluster(Z, ki, criterion='maxclust')
        k = _automatic_elbow(distance_matrix, clusters_range, db.columns)
    else:
        k = int(k)
        k = min(k, 100)
        k = min(k, db_ncol - 1)

    # Get cluster assignments
    cluster_labels = fcluster(Z, k, criterion='maxclust')

    # Create result dataframe
    rf_names = db.columns.tolist()
    result_df = pd.DataFrame({
        'rf': rf_names,
        'clusters': cluster_labels
    })

    # Calculate distance to centroid for each risk factor
    dist_to_centroid = _calc_dist_to_centroid(distance_matrix, cluster_labels, rf_names)
    result_df['dist_to_centroid'] = dist_to_centroid

    # Sort by cluster and distance
    result_df = result_df.sort_values(['clusters', 'dist_to_centroid']).reset_index(drop=True)

    return result_df


def _calc_mae_reduction(y_hat: np.ndarray, y_actual: np.ndarray) -> float:
    """Calculate mean absolute error reduction."""
    model_error = np.mean(np.abs(y_hat - y_actual))
    baseline = np.mean(y_actual)
    baseline_error = np.mean(np.abs(baseline - y_actual))

    if baseline_error == 0:
        return 0.0

    result = 1 - model_error / baseline_error
    return max(0.0, min(result, 1.0))


def _calc_misclass_reduction(y_hat: np.ndarray, y_actual: np.ndarray) -> float:
    """Calculate misclassification reduction."""
    model_error = np.mean(y_hat != y_actual)

    # Get majority class baseline error
    unique, counts = np.unique(y_actual, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    baseline_error = np.mean(majority_class != y_actual)

    if baseline_error == 0:
        return 0.0

    result = 1 - model_error / baseline_error
    return max(0.0, min(result, 1.0))


def _x2y_inner(x: np.ndarray, y: np.ndarray) -> float:
    """Inner x2y calculation using decision tree."""
    if len(np.unique(x)) == 1 or len(np.unique(y)) == 1:
        return np.nan

    min_leaf = max(30, int(0.1 * len(y)))

    x_reshaped = x.reshape(-1, 1)

    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
        # Continuous y - use regression tree
        try:
            tree = DecisionTreeRegressor(
                min_samples_split=min_leaf,
                min_samples_leaf=min_leaf
            )
            tree.fit(x_reshaped, y)
            preds = tree.predict(x_reshaped)
            return _calc_mae_reduction(preds, y)
        except Exception:
            return np.nan
    else:
        # Categorical y - use classification tree
        try:
            tree = DecisionTreeClassifier(
                min_samples_split=min_leaf,
                min_samples_leaf=min_leaf
            )
            tree.fit(x_reshaped, y)
            preds = tree.predict(x_reshaped)
            return _calc_misclass_reduction(preds, y)
        except Exception:
            return np.nan


def _x2y(x: pd.Series, y: pd.Series) -> float:
    """Calculate x2y metric between two variables."""
    # Remove missing values
    valid_mask = pd.notna(x) & pd.notna(y)
    x = x[valid_mask].values
    y = y[valid_mask].values

    if len(x) < 30:
        return np.nan

    return _x2y_inner(x, y)


def _dx2y(d: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate x2y distance matrix.

    Returns averaged x2y metric for each pair (symmetric matrix).
    """
    rf_names = d.columns.tolist()
    n_rf = len(rf_names)

    # Initialize result matrix
    x2y_raw = np.zeros((n_rf, n_rf))
    x2y_avg = np.zeros((n_rf, n_rf))

    # Calculate x2y for all pairs
    for i in range(n_rf):
        for j in range(n_rf):
            if i == j:
                x2y_raw[i, j] = 1.0
            else:
                x2y_raw[i, j] = _x2y(d.iloc[:, i], d.iloc[:, j])

    # Calculate averaged (symmetric) values
    for i in range(n_rf):
        for j in range(n_rf):
            if i == j:
                x2y_avg[i, j] = 1.0
            else:
                # Average of both directions
                vals = [x2y_raw[i, j], x2y_raw[j, i]]
                valid_vals = [v for v in vals if not np.isnan(v)]
                x2y_avg[i, j] = np.mean(valid_vals) if valid_vals else np.nan

    # Convert to DataFrames
    x2y_raw_df = pd.DataFrame(x2y_raw, index=rf_names, columns=rf_names)
    x2y_avg_df = pd.DataFrame(x2y_avg, index=rf_names, columns=rf_names)

    return x2y_raw_df, x2y_avg_df


def _automatic_elbow(
    distance_matrix: np.ndarray,
    clusters_dict: Dict[int, np.ndarray],
    rf_names: pd.Index,
) -> int:
    """
    Automatic elbow method for determining optimal number of clusters.

    Uses within-cluster sum of squares and finds the elbow point.
    """
    wss_results = []

    for nc, cluster_labels in clusters_dict.items():
        wss = _calc_wss(distance_matrix, cluster_labels)
        wss_results.append({'nc': nc, 'wss': wss})

    wss_df = pd.DataFrame(wss_results)

    if len(wss_df) < 3:
        return wss_df['nc'].iloc[0]

    # Find elbow using distance to line method
    n_points = len(wss_df)
    b = np.array([wss_df['nc'].iloc[0], wss_df['wss'].iloc[0]])
    c = np.array([wss_df['nc'].iloc[-1], wss_df['wss'].iloc[-1]])

    distances = []
    for i in range(n_points):
        a = np.array([wss_df['nc'].iloc[i], wss_df['wss'].iloc[i]])
        dist = _dist2l(a, b, c)
        distances.append(dist)

    elbow_idx = np.argmax(distances)
    return int(wss_df['nc'].iloc[elbow_idx])


def _calc_wss(distance_matrix: np.ndarray, cluster_labels: np.ndarray) -> float:
    """Calculate within-cluster sum of squares."""
    total_wss = 0

    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) <= 1:
            continue

        # Get submatrix for this cluster
        cluster_dist = distance_matrix[np.ix_(cluster_indices, cluster_indices)]

        # Calculate centroid (mean distance to all other cluster members)
        centroid = cluster_dist.mean(axis=0)

        # WSS = sum of squared distances to centroid
        wss = np.sum((cluster_dist - centroid) ** 2)
        total_wss += wss

    return total_wss


def _dist2l(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate perpendicular distance from point a to line defined by b and c."""
    v1 = b - c
    v2 = a - b

    # Cross product in 2D gives the area of parallelogram
    cross = np.abs(v1[0] * v2[1] - v1[1] * v2[0])

    # Divide by length of line segment to get perpendicular distance
    line_length = np.sqrt(np.sum(v1 ** 2))

    if line_length == 0:
        return 0

    return cross / line_length


def _calc_dist_to_centroid(
    distance_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    rf_names: List[str],
) -> List[float]:
    """Calculate distance to cluster centroid for each risk factor."""
    n_rf = len(rf_names)
    distances = []

    for i in range(n_rf):
        cluster_id = cluster_labels[i]
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        # Get distances to all cluster members
        cluster_distances = distance_matrix[i, cluster_indices]

        # Centroid distance = mean distance to all cluster members
        dist_to_centroid = np.mean(cluster_distances)
        distances.append(dist_to_centroid)

    return distances
