"""
Univariate analysis module for PDtoolkit.

This module provides functions for univariate statistical analysis of risk factors,
including summary statistics for numeric and categorical variables, and imputation
methods for special cases and outliers.

Ported from: 01_UNIVARIATE_ANALYSIS.R
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy import stats
from importlib import import_module as _import

_globals = _import('.00_globals', package='pdtoolkit')
DEFAULT_SPECIAL_CASES = _globals.DEFAULT_SPECIAL_CASES
DEFAULT_SC_THRESHOLD = _globals.DEFAULT_SC_THRESHOLD
DEFAULT_SC_REPLACEMENT = _globals.DEFAULT_SC_REPLACEMENT
DEFAULT_P_VALUE = _globals.DEFAULT_P_VALUE


def univariate(
    db: pd.DataFrame,
    sc: Optional[List[Any]] = None,
    sc_method: str = "together",
    sc_threshold: float = DEFAULT_SC_THRESHOLD,
) -> pd.DataFrame:
    """
    Perform univariate analysis on risk factors.

    For numeric risk factors, the report includes:
    - rf: Risk factor name
    - rf_type: Risk factor class (always 'numeric')
    - bin_type: Bin type - special or complete cases
    - bin: Bin label
    - pct: Percentage of observations in each bin
    - cnt_unique: Number of unique values per bin
    - min, p1, p5, p25, p50, p75, p95, p99, max: Distribution statistics
    - avg: Mean value
    - avg_se: Standard error of the mean
    - neg: Number of negative values
    - pos: Number of positive values
    - cnt_outliers: Number of outliers (Q75 +/- 1.5 * IQR)
    - sc_ind: Special case indicator (1 if share exceeds threshold)

    For categorical risk factors, the report includes:
    - rf: Risk factor name
    - rf_type: Risk factor class (character, category, or bool)
    - bin_type: Bin type - special or complete cases
    - bin: Bin label
    - pct: Percentage of observations in each bin
    - cnt_unique: Number of unique values per bin
    - sc_ind: Special case indicator

    Args:
        db: DataFrame of risk factors for univariate analysis.
        sc: List of special case elements. Default is [None, np.nan, np.inf, -np.inf].
        sc_method: How to treat special cases - "together" or "separately".
        sc_threshold: Threshold for special cases as percentage of total observations.

    Returns:
        DataFrame with univariate metrics for all analyzed risk factors.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pdtoolkit.univariate_analysis import univariate
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35, np.nan, 45],
        ...     'category': ['A', 'B', 'A', None, 'B']
        ... })
        >>> result = univariate(df)
    """
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db is not a DataFrame.")

    if sc is None:
        sc = DEFAULT_SPECIAL_CASES.copy()

    scm_opts = ["together", "separately"]
    if sc_method not in scm_opts:
        raise ValueError(f"sc_method argument has to be one of: {', '.join(scm_opts)}")

    results = []
    for col in db.columns:
        col_data = db[col]

        # Skip non-numeric and non-categorical types (like datetime)
        if not (
            pd.api.types.is_numeric_dtype(col_data)
            or pd.api.types.is_string_dtype(col_data)
            or isinstance(col_data.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(col_data)
            or pd.api.types.is_object_dtype(col_data)
        ):
            continue

        # Create bin labels
        if sc_method == "together":
            bin_labels = pd.Series(
                ["special cases" if _is_special_case(v, sc) else "complete cases"
                 for v in col_data],
                index=db.index
            )
        else:
            bin_labels = pd.Series(
                [str(v) if _is_special_case(v, sc) else "complete cases"
                 for v in col_data],
                index=db.index
            )

        bin_types = pd.Series(
            ["special cases" if _is_special_case(v, sc) else "complete cases"
             for v in col_data],
            index=db.index
        )

        if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):
            result = _univariate_num(col_data, col, bin_labels, bin_types, sc, sc_threshold)
        else:
            result = _univariate_cat(col_data, col, bin_labels, bin_types, sc, sc_threshold)

        if result is not None:
            results.append(result)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def _is_special_case(value: Any, sc: List[Any]) -> bool:
    """Check if a value is a special case."""
    if pd.isna(value):
        return any(pd.isna(s) or s is None for s in sc)
    if isinstance(value, float) and np.isinf(value):
        return np.inf in sc or -np.inf in sc
    return value in sc


def _univariate_num(
    col_data: pd.Series,
    col_name: str,
    bin_labels: pd.Series,
    bin_types: pd.Series,
    sc: List[Any],
    sc_threshold: float,
) -> pd.DataFrame:
    """Calculate univariate statistics for a numeric variable."""
    results = []
    total_n = len(col_data)

    for bin_type in ["special cases", "complete cases"]:
        mask = bin_types == bin_type
        if not mask.any():
            continue

        subset = col_data[mask]
        bin_label = bin_labels[mask].iloc[0] if mask.any() else bin_type

        # Calculate statistics
        cnt = len(subset)
        pct = cnt / total_n if total_n > 0 else 0
        cnt_unique = subset.nunique()

        if bin_type == "complete cases":
            # Filter out special cases for calculations
            valid_data = subset.dropna()
            valid_data = valid_data[~np.isinf(valid_data)]

            if len(valid_data) > 0:
                min_val = valid_data.min()
                max_val = valid_data.max()
                avg = valid_data.mean()
                avg_se = valid_data.std() / np.sqrt(len(valid_data)) if len(valid_data) > 1 else np.nan

                # Percentiles
                p1 = np.percentile(valid_data, 1)
                p5 = np.percentile(valid_data, 5)
                p25 = np.percentile(valid_data, 25)
                p50 = np.percentile(valid_data, 50)
                p75 = np.percentile(valid_data, 75)
                p95 = np.percentile(valid_data, 95)
                p99 = np.percentile(valid_data, 99)

                # Counts
                neg = (valid_data < 0).sum()
                pos = (valid_data > 0).sum()

                # Outliers (IQR method)
                iqr = p75 - p25
                lower_bound = p25 - 1.5 * iqr
                upper_bound = p75 + 1.5 * iqr
                cnt_outliers = ((valid_data < lower_bound) | (valid_data > upper_bound)).sum()
            else:
                min_val = max_val = avg = avg_se = np.nan
                p1 = p5 = p25 = p50 = p75 = p95 = p99 = np.nan
                neg = pos = cnt_outliers = 0
        else:
            min_val = max_val = avg = avg_se = np.nan
            p1 = p5 = p25 = p50 = p75 = p95 = p99 = np.nan
            neg = pos = cnt_outliers = np.nan

        results.append({
            "rf": col_name,
            "rf_type": "numeric",
            "bin_type": bin_type,
            "bin": bin_label,
            "cnt": cnt,
            "pct": pct,
            "cnt_unique": cnt_unique,
            "min": min_val,
            "p1": p1,
            "p5": p5,
            "p25": p25,
            "p50": p50,
            "avg": avg,
            "avg_se": avg_se,
            "p75": p75,
            "p95": p95,
            "p99": p99,
            "max": max_val,
            "neg": neg,
            "pos": pos,
            "cnt_outliers": cnt_outliers,
        })

    df = pd.DataFrame(results)

    # Calculate sc_ind
    if not df.empty:
        sc_pct = df.loc[df["bin_type"] == "special cases", "pct"].sum()
        df["sc_ind"] = 1 if sc_pct > sc_threshold else 0

    return df


def _univariate_cat(
    col_data: pd.Series,
    col_name: str,
    bin_labels: pd.Series,
    bin_types: pd.Series,
    sc: List[Any],
    sc_threshold: float,
) -> pd.DataFrame:
    """Calculate univariate statistics for a categorical variable."""
    results = []
    total_n = len(col_data)

    # Determine rf_type
    if pd.api.types.is_bool_dtype(col_data):
        rf_type = "logical"
    elif isinstance(col_data.dtype, pd.CategoricalDtype):
        rf_type = "factor"
    else:
        rf_type = "character"

    for bin_type in ["special cases", "complete cases"]:
        mask = bin_types == bin_type
        if not mask.any():
            continue

        subset = col_data[mask]
        bin_label = bin_labels[mask].iloc[0] if mask.any() else bin_type

        cnt = len(subset)
        pct = cnt / total_n if total_n > 0 else 0
        cnt_unique = subset.nunique()

        results.append({
            "rf": col_name,
            "rf_type": rf_type,
            "bin_type": bin_type,
            "bin": bin_label,
            "cnt": cnt,
            "pct": pct,
            "cnt_unique": cnt_unique,
        })

    df = pd.DataFrame(results)

    # Calculate sc_ind
    if not df.empty:
        sc_pct = df.loc[df["bin_type"] == "special cases", "pct"].sum()
        df["sc_ind"] = 1 if sc_pct > sc_threshold else 0

    return df


def _pearson_norm_test(x: np.ndarray) -> Dict[str, float]:
    """
    Perform Pearson's chi-squared normality test.

    Args:
        x: Array of values to test.

    Returns:
        Dictionary with 'test_stat' and 'p_val'.
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    n = len(x)

    if n < 10:
        return {"test_stat": np.nan, "p_val": np.nan}

    n_classes = int(np.ceil(2 * (n ** (2 / 5))))

    # Calculate expected frequencies under normal distribution
    mean_x = np.mean(x)
    std_x = np.std(x, ddof=0)

    if std_x == 0:
        return {"test_stat": np.nan, "p_val": np.nan}

    # Assign to classes
    num = np.floor(1 + n_classes * stats.norm.cdf(x, mean_x, std_x)).astype(int)
    num = np.clip(num, 1, n_classes)

    # Count observations in each class
    count = np.bincount(num, minlength=n_classes + 1)[1:]

    # Expected count per class
    prob = 1 / n_classes
    expected = n * prob

    # Chi-squared test statistic
    test_stat = np.sum(((count - expected) ** 2) / expected)

    # Degrees of freedom: n_classes - 2 - 1 (mean and std estimated)
    df = n_classes - 3
    if df <= 0:
        return {"test_stat": test_stat, "p_val": np.nan}

    p_val = 1 - stats.chi2.cdf(test_stat, df)

    return {"test_stat": test_stat, "p_val": p_val}


def imp_sc(
    db: pd.DataFrame,
    sc_all: Optional[List[Any]] = None,
    sc_replace: Optional[List[Any]] = None,
    method_num: str = "automatic",
    p_val: float = DEFAULT_P_VALUE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute values for special cases.

    Args:
        db: DataFrame of risk factors for imputation.
        sc_all: List of all special case elements. Default is [None, np.nan, np.inf, -np.inf].
        sc_replace: List of special case elements to replace. Default same as sc_all.
        method_num: Imputation method for numeric variables. Options:
            "automatic", "mean", "median", "zero".
        p_val: Significance level for Pearson normality test (used with "automatic").

    Returns:
        Tuple of (imputed DataFrame, imputation report DataFrame).

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pdtoolkit.univariate_analysis import imp_sc
        >>> df = pd.DataFrame({'age': [25, 30, np.nan, 40, 45]})
        >>> imputed_df, report = imp_sc(df, method_num='mean')
    """
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db is not a DataFrame.")

    if sc_all is None:
        sc_all = DEFAULT_SPECIAL_CASES.copy()
    if sc_replace is None:
        sc_replace = sc_all.copy()

    method_opts = ["automatic", "mean", "median", "zero"]
    if method_num not in method_opts:
        raise ValueError(f"method_num argument has to be one of: {', '.join(method_opts)}")

    if not 0 <= p_val <= 1:
        raise ValueError("p_val has to be between 0 and 1.")

    db = db.copy()
    reports = []

    for col in db.columns:
        col_data = db[col]

        # Get complete cases (not special)
        complete_mask = ~_series_isin_special(col_data, sc_all)
        complete_count = complete_mask.sum()

        if complete_count < 0.1 * len(col_data):
            reports.append({
                "rf": col,
                "info": "Less than 10% of complete cases.",
                "imputation_method": None,
            })
            continue

        if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):
            complete_data = col_data[complete_mask].values

            # Determine imputation value
            if method_num == "automatic":
                norm_test = _pearson_norm_test(complete_data)
                if norm_test["p_val"] < p_val:
                    imp_val = np.nanmedian(complete_data)
                    auto_algo = "median"
                else:
                    imp_val = np.nanmean(complete_data)
                    auto_algo = "mean"
            elif method_num == "mean":
                imp_val = np.nanmean(complete_data)
                auto_algo = None
            elif method_num == "median":
                imp_val = np.nanmedian(complete_data)
                auto_algo = None
            else:  # zero
                imp_val = 0
                auto_algo = None

            if pd.isna(imp_val) or np.isinf(imp_val):
                reports.append({
                    "rf": col,
                    "info": "Imputed value cannot be calculated.",
                    "imputation_method": method_num,
                })
                continue

            # Apply imputation
            replace_mask = _series_isin_special(col_data, sc_replace)
            imputation_num = replace_mask.sum()

            # Round if original data is integer
            if pd.api.types.is_integer_dtype(col_data):
                imp_val = round(imp_val)

            db.loc[replace_mask, col] = imp_val

            reports.append({
                "rf": col,
                "info": "Imputation completed.",
                "imputation_method": f"automatic - {auto_algo}" if method_num == "automatic" else method_num,
                "imputed_value": imp_val,
                "imputation_num": imputation_num,
            })

        else:
            # Categorical imputation - use mode
            complete_data = col_data[complete_mask]
            if len(complete_data) == 0:
                reports.append({
                    "rf": col,
                    "info": "Imputed value (mode) cannot be calculated.",
                    "imputation_method": "mode",
                })
                continue

            mode_val = complete_data.mode()
            if len(mode_val) == 0:
                reports.append({
                    "rf": col,
                    "info": "Imputed value (mode) cannot be calculated.",
                    "imputation_method": "mode",
                })
                continue

            mode_val = mode_val.iloc[0]

            replace_mask = _series_isin_special(col_data, sc_replace)
            imputation_num = replace_mask.sum()

            db.loc[replace_mask, col] = mode_val

            reports.append({
                "rf": col,
                "info": "Imputation completed.",
                "imputation_method": "mode",
                "imputed_mode": mode_val,
                "imputation_num": imputation_num,
            })

    report_df = pd.DataFrame(reports)
    return db, report_df


def _series_isin_special(series: pd.Series, special_cases: List[Any]) -> pd.Series:
    """Check if each element in series is a special case."""
    result = pd.Series(False, index=series.index)

    for sc in special_cases:
        if sc is None or (isinstance(sc, float) and np.isnan(sc)):
            result |= pd.isna(series)
        elif isinstance(sc, float) and np.isinf(sc):
            if sc > 0:
                result |= (series == np.inf)
            else:
                result |= (series == -np.inf)
        else:
            result |= (series == sc)

    return result


def imp_outliers(
    db: pd.DataFrame,
    sc: Optional[List[Any]] = None,
    method: str = "iqr",
    range_val: float = 1.5,
    upper_pct: float = 0.95,
    lower_pct: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replace outliers with less extreme values.

    This procedure is applicable only to numeric risk factors.

    Args:
        db: DataFrame of risk factors for imputation.
        sc: List of special case elements to exclude. Default is [None, np.nan, np.inf, -np.inf].
        method: Imputation method - "iqr" or "percentile".
        range_val: For IQR method, multiplier for IQR (default 1.5).
        upper_pct: Upper percentile limit for percentile method (default 0.95).
        lower_pct: Lower percentile limit for percentile method (default 0.05).

    Returns:
        Tuple of (imputed DataFrame, imputation report DataFrame).

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pdtoolkit.univariate_analysis import imp_outliers
        >>> df = pd.DataFrame({'value': [1, 2, 3, 4, 100]})
        >>> imputed_df, report = imp_outliers(df, method='iqr')
    """
    if not isinstance(db, pd.DataFrame):
        raise ValueError("db is not a DataFrame.")

    if sc is None:
        sc = DEFAULT_SPECIAL_CASES.copy()

    method_opts = ["iqr", "percentile"]
    if method not in method_opts:
        raise ValueError(f"method argument has to be one of: {', '.join(method_opts)}")

    if not isinstance(range_val, (int, float)):
        raise ValueError("range_val argument has to be numeric.")

    if not (0 <= upper_pct <= 1 and 0 <= lower_pct <= 1 and lower_pct < upper_pct):
        raise ValueError(
            "upper_pct and lower_pct arguments have to be between 0 and 1. "
            "Additionally, lower_pct has to be less than upper_pct."
        )

    db = db.copy()
    reports = []

    for col in db.columns:
        col_data = db[col]

        if not pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
            reports.append({
                "rf": col,
                "info": "Risk factor is not of numeric type.",
                "imputation_method": None,
            })
            continue

        # Get complete cases
        complete_mask = ~_series_isin_special(col_data, sc)
        complete_data = col_data[complete_mask].values
        complete_count = len(complete_data)

        if complete_count < 0.1 * len(col_data):
            reports.append({
                "rf": col,
                "info": "Less than 10% of complete cases.",
                "imputation_method": None,
            })
            continue

        try:
            if method == "iqr":
                q25 = np.percentile(complete_data, 25)
                q75 = np.percentile(complete_data, 75)
                iqr = q75 - q25
                lower_bound = q25 - range_val * iqr
                upper_bound = q75 + range_val * iqr
            else:  # percentile
                lower_bound = np.percentile(complete_data, lower_pct * 100)
                upper_bound = np.percentile(complete_data, upper_pct * 100)

            if pd.isna(upper_bound) or pd.isna(lower_bound) or np.isinf(upper_bound) or np.isinf(lower_bound):
                reports.append({
                    "rf": col,
                    "info": "Imputed value cannot be calculated.",
                    "imputation_method": None,
                })
                continue

        except Exception:
            reports.append({
                "rf": col,
                "info": "Imputed value cannot be calculated.",
                "imputation_method": None,
            })
            continue

        # Apply imputation
        upper_mask = complete_mask & (col_data > upper_bound)
        lower_mask = complete_mask & (col_data < lower_bound)

        num_upper = upper_mask.sum()
        num_lower = lower_mask.sum()

        # Convert to float if needed to avoid dtype incompatibility warnings
        if pd.api.types.is_integer_dtype(db[col]) and (num_upper > 0 or num_lower > 0):
            db[col] = db[col].astype(float)

        db.loc[upper_mask, col] = upper_bound
        db.loc[lower_mask, col] = lower_bound

        reports.append({
            "rf": col,
            "info": "Imputation completed.",
            "imputation_method": method,
            "imputation_val_upper": upper_bound,
            "imputation_val_lower": lower_bound,
            "imputation_num_upper": num_upper,
            "imputation_num_lower": num_lower,
        })

    report_df = pd.DataFrame(reports)
    return db, report_df
