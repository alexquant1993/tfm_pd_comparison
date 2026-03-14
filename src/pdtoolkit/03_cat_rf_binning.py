"""
Categorical risk factor binning module for PDtoolkit.

This module implements a three-stage binning procedure for categorical risk factors:
1. Correction for minimum percentage of observations
2. Correction for target rate (default rate)
3. Correction for maximum number of bins using Adjacent Pooling Algorithm (APA)

Ported from: 03_CAT_RF_BINNING.R
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from importlib import import_module as _import

_bivariate = _import('.02_bivariate_analysis', package='pdtoolkit')
woe_tbl = _bivariate.woe_tbl


def cat_bin(
    x: pd.Series,
    y: pd.Series,
    sc: Optional[Any] = None,
    sc_merge: str = "none",
    min_pct_obs: float = 0.05,
    min_avg_rate: float = 0.01,
    max_groups: Optional[int] = None,
    force_trend: str = "modalities",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Categorical risk factor binning with three-stage procedure.

    Stage 1: Correction for minimum percentage of observations
    Stage 2: Correction for target rate (default rate)
    Stage 3: Correction for maximum number of bins (APA algorithm)

    Args:
        x: Categorical risk factor (Series).
        y: Binary target variable (0/1).
        sc: Special case elements to treat separately. Default None.
        sc_merge: How to treat special cases:
            - "none": Keep in separate bin
            - "first": Merge with first bin
            - "last": Merge with last bin
            - "closest": Merge with closest bin by default rate
        min_pct_obs: Minimum percentage of observations per bin (default 0.05, min 30 obs).
        min_avg_rate: Minimum default rate (default 0.01, min 1 bad case).
        max_groups: Maximum number of bins. If None, no limit applied.
        force_trend: How to order bins for merging:
            - "modalities": Alphabetic order
            - "dr": Increasing default rate order

    Returns:
        Tuple of (summary_tbl DataFrame, x_trans Series of transformed values).

    Examples:
        >>> import pandas as pd
        >>> from pdtoolkit.cat_rf_binning import cat_bin
        >>> x = pd.Series(['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E'])
        >>> y = pd.Series([0, 0, 0, 1, 0, 1, 1, 1, 0, 1])
        >>> summary, x_trans = cat_bin(x, y, max_groups=3)
    """
    # Convert to Series if needed
    x = pd.Series(x) if not isinstance(x, pd.Series) else x.copy()
    y = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()

    # Validate target is 0/1
    y_valid = y.dropna()
    if not set(y_valid.unique()).issubset({0, 1}):
        raise ValueError("Target is not 0/1 variable.")

    # Validate x is categorical
    is_categorical = (
        pd.api.types.is_string_dtype(x)
        or isinstance(x.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(x)
        or pd.api.types.is_object_dtype(x)
    )
    if not is_categorical:
        raise ValueError("Inappropriate class. It has to be one of: character, factor or logical.")

    # Validate arguments
    if not isinstance(min_pct_obs, (int, float)) or not isinstance(min_avg_rate, (int, float)):
        raise ValueError("min_pct_obs and min_avg_rate have to be numeric.")

    force_trend_opt = ["modalities", "dr"]
    if force_trend not in force_trend_opt:
        raise ValueError(f"force_trend argument has to be one of: {', '.join(force_trend_opt)}.")

    sc_merge_opt = ["none", "first", "last", "closest"]
    if sc_merge not in sc_merge_opt:
        raise ValueError(f"sc_merge argument has to be one of: {', '.join(sc_merge_opt)}.")

    # Normalize special cases to a list
    if sc is None:
        sc_list = [None]
    elif isinstance(sc, (list, tuple)):
        sc_list = list(sc)
    else:
        sc_list = [sc]

    # Create working dataframe, exclude missing y
    d = pd.DataFrame({'y': y, 'x': x})
    d = d[d['y'].notna()].copy()
    d = d.reset_index(drop=True)

    # Identify complete cases (not special)
    d_cc = d[~_is_in_special(d['x'], sc_list)].copy()

    # Run checks
    check_result = _checks(d, d_cc)
    if check_result[0] > 0:
        # Return error message as dataframe
        error_df = pd.DataFrame({'bin': [check_result[1]]})
        return error_df, pd.Series(dtype=object)

    # Calculate minimum observations and rate thresholds
    nr = len(d)
    min_obs = int(np.ceil(max(30, nr * min_pct_obs)))
    nd = d['y'].sum()
    min_rate = int(np.ceil(max(1, nd * min_avg_rate)))

    # Initial summary table
    ds = d.groupby('x', dropna=False).agg(
        no=('y', 'count'),
        nb=('y', 'sum')
    ).reset_index()
    ds.columns = ['bin', 'no', 'nb']
    ds['dr'] = ds['nb'] / ds['no']

    # Sort based on force_trend
    if force_trend == "dr":
        ds = ds.sort_values('dr').reset_index(drop=True)
    else:
        ds = ds.sort_values('bin').reset_index(drop=True)

    # Handle special cases merging
    sc_replace = None
    has_sc = _is_in_special(d['x'], sc_list).any()

    if has_sc:
        if sc_merge == "none":
            sc_replace = d.loc[_is_in_special(d['x'], sc_list), 'x'].values
        elif sc_merge == "first":
            non_sc_bins = ds.loc[~_is_in_special(ds['bin'], sc_list), 'bin']
            sc_replace = non_sc_bins.iloc[0] if len(non_sc_bins) > 0 else None
        elif sc_merge == "last":
            non_sc_bins = ds.loc[~_is_in_special(ds['bin'], sc_list), 'bin']
            sc_replace = non_sc_bins.iloc[-1] if len(non_sc_bins) > 0 else None
        elif sc_merge == "closest":
            sc_replace = _find_closest(ds, sc_list)

        if sc_replace is not None and sc_merge != "none":
            d.loc[_is_in_special(d['x'], sc_list), 'x'] = sc_replace

            # Rebuild summary table after merging
            ds = d.groupby('x', dropna=False).agg(
                no=('y', 'count'),
                nb=('y', 'sum')
            ).reset_index()
            ds.columns = ['bin', 'no', 'nb']
            ds['dr'] = ds['nb'] / ds['no']

            if force_trend == "dr":
                ds = ds.sort_values('dr').reset_index(drop=True)
            else:
                ds = ds.sort_values('bin').reset_index(drop=True)

    ds['group'] = range(1, len(ds) + 1)

    # Stage 1: Correction for number of observations
    ds_no = _tbl_correction(
        ds[~_is_in_special(ds['bin'], sc_list)].copy(),
        min_obs, min_rate, what="obs"
    )
    ds_no_1 = ds_no[0]

    # Stage 2: Correction for minimum number of bads
    ds_nb = _tbl_correction(ds_no[1].copy(), min_obs, min_rate, what="bad")
    ds_nb_1 = ds_nb[0]

    # Merge labels
    ds_cor = ds_no_1.merge(
        ds_nb_1[['bin', 'group']],
        left_on='label',
        right_on='bin',
        how='left',
        suffixes=('', '_y')
    )
    ds_cor['label'] = _format_labels(ds_cor['group_y'].values, ds_cor['bin'].values)

    # Apply labels to data
    d = d.merge(ds_cor[['bin', 'label']], left_on='x', right_on='bin', how='left')
    d.loc[_is_in_special(d['x'], sc_list), 'label'] = d.loc[_is_in_special(d['x'], sc_list), 'x']

    # Stage 3: Correction for max number of bins
    woe_df = pd.DataFrame({'label': d['label'], 'y': d['y']})
    ds = woe_tbl(woe_df, x='label', y='y')

    if max_groups is None or len(ds) <= max_groups:
        summary_tbl = ds
        x_trans = d['label']
    else:
        # Apply APA algorithm
        ds_sc = ds[_is_in_special(ds['bin'], sc_list)]
        ds_cc = ds[~_is_in_special(ds['bin'], sc_list)].copy()

        no_sc = ds_sc['no'].sum() if len(ds_sc) > 0 else 0
        ng_sc = ds_sc['ng'].sum() if len(ds_sc) > 0 else 0
        nb_sc = ds_sc['nb'].sum() if len(ds_sc) > 0 else 0

        tbl_apa = _apa(ds_cc, no_sc, ng_sc, nb_sc, max_groups)

        # Merge APA results back
        d = d.merge(tbl_apa[['bin', 'label']], left_on='label', right_on='bin',
                    how='left', suffixes=('', '_apa'))

        # Rebuild labels
        ds_new = d[~_is_in_special(d['x'], sc_list)].groupby('x').agg(
            g=('label_apa', 'first')
        ).reset_index()
        ds_new.columns = ['bin', 'g']
        ds_new['label'] = _format_labels(
            pd.factorize(ds_new['g'])[0] + 1,
            ds_new['bin'].values
        )

        d = d[['x', 'y']].merge(ds_new[['bin', 'label']], left_on='x', right_on='bin', how='left')
        d.loc[_is_in_special(d['x'], sc_list), 'label'] = d.loc[_is_in_special(d['x'], sc_list), 'x']

        woe_df = pd.DataFrame({'label': d['label'], 'y': d['y']})
        summary_tbl = woe_tbl(woe_df, x='label', y='y')
        x_trans = d['label']

    # Add sc.bin column if special cases exist
    if has_sc:
        if sc_merge == "none":
            summary_tbl['sc_bin'] = "none"
        else:
            summary_tbl['sc_bin'] = f"{sc_merge} & {sc_replace}"

    return summary_tbl, x_trans


def _is_in_special(series: pd.Series, sc_list: List[Any]) -> pd.Series:
    """Check if values in series are in the special cases list."""
    result = pd.Series(False, index=series.index)
    for sc in sc_list:
        if sc is None or (isinstance(sc, float) and np.isnan(sc)):
            result |= pd.isna(series)
        else:
            result |= (series == sc)
    return result


def _find_closest(ds: pd.DataFrame, sc_list: List[Any]) -> Any:
    """Find the bin closest to special cases by default rate."""
    sc_mask = _is_in_special(ds['bin'], sc_list)
    sc_rows = ds[sc_mask]
    cc_rows = ds[~sc_mask]

    if len(sc_rows) == 0 or len(cc_rows) == 0:
        return None

    dr_sc = sc_rows['nb'].sum() / sc_rows['no'].sum() if sc_rows['no'].sum() > 0 else 0
    dr_diff = np.abs(cc_rows['dr'] - dr_sc)

    return cc_rows.loc[dr_diff.idxmin(), 'bin']


def _checks(d: pd.DataFrame, d_cc: pd.DataFrame) -> Tuple[int, str]:
    """Run validation checks on the data."""
    if len(d_cc) == 0:
        return (4, "no complete cases")

    if d_cc['y'].nunique() == 1:
        return (1, "y has single unique value for complete cases")

    if d_cc['x'].nunique() == 1:
        return (2, "x has single unique value for complete cases")

    y_valid = d['y'].dropna()
    if not set(y_valid.unique()).issubset({0, 1}):
        return (3, "y is not 0/1 variable")

    return (0, "")


def _tbl_correction(
    tbl: pd.DataFrame,
    mno: int,
    mrate: int,
    what: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Correct table for minimum observations or bad rate.

    Args:
        tbl: Summary table with bin, no, nb columns
        mno: Minimum number of observations
        mrate: Minimum number of bads
        what: "obs" for observations or "bad" for bad rate correction

    Returns:
        Tuple of (corrected table with labels, new aggregated table)
    """
    if what == "obs":
        cn = "no"
        thr = mno
    else:
        cn = "nb"
        thr = mrate

    tbl = tbl.copy()
    tbl['group'] = range(1, len(tbl) + 1)
    tbl_s = tbl.copy()

    while True:
        if len(tbl) == 1:
            break

        values = tbl[cn].values
        if all(values >= thr):
            break

        gap = np.argmin(values >= thr)

        if gap == len(tbl) - 1:
            # Merge with previous
            gm = tbl['group'].iloc[-2:].values
            gr = tbl['group'].iloc[-2]
            tbl_s.loc[tbl_s['group'].isin(gm), 'group'] = gr
            tbl.iloc[-2:, tbl.columns.get_loc('group')] = gr
        else:
            # Merge with next
            gm = tbl['group'].iloc[gap:gap+2].values
            gr = tbl['group'].iloc[gap + 1]
            tbl_s.loc[tbl_s['group'].isin(gm), 'group'] = gr
            tbl.iloc[gap:gap+2, tbl.columns.get_loc('group')] = gr

        tbl = tbl.groupby('group').agg(
            no=('no', 'sum'),
            nb=('nb', 'sum')
        ).reset_index()

    tbl_s['label'] = _format_labels(tbl_s['group'].values, tbl_s['bin'].values)

    tbl_np = tbl_s.groupby('label').agg(
        no=('no', 'sum'),
        nb=('nb', 'sum')
    ).reset_index()
    tbl_np.columns = ['bin', 'no', 'nb']
    tbl_np['group'] = range(1, len(tbl_np) + 1)

    return tbl_s, tbl_np


def _format_labels(g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Format group labels as 'XX [bin1,bin2,...]'."""
    d = pd.DataFrame({'g': g, 'b': b})
    ds = d.groupby('g').agg(
        bin=('b', lambda x: '[' + ','.join(map(str, sorted(x))) + ']')
    ).reset_index()
    ds['gn'] = range(1, len(ds) + 1)

    nd = len(str(ds['gn'].max()))
    ds['gn_str'] = ds['gn'].apply(lambda x: f"{x:0{nd}d}")
    ds['label'] = ds['gn_str'] + ' ' + ds['bin']

    label_map = dict(zip(ds['g'], ds['label']))
    return np.array([label_map[gi] for gi in g])


def _woe_calc(
    tbl: pd.DataFrame,
    no_sc: int,
    ng_sc: int,
    nb_sc: int,
) -> np.ndarray:
    """Calculate information value per bin."""
    so = tbl['no'].sum() + no_sc
    sg = tbl['ng'].sum() + ng_sc
    sb = tbl['nb'].sum() + nb_sc

    eps = 1e-10
    dist_g = np.maximum(tbl['ng'].values / sg, eps)
    dist_b = np.maximum(tbl['nb'].values / sb, eps)

    woe = np.log(dist_g / dist_b)
    iv_b = (dist_g - dist_b) * woe

    return iv_b


def _woe_adjacent(
    tbl: pd.DataFrame,
    no_sc: int,
    ng_sc: int,
    nb_sc: int,
) -> np.ndarray:
    """Calculate WoE for adjacent bins when merged."""
    tbl_nr = len(tbl)
    so = tbl['no'].sum() + no_sc
    sg = tbl['ng'].sum() + ng_sc
    sb = tbl['nb'].sum() + nb_sc

    res = np.full(tbl_nr, np.nan)
    eps = 1e-10

    for i in range(1, tbl_nr):
        ng_sum = tbl['ng'].iloc[i-1] + tbl['ng'].iloc[i]
        nb_sum = tbl['nb'].iloc[i-1] + tbl['nb'].iloc[i]

        dist_g = max(ng_sum / sg, eps)
        dist_b = max(nb_sum / sb, eps)

        woe = np.log(dist_g / dist_b)
        iv_b = (dist_g - dist_b) * woe
        res[i] = iv_b

    return res


def _apa(
    tbl: pd.DataFrame,
    no_sc: int,
    ng_sc: int,
    nb_sc: int,
    mg: int,
) -> pd.DataFrame:
    """
    Adjacent Pooling Algorithm (APA) for merging bins.

    Minimizes information loss while iteratively merging bins to reach
    the target number of groups.
    """
    if len(tbl) <= mg:
        return tbl

    tbl = tbl.copy()
    tbl['group'] = range(1, len(tbl) + 1)
    tbl_s = tbl.copy()

    # Ensure ng column exists
    if 'ng' not in tbl.columns:
        tbl['ng'] = tbl['no'] - tbl['nb']
        tbl_s['ng'] = tbl_s['no'] - tbl_s['nb']

    while True:
        if len(tbl) == mg:
            break

        # Calculate IV for each bin
        tbl['iv_b'] = _woe_calc(tbl, no_sc, ng_sc, nb_sc)

        # Sum of adjacent IV values
        iv_b = tbl['iv_b'].values
        f2_1 = np.full(len(tbl), np.nan)
        for i in range(1, len(tbl)):
            f2_1[i] = iv_b[i-1] + iv_b[i]

        # IV if bins were merged
        f21 = _woe_adjacent(tbl, no_sc, ng_sc, nb_sc)

        # Information loss
        iv_loss = f2_1 - f21

        # Find minimum loss position
        mg_loc = np.nanargmin(iv_loss)
        mg_idx = [mg_loc - 1, mg_loc]

        # Update groups
        new_group = tbl['group'].iloc[mg_loc]
        old_groups = tbl['group'].iloc[mg_idx].values
        tbl_s.loc[tbl_s['group'].isin(old_groups), 'group'] = new_group
        tbl.iloc[mg_idx, tbl.columns.get_loc('group')] = new_group

        # Reaggregate
        tbl = tbl.groupby('group').agg(
            no=('no', 'sum'),
            ng=('ng', 'sum'),
            nb=('nb', 'sum')
        ).reset_index()

    tbl_s['label'] = _format_labels(tbl_s['group'].values, tbl_s['bin'].values)
    return tbl_s
