"""Load German Credit dataset and metadata from the parent repo."""
from __future__ import annotations
import re
import math
from pathlib import Path
import pandas as pd
import numpy as np

TARGET = "Creditability"

def _parse_encoding(encoding_str) -> dict[int, str]:
    """Parse the Encoding column into {int_code: human_label}.

    Format examples (raw CSV cells):
      '1=<0 DM (A11), 2=0-200 DM (A12), 3=>=200 DM/salary (A13). Special: 4=No checking account (A14)'
      '0=New car, 1=Used car, 2=Furniture, 3=Radio/TV, ...'
    """
    if not isinstance(encoding_str, str):
        return {}
    mapping: dict[int, str] = {}
    cleaned = re.sub(r"\([^)]*\)", "", encoding_str)
    cleaned = cleaned.replace("Special:", ",")
    for part in re.split(r"[,.]", cleaned):
        m = re.match(r"\s*(-?\d+)\s*=\s*(.+?)\s*$", part)
        if m:
            mapping[int(m.group(1))] = m.group(2).strip()
    return mapping

def _parse_special_codes(s) -> list[int]:
    """Parse a SpecialCodes cell that may arrive as int, float, NaN, str, or None.

    Examples:
      "4"     -> [4]
      "4,5"   -> [4, 5]
      4       -> [4]
      4.0     -> [4]
      NaN     -> []
      None    -> []
      ""      -> []
    """
    if s is None:
        return []
    # NaN check (works for float NaN) - must run BEFORE the generic float branch
    if isinstance(s, float) and math.isnan(s):
        return []
    # Integer cell from pandas
    if isinstance(s, (int, np.integer)):
        return [int(s)]
    # Float cell from pandas (most common when SpecialCodes col is read as float64)
    if isinstance(s, float):
        return [int(s)]
    # String cell (including comma-separated like "4,5")
    if isinstance(s, str):
        s = s.strip()
        if not s:
            return []
        return [int(c.strip()) for c in s.split(",") if c.strip()]
    return []

def load(repo_root: Path | str) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load loans.csv + variable_types.csv from the parent repo.

    Returns:
        X: DataFrame (1000, 20)
        y: Series (1000,) of 0/1 int
        meta: dict[col_name, {type, dtype, monotonicity, special_codes, encoding}]
    """
    repo_root = Path(repo_root)
    df = pd.read_csv(repo_root / "data" / "loans.csv")
    types_df = pd.read_csv(repo_root / "data" / "variable_types.csv")

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    type_map = {"Nominal": "nominal", "Ordinal": "ordinal",
                "Continuous": "continuous", "Numerical": "continuous",
                "Binary": "nominal", "Categorical": "nominal"}

    meta: dict = {}
    for _, row in types_df.iterrows():
        col = row["Variable"]
        if col == TARGET or col not in X.columns:
            continue
        meta[col] = {
            "type": type_map.get(row["Type"], "continuous"),
            "dtype": row["dtype"] if isinstance(row["dtype"], str) else "numerical",
            "monotonicity": row["Monotonicity"] if isinstance(row["Monotonicity"], str) else "no",
            "special_codes": _parse_special_codes(row.get("SpecialCodes")),
            "encoding": _parse_encoding(row.get("Encoding")),
        }
    return X, y, meta

def carte_decode(X: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Replace integer codes with human-readable strings for nominal/ordinal columns.

    Continuous columns are passed through unchanged. CARTE consumes the resulting
    string-labelled DataFrame so its graph-attention mechanism can use the actual
    semantic labels rather than meaningless integer codes.
    """
    out = X.copy()
    for col, info in meta.items():
        if info["type"] == "continuous":
            continue
        encoding = info["encoding"]
        if not encoding:
            continue
        out[col] = out[col].map(lambda v: encoding.get(int(v), str(v)) if pd.notna(v) else v)
        out[col] = out[col].astype(object)
    return out
