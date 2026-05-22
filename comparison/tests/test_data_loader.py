from pathlib import Path
import pandas as pd
import numpy as np
from src.data_loader import load, carte_decode, _parse_special_codes

REPO_ROOT = Path(__file__).resolve().parents[2]

def test_load_shape_and_target():
    X, y, meta = load(repo_root=REPO_ROOT)
    assert X.shape == (1000, 20)
    assert y.shape == (1000,)
    assert set(y.unique()) == {0, 1}
    assert abs(y.mean() - 0.30) < 0.01
    assert "Creditability" not in X.columns

def test_meta_has_all_columns():
    X, y, meta = load(repo_root=REPO_ROOT)
    assert set(meta.keys()) == set(X.columns)
    for col, info in meta.items():
        assert info["type"] in {"nominal", "ordinal", "continuous"}
        assert info["dtype"] in {"numerical", "categorical"}

def test_parse_special_codes_handles_all_types():
    # The bug v1 had: only str was handled.
    assert _parse_special_codes("4") == [4]
    assert _parse_special_codes("4,5") == [4, 5]
    assert _parse_special_codes(4) == [4]
    assert _parse_special_codes(4.0) == [4]
    assert _parse_special_codes(np.nan) == []
    assert _parse_special_codes(None) == []
    assert _parse_special_codes("") == []

def test_special_codes_actually_loaded_for_three_known_vars():
    X, y, meta = load(repo_root=REPO_ROOT)
    assert meta["Account Balance"]["special_codes"] == [4]
    assert meta["Value Savings/Stocks"]["special_codes"] == [5]
    assert meta["Most valuable available asset"]["special_codes"] == [4]
    # And vars without special codes report []
    assert meta["Purpose"]["special_codes"] == []

def test_carte_decode_replaces_purpose():
    X, y, meta = load(repo_root=REPO_ROOT)
    X_str = carte_decode(X, meta)
    assert X_str["Purpose"].dtype == object
    assert not (X_str["Purpose"] == 3).any()
    assert "Radio/TV" in X_str["Purpose"].astype(str).str.cat(sep="|")

def test_carte_decode_keeps_continuous_numeric():
    X, y, meta = load(repo_root=REPO_ROOT)
    X_str = carte_decode(X, meta)
    assert pd.api.types.is_numeric_dtype(X_str["Age (years)"])
    assert pd.api.types.is_numeric_dtype(X_str["Credit Amount"])
