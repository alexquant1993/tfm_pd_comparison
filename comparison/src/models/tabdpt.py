"""TabDPT wrapper (in-context tabular foundation model from Layer 6 AI)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.models.tabpfn import _device

class TabDPTWrapper:
    name = "tabdpt"
    requires_string_labels = False

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        from tabdpt import TabDPTClassifier
        device = _device()
        # MPS does not support float64; cast inputs to float32 to avoid TypeError.
        dtype = np.float32 if device == "mps" else None
        Xtr = X_train.values.astype(dtype) if dtype is not None else X_train.values
        Xte = X_test.values.astype(dtype) if dtype is not None else X_test.values
        clf = TabDPTClassifier(device=device)
        clf.fit(Xtr, y_train.values)
        proba = clf.predict_proba(Xte)
        # Some versions return (n, 2), others (n,) for binary; normalise:
        if proba.ndim == 2:
            return proba[:, 1]
        return proba
