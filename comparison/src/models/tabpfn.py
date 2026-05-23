"""TabPFN v2 wrapper."""
from __future__ import annotations
import numpy as np
import pandas as pd

def _device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

class TabPFNWrapper:
    name = "tabpfn_v2"
    requires_string_labels = False

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        from tabpfn import TabPFNClassifier
        cat_idx = [X_train.columns.get_loc(c)
                   for c, i in types_meta.items()
                   if i["type"] == "nominal" and c in X_train.columns]
        clf = TabPFNClassifier(
            device=_device(),
            categorical_features_indices=cat_idx or None,
            random_state=seed,
        )
        clf.fit(X_train.values, y_train.values)
        return clf.predict_proba(X_test.values)[:, 1]
