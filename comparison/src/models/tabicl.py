"""TabICL wrapper (in-context tabular learner)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.models.tabpfn import _device

class TabICLWrapper:
    name = "tabicl"
    requires_string_labels = False

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        from tabicl import TabICLClassifier
        cat_idx = [X_train.columns.get_loc(c)
                   for c, i in types_meta.items()
                   if i["type"] == "nominal" and c in X_train.columns]
        clf = TabICLClassifier(device=_device(), random_state=seed)
        try:
            clf.fit(X_train.values, y_train.values, categorical_features=cat_idx or None)
        except TypeError:
            clf.fit(X_train.values, y_train.values)
        return clf.predict_proba(X_test.values)[:, 1]
