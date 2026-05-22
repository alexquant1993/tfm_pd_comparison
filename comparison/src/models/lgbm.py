"""LightGBM with categorical_feature index list."""
from __future__ import annotations
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

class LGBMWrapper:
    name = "lgbm"
    requires_string_labels = False

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        cat_cols = [c for c, i in types_meta.items() if i["type"] == "nominal" and c in X_train.columns]
        cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]
        X_tr = X_train.copy()
        X_te = X_test.copy()
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype("category")
            X_te[c] = X_te[c].astype(pd.CategoricalDtype(categories=X_tr[c].cat.categories))
        params = dict(
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        params.update(self.params)
        clf = LGBMClassifier(**params)
        clf.fit(X_tr, y_train.values, categorical_feature=cat_idx)
        return clf.predict_proba(X_te)[:, 1]
