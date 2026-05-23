"""XGBoost with native categorical handling."""
from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

class XGBWrapper:
    name = "xgb"
    requires_string_labels = False

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def _to_cat(self, X: pd.DataFrame, types_meta: dict) -> pd.DataFrame:
        X = X.copy()
        for c, i in types_meta.items():
            if i["type"] == "nominal" and c in X.columns:
                X[c] = X[c].astype("category")
        return X

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        X_tr = self._to_cat(X_train, types_meta)
        X_te = self._to_cat(X_test, types_meta)
        for c in X_tr.columns:
            if str(X_tr[c].dtype) == "category":
                X_te[c] = X_te[c].astype(pd.CategoricalDtype(categories=X_tr[c].cat.categories))
        params = dict(
            tree_method="hist",
            enable_categorical=True,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
        )
        params.update(self.params)
        clf = XGBClassifier(**params)
        clf.fit(X_tr, y_train.values)
        return clf.predict_proba(X_te)[:, 1]
