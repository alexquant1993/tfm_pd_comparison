"""Plain logistic regression on one-hot nominals + standardised continuous."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class PlainLogit:
    name = "logit"
    requires_string_labels = False

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        nominal = [c for c, i in types_meta.items() if i["type"] == "nominal" and c in X_train.columns]
        other = [c for c in X_train.columns if c not in nominal]
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal),
            ("num", StandardScaler(), other),
        ])
        lr_kwargs = dict(solver="lbfgs", max_iter=2000, random_state=seed)
        lr_kwargs.update(self.params)
        pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(**lr_kwargs))])
        pipe.fit(X_train, y_train.values)
        return pipe.predict_proba(X_test)[:, 1]
