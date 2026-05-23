import numpy as np
import pandas as pd
from src.models.base import PDModel

class _Dummy:
    name = "dummy"
    requires_string_labels = False
    def fit_predict(self, X_train, y_train, X_test, types_meta, seed=42):
        return np.full(len(X_test), 0.5)

def test_dummy_satisfies_protocol():
    m: PDModel = _Dummy()
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    out = m.fit_predict(X, y, X, {"a": {"type": "continuous"}})
    assert out.shape == (3,)
    assert np.all((out >= 0) & (out <= 1))
