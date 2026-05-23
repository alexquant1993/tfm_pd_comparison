from __future__ import annotations
from typing import Protocol
import numpy as np
import pandas as pd

class PDModel(Protocol):
    """Uniform interface every model wrapper implements.

    The runner applies fold-safe preprocessing BEFORE calling fit_predict, so
    X_train and X_test arrive already capped. Wrappers should not re-cap.
    """
    name: str
    requires_string_labels: bool

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        types_meta: dict,
        seed: int = 42,
    ) -> np.ndarray:
        """Return P(default=1) for each row of X_test, shape (len(X_test),)."""
        ...
