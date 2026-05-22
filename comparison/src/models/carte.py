"""CARTE wrapper — consumes string-decoded inputs from carte_decode().

CARTE (Soda/INRIA 2024) embeds each table row as a star-shaped graph where the
center is the row and column-name / value pairs are leaves. The pretrained
graph-attention transformer requires:
  - the bundled `kg_pretrained.pt` (ships with `carte-ai`)
  - a FastText `cc.en.300.bin` file (~7 GB) for embedding column names and
    string cell values. Download once with:

        cd comparison && uv run python -c \\
          "from carte_ai.scripts.download_data import _download_fasttext; \\
           _download_fasttext()"

    If the auto-download hits an SSL error on macOS, fetch manually:

        curl -L -o \\
          .venv/lib/python3.12/site-packages/carte_ai/data/etc/cc.en.300.bin.gz \\
          https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
        gunzip .venv/lib/python3.12/site-packages/carte_ai/data/etc/cc.en.300.bin.gz
"""
from __future__ import annotations
from pathlib import Path
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


class CARTEWrapper:
    name = "carte"
    requires_string_labels = True

    def fit_predict(self, X_train, y_train, X_test, types_meta, seed: int = 42) -> np.ndarray:
        from carte_ai import CARTEClassifier, Table2GraphTransformer, config_directory

        ft_path = Path(config_directory["fasttext"])
        if not ft_path.exists():
            raise FileNotFoundError(
                f"FastText model not found at {ft_path}. "
                "Run once: "
                "uv run python -c 'from carte_ai.scripts.download_data import "
                "_download_fasttext; _download_fasttext()' "
                "(or curl the .bin.gz manually — see module docstring)."
            )

        # CARTE expects DataFrames where categorical cells are strings (object
        # dtype) and numeric cells are numeric. `carte_decode` already mapped
        # nominal/ordinal codes to human labels in the caller; we just need to
        # make sure those columns are object dtype.
        X_train_in = X_train.copy()
        X_test_in = X_test.copy()
        for col, info in types_meta.items():
            if col not in X_train_in.columns:
                continue
            if info["type"] in ("nominal", "ordinal"):
                X_train_in[col] = X_train_in[col].astype(object)
                X_test_in[col] = X_test_in[col].astype(object)

        t2g = Table2GraphTransformer(fasttext_model_path=str(ft_path))
        X_train_g = t2g.fit_transform(X_train_in, y=np.asarray(y_train.values, dtype=float))
        X_test_g = t2g.transform(X_test_in)

        clf = CARTEClassifier(random_state=seed, device=_device(), disable_pbar=True)
        clf.fit(X_train_g, np.asarray(y_train.values, dtype=float))
        proba = clf.predict_proba(X_test_g)
        proba = np.asarray(proba)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba
