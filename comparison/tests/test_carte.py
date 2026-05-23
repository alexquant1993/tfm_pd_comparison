from pathlib import Path
import numpy as np
import pytest
from src.data_loader import load, carte_decode
from src.preprocessing import fold_safe_iqr_cap
from src.models.carte import CARTEWrapper

REPO_ROOT = Path(__file__).resolve().parents[2]


def _fasttext_present() -> bool:
    """CARTE's Table2GraphTransformer needs the ~7GB FastText cc.en.300.bin.
    Check the canonical location inside the carte-ai package."""
    try:
        from carte_ai import config_directory
    except ImportError:
        return False
    return Path(config_directory["fasttext"]).exists()


@pytest.mark.slow
def test_carte_requires_string_labels_flag():
    assert CARTEWrapper.requires_string_labels is True


@pytest.mark.slow
@pytest.mark.skipif(
    not _fasttext_present(),
    reason="CARTE requires ~7GB FastText cc.en.300.bin — "
           "run 'uv run python -c \"from carte_ai.scripts.download_data import _download_fasttext; _download_fasttext()\"' once to enable",
)
def test_carte_runs_on_decoded_input():
    X, y, meta = load(repo_root=REPO_ROOT)
    # Apply fold-safe cap THEN string-decode (decode is a no-op on continuous cols)
    X_tr, X_te, _ = fold_safe_iqr_cap(X.iloc[:800], X.iloc[800:], meta)
    X_tr_str = carte_decode(X_tr, meta)
    X_te_str = carte_decode(X_te, meta)
    m = CARTEWrapper()
    p = m.fit_predict(X_tr_str, y.iloc[:800], X_te_str, meta)
    assert p.shape == (200,)
    assert np.all((p >= 0) & (p <= 1))
