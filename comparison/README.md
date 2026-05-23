# TFM vs WoE-Logit Comparison

Benchmarks **3 in-context tabular foundation models** (TabPFN v2, TabICL,
TabDPT) against the pd-autopilot WoE-logit champion on German Credit, plus
plain logit / XGBoost / LightGBM as classical baselines. See
[design spec](../docs/superpowers/specs/2026-05-22-tfm-comparison-design.md).

**Scope (narrow):** Pure discrimination only. Does NOT support the claim that
"TFMs are better PD models" — that would require calibration, rating grades,
PSI, and OOT validation, all out of scope here.

**Note on CARTE:** the CARTE wrapper (`src/models/carte.py`) is implemented
and tested but excluded from the primary lineup because it needs a ~7 GB
FastText embedding file and the marginal value on German Credit's
semantically-thin categorical labels is likely small. See "Re-enabling CARTE"
below if you want to include it.

## Setup

```bash
brew install uv      # one-time
uv sync              # creates .venv from uv.lock
```

## Run

```bash
# Always use `uv run` so you get this project's venv, not your system/conda Python.
uv run pytest -m "not slow"          # fast tests
uv run pytest -m slow                # TFM smoke tests (downloads weights ~1 GB)
uv run python -m src.runner --pass defaults
uv run python -m src.runner --pass tuned
uv run python -m src.summary         # build summary_*.md from CSVs + parquets
uv run jupyter lab notebooks/        # interactive analysis
```

## Outputs

- `results/per_fold_defaults.csv`, `results/per_fold_tuned.csv` — metrics × fold × model
- `results/predictions_defaults.parquet`, `results/predictions_tuned.parquet` — OOF predictions
- `results/summary_defaults.md`, `results/summary_tuned.md` — paired tests, mean ± std tables
- `results/figures/` — ROC overlays, reliability curves, AUC boxplots, forest plots

## Re-enabling CARTE (optional)

Two steps if you want CARTE back in the lineup:

**1. Add CARTE to the runner's model list.** Edit `src/runner.py`:
in `_build_models_for_fold` add `CARTEWrapper()` to the returned list
(import it at the top of the function), and add `"carte"` to the
`COPY_FROM_PASS1` set so pass 2 reuses its pass-1 rows.

**2. Download the FastText embedding file** (~7 GB, ~10 min on a fast
connection):

```bash
uv run python -c \
  "from carte_ai.scripts.download_data import _download_fasttext; _download_fasttext()"
```

Or manual:

```bash
curl -L -o \
  .venv/lib/python3.12/site-packages/carte_ai/data/etc/cc.en.300.bin.gz \
  https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip .venv/lib/python3.12/site-packages/carte_ai/data/etc/cc.en.300.bin.gz
```

Without FastText the CARTE wrapper raises a clear `FileNotFoundError` with
the download command in the message; the slow test for CARTE auto-skips.
