# TFM vs WoE-Logit Comparison

Benchmarks 4 tabular foundation models against the pd-autopilot WoE-logit
champion on German Credit. See
[design spec](../docs/superpowers/specs/2026-05-22-tfm-comparison-design.md).

**Scope (narrow):** Pure discrimination only. Does NOT support the claim that
"TFMs are better PD models" — that would require calibration, rating grades,
PSI, and OOT validation, all out of scope here.

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
