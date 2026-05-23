"""Orchestration: outer loop over folds, inner loop over models.

Per-fold pipeline (in order):
  1. Slice X, y by (train_idx, test_idx) from RAW loans.csv.
  2. Apply fold_safe_iqr_cap.
  3. (Pass 2 only) Run tune_classicals_for_fold on the capped training fold.
  4. For each model: invoke fit_predict, append metrics row to CSV,
     append predictions to parquet.
"""
from __future__ import annotations

# IMPORTANT: set BEFORE any numerical libs are imported. PyTorch (via TFM
# wrappers) bundles its own libomp; LightGBM links against Homebrew libomp.
# When both load into the same process, the second can either segfault
# (LightGBM-after-torch) or deadlock (TabPFN-after-LightGBM). This env var
# tells OpenMP's runtime to tolerate co-loading and pick one set of threads.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import load, carte_decode
from src.preprocessing import fold_safe_iqr_cap
from src.cv import make_folds
from src.metrics import compute_all

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _log(msg: str) -> None:
    """Timestamped stderr line that survives buffering.

    Use this for "I'm about to do something that might take a while"
    breadcrumbs. tqdm bars buffer per-line and can look frozen during long
    fit_predict calls; explicit timestamped prints prove the process is alive.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_one_fold_one_model(model, X_train, y_train, X_test, y_test, meta) -> tuple[dict, np.ndarray]:
    """Fit on already-sliced training data, predict on already-sliced test data.

    The caller is responsible for slicing — this function does NOT do iloc on
    full-dataset indices. (Previous version had a label-alignment bug where it
    used local positional indices on the full y, which silently returned the
    wrong labels for every fold after fold 0.)
    """
    t0 = time.perf_counter()
    proba = model.fit_predict(X_train, y_train, X_test, meta)
    runtime = time.perf_counter() - t0
    m = compute_all(y_test.values, proba)
    m.update(model=model.name, n_train=len(X_train), n_test=len(X_test),
             runtime_s=runtime, fold_idx=-1)
    return m, proba


def _build_models_for_fold(pass_name: str, tuned_params: dict | None) -> list:
    """Construct one instance per model for this outer fold.

    Model order matters: TFMs (torch-bundled libomp) run BEFORE the
    OpenMP-heavy gradient boosters (xgb, lgbm) to reduce the chance of
    OpenMP runtime contention. CARTE is excluded from the primary thesis
    scope (see README).

    For Pass 1: all 7 models with defaults.
    For Pass 2: classicals use tuned_params from this fold's nested study;
                woe_logit + 3 TFMs unchanged (their pass-1 rows will be reused).
    """
    from src.models.woe_logit import WoELogitChampion
    from src.models.logit import PlainLogit
    from src.models.tabpfn import TabPFNWrapper
    from src.models.tabicl import TabICLWrapper
    from src.models.tabdpt import TabDPTWrapper
    from src.models.xgb import XGBWrapper
    from src.models.lgbm import LGBMWrapper

    tp = tuned_params or {}
    return [
        WoELogitChampion(),
        PlainLogit(params=tp.get("logit")),
        TabPFNWrapper(),
        TabICLWrapper(),
        TabDPTWrapper(),
        XGBWrapper(params=tp.get("xgb")),
        LGBMWrapper(params=tp.get("lgbm")),
    ]


# Which models are unchanged between Pass 1 and Pass 2 → their pass-2 rows are
# copied from the pass-1 outputs rather than re-computed.
# CARTE removed from the primary thesis lineup (3 in-context TFMs only).
COPY_FROM_PASS1 = {"woe_logit", "tabpfn_v2", "tabicl", "tabdpt"}


def run_pass(pass_name: str, n_splits: int = 10, seed: int = 42) -> None:
    _log(f"=== Pass {pass_name} starting (n_splits={n_splits}, seed={seed}) ===")
    X, y, meta = load(REPO_ROOT)
    folds = make_folds(y, n_splits=n_splits, seed=seed)

    out_csv = RESULTS_DIR / f"per_fold_{pass_name}.csv"
    out_parquet = RESULTS_DIR / f"predictions_{pass_name}.parquet"
    parts_dir = RESULTS_DIR / f"predictions_{pass_name}_parts"
    # Clean prior outputs (full re-run)
    if out_csv.exists():
        out_csv.unlink()
    if out_parquet.exists():
        out_parquet.unlink()
    if parts_dir.exists():
        for f in parts_dir.iterdir():
            f.unlink()
        parts_dir.rmdir()
    parts_dir.mkdir(parents=True, exist_ok=True)

    pass1_preds = None
    pass1_metrics = None
    if pass_name == "tuned":
        p1_parquet = RESULTS_DIR / "predictions_defaults.parquet"
        p1_csv = RESULTS_DIR / "per_fold_defaults.csv"
        if not (p1_parquet.exists() and p1_csv.exists()):
            raise RuntimeError("Pass 2 requires Pass 1 outputs. Run --pass defaults first.")
        pass1_preds = pd.read_parquet(p1_parquet)
        pass1_metrics = pd.read_csv(p1_csv)

    for fold_i, (tr, te) in enumerate(folds):
        _log(f"--- fold {fold_i+1}/{n_splits} ---")
        X_tr_raw = X.iloc[tr].reset_index(drop=True)
        y_tr     = y.iloc[tr].reset_index(drop=True)
        X_te_raw = X.iloc[te].reset_index(drop=True)
        y_te     = y.iloc[te].reset_index(drop=True)

        X_tr_capped, X_te_capped, _ = fold_safe_iqr_cap(X_tr_raw, X_te_raw, meta)

        tuned_params = None
        if pass_name == "tuned":
            from src.tuning import tune_classicals_for_fold
            _log(f"fold {fold_i+1}: tuning classicals (nested Optuna)...")
            tuned_params = tune_classicals_for_fold(X_tr_raw, y_tr, meta, seed=seed)
            _log(f"fold {fold_i+1}: tuning done")

        models = _build_models_for_fold(pass_name, tuned_params)
        fold_preds: list[pd.DataFrame] = []
        for model in tqdm(models, desc=f"fold {fold_i+1}/{n_splits}",
                          leave=True, file=sys.stderr):
            # Pass 2 shortcut: reuse pass-1 rows for unchanged models
            if pass_name == "tuned" and model.name in COPY_FROM_PASS1:
                _log(f"fold {fold_i+1}: {model.name} → reusing pass-1 row")
                row = pass1_metrics[(pass1_metrics["model"] == model.name) &
                                    (pass1_metrics["fold_idx"] == fold_i)].iloc[0].to_dict()
                row["pass"] = "tuned"
                append_row(out_csv, row)
                copy_preds = pass1_preds[(pass1_preds["model"] == model.name) &
                                         (pass1_preds["fold_idx"] == fold_i)].copy()
                fold_preds.append(copy_preds)
                continue

            X_tr_use, X_te_use = X_tr_capped, X_te_capped

            _log(f"fold {fold_i+1}: {model.name} → starting fit_predict")
            t0 = time.perf_counter()
            row, proba = run_one_fold_one_model(
                model, X_tr_use, y_tr, X_te_use, y_te, meta,
            )
            _log(f"fold {fold_i+1}: {model.name} → done in {time.perf_counter() - t0:.1f}s "
                 f"(auc={row['auc']:.4f})")
            row["fold_idx"] = fold_i
            row["pass"] = pass_name
            append_row(out_csv, row)

            fold_preds.append(pd.DataFrame({
                "model": model.name,
                "fold_idx": fold_i,
                "test_idx": te,
                "y": y_te.values,
                "proba": proba,
            }))

        part_path = parts_dir / f"fold_{fold_i:02d}.parquet"
        pd.concat(fold_preds, ignore_index=True).to_parquet(part_path, index=False)
        _log(f"fold {fold_i+1}: wrote {part_path.name}")

    parts = sorted(parts_dir.glob("fold_*.parquet"))
    combined = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    combined.to_parquet(out_parquet, index=False)
    _log(f"=== Pass {pass_name} complete ===")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_parquet} (combined from {len(parts)} parts in {parts_dir})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pass", dest="pass_name", choices=["defaults", "tuned"], required=True)
    args = p.parse_args()
    run_pass(args.pass_name)


if __name__ == "__main__":
    main()
