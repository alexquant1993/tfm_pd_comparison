"""Orchestration: outer loop over folds, inner loop over models.

Per-fold pipeline (in order):
  1. Slice X, y by (train_idx, test_idx) from RAW loans.csv.
  2. Apply fold_safe_iqr_cap.
  3. (Pass 2 only) Run tune_classicals_for_fold on the capped training fold.
  4. For each model: invoke fit_predict, append metrics row to CSV,
     append predictions to parquet.
"""
from __future__ import annotations
import argparse
import csv
import time
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

    For Pass 1: all models with defaults.
    For Pass 2: classicals use tuned_params from this fold's nested study;
                woe_logit + 4 TFMs unchanged (their pass-1 rows will be reused).
    """
    from src.models.woe_logit import WoELogitChampion
    from src.models.logit import PlainLogit
    from src.models.xgb import XGBWrapper
    from src.models.lgbm import LGBMWrapper
    from src.models.tabpfn import TabPFNWrapper
    from src.models.tabicl import TabICLWrapper
    from src.models.tabdpt import TabDPTWrapper
    from src.models.carte import CARTEWrapper

    tp = tuned_params or {}
    return [
        WoELogitChampion(),
        PlainLogit(params=tp.get("logit")),
        XGBWrapper(params=tp.get("xgb")),
        LGBMWrapper(params=tp.get("lgbm")),
        TabPFNWrapper(),
        TabICLWrapper(),
        TabDPTWrapper(),
        CARTEWrapper(),
    ]

# Which models are unchanged between Pass 1 and Pass 2 → their pass-2 rows are
# copied from the pass-1 outputs rather than re-computed.
COPY_FROM_PASS1 = {"woe_logit", "tabpfn_v2", "tabicl", "tabdpt", "carte"}

def run_pass(pass_name: str, n_splits: int = 10, seed: int = 42) -> None:
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
        # Reuse pass-1 outputs for the unchanged models
        p1_parquet = RESULTS_DIR / "predictions_defaults.parquet"
        p1_csv = RESULTS_DIR / "per_fold_defaults.csv"
        if not (p1_parquet.exists() and p1_csv.exists()):
            raise RuntimeError("Pass 2 requires Pass 1 outputs. Run --pass defaults first.")
        pass1_preds = pd.read_parquet(p1_parquet)
        pass1_metrics = pd.read_csv(p1_csv)

    for fold_i, (tr, te) in enumerate(folds):
        # Step 1: slice raw, uncapped data and labels by the fold's positional indices
        X_tr_raw = X.iloc[tr].reset_index(drop=True)
        y_tr     = y.iloc[tr].reset_index(drop=True)
        X_te_raw = X.iloc[te].reset_index(drop=True)
        y_te     = y.iloc[te].reset_index(drop=True)

        # Step 2: outer fold-safe cap (caps fit on X_tr_raw, applied to both)
        X_tr_capped, X_te_capped, _ = fold_safe_iqr_cap(X_tr_raw, X_te_raw, meta)
        # CARTE inputs: cap (numeric only) then string-decode
        X_tr_carte = carte_decode(X_tr_capped, meta)
        X_te_carte = carte_decode(X_te_capped, meta)

        # Step 3: nested Optuna tuning (Pass 2 only).
        # Tuning receives the UNCAPPED outer-training fold; it recomputes caps
        # inside each inner split (strict nested preprocessing — see spec §4).
        tuned_params = None
        if pass_name == "tuned":
            from src.tuning import tune_classicals_for_fold
            tuned_params = tune_classicals_for_fold(X_tr_raw, y_tr, meta, seed=seed)

        # Step 4: per-model fits on outer-capped data
        models = _build_models_for_fold(pass_name, tuned_params)
        fold_preds: list[pd.DataFrame] = []
        for model in tqdm(models, desc=f"fold {fold_i+1}/{n_splits}", leave=False):
            # Pass 2 shortcut: reuse pass-1 rows for unchanged models
            if pass_name == "tuned" and model.name in COPY_FROM_PASS1:
                row = pass1_metrics[(pass1_metrics["model"] == model.name) &
                                    (pass1_metrics["fold_idx"] == fold_i)].iloc[0].to_dict()
                row["pass"] = "tuned"
                append_row(out_csv, row)
                copy_preds = pass1_preds[(pass1_preds["model"] == model.name) &
                                         (pass1_preds["fold_idx"] == fold_i)].copy()
                fold_preds.append(copy_preds)
                continue

            # Choose feature view (CARTE wants strings)
            if getattr(model, "requires_string_labels", False):
                X_tr_use, X_te_use = X_tr_carte, X_te_carte
            else:
                X_tr_use, X_te_use = X_tr_capped, X_te_capped

            row, proba = run_one_fold_one_model(
                model, X_tr_use, y_tr, X_te_use, y_te, meta,
            )
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

        # Crash-safe: write this fold's predictions as a parquet part immediately.
        # If the run dies mid-pass, all completed folds' parts are intact and
        # can be combined with the snippet below.
        part_path = parts_dir / f"fold_{fold_i:02d}.parquet"
        pd.concat(fold_preds, ignore_index=True).to_parquet(part_path, index=False)

    # Combine parts into the canonical predictions parquet
    parts = sorted(parts_dir.glob("fold_*.parquet"))
    combined = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    combined.to_parquet(out_parquet, index=False)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_parquet} (combined from {len(parts)} parts in {parts_dir})")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pass", dest="pass_name", choices=["defaults", "tuned"], required=True)
    args = p.parse_args()
    run_pass(args.pass_name)

if __name__ == "__main__":
    main()
