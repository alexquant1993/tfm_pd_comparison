"""Render per-fold CSV + predictions parquet into a thesis-ready summary."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src.metrics import delong_test, wilcoxon_paired, bonferroni

CHAMPION = "woe_logit"
# Primary discrimination metrics — the headline of the benchmark.
DISCRIMINATION_METRICS = ["auc", "gini", "ks"]
# Secondary probability-quality metrics — sensitive to calibration as well as
# discrimination. Reported for completeness but NOT part of the primary claim.
PROBABILITY_QUALITY_METRICS = ["brier", "logloss"]
METRICS = DISCRIMINATION_METRICS + PROBABILITY_QUALITY_METRICS

DISCLAIMER = (
    "> **Scope of claim (narrow):** This benchmark addresses *discriminatory power only*. "
    "It does NOT support the broader claim that \"TFMs are better PD models\" — that would "
    "require calibration quality, rating-grade homogeneity/heterogeneity, PSI stability, and "
    "out-of-time validation, all of which the `pd-autopilot` pipeline provides for the incumbent "
    "but are out of scope here. Brier and log-loss are reported as **probability-quality "
    "(secondary)** metrics — they combine discrimination with calibration, so a challenger that "
    "ties on AUC but loses on Brier is producing well-ranked but mis-scaled probabilities, "
    "which is a calibration story outside this benchmark's primary scope."
)

CONTEXT = """## Context (not directly comparable — different preprocessing path / fold seed)
- Report dev AUC (in-sample, loans_clean): 0.8109
- Report 10-fold CV AUC (loans_clean): 0.8065
- Report bootstrap AUC: 0.8026
"""

def _paired_tests(df: pd.DataFrame, preds: pd.DataFrame, champion: str) -> dict[str, dict]:
    """For each non-champion model, compute DeLong and Wilcoxon p-values.

    Critical: the predictions parquet must be aligned across models — same
    (fold_idx, test_idx) rows with identical y labels for each model. We
    assert that explicitly before running DeLong; a misaligned parquet would
    otherwise produce silently-wrong p-values.
    """
    models = list(df["model"].unique())
    n_comp = sum(1 for m in models if m != champion)

    ch_preds = preds[preds.model == champion].sort_values(["fold_idx", "test_idx"]).reset_index(drop=True)
    y_all = ch_preds["y"].values
    p_ch = ch_preds["proba"].values
    ch_keys = ch_preds[["fold_idx", "test_idx", "y"]].values
    ch_aucs = df[df.model == champion].sort_values("fold_idx")["auc"].values

    out: dict[str, dict] = {}
    for m in models:
        if m == champion:
            continue
        m_preds = preds[preds.model == m].sort_values(["fold_idx", "test_idx"]).reset_index(drop=True)
        m_keys = m_preds[["fold_idx", "test_idx", "y"]].values
        # Hard alignment check — DeLong silently misreports if labels are off.
        if not np.array_equal(ch_keys, m_keys):
            raise AssertionError(
                f"predictions parquet misaligned: model={m} has different "
                f"(fold_idx, test_idx, y) rows than champion={champion}. "
                f"DeLong would be invalid. Inspect predictions_*.parquet."
            )
        p_m = m_preds["proba"].values
        _, p_delong = delong_test(y_all, p_m, p_ch)
        m_aucs = df[df.model == m].sort_values("fold_idx")["auc"].values
        try:
            _, p_wilcox = wilcoxon_paired(m_aucs, ch_aucs)
        except ValueError:
            p_wilcox = 1.0
        out[m] = {
            "delong_p": p_delong,
            "delong_bonf": bonferroni(p_delong, n_comp),
            "wilcoxon_p": p_wilcox,
            "wilcoxon_bonf": bonferroni(p_wilcox, n_comp),
        }
    return out

def build_summary_md(df: pd.DataFrame, preds: pd.DataFrame, champion: str = CHAMPION) -> str:
    g = df.groupby("model")
    agg = g[METRICS].agg(["mean", "std"]).round(4)
    tests = _paired_tests(df, preds, champion)

    # Build the header in two visually grouped sections.
    discrim_hdr = " | ".join(m.upper() for m in DISCRIMINATION_METRICS)
    proba_hdr   = " | ".join(m.upper() + " (sec.)" for m in PROBABILITY_QUALITY_METRICS)
    header = (f"| Model | {discrim_hdr} | {proba_hdr} | "
              f"DeLong p | DeLong (bonf) | Wilcoxon p | Wilcoxon (bonf) |")
    sep = "|" + "|".join(["---"] * (len(METRICS) + 5)) + "|"

    rows = []
    for model in df["model"].unique():
        cells = [f"{agg.loc[model, (m, 'mean')]:.4f} ± {agg.loc[model, (m, 'std')]:.4f}" for m in METRICS]
        if model == champion:
            tail = ["reference"] * 4
        else:
            t = tests[model]
            tail = [f"{t['delong_p']:.3f}", f"{t['delong_bonf']:.3f}",
                    f"{t['wilcoxon_p']:.3f}", f"{t['wilcoxon_bonf']:.3f}"]
        rows.append("| " + model + " | " + " | ".join(cells) + " | " + " | ".join(tail) + " |")

    legend = (
        "*Primary discrimination metrics: AUC, Gini, KS. "
        "Probability-quality (secondary) metrics: Brier, log-loss — sensitive to calibration, "
        "not part of the primary claim (see disclaimer above).*"
    )
    return "\n".join([
        DISCLAIMER, "",
        "## Headline (10-fold CV)", "",
        header, sep, *rows, "",
        legend, "",
        CONTEXT,
    ])

def main():
    here = Path(__file__).resolve().parents[1]
    for pass_name in ["defaults", "tuned"]:
        csv_path = here / "results" / f"per_fold_{pass_name}.csv"
        pred_path = here / "results" / f"predictions_{pass_name}.parquet"
        if not (csv_path.exists() and pred_path.exists()):
            print(f"skip {pass_name}: missing artefact(s)")
            continue
        df = pd.read_csv(csv_path)
        preds = pd.read_parquet(pred_path)
        md = build_summary_md(df, preds)
        out = here / "results" / f"summary_{pass_name}.md"
        out.write_text(md)
        print(f"wrote {out}")

if __name__ == "__main__":
    main()
