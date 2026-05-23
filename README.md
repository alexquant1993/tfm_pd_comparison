# TFM vs Production PD Scorecard — A Fair Benchmark

Can Tabular Foundation Models beat a production-grade PD scorecard built by an agentic AI system?

This project benchmarks three Tabular Foundation Models (TabPFN v2, TabICL, TabDPT) against classical ML models and a WoE-logistic regression champion scorecard — all on the same data, same folds, same evaluation protocol.

The champion scorecard is built automatically by [pd-autopilot](https://github.com/at621/pd-autopilot), an agentic PD model development pipeline powered by Claude Code. The comparison framework then puts every model on equal footing using 10-fold stratified CV on the raw German Credit dataset.

**[View results](https://alexquant1993.github.io/tfm_pd_comparison/)**

## Why this benchmark exists

Published TFM papers evaluate against default-config baselines on raw data — not against the kind of model a bank would actually deploy. That comparison flatters TFMs by beating a straw man. Here we compare against a full production pipeline: WoE binning with monotonicity constraints, IV-based variable selection, stepwise logistic regression, and expert-level calibration.

## Key findings

| Model | AUC (10-fold CV) | vs Champion |
|---|---|---|
| TabPFN v2 | 0.800 ± 0.051 | +1.6 pp |
| TabICL | 0.800 ± 0.053 | +1.6 pp |
| TabDPT | 0.794 ± 0.061 | +1.0 pp |
| XGBoost (tuned) | 0.787 ± 0.056 | +0.3 pp |
| WoE-Logistic | 0.784 ± 0.051 | ref |

TFMs show a consistent edge, but it is **not statistically significant** (DeLong p ≈ 0.10, Bonferroni-corrected ≈ 0.58). Even after Optuna tuning the classical models, the TFM advantage persists but stays within the noise band of a 1,000-observation dataset.

## Project structure

```
comparison/
├── src/
│   ├── runner.py          # 10-fold CV orchestrator
│   ├── tuning.py          # Nested Optuna tuning for classicals
│   ├── summary.py         # Generates summary tables + statistical tests
│   └── models/            # Thin wrappers: woe_logit, tabpfn, tabicl, tabdpt, xgb, lgbm
├── notebooks/
│   ├── 01_results_analysis.ipynb        # Figures for default hyperparameters
│   └── 02_results_analysis_tuned.ipynb  # Figures for tuned hyperparameters
└── results/
    ├── per_fold_{defaults,tuned}.csv    # Per-fold metrics (70 rows each)
    ├── predictions_{defaults,tuned}.parquet
    ├── summary_{defaults,tuned}.md      # Full tables + DeLong/Wilcoxon tests
    └── figures/                         # ROC, boxplot, forest plot, reliability
```

## Attribution

- **[pd-autopilot](https://github.com/at621/pd-autopilot)** by [@at621](https://github.com/at621) — the agentic PD model development pipeline that builds the champion scorecard. This repository is a fork; the comparison framework in `comparison/` is the only addition.
- **[pdtoolkit](https://github.com/at621/pdtoolkit_py)** by [@at621](https://github.com/at621) — the PD modelling library used by pd-autopilot.
- **Dataset:** [German Credit (Statlog)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), UCI Machine Learning Repository.
- **TFM implementations:** [TabPFN v2](https://github.com/PriorLabs/TabPFN), [TabICL](https://github.com/snovaisg/TabICL), [TabDPT](https://github.com/layer6ai-labs/TabDPT).

## License

The comparison framework follows the license of the upstream [pd-autopilot](https://github.com/at621/pd-autopilot) repository.
