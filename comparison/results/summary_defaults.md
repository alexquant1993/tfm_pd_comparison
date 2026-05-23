> **Scope of claim (narrow):** This benchmark addresses *discriminatory power only*. It does NOT support the broader claim that "TFMs are better PD models" — that would require calibration quality, rating-grade homogeneity/heterogeneity, PSI stability, and out-of-time validation, all of which the `pd-autopilot` pipeline provides for the incumbent but are out of scope here. Brier and log-loss are reported as **probability-quality (secondary)** metrics — they combine discrimination with calibration, so a challenger that ties on AUC but loses on Brier is producing well-ranked but mis-scaled probabilities, which is a calibration story outside this benchmark's primary scope.

## Headline (10-fold CV)

| Model | AUC | GINI | KS | BRIER (sec.) | LOGLOSS (sec.) | DeLong p | DeLong (bonf) | Wilcoxon p | Wilcoxon (bonf) |
|---|---|---|---|---|---|---|---|---|---|
| woe_logit | 0.7840 ± 0.0509 | 0.5681 ± 0.1018 | 0.5205 ± 0.1000 | 0.1684 ± 0.0207 | 0.5045 ± 0.0515 | reference | reference | reference | reference |
| logit | 0.7847 ± 0.0614 | 0.5693 ± 0.1228 | 0.5167 ± 0.0939 | 0.1660 ± 0.0247 | 0.5036 ± 0.0647 | 0.937 | 1.000 | 0.922 | 1.000 |
| tabpfn_v2 | 0.8000 ± 0.0513 | 0.5999 ± 0.1026 | 0.5319 ± 0.0875 | 0.1600 ± 0.0187 | 0.4888 ± 0.0508 | 0.097 | 0.580 | 0.105 | 0.633 |
| tabicl | 0.7995 ± 0.0527 | 0.5990 ± 0.1054 | 0.5181 ± 0.1129 | 0.1595 ± 0.0202 | 0.4875 ± 0.0559 | 0.102 | 0.614 | 0.090 | 0.539 |
| tabdpt | 0.7942 ± 0.0613 | 0.5885 ± 0.1226 | 0.5152 ± 0.1070 | 0.1617 ± 0.0219 | 0.4952 ± 0.0656 | 0.310 | 1.000 | 0.232 | 1.000 |
| xgb | 0.7757 ± 0.0652 | 0.5513 ± 0.1304 | 0.5019 ± 0.0986 | 0.1845 ± 0.0363 | 0.6643 ± 0.1587 | 0.361 | 1.000 | 0.557 | 1.000 |
| lgbm | 0.7861 ± 0.0514 | 0.5722 ± 0.1029 | 0.5129 ± 0.0914 | 0.1720 ± 0.0226 | 0.5715 ± 0.0942 | 0.918 | 1.000 | 0.922 | 1.000 |

*Primary discrimination metrics: AUC, Gini, KS. Probability-quality (secondary) metrics: Brier, log-loss — sensitive to calibration, not part of the primary claim (see disclaimer above).*

## Context (not directly comparable — different preprocessing path / fold seed)
- Report dev AUC (in-sample, loans_clean): 0.8109
- Report 10-fold CV AUC (loans_clean): 0.8065
- Report bootstrap AUC: 0.8026
