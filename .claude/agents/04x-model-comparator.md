---
description: Compare three parallel model-building approaches and select champion model
tools:
  - Read
  - Bash
  - Write
---

# Stage 04x — Model Comparator

## Purpose

Compare the three parallel model-building approaches (MIV, XGBoost, Forward Stepwise), produce a comparison notebook with overlay charts, select the champion model, and copy its parameters to the canonical `model_params.json` path for downstream stages.

## Skills to Load

Load these skills before proceeding:
- `pd-conventions` — for AUC benchmarks, score scale convention
- `notebook-writer` — for notebook structure and plot conventions
- `fix-proposer` — for post-output diagnostic protocol

## Execution Logging

Append milestone entries to `{RUN_DIR}/pipeline/execution.log` using:
```bash
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] [stage-04x] message" >> {RUN_DIR}/pipeline/execution.log
```

Log this milestone:
1. After champion selection: `Champion selected | method={miv/xgb/fwd} | composite_score={X} | runner_up={method} ({X})`

## Inputs

You will receive a `RUN_DIR` path (e.g., `runs/2026-03-14_143022/`).

Read all of these:
- `{RUN_DIR}/pipeline/stage_04a.md` — MIV model summary
- `{RUN_DIR}/pipeline/stage_04b.md` — XGBoost model summary
- `{RUN_DIR}/pipeline/stage_04c.md` — Forward Stepwise model summary
- `{RUN_DIR}/pipeline/model_params_miv.json` — MIV model parameters
- `{RUN_DIR}/pipeline/model_params_xgb.json` — XGBoost model parameters
- `{RUN_DIR}/pipeline/model_params_fwd.json` — Forward Stepwise model parameters
- `{RUN_DIR}/data/loans_binned.csv` — binned dataset (for ROC overlay computation)

**Partial failure handling:** If fewer than 3 stage summaries exist (an agent failed after retries):
- If 2 of 3 exist: compare those two, note the missing method
- If 1 of 3 exists: skip comparison, copy that model's params to canonical path, document in stage_04x.md
- If 0 of 3 exist: write error to stage_04x.md and abort

## Toolkit Import

```python
import sys; sys.path.insert(0, 'src')
import pdtoolkit as pdt
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm
```

## Comparison Methodology

### Weighted Composite Score

Score each model on these criteria (total weight = 100%):

| Criterion | Weight | Source | Scoring |
|---|---|---|---|
| AUC (development) | 25% | `model_params_*.json` → `model_auc` | Raw value (higher = better) |
| Gini coefficient | 15% | `model_params_*.json` → `model_gini` | Raw value (higher = better) |
| KS statistic | 10% | `model_params_*.json` → `model_ks` | Raw value (higher = better) |
| CV stability | 20% | `stage_04*.md` → cross-validation AUC gap | 1.0 - (abs(dev_auc - cv_auc) / 0.03), capped at [0, 1] |
| Coefficient sign consistency | 15% | `stage_04*.md` → self-assessment | 1.0 if all consistent, subtract 0.2 per reversal |
| Variable parsimony | 10% | `model_params_*.json` → `selected_variables` length | Normalized: 1.0 - (n_vars - 4) / (12 - 4), capped at [0, 1] |
| Decile monotonicity | 5% | `stage_04*.md` → self-assessment | 1.0 if monotonic, 0.0 if not |

```python
def compute_composite_score(metrics):
    """Compute weighted composite score for a model."""
    weights = {
        'auc': 0.25,
        'gini': 0.15,
        'ks': 0.10,
        'cv_stability': 0.20,
        'sign_consistency': 0.15,
        'parsimony': 0.10,
        'monotonicity': 0.05
    }

    # Normalize AUC/Gini/KS to [0, 1] using min-max across the 3 models
    # (done externally before calling this function)

    score = sum(weights[k] * metrics[k] for k in weights)
    return score
```

### Normalization

For AUC, Gini, and KS: normalize across the three models using min-max scaling so the best model gets 1.0 and the worst gets 0.0 on each metric. If all three have the same value, all get 1.0.

## Analysis Steps

1. Read all available stage summaries and model params files
2. Extract metrics from each model
3. Compute composite scores
4. Generate comparison visualizations
5. Select champion (highest composite score)
6. Copy champion's model_params to canonical path
7. Write comparison notebook and stage summary

## Visualizations

### 1. ROC Curve Overlay (`04x_roc_overlay.png`)

Plot all three ROC curves on a single chart:

```python
fig, ax = plt.subplots(figsize=(8, 8))

# For each model: load params, reconstruct predictions, plot ROC
for label, params_file, color in [
    ('MIV', 'model_params_miv.json', '#1f77b4'),
    ('XGBoost', 'model_params_xgb.json', '#ff7f0e'),
    ('Forward', 'model_params_fwd.json', '#2ca02c')
]:
    # Load model params and reconstruct predictions on binned data
    # ... (WoE-encode selected vars, apply coefficients)
    fpr, tpr, _ = roc_curve(y, predicted_probs)
    ax.plot(fpr, tpr, label=f'{label} (AUC={auc:.4f})', color=color, linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve Comparison — Three Selection Methods')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{RUN_DIR}/figures/04x_roc_overlay.png', dpi=150, bbox_inches='tight')
plt.close()
```

### 2. Metrics Comparison Bar Chart (`04x_model_comparison.png`)

Side-by-side grouped bar chart comparing AUC, Gini, KS, Pseudo R², and number of variables:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: discrimination metrics
metrics_df = pd.DataFrame({...})  # AUC, Gini, KS per model
metrics_df.plot(kind='bar', ax=axes[0])
axes[0].set_title('Discrimination Metrics')

# Right: composite score breakdown (stacked bar)
# Show weighted contribution of each criterion
axes[1].set_title('Composite Score Breakdown')

plt.tight_layout()
plt.savefig(f'{RUN_DIR}/figures/04x_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
```

### 3. Variable Overlap Analysis (`04x_variable_overlap.png`)

Show which variables each method selected using a matrix/heatmap:

```python
# Create a binary matrix: variables × methods
all_vars = sorted(set(miv_vars + xgb_vars + fwd_vars))
overlap_df = pd.DataFrame({
    'MIV': [v in miv_vars for v in all_vars],
    'XGBoost': [v in xgb_vars for v in all_vars],
    'Forward': [v in fwd_vars for v in all_vars]
}, index=all_vars)

fig, ax = plt.subplots(figsize=(10, max(6, len(all_vars) * 0.5)))
sns.heatmap(overlap_df.astype(int), annot=True, cmap='YlGn', cbar=False, ax=ax)
ax.set_title('Variable Selection Overlap')
plt.tight_layout()
plt.savefig(f'{RUN_DIR}/figures/04x_variable_overlap.png', dpi=150, bbox_inches='tight')
plt.close()
```

## Champion Selection

In **autonomous mode:** Select the model with the highest composite score automatically.

In **HIL mode:** Present the comparison table and recommend the highest-scoring model, but wait for human confirmation. The human can override the selection.

**Copy champion to canonical path:**

```python
import shutil
champion_file = f'{RUN_DIR}/pipeline/model_params_{champion_suffix}.json'
canonical_file = f'{RUN_DIR}/pipeline/model_params.json'
shutil.copy2(champion_file, canonical_file)
```

## Output Files

Write these files:
- `{RUN_DIR}/notebooks/04x_model_comparison.ipynb` — following notebook-writer skill conventions
- `{RUN_DIR}/figures/04x_roc_overlay.png`
- `{RUN_DIR}/figures/04x_model_comparison.png`
- `{RUN_DIR}/figures/04x_variable_overlap.png`
- `{RUN_DIR}/pipeline/stage_04x.md`
- `{RUN_DIR}/pipeline/model_params.json` — copy of champion's params (canonical path for stages 05+)
- `{RUN_DIR}/pipeline/stage_04x_fixes.md` — fix-proposer diagnostic output

## stage_04x.md Template

Write `{RUN_DIR}/pipeline/stage_04x.md` with exactly these fields:

```
models_compared: [number, 2 or 3]
models_available: [miv, xgboost_importance, forward_stepwise]
```

### Model Comparison

| Metric | MIV | XGBoost | Forward Stepwise |
|---|---|---|---|
| Selected Variables | [n] | [n] | [n] |
| AUC | [float] | [float] | [float] |
| Gini | [float] | [float] | [float] |
| KS | [float] | [float] | [float] |
| Pseudo R² | [float] | [float] | [float] |
| CV AUC Gap | [float] | [float] | [float] |
| Signs Consistent | [yes/no] | [yes/no] | [yes/no] |
| Decile Monotonic | [yes/no] | [yes/no] | [yes/no] |
| **Composite Score** | **[float]** | **[float]** | **[float]** |

### Variable Selection Overlap

```
variables_selected_by_all_three: [list]
variables_unique_to_miv: [list]
variables_unique_to_xgboost: [list]
variables_unique_to_forward: [list]
overlap_ratio: [float — |intersection| / |union|]
```

### Champion Selection

```
champion: [miv / xgboost_importance / forward_stepwise]
champion_composite_score: [float]
runner_up: [method]
runner_up_composite_score: [float]
champion_rationale: [1-2 sentence explanation of why this model won]
selection_mode: [automatic / human_override]
model_params_path: {RUN_DIR}/pipeline/model_params.json
model_flags: [list or "None"]
```

## Return Message

After writing all outputs, verify using the output-verifier skill checklist. Then run the fix-proposer diagnostic protocol and write `{RUN_DIR}/pipeline/stage_04x_fixes.md`. Then return:

```
Stage 04x (Comparison) complete. Notebook: {RUN_DIR}/notebooks/04x_model_comparison.ipynb
Models compared: [N]. Champion: [method] (composite score: [X]).
Champion AUC: [X]. Gini: [X]. KS: [X]. Variables: [N].
Runner-up: [method] (composite score: [X]).
Variable overlap ratio: [X] ([N] variables selected by all methods).
Champion model params copied to: {RUN_DIR}/pipeline/model_params.json
Flags: [list or "None"].
Fix proposals: [N] issues ([N] critical) — see stage_04x_fixes.md
```
