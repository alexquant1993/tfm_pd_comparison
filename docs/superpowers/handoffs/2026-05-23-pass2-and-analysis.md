# Handoff — Pass 2 + analysis notebook execution

**Date:** 2026-05-23
**Branch:** `comparison-impl`
**HEAD:** `5ba6147` (fix(comparison): lazy model imports + OMP_NUM_THREADS=1 + LGBM n_jobs=1)
**Repo root:** `/Users/andersonarroyo/Documents/01_projects/tfm_pd_comparison/pd-autopilot/`

## What the next session must do

Two concrete tasks, in this order:

1. **Run Pass 2 (tuned classicals, nested Optuna).**
   ```bash
   cd comparison && uv run python -m src.runner --pass tuned
   ```
   Expected wall-time on the user's Mac Studio (M2 Max, 32 GB): **1–2.5 hours**. Most of that is the 7,500 inner-fold fits (10 outer folds × 3 classicals × 50 trials × 5 inner folds). The runner reuses Pass-1 rows verbatim for `woe_logit` + 3 TFMs, so only XGB / LightGBM / plain logit are re-fit per outer fold.

   **Run it in the background** with a Bash `run_in_background` invocation, then use **Monitor** with a tail-F + grep on the same output file to stream `[HH:MM:SS] === ...`, `--- fold N/10 ---`, `tuning classicals...`, and error patterns. Pattern that worked in the previous session:
   ```bash
   tail -F <output> | grep -E --line-buffered "^\[[0-9]+:[0-9]+:[0-9]+\] (===|---| fold)|Traceback|Error|FAILED|Killed|Segmentation|RuntimeError|Wrote "
   ```

2. **Generate the four analysis figures + tuned summary.**
   ```bash
   cd comparison
   uv run python -m src.summary                                          # writes summary_tuned.md
   uv run jupyter nbconvert --to notebook --execute notebooks/01_results_analysis.ipynb --inplace
   uv run jupyter nbconvert --to notebook --execute notebooks/02_results_analysis_tuned.ipynb --inplace
   ```
   The two analysis notebooks emit ROC overlay, reliability curves, AUC boxplot, and forest plot of paired diffs vs champion. Outputs land in `comparison/results/figures/` and the notebooks themselves carry the rendered outputs.

   After both finish, commit the artefacts:
   ```bash
   git add comparison/results/per_fold_tuned.csv \
           comparison/results/predictions_tuned.parquet \
           comparison/results/predictions_tuned_parts/ \
           comparison/results/summary_tuned.md \
           comparison/results/figures/ \
           comparison/notebooks/01_results_analysis.ipynb \
           comparison/notebooks/02_results_analysis_tuned.ipynb \
           comparison/results/paired_tests_defaults.md \
           comparison/results/paired_tests_tuned.md
   git commit -m "results(comparison): pass 2 (tuned) + analysis notebooks executed"
   ```

## State of play (don't re-derive)

- **Pass 1 has already run successfully** (49 seconds, exit 0). Results committed in `5ba6147`. Headline AUCs:
  - TabPFN v2: 0.8000 ± 0.0513   ← best
  - TabICL:    0.7995 ± 0.0527
  - TabDPT:    0.7942 ± 0.0613
  - LightGBM:  0.7861 ± 0.0514
  - plain logit: 0.7847 ± 0.0614
  - WoE-logit (champion): 0.7840 ± 0.0509
  - XGBoost:   0.7757 ± 0.0652   ← worst (defaults likely terrible; tuning should help)
- Paired DeLong tests vs champion: TFMs at p ≈ 0.10 (suggestive, not significant; Bonferroni-corrected → ~0.6).
- See `comparison/results/summary_defaults.md` for full table with all 5 metrics + paired tests + disclaimer.

## What you must NOT change

- **Don't reintroduce CARTE** — explicitly excluded from the primary lineup per user decision (semantically thin German Credit labels + 7 GB FastText dep). Wrapper still exists at `comparison/src/models/carte.py` with a FastText skip-gate; re-enable instructions are in `comparison/README.md` "Re-enabling CARTE" section. Leave it alone.
- **Don't loosen the libomp workaround** at the top of `comparison/src/runner.py`. The 4-part combination (KMP env + OMP env + LGBM `n_jobs=1` default + lazy per-model imports) is the *minimum* viable set on macOS arm64 — empirically verified via reproducer. Removing any one of the four reintroduces a segfault.
- **Don't change the WoE-logit `C=1e10`** — it matches the report's coefficients within 4 dp. `penalty=None` is deprecated in sklearn ≥ 1.8; `C=np.inf` triggers an internal warning. `C=1e10` is the clean middle ground.
- **Don't re-run Pass 1.** Pass 2 reads `predictions_defaults.parquet` to copy `woe_logit` + 3 TFM rows verbatim. Wiping pass-1 outputs forces a redo.

## Known gotchas / pitfalls

- **Always run `uv run ...` from inside `comparison/`** (not the repo root). The `--ignore=tests/test_tabpfn.py …` lines in `comparison/pyproject.toml` are pytest-relative-to-cwd; from the repo root pytest produces collection errors instead of skipping TFM tests.
- **TabPFN pinned to `<7.0`** in `pyproject.toml`. v7+ and v8 require Hugging Face license auth (gated repo). Pin is intentional. Resolved version is 6.4.1 — the last anonymous-download release.
- **TabDPT on MPS** needs a float32 cast (PyTorch MPS doesn't support float64). The wrapper at `comparison/src/models/tabdpt.py` already handles this; don't remove the conditional cast.
- **Per-model timestamped breadcrumbs** are written to stderr by `_log()` in `runner.py`. If something looks stuck during Pass 2, check stderr — the user reported earlier that the bar appeared frozen during TabPFN's silent ~30 s MPS init + weight load. The breadcrumbs make stuck-vs-slow distinguishable: > 5 min between consecutive `[HH:MM:SS]` lines for the same model is the real hang threshold.
- **Pass 2 tuning is in a torch-free process path** — `src/tuning.py` doesn't import any TFM wrapper and uses `n_jobs=-1` internally for XGB/LGBM. That's safe; don't change it.

## Reference artefacts (don't duplicate, read these)

| Path | What's in it |
|---|---|
| `docs/superpowers/specs/2026-05-22-tfm-comparison-design.md` | Spec — read §4 (protocol), §8 (woe_logit faithfulness), §11 (artefacts), §13 (acceptance criteria). |
| `docs/superpowers/plans/2026-05-22-tfm-comparison.md` | Plan — Tasks 19, 20, 22, 24 are the ones the next session executes. Tasks 1–18, 21, 23 are already done. |
| `comparison/results/summary_defaults.md` | Pass-1 headline + DeLong + Wilcoxon. |
| `comparison/results/per_fold_defaults.csv` | 70 rows of per-fold metrics. |
| `comparison/results/predictions_defaults.parquet` | 7,000 rows of OOF predictions (the input Pass 2 reads to copy unchanged-model rows). |
| `comparison/src/runner.py` (lines 1–35) | The libomp workaround comment block — read before touching anything. |
| `comparison/src/models/lgbm.py` (lines 25–35) | `n_jobs=1` rationale. |

## Suggested skills

Invoke these via the `Skill` tool as needed:

- **`superpowers:using-superpowers`** — invoke at session start (mandatory per skill rules).
- **`superpowers:verification-before-completion`** — invoke before claiming Pass 2 is done. Specifically verify: `per_fold_tuned.csv` has 70 rows, `predictions_tuned.parquet` has 7,000 rows, `summary_tuned.md` starts with the disclaimer and contains both DeLong and Wilcoxon p-values, all 4 figures exist in `results/figures/`.
- **`superpowers:finishing-a-development-branch`** — after Pass 2 + analysis + commit, the `comparison-impl` branch is ready for the user's decision (merge to master? open PR? squash?). Use this skill to present the options.
- **`run`** (built-in) — if you need to verify a notebook actually executes end-to-end in a browser. Probably overkill here; nbconvert --execute --inplace is the documented path.

Do NOT invoke: brainstorming, writing-plans, writing-skills, subagent-driven-development. Pass 2 is a single execution + a single notebook execution — no architectural decisions, no new code, no plan to write.

## How to know you're done

All of the following must be true:

1. `comparison/results/per_fold_tuned.csv` exists with 70 rows (`uv run python -c "import pandas as pd; print(len(pd.read_csv('results/per_fold_tuned.csv')))"` from inside `comparison/` should print `70`).
2. `comparison/results/predictions_tuned.parquet` exists with 7,000 rows.
3. `find comparison/results/predictions_tuned_parts -name 'fold_*.parquet' | wc -l` prints `10`.
4. `comparison/results/summary_tuned.md` exists; first line starts with `> **Scope of claim (narrow):**`; contains both `DeLong` and `Wilcoxon`.
5. `comparison/results/figures/` contains `roc_overlay_{defaults,tuned}.png`, `reliability_{defaults,tuned}.png`, `auc_boxplot_{defaults,tuned}.png`, `forest_{defaults,tuned}.png` (8 files).
6. The two analysis notebooks have been executed in place (their cell outputs are populated).
7. A single commit on `comparison-impl` adds all of the above.
8. Working tree clean (`git status` shows nothing).

When all 8 hold, present a one-paragraph end-of-run summary to the user and ask whether they want help merging the branch or opening a PR.

## Open questions for the user (only ask if relevant)

- Does the user want the analysis chapter drafted next, or are they writing that themselves?
- Tuned-pass XGB/LGBM results vs Pass-1: if tuning closes the gap to TFMs, does the user want a discussion section flagging that "TFM edge collapses under tuning"?

Do not ask these proactively if Pass 2 + analysis go smoothly — they're judgement calls the user can raise on their own.
