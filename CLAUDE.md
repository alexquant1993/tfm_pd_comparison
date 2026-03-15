# pd-autopilot

Automated PD (Probability of Default) model development on the German Credit dataset using the `pdtoolkit` library. Produces six Jupyter notebooks (one per pipeline stage) and a comprehensive DOCX (Word) report. Supports two execution modes.

## Execution Modes

### Autonomous Mode
Trigger: user says "build model", "run pipeline", "autonomous", or similar without requesting review.

- Run all 6 stages end-to-end without stopping
- Each stage still writes its notebook, pipeline/*.md, and figures as normal
- Output-verifier runs after each stage — on failure, retry the stage (max 2 retries), then abort the run with an error summary
- If any stage produces regulatory flags, collect them and present a consolidated flag report at the end
- Notebook-reviewer is invoked automatically for any stage with flags
- After stage 06, invoke report-writer (stage 07) to generate the DOCX report
- At the end, present a single summary of all 6 stages with overall assessment and the report path

### HIL (Human-in-the-Loop) Mode
Trigger: user says "HIL", "review mode", "step by step", or explicitly asks to review each stage.

- Stop after every stage and present the summary to the human
- Wait for human approval before proceeding to the next stage
- Human can: approve, reject with feedback (re-run same stage), or modify parameters for the next stage
- The human can also switch to autonomous mode mid-pipeline (e.g., "looks good, finish the rest autonomously")

### Mode Selection

**Step 1 — Detect or ask.** At the start of a pipeline run:
- If the user's message contains "autonomous", "run all", "build model", "full pipeline" → set mode to `autonomous`
- If the user's message contains "HIL", "review", "step by step", "one at a time" → set mode to `hil`
- If ambiguous → ask the user before creating the run directory:
  > Would you like to review each stage (HIL mode) or run the full pipeline autonomously?

**Step 2 — Record.** Write `{RUN_DIR}/pipeline/run_config.md` immediately after creating the run directory:
```
mode: [autonomous / hil]
started: [YYYY-MM-DD HH:MM:SS]
dataset: data/loans.csv
target: Creditability
current_stage: 01
```

**Step 3 — Update on progress.** After each stage completes, update `current_stage` in `run_config.md` to the next stage number. This allows a resumed conversation to know where the pipeline left off. For the parallel model-building phase, use `current_stage: 04a,04b,04c` while the three agents are running, then `04x` during comparison, then `05` after champion selection.

**Mid-run mode switch.** In HIL mode, if the user says "finish autonomously" or "run the rest", update `mode: autonomous` in `run_config.md` and continue without further checkpoints. The reverse (autonomous → HIL) is not supported since autonomous stages have already proceeded.

---

## Dataset

- **Location:** `data/loans.csv`
- **Target variable:** `Creditability` (1 = default)

## Toolkit

- **Source:** `src/pdtoolkit/` — local copy of the pdtoolkit package
- **Import:** `import sys; sys.path.insert(0, 'src'); import pdtoolkit as pdt`
- **optbinning:** `pip install optbinning` — used by Stage 03 for constraint-programming optimal binning with monotonicity enforcement. Import: `from optbinning import OptimalBinning`
- **xgboost:** `pip install xgboost` — used by Stage 04b for XGBoost feature importance-based variable selection. Import: `from xgboost import XGBClassifier`
- **Pipeline fixes:** `lib/pipeline_fixes/` — utility functions proposed by the fix-proposer diagnostic. Import: `sys.path.insert(0, 'lib'); from pipeline_fixes.X import Y`

## Run Directory Structure

Each pipeline execution creates a timestamped run directory under `runs/`:

```
runs/
  2026-03-14_143022/
    notebooks/
      01_data_quality.ipynb
      02_data_preparation.ipynb
      03_bivariate_analysis.ipynb
      04a_model_building_miv.ipynb
      04b_model_building_xgb.ipynb
      04c_model_building_fwd.ipynb
      04x_model_comparison.ipynb
      05_calibration.ipynb
      06_validation.ipynb
    pipeline/
      run_config.md
      stage_01.md
      stage_02.md
      stage_03.md
      stage_04a.md
      stage_04b.md
      stage_04c.md
      stage_04x.md
      stage_05.md
      stage_06.md
      stage_XX_fixes.md
      fixes_summary.md
      model_params_miv.json
      model_params_xgb.json
      model_params_fwd.json
      model_params.json          # copy of champion (canonical path for stages 05+)
    figures/
      01_missing_rates.png
      04a_*.png, 04b_*.png, 04c_*.png, 04x_*.png
      ...
    data/
      loans_clean.csv
```

**At the start of each new pipeline run:**
1. Generate a timestamp: `YYYY-MM-DD_HHMMSS`
2. Create the run directory: `runs/<timestamp>/`
3. Create subdirectories: `notebooks/`, `pipeline/`, `figures/`, `data/`
4. Write `{RUN_DIR}/pipeline/run_config.md` with mode and metadata
5. Pass the run directory path (e.g., `runs/2026-03-14_143022/`) to every subagent as `RUN_DIR`

All subagent file paths are relative to `RUN_DIR`. For example:
- `{RUN_DIR}/notebooks/01_data_quality.ipynb`
- `{RUN_DIR}/pipeline/stage_01.md`
- `{RUN_DIR}/figures/01_missing_rates.png`
- `{RUN_DIR}/data/loans_clean.csv`

The raw dataset remains at `data/loans.csv` (shared across runs). The clean dataset is written inside the run directory.

## Pipeline Stages

| Stage | Subagent | Purpose |
|---|---|---|
| 01 | `01-data-explorer` | Univariate analysis, data quality assessment |
| 02 | `02-data-preparer` | Imputation and outlier treatment |
| 03 | `03-bivariate-analyst` | WoE, IV, AUC analysis, variable shortlisting |
| 04a | `04a-model-builder-miv` | MIV stepwise variable selection + logistic regression |
| 04b | `04b-model-builder-xgb` | XGBoost importance variable selection + logistic regression |
| 04c | `04c-model-builder-fwd` | Forward stepwise variable selection + logistic regression |
| 04x | `04x-model-comparator` | Compare three models, select champion |
| 05 | `05-calibrator` | Rating scale calibration, stress testing |
| 06 | `06-validator` | Full validation suite, overall assessment |
| 07 | `07-report-writer` | Generate DOCX report from all pipeline outputs |
| -- | `notebook-reviewer` | Optional second-pass domain review |

**Stage 04 parallel execution:** Stages 04a, 04b, and 04c run **simultaneously** (launched as three parallel Agent invocations). After all three complete, stage 04x compares them and copies the champion's `model_params.json` to the canonical path. Stages 05+ read only the canonical `model_params.json` — no changes needed downstream.

## Orchestration Rules

These rules apply in both modes unless noted otherwise.

1. **Always load the `pd-conventions` skill** before making any threshold or classification decisions
2. **Every subagent writes a notebook and a `{RUN_DIR}/pipeline/stage_XX.md`** before returning — these are the only inter-stage communication mechanism
3. **Notebooks are executed via `nbconvert` and cleared of outputs** before saving — the notebook on disk is always clean source, all plots are in `{RUN_DIR}/figures/`
4. **Run the `output-verifier` skill checklist inline** after each subagent completes. If verification fails, re-invoke the subagent with the failure details (max 2 retries).
5. **Run the `fix-proposer` diagnostic** within each subagent (stages 01-06, including 04a/04b/04c/04x) after output-verifier passes. Each subagent writes `{RUN_DIR}/pipeline/stage_XX_fixes.md`. This is advisory — it does not block the pipeline.
6. **On stage completion, pass only the stage summary path** (`{RUN_DIR}/pipeline/stage_XX.md`) to the next subagent
6a. **Stage 04 parallel launch:** After stage 03 completes, launch stages 04a, 04b, and 04c **simultaneously** as three parallel Agent invocations. Wait for all three to complete. Then invoke stage 04x (model comparator) which reads all three summaries, selects the champion, and copies its `model_params_*.json` to the canonical `model_params.json`. Only then proceed to stage 05.
6b. **Stage 04 partial failure:** If one of the three model builders fails after 2 retries, proceed with whichever completed. The comparator handles 2-of-3 or 1-of-3 completion. If all three fail, abort the run.

### HIL-only rules
7. **Stop after every stage.** Present the stage summary, fix proposals count, and notebook path to the human. Wait for approval before invoking the next stage.
8. **On human rejection:** re-invoke the same subagent with the reviewer's specific feedback
9. **Optionally invoke `notebook-reviewer`** if the stage summary contains flags or if the human requests a second opinion

### Autonomous-only rules
7. **Proceed to the next stage immediately** after output-verifier passes — do not wait for human input
8. **If a stage produces flags:** invoke `notebook-reviewer` automatically and record its assessment in `{RUN_DIR}/pipeline/stage_XX_review.md`
9. **If output-verifier fails after 2 retries:** abort the run, present the error, and save abort reason to `{RUN_DIR}/pipeline/run_aborted.md`
10. **After stage 06 completes, before invoking stage 07:** read all `stage_XX_fixes.md` files and generate `{RUN_DIR}/pipeline/fixes_summary.md` (consolidated fix proposals grouped by category and severity)
11. **At the end of the pipeline:** present a consolidated summary of all 6 stages, all flags, all reviewer assessments, fix proposal counts, and the final overall assessment

## Context Management Rules

| Rule | Rationale |
|---|---|
| Subagents write to disk, return only short summaries | Keeps main agent context clean |
| Notebooks never embed base64 — all plots saved to `{RUN_DIR}/figures/` | Single plot can be 50K+ characters |
| Convert notebooks to `.py` before reading back | Removes output metadata bloat |
| Inter-stage data via `{RUN_DIR}/pipeline/*.md` files, not prompt strings | Stage summaries are small structured text |
| `pdtoolkit-api` skill prevents hallucinated function calls | Wasted tool calls consume context fast |
| Stage 03 processes variables in batches for wide datasets | Prevents bivariate analyst context from filling |

## Pipeline Execution Flow

### Autonomous Mode

```
User: "Build PD model on loans.csv"
  │
  ├─► Create run directory: runs/<timestamp>/
  ├─► Write run_config.md (mode: autonomous)
  │
  ├─► Invoke data-explorer (stage 01)
  │     └─► output-verifier → pass? continue : retry (max 2) or abort
  │     └─► fix-proposer → write stage_01_fixes.md
  │
  ├─► Invoke data-preparer (stage 02)
  │     └─► output-verifier → pass? continue : retry or abort
  │     └─► fix-proposer → write stage_02_fixes.md
  │
  ├─► Invoke bivariate-analyst (stage 03)
  │     └─► output-verifier
  │     └─► fix-proposer → write stage_03_fixes.md
  │     └─► flags? → invoke notebook-reviewer, save review
  │
  ├─► [PARALLEL] Launch three model builders simultaneously:
  │     ├─► 04a-model-builder-miv
  │     │     └─► output-verifier → pass? continue : retry (max 2)
  │     │     └─► fix-proposer → write stage_04a_fixes.md
  │     │     └─► flags? → invoke notebook-reviewer, save review
  │     ├─► 04b-model-builder-xgb
  │     │     └─► output-verifier → pass? continue : retry (max 2)
  │     │     └─► fix-proposer → write stage_04b_fixes.md
  │     │     └─► flags? → invoke notebook-reviewer, save review
  │     └─► 04c-model-builder-fwd
  │           └─► output-verifier → pass? continue : retry (max 2)
  │           └─► fix-proposer → write stage_04c_fixes.md
  │           └─► flags? → invoke notebook-reviewer, save review
  │
  ├─► [WAIT] All three model builders must complete (or fail) before proceeding
  │
  ├─► Invoke model-comparator (stage 04x)
  │     └─► Reads stage_04a.md, stage_04b.md, stage_04c.md
  │     └─► Reads model_params_miv.json, model_params_xgb.json, model_params_fwd.json
  │     └─► Produces comparison notebook + stage_04x.md
  │     └─► Copies champion model_params to canonical model_params.json
  │     └─► output-verifier
  │     └─► fix-proposer → write stage_04x_fixes.md
  │
  ├─► Invoke calibrator (stage 05)
  │     └─► output-verifier
  │     └─► fix-proposer → write stage_05_fixes.md
  │
  ├─► Invoke validator (stage 06)
  │     └─► output-verifier
  │     └─► fix-proposer → write stage_06_fixes.md
  │     └─► flags? → invoke notebook-reviewer, save review
  │
  ├─► Generate fixes_summary.md from all stage_XX_fixes.md (including 04a, 04b, 04c, 04x)
  │
  └─► Invoke report-writer (stage 07)
        └─► Reads all stage_*.md (including 04a/04b/04c/04x), fixes_summary.md, model_params.json, figures/*.png
        └─► Writes report.md → pandoc → report.docx
        │
        └─► Present consolidated summary:
              - All stage summaries (01-03, 04a/04b/04c/04x, 05-06)
              - Model comparison: champion method, composite scores
              - All flags and reviewer assessments
              - Fix proposals: [N] total ([N] critical)
              - Overall assessment: PASS / PASS WITH FLAGS / FAIL
              - Report: {RUN_DIR}/report.docx
```

### HIL (Human-in-the-Loop) Mode

```
User: "Build PD model on loans.csv, I want to review each stage"
  │
  ├─► Create run directory: runs/<timestamp>/
  ├─► Write run_config.md (mode: hil)
  │
  ├─► Invoke data-explorer (stage 01)
  │     └─► output-verifier
  │     └─► fix-proposer → write stage_01_fixes.md
  │     └─► Present summary + fix proposals to human
  │     └─► [CHECKPOINT 01] ◄── Approve / Reject / Modify
  │
  ├─► Invoke data-preparer (stage 02)
  │     └─► output-verifier
  │     └─► fix-proposer → write stage_02_fixes.md
  │     └─► Present summary + fix proposals to human
  │     └─► [CHECKPOINT 02] ◄── Approve / Reject
  │
  ├─► Invoke bivariate-analyst (stage 03)
  │     └─► output-verifier
  │     └─► fix-proposer → write stage_03_fixes.md
  │     └─► [optional] notebook-reviewer if flags
  │     └─► Present summary + shortlist + fix proposals to human
  │     └─► [CHECKPOINT 03] ◄── Approve/modify shortlist (most critical gate)
  │
  ├─► [PARALLEL] Launch three model builders simultaneously:
  │     ├─► 04a-model-builder-miv
  │     │     └─► output-verifier
  │     │     └─► fix-proposer → write stage_04a_fixes.md
  │     ├─► 04b-model-builder-xgb
  │     │     └─► output-verifier
  │     │     └─► fix-proposer → write stage_04b_fixes.md
  │     └─► 04c-model-builder-fwd
  │           └─► output-verifier
  │           └─► fix-proposer → write stage_04c_fixes.md
  │
  ├─► [WAIT] All three model builders must complete (or fail) before proceeding
  │
  ├─► Invoke model-comparator (stage 04x)
  │     └─► Reads all three stage summaries + model params
  │     └─► Produces comparison notebook + stage_04x.md
  │     └─► fix-proposer → write stage_04x_fixes.md
  │     └─► Present comparison table + champion recommendation to human
  │     └─► [CHECKPOINT 04] ◄── Confirm/override champion model selection
  │
  ├─► Invoke calibrator (stage 05)
  │     └─► output-verifier
  │     └─► fix-proposer → write stage_05_fixes.md
  │     └─► Present summary + fix proposals to human
  │     └─► [CHECKPOINT 05] ◄── Approve rating scale
  │
  ├─► Invoke validator (stage 06)
  │     └─► output-verifier
  │     └─► fix-proposer → write stage_06_fixes.md
  │     └─► [optional] notebook-reviewer for flagged tests
  │     └─► Present full validation summary + fix proposals to human
  │     └─► [CHECKPOINT 06] ◄── Final sign-off
  │
  ├─► Generate fixes_summary.md from all stage_XX_fixes.md (including 04a, 04b, 04c, 04x)
  │
  └─► Invoke report-writer (stage 07)
        └─► Writes report.md → pandoc → report.docx
        └─► Present report path to human: {RUN_DIR}/report.docx
```

## Deliverables

After the pipeline completes, the run directory contains:

| File | Contents |
|---|---|
| `{RUN_DIR}/pipeline/run_config.md` | Run metadata (mode, timestamp, dataset) |
| `{RUN_DIR}/notebooks/01_data_quality.ipynb` | Univariate analysis, missing rates, correlations |
| `{RUN_DIR}/notebooks/02_data_preparation.ipynb` | Imputation results, before/after distributions |
| `{RUN_DIR}/notebooks/03_bivariate_analysis.ipynb` | WoE profiles, IV ranking, shortlist |
| `{RUN_DIR}/notebooks/04a_model_building_miv.ipynb` | MIV stepwise selection, coefficients, score distribution |
| `{RUN_DIR}/notebooks/04b_model_building_xgb.ipynb` | XGBoost importance selection, coefficients, score distribution |
| `{RUN_DIR}/notebooks/04c_model_building_fwd.ipynb` | Forward stepwise selection, coefficients, score distribution |
| `{RUN_DIR}/notebooks/04x_model_comparison.ipynb` | Model comparison, ROC overlay, champion selection |
| `{RUN_DIR}/notebooks/05_calibration.ipynb` | Rating scale, grade distribution, stress test |
| `{RUN_DIR}/notebooks/06_validation.ipynb` | Full validation suite, overall assessment |
| `{RUN_DIR}/pipeline/stage_01.md` through `stage_06.md` | Structured stage summaries — model audit trail |
| `{RUN_DIR}/pipeline/stage_04a.md`, `stage_04b.md`, `stage_04c.md`, `stage_04x.md` | Triple-track model building summaries + comparison |
| `{RUN_DIR}/pipeline/model_params_miv.json`, `model_params_xgb.json`, `model_params_fwd.json` | Per-method model parameters |
| `{RUN_DIR}/pipeline/stage_XX_review.md` | Notebook reviewer assessments (if flags were present) |
| `{RUN_DIR}/pipeline/stage_XX_fixes.md` | Fix-proposer diagnostic output per stage |
| `{RUN_DIR}/pipeline/fixes_summary.md` | Consolidated fix proposals (generated before stage 07) |
| `{RUN_DIR}/pipeline/model_params.json` | Final model parameters |
| `{RUN_DIR}/data/loans_clean.csv` | Prepared dataset |
| `{RUN_DIR}/figures/*.png` | All plots referenced by notebooks |
| `{RUN_DIR}/report.md` | Full report in markdown (source for DOCX) |
| `{RUN_DIR}/report.docx` | Comprehensive Word model development report |
