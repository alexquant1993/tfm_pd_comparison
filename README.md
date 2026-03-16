# pd-autopilot

Automated Probability of Default (PD) model development pipeline powered by Claude Code agents and the [pdtoolkit](https://github.com/at621/pdtoolkit_py) library.

## What it does

Takes a loan dataset and produces a fully documented PD model through automated stages, including parallel model comparison and a comprehensive Word report:

1. **Data Quality** — univariate analysis, missing values, correlations, outlier detection
2. **Data Preparation** — imputation, outlier capping
3. **Bivariate Analysis** — WoE/IV via OptimalBinning, variable shortlisting
4. **Model Building** (three methods in parallel):
   - **04a** MIV stepwise selection → logistic regression
   - **04b** XGBoost feature importance → logistic regression
   - **04c** Forward stepwise selection → logistic regression
   - **04x** Model comparison → champion selection via weighted composite score
5. **Calibration** — rating grades, calibrated PDs, stress testing
6. **Validation** — discriminatory power, predictive power, homogeneity, heterogeneity, PSI
7. **Report** — comprehensive Word document generated via pandoc

Each stage produces a Jupyter notebook, figures, a structured summary, and diagnostic fix proposals. A timestamped execution log tracks durations and key metrics.

## Execution modes

- **Autonomous** — runs all stages end-to-end, presents consolidated results at the end
- **HIL (Human-in-the-Loop)** — stops after each stage for review and approval; human can override shortlists, champion selection, or switch to autonomous mid-run

## Project structure

```
├── CLAUDE.md                        # Orchestrator instructions
├── data/
│   └── loans.csv                    # Input dataset
├── src/
│   ├── pdtoolkit/                   # PD toolkit library
│   ├── pipeline_fixes/              # Utility functions from fix-proposer diagnostics
│   ├── scripts/
│   │   └── format_report.py         # DOCX table formatting post-processor
│   └── templates/
│       └── doc_template.docx        # Word report style template
├── runs/                            # Pipeline outputs (timestamped)
│   └── YYYY-MM-DD_HHMMSS/
│       ├── notebooks/               # 9 Jupyter notebooks (01-06, 04a/04b/04c/04x)
│       ├── pipeline/                # Stage summaries, model params, execution.log
│       ├── figures/                  # All plots
│       ├── data/                    # Clean + binned datasets
│       ├── report.md                # Report source
│       └── report.docx              # Final Word report
├── .claude/
│   ├── agents/                      # 11 subagent definitions
│   │   ├── 01-data-explorer.md
│   │   ├── 02-data-preparer.md
│   │   ├── 03-bivariate-analyst.md
│   │   ├── 04a-model-builder-miv.md
│   │   ├── 04b-model-builder-xgb.md
│   │   ├── 04c-model-builder-fwd.md
│   │   ├── 04x-model-comparator.md
│   │   ├── 05-calibrator.md
│   │   ├── 06-validator.md
│   │   ├── 07-report-writer.md
│   │   └── notebook-reviewer.md
│   └── skills/                      # 5 shared skill definitions
│       ├── pd-conventions/          # Domain thresholds (IV, AUC, WoE rules)
│       ├── pdtoolkit-api/           # Complete function reference
│       ├── notebook-writer/         # Notebook format conventions
│       ├── output-verifier/         # Post-stage verification checklist
│       └── fix-proposer/            # Diagnostic protocol for each stage
```

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
- Python 3.10+ with: pandas, numpy, matplotlib, scikit-learn, statsmodels, jupyter, nbconvert, optbinning, xgboost, python-docx
- [pandoc](https://pandoc.org/installing.html) (for Word report generation)

## Usage

Place your dataset at `data/loans.csv` and start Claude Code in the project directory.

**Autonomous mode:**
```
Build a PD model on loans.csv
```

**HIL mode:**
```
Build a PD model on loans.csv, I want to review each stage
```

Results are saved to `runs/<timestamp>/`, including notebooks, figures, model parameters, execution log, and a Word report.
