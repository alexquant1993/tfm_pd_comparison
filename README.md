# pd-autopilot

Automated Probability of Default (PD) model development pipeline powered by Claude Code agents and the [pdtoolkit](https://github.com/at621/pdtoolkit_py) library.

## What it does

Takes a loan dataset and produces a fully documented PD model through six automated stages:

1. **Data Quality** — univariate analysis, missing values, correlations
2. **Data Preparation** — imputation, outlier treatment
3. **Bivariate Analysis** — WoE, IV, variable shortlisting
4. **Model Building** — stepwise logistic regression, score scaling
5. **Calibration** — rating grades, calibrated PDs, stress testing
6. **Validation** — discriminatory power, predictive power, homogeneity, heterogeneity, PSI
7. **Report** — comprehensive PDF generated via pandoc

Each stage produces a Jupyter notebook, figures, and a structured summary. A final PDF report consolidates everything.

## Execution modes

- **Autonomous** — runs all stages end-to-end, presents results at the end
- **HIL (Human-in-the-Loop)** — stops after each stage for review and approval

## Project structure

```
├── CLAUDE.md                    # Orchestrator instructions
├── data/
│   └── loans.csv                # Input dataset
├── src/
│   └── pdtoolkit/               # PD toolkit library (40 modules)
├── runs/                        # Pipeline outputs (timestamped)
│   └── YYYY-MM-DD_HHMMSS/
│       ├── notebooks/           # 6 Jupyter notebooks
│       ├── pipeline/            # Stage summaries, model params
│       ├── figures/             # All plots
│       ├── data/                # Clean dataset
│       ├── report.md            # Report source
│       └── report.pdf           # Final PDF report
├── .claude/
│   ├── agents/                  # 8 subagent definitions
│   │   ├── 01-data-explorer.md
│   │   ├── 02-data-preparer.md
│   │   ├── 03-bivariate-analyst.md
│   │   ├── 04-model-builder.md
│   │   ├── 05-calibrator.md
│   │   ├── 06-validator.md
│   │   ├── 07-report-writer.md
│   │   └── notebook-reviewer.md
│   └── skills/                  # 4 shared skill definitions
│       ├── pd-conventions/      # Domain thresholds (IV, AUC, WoE rules)
│       ├── pdtoolkit-api/       # Complete function reference
│       ├── notebook-writer/     # Notebook format conventions
│       └── output-verifier/     # Post-stage verification checklist
```

## Prerequisites

- [Claude Code](https://claude.ai/claude-code)
- Python 3.10+
- pandas, numpy, matplotlib, scikit-learn, jupyter, nbconvert
- [pandoc](https://pandoc.org/installing.html) (for PDF report generation)

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

Results are saved to `runs/<timestamp>/`.
