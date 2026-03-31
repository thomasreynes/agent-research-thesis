# Agent Research Thesis

**Interpreting and Stress-Testing Context-Aware Asset Pricing with Linear Attention**

Author: Thomas Reynes

## Overview

This repository contains the complete AI agent system and codebase for a master's thesis that:

1. Implements the LINEAR transformer case from Kelly et al. (NBER WP 33351, 2025)
2. Decomposes attention into symmetric and antisymmetric components
3. Tests three hypotheses about what these components capture economically
4. Analyzes variation across market regimes
5. Validates findings with placebo and stress tests

## Getting Started

### 1. Clone and open in VS Code

```bash
git clone https://github.com/thomasreynes/agent-research-thesis.git
cd agent-research-thesis
code .
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Add raw data

Place your CRSP monthly returns and Compustat annual fundamentals files in `data/raw/`:

```
data/raw/crsp_monthly.parquet
data/raw/compustat_annual.parquet
```

File paths and cleaning parameters are configured in `config/default.yaml`.

### 4. Run the audit script

```bash
python scripts/audit.py
```

### 5. Start coding with Copilot agents

Open Copilot Chat in **Agent mode** in VS Code and invoke the thesis-specific agents:

| Agent | Invocation | Role |
|---|---|---|
| Memory | `@memory` | Initialise / query the thesis knowledge graph |
| Linear Attention | `@linear-attention-agent` | PyTorch model changes |
| Data Pipeline | `@data-agent` | CRSP/Compustat pipeline and characteristics |
| Hypothesis Tests | `@hypothesis-agent` | Panel regressions (H1/H2/H3) |
| Visualisation | `@thesis-agent` | Figures and LaTeX tables |
| Code Quality | `@auditor-agent` | Naming, dead code, equation traceability |

## Project Structure

```
agent-research-thesis/
├── config/
│   └── default.yaml          # All hyperparameters, data paths, regime thresholds
├── data/
│   ├── raw/                  # Place CRSP / Compustat files here (not committed)
│   └── processed/            # Cleaned panel data (not committed)
├── outputs/
│   ├── figures/              # Saved plots (.png / .pdf / .pgf)
│   └── tables/               # LaTeX tables (.tex)
├── scripts/
│   └── audit.py              # Automated code-quality auditor
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── attention.py      # LinearAttention, decompose_attention, compute_sdf_weights
│   │   └── sdf.py            # SDFModel, pricing_error
│   ├── data/
│   │   ├── loader.py         # load_config, CRSPLoader, CompustatLoader, merge_crsp_compustat
│   │   └── features.py       # compute_firm_characteristics, normalize_features, construct_pairs
│   ├── analysis/
│   │   ├── regressions.py    # run_h1_regression, run_h2_regression, run_h3_placebo, RegressionResults
│   │   └── diagnostics.py    # VIF, heteroskedasticity, serial-correlation checks
│   └── viz/
│       ├── plots.py          # setup_style, plot_attention_heatmap, plot_symmetric_vs_antisymmetric, …
│       └── tables.py         # regression_table, summary_statistics_table
├── context/                  # Agent memory / knowledge graph artefacts
├── requirements.txt
└── README.md
```

### Subpackage descriptions

| Subpackage | Purpose |
|---|---|
| `src/model` | Linear attention model `A_t = X_t W X_t^T` and SDF computation |
| `src/data` | CRSP/Compustat loading, cleaning, feature engineering for X_t |
| `src/analysis` | Panel regressions for H1, H2, H3 with clustered standard errors |
| `src/viz` | 300 DPI publication figures and LaTeX table export |
