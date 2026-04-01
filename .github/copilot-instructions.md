# Copilot Instructions for agent-research-thesis

## Project
This repository implements the thesis **"Interpreting and Stress-Testing Context-Aware Asset Pricing with Linear Attention"**, based on Kelly, Malamud, Ramirez & Zhou (NBER WP 33351, 2025).

## Key Concepts
- **Linear transformer attention:** A = Q K^T (no softmax).
- **Symmetric component S = (A + A^T)/2** → factor structure (IPCA-equivalent).
- **Antisymmetric component AS = (A - A^T)/2** → mispricing / anomaly signals.
- **Three hypotheses:** H1 (symmetric ↔ factors), H2 (antisymmetric ↔ mispricing), H3 (regime variation).

## Code Conventions
- Python 3.10+, PyTorch for models, pandas/numpy for data, matplotlib/seaborn for plots.
- Scripts live in `scripts/` and accept `--config` YAML files from `config/`.
- Raw data → `data/raw/`, processed → `data/processed/`, results → `outputs/`.
- PEP 8, type hints, docstrings that reference the paper methodology.
- Use `logging` module, not print statements.
- Reproducibility: set seeds, log all hyperparameters, save model checkpoints.

## Agent
Use `@thesis-expert` for deep methodological guidance on the paper, hypotheses, and analysis pipeline.
