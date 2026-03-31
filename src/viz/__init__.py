"""Visualization subpackage for the AIPM Linear Attention thesis.

Exposes publication-quality plot functions and LaTeX table generators.
"""

from src.viz.plots import (
    setup_style,
    plot_attention_heatmap,
    plot_symmetric_vs_antisymmetric,
    plot_regression_diagnostics,
    plot_sdf_timeseries,
)
from src.viz.tables import regression_table, summary_statistics_table

__all__ = [
    "setup_style",
    "plot_attention_heatmap",
    "plot_symmetric_vs_antisymmetric",
    "plot_regression_diagnostics",
    "plot_sdf_timeseries",
    "regression_table",
    "summary_statistics_table",
]
