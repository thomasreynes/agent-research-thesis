"""Analysis subpackage for the AIPM Linear Attention thesis.

Exposes panel regression runners (H1, H2, H3) and diagnostic utilities.
"""

from src.analysis.regressions import (
    RegressionResults,
    run_h1_regression,
    run_h2_regression,
    run_h3_placebo,
)
from src.analysis.diagnostics import run_all_diagnostics

__all__ = [
    "RegressionResults",
    "run_h1_regression",
    "run_h2_regression",
    "run_h3_placebo",
    "run_all_diagnostics",
]
