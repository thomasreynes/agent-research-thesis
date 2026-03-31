"""AIPM Linear Attention Thesis — source package.

This package contains the core modules for the master's thesis:
"Interpreting Linear Attention in Asset Pricing: Symmetric vs.
Antisymmetric Components as Economic Signals"

Subpackages
-----------
model     : Linear attention mechanism and SDF computation
data      : CRSP/Compustat data loading and feature engineering
analysis  : Panel regressions (H1, H2, H3) and diagnostics
viz       : Publication-quality plots and LaTeX tables
"""

__version__ = "0.1.0"
__author__ = "Thomas Reynes"

__all__ = ["model", "data", "analysis", "viz", "__version__"]
