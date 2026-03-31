"""Model diagnostics for the AIPM Linear Attention thesis.

Provides checks for multicollinearity (VIF), heteroskedasticity, and
serial correlation that are run after each panel regression to validate
model assumptions.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Compute Variance Inflation Factors for multicollinearity detection.

    A VIF > 10 is typically considered problematic.

    Args:
        X: Regressor matrix (excluding constant) with shape (N, K).

    Returns:
        Series indexed by regressor name with VIF values.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        raise ImportError("statsmodels is required. Run: pip install statsmodels")

    vif_data = {col: variance_inflation_factor(X.values, i) for i, col in enumerate(X.columns)}
    return pd.Series(vif_data, name="VIF")


def test_heteroskedasticity(residuals: pd.Series, X: pd.DataFrame) -> Dict[str, Any]:
    """Breusch-Pagan test for heteroskedasticity.

    Args:
        residuals: Regression residuals of shape (N,).
        X: Regressor matrix used in the regression of shape (N, K).

    Returns:
        Dictionary with keys:
        - ``bp_stat``: Breusch-Pagan LM statistic
        - ``bp_pvalue``: p-value (small → reject homoskedasticity)
        - ``reject_h0``: bool, True if heteroskedasticity detected at 5%
    """
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels is required. Run: pip install statsmodels")

    X_with_const = sm.add_constant(X)
    lm_stat, lm_pvalue, _, _ = het_breuschpagan(residuals, X_with_const)
    return {
        "bp_stat": float(lm_stat),
        "bp_pvalue": float(lm_pvalue),
        "reject_h0": bool(lm_pvalue < 0.05),
    }


def test_serial_correlation(residuals: pd.Series, nlags: int = 4) -> Dict[str, Any]:
    """Ljung-Box test for serial correlation in residuals.

    Args:
        residuals: Regression residuals of shape (N,).
        nlags: Number of lags to test.  Defaults to 4.

    Returns:
        Dictionary with keys:
        - ``lb_stats``: array of Ljung-Box statistics for each lag
        - ``lb_pvalues``: array of p-values
        - ``reject_h0``: bool, True if serial correlation detected at 5%
          in any lag
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        raise ImportError("statsmodels is required. Run: pip install statsmodels")

    result = acorr_ljungbox(residuals.dropna(), lags=nlags, return_df=True)
    lb_stats = result["lb_stat"].values
    lb_pvalues = result["lb_pvalue"].values
    return {
        "lb_stats": lb_stats.tolist(),
        "lb_pvalues": lb_pvalues.tolist(),
        "reject_h0": bool((lb_pvalues < 0.05).any()),
    }


def run_all_diagnostics(
    residuals: pd.Series,
    X: pd.DataFrame,
    nlags: int = 4,
) -> Dict[str, Any]:
    """Run the full diagnostic battery on a fitted regression.

    Combines VIF, Breusch-Pagan, and Ljung-Box tests into a single
    report dictionary.

    Args:
        residuals: Regression residuals of shape (N,).
        X: Regressor matrix (excluding constant) of shape (N, K).
        nlags: Number of lags for the Ljung-Box test.

    Returns:
        Dictionary with keys ``'vif'``, ``'heteroskedasticity'``, and
        ``'serial_correlation'``, each containing the output of the
        respective diagnostic function.
    """
    return {
        "vif": compute_vif(X).to_dict(),
        "heteroskedasticity": test_heteroskedasticity(residuals, X),
        "serial_correlation": test_serial_correlation(residuals, nlags=nlags),
    }
