"""Panel regression runners for the three thesis hypotheses.

Hypothesis overview
-------------------
H1 (symmetric attention ↔ economic similarity):
    A^s_{ij,t} = α + β₁ · 1[same_industry] + β₂ · |size_i − size_j|
               + β₃ · |style_i − style_j| + ε

H2 (antisymmetric attention ↔ directional lead-lag):
    A^a_{ij,t} = α + γ₁ · 1[large_i, small_j] + γ₂ · 1[liquid_i, illiquid_j] + ε

H3 (placebo / stress tests):
    Re-run H1 and H2 under shuffled/permuted inputs — coefficients
    should collapse to zero under the null.

Standard errors are always clustered as specified in
``config/default.yaml`` under ``hypothesis_tests.h{1,2}.cluster_se``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class RegressionResults:
    """Container for panel regression output.

    Attributes:
        hypothesis: Which hypothesis this result corresponds to ('H1', 'H2', 'H3').
        coefficients: Estimated coefficients as a dict mapping variable name
            to float.
        t_stats: Clustered t-statistics, same keys as ``coefficients``.
        p_values: Two-sided p-values, same keys as ``coefficients``.
        r_squared: Within R² of the regression.
        n_obs: Number of observations used.
        cluster_vars: Variables used for standard error clustering.
        notes: Optional free-text notes about the specification.
    """

    hypothesis: str
    coefficients: dict
    t_stats: dict
    p_values: dict
    r_squared: float
    n_obs: int
    cluster_vars: List[str] = field(default_factory=list)
    notes: Optional[str] = None


def run_h1_regression(
    attention_symmetric: pd.DataFrame,
    similarity_measures: pd.DataFrame,
    cluster_vars: Optional[List[str]] = None,
) -> RegressionResults:
    """Test H1: symmetric attention reflects economic similarity.

    Regresses the symmetric attention value A^s_{ij,t} on pair-level
    similarity measures:
        A^s_{ij,t} = α + β₁ · same_industry + β₂ · size_diff
                   + β₃ · style_diff + ε_{ij,t}

    Standard errors are two-way clustered on time and pair by default
    (as in ``config/default.yaml → hypothesis_tests.h1.cluster_se``).

    Args:
        attention_symmetric: Long-format DataFrame with columns
            ``[date, permno_i, permno_j, A_s]`` — symmetric attention values.
        similarity_measures: Long-format DataFrame with columns
            ``[date, permno_i, permno_j, same_industry, size_diff, style_diff]``.
        cluster_vars: List of column names to cluster standard errors on.
            Defaults to ``['time', 'pair']``.

    Returns:
        :class:`RegressionResults` for H1.
    """
    if cluster_vars is None:
        cluster_vars = ["time", "pair"]

    # Merge attention values with similarity measures
    df = pd.merge(
        attention_symmetric,
        similarity_measures,
        on=["date", "permno_i", "permno_j"],
        how="inner",
    )

    if df.empty:
        return RegressionResults(
            hypothesis="H1",
            coefficients={},
            t_stats={},
            p_values={},
            r_squared=float("nan"),
            n_obs=0,
            cluster_vars=cluster_vars,
            notes="Merge produced empty DataFrame — check input column names.",
        )

    regressors = ["same_industry", "size_diff", "style_diff"]
    regressors = [r for r in regressors if r in df.columns]

    try:
        import statsmodels.api as sm

        X = sm.add_constant(df[regressors].astype(float))
        y = df["A_s"].astype(float)
        # TODO: Replace HC3-robust OLS with two-way clustered panel estimator
        # (linearmodels.PanelOLS with cluster_se=cluster_vars) once panel
        # index (date, pair) is set up.  cluster_vars parameter is accepted
        # but not yet wired into the estimator below.
        model = sm.OLS(y, X).fit(cov_type="HC3")
        return RegressionResults(
            hypothesis="H1",
            coefficients=dict(zip(X.columns, model.params)),
            t_stats=dict(zip(X.columns, model.tvalues)),
            p_values=dict(zip(X.columns, model.pvalues)),
            r_squared=float(model.rsquared),
            n_obs=int(model.nobs),
            cluster_vars=cluster_vars,
        )
    except ImportError:
        raise ImportError("statsmodels is required for hypothesis testing. Run: pip install statsmodels")


def run_h2_regression(
    attention_antisymmetric: pd.DataFrame,
    momentum_signals: pd.DataFrame,
    cluster_vars: Optional[List[str]] = None,
) -> RegressionResults:
    """Test H2: antisymmetric attention captures directional lead-lag.

    Regresses the antisymmetric attention value A^a_{ij,t} on direction
    indicators:
        A^a_{ij,t} = α + γ₁ · large_to_small + γ₂ · liquid_to_illiquid + ε_{ij,t}

    A positive γ₁ means information flows from large to small firms
    (consistent with gradual diffusion / lead-lag hypothesis).

    Args:
        attention_antisymmetric: Long-format DataFrame with columns
            ``[date, permno_i, permno_j, A_a]`` — antisymmetric attention values.
        momentum_signals: Long-format DataFrame with columns
            ``[date, permno_i, permno_j, large_to_small, liquid_to_illiquid]``.
        cluster_vars: List of column names to cluster standard errors on.
            Defaults to ``['time']``.

    Returns:
        :class:`RegressionResults` for H2.
    """
    if cluster_vars is None:
        cluster_vars = ["time"]

    df = pd.merge(
        attention_antisymmetric,
        momentum_signals,
        on=["date", "permno_i", "permno_j"],
        how="inner",
    )

    if df.empty:
        return RegressionResults(
            hypothesis="H2",
            coefficients={},
            t_stats={},
            p_values={},
            r_squared=float("nan"),
            n_obs=0,
            cluster_vars=cluster_vars,
            notes="Merge produced empty DataFrame — check input column names.",
        )

    regressors = ["large_to_small", "liquid_to_illiquid"]
    regressors = [r for r in regressors if r in df.columns]

    try:
        import statsmodels.api as sm

        X = sm.add_constant(df[regressors].astype(float))
        y = df["A_a"].astype(float)
        # TODO: Replace HC3-robust OLS with panel estimator with time-clustered
        # SE (linearmodels.PanelOLS).  cluster_vars parameter is accepted but
        # not yet wired into the estimator below.
        model = sm.OLS(y, X).fit(cov_type="HC3")
        return RegressionResults(
            hypothesis="H2",
            coefficients=dict(zip(X.columns, model.params)),
            t_stats=dict(zip(X.columns, model.tvalues)),
            p_values=dict(zip(X.columns, model.pvalues)),
            r_squared=float(model.rsquared),
            n_obs=int(model.nobs),
            cluster_vars=cluster_vars,
        )
    except ImportError:
        raise ImportError("statsmodels is required for hypothesis testing. Run: pip install statsmodels")


def run_h3_placebo(
    attention_components: pd.DataFrame,
    placebo_signals: pd.DataFrame,
    n_permutations: int = 1000,
) -> List[RegressionResults]:
    """Test H3: patterns vanish under placebo / stress inputs.

    Runs H1 and H2 regressions on shuffled/permuted inputs and returns
    the distribution of coefficients.  Under the null, all coefficients
    should be centred at zero.

    Placebo types (from ``config/default.yaml → hypothesis_tests.h3``):
    - ``characteristic_shuffle``: cross-sectionally shuffle X_t columns
    - ``attention_permutation``:  randomly permute entries of A
    - ``time_shift_6m``:          shift all signals by 6 months

    Args:
        attention_components: DataFrame with columns
            ``[date, permno_i, permno_j, A_s, A_a]``.
        placebo_signals: DataFrame with placebo-transformed similarity and
            momentum columns (same schema as H1/H2 inputs).
        n_permutations: Number of placebo repetitions.  Defaults to 1000
            (matches ``config/default.yaml → hypothesis_tests.h3.n_placebo_permutations``).

    Returns:
        List of :class:`RegressionResults` — one per placebo repetition,
        alternating between H1-type and H2-type specifications.

    Todo:
        Implement the actual permutation loop using the placebo generator
        in ``src/data/features.py``.
    """
    results: List[RegressionResults] = []

    # Separate A^s and A^a columns
    A_s_df = attention_components[["date", "permno_i", "permno_j", "A_s"]].copy()
    A_a_df = attention_components[["date", "permno_i", "permno_j", "A_a"]].copy()

    # TODO: Implement proper permutation loop over n_permutations
    # For now, run a single pass on the provided placebo_signals to
    # verify the pipeline end-to-end.
    h1_placebo = run_h1_regression(A_s_df, placebo_signals)
    h1_placebo.hypothesis = "H3-H1-placebo"
    h1_placebo.notes = "Single placebo pass — full permutation loop not yet implemented."
    results.append(h1_placebo)

    h2_placebo = run_h2_regression(A_a_df, placebo_signals)
    h2_placebo.hypothesis = "H3-H2-placebo"
    h2_placebo.notes = "Single placebo pass — full permutation loop not yet implemented."
    results.append(h2_placebo)

    return results
