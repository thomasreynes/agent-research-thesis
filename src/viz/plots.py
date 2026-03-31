"""Publication-quality visualisations for the AIPM Linear Attention thesis.

All plots default to saving in ``outputs/figures/`` (configurable via
``config/default.yaml → visualization.figure_dir``) at 300 DPI with a
colourblind-safe palette.

Usage
-----
>>> from src.viz.plots import setup_style, plot_attention_heatmap
>>> setup_style()
>>> plot_attention_heatmap(A, title="Attention t=2000-06", save_path="outputs/figures/attn.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


_FIGURE_DIR = Path("outputs/figures")
_DPI = 300
_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink/purple
    "#56B4E9",  # light blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
]


def setup_style() -> None:
    """Configure matplotlib for publication-quality output.

    Sets 300 DPI, a colourblind-safe discrete palette (Wong 2011), and
    font sizes consistent with a master's thesis.  Call once at the top
    of any script or notebook before plotting.

    The pgf backend is NOT activated here by default because it requires
    a full TeX installation.  To enable it, set
    ``matplotlib.use('pgf')`` before importing pyplot.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.dpi": _DPI,
            "savefig.dpi": _DPI,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.prop_cycle": mpl.cycler("color", _PALETTE),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.constrained_layout.use": True,
        }
    )


def plot_attention_heatmap(
    A: "np.ndarray",
    title: str = "Attention matrix",
    save_path: Optional[str] = None,
) -> None:
    """Plot a heatmap of the attention matrix A_t.

    Args:
        A: Attention matrix of shape (N, N) as a numpy array or torch
            tensor (converted automatically).
        title: Plot title.
        save_path: File path to save the figure.  Defaults to
            ``outputs/figures/attention_heatmap.png``.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if hasattr(A, "detach"):
        A = A.detach().cpu().numpy()
    A = np.asarray(A, dtype=float)

    save_path = save_path or str(_FIGURE_DIR / "attention_heatmap.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        A,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"shrink": 0.75, "label": "Attention weight"},
    )
    ax.set_title(title)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_symmetric_vs_antisymmetric(
    A_s: "np.ndarray",
    A_a: "np.ndarray",
    save_path: Optional[str] = None,
) -> None:
    """Plot symmetric and antisymmetric attention components side by side.

    Visualises the decomposition:
        A^s = (A + A^T) / 2   — mutual similarity (H1)
        A^a = (A - A^T) / 2   — directional lead-lag (H2)

    Args:
        A_s: Symmetric component of shape (N, N).
        A_a: Antisymmetric component of shape (N, N).
        save_path: Output file path.  Defaults to
            ``outputs/figures/symmetric_vs_antisymmetric.png``.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    for arr in (A_s, A_a):
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()

    A_s = np.asarray(A_s, dtype=float)
    A_a = np.asarray(A_a, dtype=float)

    save_path = save_path or str(_FIGURE_DIR / "symmetric_vs_antisymmetric.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, label in zip(axes, [A_s, A_a], ["$A^s$ (symmetric)", "$A^a$ (antisymmetric)"]):
        sns.heatmap(
            mat,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={"shrink": 0.75},
        )
        ax.set_title(label)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_regression_diagnostics(
    results: "src.analysis.regressions.RegressionResults",
    save_path: Optional[str] = None,
) -> None:
    """Bar chart of regression coefficients with t-stat shading.

    Args:
        results: :class:`src.analysis.regressions.RegressionResults` object.
        save_path: Output file path.  Defaults to
            ``outputs/figures/regression_<hypothesis>.png``.
    """
    import matplotlib.pyplot as plt

    coefs = results.coefficients
    tstats = results.t_stats

    vars_ = [k for k in coefs if k != "const"]
    vals = [coefs[k] for k in vars_]
    ts = [abs(tstats.get(k, 0.0)) for k in vars_]

    save_path = save_path or str(_FIGURE_DIR / f"regression_{results.hypothesis}.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = [_PALETTE[0] if t >= 1.96 else "#AAAAAA" for t in ts]
    ax.barh(vars_, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient")
    ax.set_title(f"{results.hypothesis} — n={results.n_obs:,}  R²={results.r_squared:.3f}")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_sdf_timeseries(
    sdf_weights: "np.ndarray",
    dates: Sequence,
    save_path: Optional[str] = None,
) -> None:
    """Plot the SDF portfolio weight time series.

    Args:
        sdf_weights: Array of shape (T,) or (T, N) containing SDF weights
            over time.  If 2-D, the cross-sectional mean is plotted.
        dates: Sequence of T date values compatible with matplotlib.
        save_path: Output file path.  Defaults to
            ``outputs/figures/sdf_timeseries.png``.
    """
    import matplotlib.pyplot as plt

    w = np.asarray(sdf_weights, dtype=float)
    if w.ndim == 2:
        w = w.mean(axis=1)

    save_path = save_path or str(_FIGURE_DIR / "sdf_timeseries.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(dates, w, color=_PALETTE[0], linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean SDF weight")
    ax.set_title("SDF portfolio weights over time")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
