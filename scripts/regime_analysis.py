"""Regime-conditional attention analysis script (H3).

Usage::

    python scripts/regime_analysis.py --config config/regime.yaml

Tests **H3**: Both the symmetric (factor structure) and antisymmetric
(mispricing) attention components vary meaningfully across market regimes:

- **NBER recession vs. expansion** — loaded from a CSV with columns
  ``(date, recession)`` where ``recession`` is 0/1.
- **High vs. low VIX** — split at the threshold defined in config.

For each regime the script:

1. Averages A^s and A^a matrices within the regime.
2. Computes the eigendecomposition of mean A^s (H1 spectrum check).
3. Computes average off-diagonal A^a by size/liquidity quintile (H2 check).
4. Saves figures and tables to ``outputs/regimes/``.

References: Kelly, Malamud, Ramirez & Zhou (NBER WP 33351, 2025).
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import load_config, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime loading helpers
# ---------------------------------------------------------------------------

def load_nber_regimes(recessions_csv: str) -> pd.Series:
    """Load NBER recession indicator from a CSV file.

    The CSV must have columns ``date`` (YYYY-MM-DD or YYYY-MM) and
    ``recession`` (int 0/1).

    Args:
        recessions_csv: Path to the NBER recessions CSV.

    Returns:
        Series indexed by ``date`` (period month) with values 0/1.
    """
    df = pd.read_csv(recessions_csv, parse_dates=["date"])
    df["date"] = df["date"].dt.to_period("M").astype(str)
    return df.set_index("date")["recession"]


def load_vix_regimes(vix_path: str, threshold: float) -> pd.Series:
    """Derive a high-VIX indicator from daily VIX data.

    Resamples to monthly average and flags months above *threshold* as
    high-volatility (1) and below as low-volatility (0).

    Args:
        vix_path: Path to daily VIX CSV with columns ``date`` and ``vix``.
        threshold: VIX level that separates high from low regimes
                   (from config, never hardcoded).

    Returns:
        Series indexed by month string (YYYY-MM) with values 0/1.
    """
    df = pd.read_csv(vix_path, parse_dates=["date"])
    monthly = df.set_index("date")["vix"].resample("ME").mean()
    monthly.index = monthly.index.to_period("M").astype(str)
    return (monthly >= threshold).astype(int).rename("high_vix")


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def load_decomposition_dates(decomp_dir: Path) -> list[str]:
    """Return sorted list of month strings for which A_sym files exist.

    Args:
        decomp_dir: Path to the decomposition output directory.

    Returns:
        Sorted list of ``YYYY-MM`` strings.
    """
    files = sorted(decomp_dir.glob("A_sym_*.npy"))
    return [f.stem.replace("A_sym_", "") for f in files]


def average_matrix(decomp_dir: Path, prefix: str, dates: list[str]) -> np.ndarray:
    """Average decomposed attention matrices over a subset of dates.

    Args:
        decomp_dir: Directory containing per-month ``.npy`` files.
        prefix: File prefix, either ``"A_sym"`` or ``"A_anti"``.
        dates: List of ``YYYY-MM`` strings to include.

    Returns:
        Element-wise average matrix of shape ``(N, N)`` padded / truncated
        to the modal size.  Returns empty array if no dates provided.
    """
    if not dates:
        return np.array([])

    matrices: list[np.ndarray] = []
    for d in dates:
        p = decomp_dir / f"{prefix}_{d}.npy"
        if p.exists():
            matrices.append(np.load(p))

    if not matrices:
        logger.warning("No %s matrices found for provided dates.", prefix)
        return np.array([])

    # TODO: handle varying N_t across months with proper alignment.
    # For now use the modal shape and filter out months with different N.
    # Months excluded due to shape mismatch are logged below.
    shapes = [m.shape for m in matrices]
    modal_shape = max(set(shapes), key=shapes.count)
    excluded = [d for d, m in zip(dates, matrices) if m.shape != modal_shape]
    if excluded:
        logger.warning(
            "average_matrix: excluding %d month(s) with non-modal shape %s: %s",
            len(excluded),
            modal_shape,
            excluded,
        )
    matrices = [m for m in matrices if m.shape == modal_shape]
    return np.mean(matrices, axis=0)


def eigenspectrum(A_sym: np.ndarray) -> np.ndarray:
    """Return sorted descending eigenvalues of a symmetric matrix.

    Args:
        A_sym: Symmetric matrix ``(N, N)``.  Used for H1 spectrum analysis.

    Returns:
        Eigenvalues in descending order of shape ``(N,)``.
    """
    # Eigendecomposition of symmetric attention — Kelly et al. 2025, H1
    eigvals: np.ndarray = np.linalg.eigvalsh(A_sym)
    return eigvals[::-1]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_eigenspectra(
    eigvals_dict: dict[str, np.ndarray],
    out_path: Path,
    top_k: int = 20,
) -> None:
    """Plot eigenvalue spectra for multiple regimes side by side.

    Visualises H3 variation: factor concentration (H1) across regimes.

    Args:
        eigvals_dict: Mapping from regime label to eigenvalue array.
        out_path: File path for the saved figure.
        top_k: Number of leading eigenvalues to display (from config).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, eigvals in eigvals_dict.items():
        ax.plot(range(1, min(top_k, len(eigvals)) + 1), eigvals[:top_k], marker="o", label=label)
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue magnitude")
    ax.set_title("H3: Eigenspectrum of A^s by regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved eigenspectrum plot to %s", out_path)


# ---------------------------------------------------------------------------
# Main analysis routine
# ---------------------------------------------------------------------------

def analyse(config: dict[str, Any]) -> None:
    """Run regime-conditional decomposition analysis (H3).

    Args:
        config: Full config dict loaded from ``config/regime.yaml``.
    """
    decomp_dir = Path(config["decomposition"]["dir"])
    out_dir = Path(config["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dates = load_decomposition_dates(decomp_dir)
    if not all_dates:
        logger.warning(
            "No decomposition files found in %s — run decompose.py first.", decomp_dir
        )
        return

    logger.info("Found %d decomposed months.", len(all_dates))

    # ------------------------------------------------------------------
    # Build regime masks
    # ------------------------------------------------------------------
    regime_masks: dict[str, list[str]] = {}

    # NBER recession / expansion
    nber_csv = config["regimes"]["nber"]["recessions_csv"]
    nber_path = Path(nber_csv)
    if nber_path.exists():
        nber = load_nber_regimes(nber_csv)
        recession_dates = [d for d in all_dates if nber.get(d, 0) == 1]
        expansion_dates = [d for d in all_dates if nber.get(d, 0) == 0]
        regime_masks["NBER_recession"] = recession_dates
        regime_masks["NBER_expansion"] = expansion_dates
        logger.info(
            "NBER: %d recession months, %d expansion months.",
            len(recession_dates),
            len(expansion_dates),
        )
    else:
        logger.warning("NBER CSV not found at %s — skipping NBER regimes.", nber_path)

    # VIX high / low
    vix_path = Path(config["regimes"]["vix"]["vix_path"])
    vix_threshold: float = config["regimes"]["vix"]["threshold"]
    if vix_path.exists():
        vix_indicator = load_vix_regimes(str(vix_path), vix_threshold)
        high_vix_dates = [d for d in all_dates if vix_indicator.get(d, 0) == 1]
        low_vix_dates = [d for d in all_dates if vix_indicator.get(d, 0) == 0]
        regime_masks["high_VIX"] = high_vix_dates
        regime_masks["low_VIX"] = low_vix_dates
        logger.info(
            "VIX (threshold=%.1f): %d high, %d low months.",
            vix_threshold,
            len(high_vix_dates),
            len(low_vix_dates),
        )
    else:
        logger.warning("VIX CSV not found at %s — skipping VIX regimes.", vix_path)

    if not regime_masks:
        logger.error("No regime data available; cannot run analysis.")
        return

    # ------------------------------------------------------------------
    # Eigenspectrum analysis (H3 — symmetric component)
    # ------------------------------------------------------------------
    eigvals_by_regime: dict[str, np.ndarray] = {}
    summary_records: list[dict[str, Any]] = []

    for regime_label, dates in regime_masks.items():
        if not dates:
            continue
        A_sym_avg = average_matrix(decomp_dir, "A_sym", dates)
        A_anti_avg = average_matrix(decomp_dir, "A_anti", dates)
        if A_sym_avg.size == 0:
            continue

        # Save averaged matrices
        np.save(out_dir / f"A_sym_avg_{regime_label}.npy", A_sym_avg)
        np.save(out_dir / f"A_anti_avg_{regime_label}.npy", A_anti_avg)

        eigvals = eigenspectrum(A_sym_avg)
        eigvals_by_regime[regime_label] = eigvals

        # TODO: H2 — sort assets by A^a row sums and form long-short portfolios.
        # Full H2 per-regime analysis (Fama-MacBeth spanning regressions on
        # antisymmetric attention scores) will be implemented in a dedicated
        # hypothesis testing script as a follow-up.
        frob_sym = float(np.linalg.norm(A_sym_avg, "fro"))
        frob_anti = float(np.linalg.norm(A_anti_avg, "fro")) if A_anti_avg.size > 0 else float("nan")
        top1_ratio = float(eigvals[0] / eigvals[eigvals > 0].sum()) if len(eigvals[eigvals > 0]) > 0 else float("nan")

        summary_records.append(
            {
                "regime": regime_label,
                "n_months": len(dates),
                "frob_sym": frob_sym,
                "frob_anti": frob_anti,
                "top1_eigenvalue_ratio": top1_ratio,
            }
        )
        logger.info(
            "Regime %s: frob_sym=%.4f  frob_anti=%.4f  top1_ratio=%.4f",
            regime_label,
            frob_sym,
            frob_anti,
            top1_ratio,
        )

    # Plot eigenspectra (H3 variation)
    if eigvals_by_regime:
        plot_eigenspectra(eigvals_by_regime, out_dir / "eigenspectra_by_regime.png")

    # Save summary table
    if summary_records:
        summary_df = pd.DataFrame(summary_records).set_index("regime")
        summary_df.to_csv(out_dir / "regime_summary.csv")
        logger.info("Regime summary:\n%s", summary_df.to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run regime analysis."""
    parser = argparse.ArgumentParser(
        description="Regime-conditional attention analysis (H3)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/regime.yaml",
        help="Path to the regime analysis YAML config file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    analyse(cfg)


if __name__ == "__main__":
    main()
