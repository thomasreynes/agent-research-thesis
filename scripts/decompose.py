"""Attention decomposition script for the linear attention transformer.

Usage::

    python scripts/decompose.py --config config/decompose.yaml

For each month t in the sample this script computes the linear attention
matrix and decomposes it into symmetric and antisymmetric components
following Kelly, Malamud, Ramirez & Zhou (NBER WP 33351, 2025):

    A_t   = Q X_t^T X_t K^T      (Kelly et al. 2025, linear case)
    A^s_t = (A_t + A_t^T) / 2    (symmetric  — H1: factor structure)
    A^a_t = (A_t - A_t^T) / 2    (antisymmetric — H2: mispricing signals)

Decomposed matrices are saved as ``.npy`` files under
``outputs/decomposition/``, together with summary statistics.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from utils import load_config, load_data, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decomposition helpers
# ---------------------------------------------------------------------------

def compute_attention(
    X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray
) -> np.ndarray:
    """Compute the linear attention matrix for one cross-section.

    Implements the linear attention (no softmax) from Kelly et al. (2025):

        A_t = (X_t W_Q^T)(X_t W_K^T)^T = Q_t K_t^T    (Kelly et al. 2025)

    Args:
        X: Characteristics matrix of shape ``(N_t, D)``.
        W_Q: Query weight matrix of shape ``(embed_dim, D)``.
        W_K: Key weight matrix of shape ``(embed_dim, D)``.

    Returns:
        Attention matrix ``A_t`` of shape ``(N_t, N_t)``.
    """
    # Q_t = X_t W_Q^T,  K_t = X_t W_K^T — Kelly et al. 2025, linear case
    Q_t = X @ W_Q.T  # (N_t, embed_dim)
    K_t = X @ W_K.T  # (N_t, embed_dim)
    A_t: np.ndarray = Q_t @ K_t.T  # Eq. (linear case) Kelly et al. 2025
    return A_t


def decompose_attention(
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose attention matrix into symmetric and antisymmetric parts.

    Following Kelly et al. (2025):

        A^s = (A + A^T) / 2     (symmetric  — factor structure, H1)
        A^a = (A - A^T) / 2     (antisymmetric — mispricing, H2)

    Args:
        A: Attention matrix of shape ``(N_t, N_t)``.

    Returns:
        Tuple ``(A_sym, A_anti)`` each of shape ``(N_t, N_t)``.
    """
    A_sym: np.ndarray = (A + A.T) / 2.0   # Kelly et al. 2025
    A_anti: np.ndarray = (A - A.T) / 2.0  # Kelly et al. 2025
    return A_sym, A_anti


def summarise(A_sym: np.ndarray, A_anti: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for one month's decomposed matrices.

    Args:
        A_sym: Symmetric attention matrix of shape ``(N_t, N_t)``.
        A_anti: Antisymmetric attention matrix of shape ``(N_t, N_t)``.

    Returns:
        Dict with Frobenius norms, rank of ``A_sym``, and ratio of the
        top-1 eigenvalue of ``A_sym`` to the total trace.
    """
    frob_sym = float(np.linalg.norm(A_sym, "fro"))
    frob_anti = float(np.linalg.norm(A_anti, "fro"))

    # TODO: use full eigendecomposition for H1 analysis in a dedicated
    # hypothesis testing script (to be added as a follow-up).
    eigvals = np.linalg.eigvalsh(A_sym)
    eigvals_pos = eigvals[eigvals > 0]
    rank_sym = int((eigvals > 1e-8 * eigvals.max()).sum()) if len(eigvals) > 0 else 0
    trace = float(eigvals_pos.sum()) if len(eigvals_pos) > 0 else 1.0
    top1_ratio = float(eigvals_pos[-1] / trace) if len(eigvals_pos) > 0 else float("nan")

    return {
        "frob_sym": frob_sym,
        "frob_anti": frob_anti,
        "rank_sym": rank_sym,
        "top1_eigenvalue_ratio": top1_ratio,
    }


# ---------------------------------------------------------------------------
# Main decomposition routine
# ---------------------------------------------------------------------------

def decompose(config: dict[str, Any]) -> None:
    """Run the full decomposition pipeline over all months in the sample.

    Steps
    -----
    1. Load W_Q and W_K saved by ``train.py``.
    2. Load processed panel data.
    3. For each month t: compute A_t, decompose into A^s_t and A^a_t.
    4. Save per-month ``.npy`` files and an aggregate summary CSV.

    Args:
        config: Full config dict loaded from ``config/decompose.yaml``.
    """
    checkpoint_path = Path(config["model"]["checkpoint"])
    models_dir = checkpoint_path.parent

    W_Q = np.load(models_dir / "W_Q.npy")
    W_K = np.load(models_dir / "W_K.npy")
    logger.info("Loaded W_Q %s and W_K %s", W_Q.shape, W_K.shape)

    df = load_data(config)
    date_col: str = config["data"]["date_col"]
    char_cols: list[str] = config["data"]["characteristics"]

    out_dir = Path(config["output"]["decomposition_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for date, grp in df.groupby(date_col, sort=True):
        X = grp[char_cols].values.astype(np.float32)  # (N_t, D)

        # A_t = Q_t K_t^T  — Kelly et al. 2025, linear case
        A_t = compute_attention(X, W_Q, W_K)

        # Symmetric / antisymmetric decomposition — Kelly et al. 2025
        A_sym, A_anti = decompose_attention(A_t)

        # Persist per-month matrices
        date_str = pd.Timestamp(date).strftime("%Y-%m")
        np.save(out_dir / f"A_sym_{date_str}.npy", A_sym)
        np.save(out_dir / f"A_anti_{date_str}.npy", A_anti)

        stats = summarise(A_sym, A_anti)
        stats["date"] = date_str
        stats["n_assets"] = int(X.shape[0])
        records.append(stats)

    summary_df = pd.DataFrame(records).set_index("date")
    summary_path = out_dir / "decomposition_summary.csv"
    summary_df.to_csv(summary_path)
    logger.info("Saved decomposition summary to %s", summary_path)
    logger.info(summary_df.describe())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run decomposition."""
    parser = argparse.ArgumentParser(
        description="Decompose linear attention into symmetric/antisymmetric parts."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/decompose.yaml",
        help="Path to the decomposition YAML config file.",
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

    decompose(cfg)


if __name__ == "__main__":
    main()
