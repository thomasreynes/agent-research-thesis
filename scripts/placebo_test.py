"""Placebo and robustness tests for the linear attention decomposition (H3).

Usage::

    python scripts/placebo_test.py --config config/placebo.yaml

Tests **H3** (null hypothesis variant): When firm characteristics are
randomly permuted across assets within each month, the attention
decomposition should lose its economic meaning — symmetric norms,
eigenvalue concentration, and antisymmetric norms should collapse toward
values indistinguishable from a random baseline.

Procedure
---------
For each of ``n_iterations`` placebo trials:

1. Shuffle the characteristics matrix cross-sectionally within each month
   (i.e., randomly reassign characteristic vectors across firms but keep the
   time structure intact).
2. Recompute the attention matrices from the shuffled data using the saved
   Q and K weight matrices.
3. Decompose into A^s and A^a.
4. Record summary statistics (Frobenius norms, eigenvalue concentration).

The resulting distribution is compared to the *actual* statistics from
``decompose.py`` to derive empirical p-values.

References: Kelly, Malamud, Ramirez & Zhou (NBER WP 33351, 2025), H3.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decompose import compute_attention, decompose_attention, summarise
from utils import load_config, load_data, set_seed, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Placebo helpers
# ---------------------------------------------------------------------------

def permute_characteristics(
    X: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle rows of the characteristics matrix within one month.

    Randomly permutes firm indices so that each row (firm characteristic
    vector) is reassigned to a different firm.  The cross-sectional
    distribution of each characteristic is preserved; only the firm-level
    assignments are randomised.

    Args:
        X: Characteristics matrix of shape ``(N_t, D)``.
        rng: NumPy random Generator (seeded from config).

    Returns:
        Permuted characteristics matrix of shape ``(N_t, D)``.
    """
    perm = rng.permutation(X.shape[0])
    return X[perm]


def run_placebo_iteration(
    df: pd.DataFrame,
    char_cols: list[str],
    date_col: str,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Run one placebo trial over all months and return aggregate statistics.

    For each month, the characteristics are shuffled and A_t is recomputed
    from the shuffled data.  Statistics are averaged across months.

    Args:
        df: Panel DataFrame produced by :func:`utils.load_data`.
        char_cols: List of characteristic column names.
        date_col: Name of the date column in ``df``.
        W_Q: Query weight matrix ``(embed_dim, D)``.
        W_K: Key weight matrix ``(embed_dim, D)``.
        rng: Seeded random Generator for reproducibility.

    Returns:
        Dict with mean ``frob_sym``, ``frob_anti``, ``rank_sym``, and
        ``top1_eigenvalue_ratio`` across all months.
    """
    records: list[dict[str, float]] = []
    for _, grp in df.groupby(date_col, sort=True):
        X = grp[char_cols].values.astype(np.float32)  # (N_t, D)
        X_perm = permute_characteristics(X, rng)

        # A_t from permuted characteristics — Kelly et al. 2025, linear case
        A_t = compute_attention(X_perm, W_Q, W_K)

        # Decomposition — Kelly et al. 2025
        A_sym, A_anti = decompose_attention(A_t)
        records.append(summarise(A_sym, A_anti))

    mean_stats: dict[str, float] = {
        key: float(np.mean([r[key] for r in records]))
        for key in records[0]
    }
    return mean_stats


# ---------------------------------------------------------------------------
# Main placebo routine
# ---------------------------------------------------------------------------

def run_placebo_tests(config: dict[str, Any]) -> None:
    """Run all placebo iterations and report empirical p-values.

    Steps
    -----
    1. Load actual decomposition summary (produced by ``decompose.py``).
    2. Load W_Q and W_K weight matrices.
    3. For ``n_iterations`` trials: permute characteristics, recompute
       attention, record statistics.
    4. Compare placebo distribution to actual statistics.
    5. Save results to ``outputs/placebo/``.

    Args:
        config: Full config dict loaded from ``config/placebo.yaml``.
    """
    seed: int = config["placebo"]["seed"]
    n_iter: int = config["placebo"]["n_iterations"]
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # Load actual summary statistics produced by decompose.py
    checkpoint_path = Path(config["model"]["checkpoint"])
    models_dir = checkpoint_path.parent
    decomp_summary_path = (
        models_dir.parent / "decomposition" / "decomposition_summary.csv"
    )

    actual_stats: dict[str, float] | None = None
    if decomp_summary_path.exists():
        summary_df = pd.read_csv(decomp_summary_path, index_col="date")
        actual_stats = summary_df.mean().to_dict()
        logger.info("Actual stats (mean across months): %s", actual_stats)
    else:
        logger.warning(
            "Decomposition summary not found at %s. "
            "Run decompose.py first to enable p-value computation.",
            decomp_summary_path,
        )

    # Load model weights
    W_Q = np.load(models_dir / "W_Q.npy")
    W_K = np.load(models_dir / "W_K.npy")
    logger.info("Loaded W_Q %s and W_K %s", W_Q.shape, W_K.shape)

    # Load panel data
    df = load_data(config)
    char_cols: list[str] = config["data"]["characteristics"]
    date_col: str = config["data"]["date_col"]

    out_dir = Path(config["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Placebo iterations
    # ------------------------------------------------------------------
    placebo_records: list[dict[str, Any]] = []
    for i in range(n_iter):
        if (i + 1) % max(1, n_iter // 10) == 0:
            logger.info("Placebo iteration %d / %d", i + 1, n_iter)

        stats = run_placebo_iteration(df, char_cols, date_col, W_Q, W_K, rng)
        stats["iteration"] = i
        placebo_records.append(stats)

    placebo_df = pd.DataFrame(placebo_records).set_index("iteration")
    placebo_path = out_dir / "placebo_statistics.csv"
    placebo_df.to_csv(placebo_path)
    logger.info("Saved placebo statistics to %s", placebo_path)

    # ------------------------------------------------------------------
    # Empirical p-values
    # ------------------------------------------------------------------
    if actual_stats is not None:
        stat_keys = [k for k in actual_stats if k in placebo_df.columns]
        pval_records: list[dict[str, Any]] = []
        for key in stat_keys:
            actual_val = actual_stats[key]
            placebo_vals = placebo_df[key].values
            # One-sided p-value: fraction of placebo stats >= actual
            pval = float((placebo_vals >= actual_val).mean())
            pval_records.append(
                {
                    "statistic": key,
                    "actual": actual_val,
                    "placebo_mean": float(placebo_vals.mean()),
                    "placebo_std": float(placebo_vals.std()),
                    "p_value": pval,
                }
            )
            logger.info(
                "%s: actual=%.4f  placebo_mean=%.4f  p=%.4f",
                key,
                actual_val,
                placebo_vals.mean(),
                pval,
            )

        pval_df = pd.DataFrame(pval_records).set_index("statistic")
        pval_path = out_dir / "placebo_pvalues.csv"
        pval_df.to_csv(pval_path)
        logger.info("Saved p-values to %s", pval_path)
    else:
        logger.info(
            "Skipping p-value computation (actual stats not available)."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run placebo tests."""
    parser = argparse.ArgumentParser(
        description="Placebo / robustness tests for the attention decomposition (H3)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/placebo.yaml",
        help="Path to the placebo YAML config file.",
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

    run_placebo_tests(cfg)


if __name__ == "__main__":
    main()
