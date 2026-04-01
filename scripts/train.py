"""Training script for the single-head, single-layer linear attention transformer.

Usage::

    python scripts/train.py --config config/train.yaml

The model implements the *linear* attention variant from Kelly, Malamud,
Ramirez & Zhou (NBER WP 33351, 2025):

    A_t = Q_t K_t^T   where Q_t = X_t W_Q^T,  K_t = X_t W_K^T
                                          (Kelly et al. 2025, linear case)

where W_Q, W_K ∈ R^{embed_dim×D} are learned weight matrices and
X_t ∈ R^{N_t×D} is the cross-section of firm characteristics at month t.
No softmax is applied (linear attention).
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils import load_config, load_data, set_seed, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MonthlyPanelDataset(Dataset):
    """PyTorch Dataset wrapping one cross-section per month.

    Each item is a tuple ``(X, y)`` where ``X ∈ R^{N_t × D}`` is the
    characteristics matrix and ``y ∈ R^{N_t}`` contains excess returns.

    Args:
        df: Panel DataFrame produced by :func:`utils.load_data`.
        config: Full config dict (uses ``data`` sub-dict).
    """

    def __init__(self, df: pd.DataFrame, config: dict[str, Any]) -> None:
        data_cfg = config["data"]
        char_cols: list[str] = data_cfg["characteristics"]
        ret_col: str = data_cfg["return_col"]
        date_col: str = data_cfg["date_col"]

        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _, grp in df.groupby(date_col, sort=True):
            X = torch.tensor(grp[char_cols].values, dtype=torch.float32)
            y = torch.tensor(grp[ret_col].values, dtype=torch.float32)
            self.samples.append((X, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate a list of (X, y) pairs into padded batch tensors.

    Because the cross-sectional size N_t varies across months we simply
    return the single-month tensors as-is (batch_size = 1 in the DataLoader).

    Args:
        batch: List containing exactly one ``(X, y)`` tuple.

    Returns:
        Tuple of ``(X, y)`` tensors for the single month.
    """
    X, y = batch[0]
    return X, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LinearAttentionTransformer(nn.Module):
    """Single-head, single-layer linear attention transformer.

    Implements the model from Kelly et al. (2025, NBER WP 33351).
    The attention matrix is computed *without* softmax (linear attention):

        A_t = X_t W_Q^T W_K X_t^T          (Kelly et al. 2025, linear case)

    The decomposition into symmetric and antisymmetric parts is:

        A^s_t = (A_t + A_t^T) / 2          (symmetric  — factor structure)
        A^a_t = (A_t - A_t^T) / 2          (antisymmetric — mispricing)

    The value projection and output head follow the standard single-layer
    transformer architecture.

    Args:
        d_in: Number of input characteristics (P).
        embed_dim: Internal embedding dimension (D).
        dropout: Dropout probability applied after attention (set 0 in paper).
    """

    def __init__(self, d_in: int, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        # Linear projections for Q, K, V — no bias to keep the model
        # interpretable (Kelly et al. 2025 prescribe weight-only attention)
        self.W_Q = nn.Linear(d_in, embed_dim, bias=False)
        self.W_K = nn.Linear(d_in, embed_dim, bias=False)
        self.W_V = nn.Linear(d_in, embed_dim, bias=False)
        self.head = nn.Linear(embed_dim, 1, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute predicted excess returns for a cross-section.

        Args:
            X: Characteristics matrix of shape ``(N_t, d_in)``.

        Returns:
            Predicted excess returns of shape ``(N_t,)``.
        """
        # Q ∈ R^{N_t × D},  K ∈ R^{N_t × D},  V ∈ R^{N_t × D}
        Q = self.W_Q(X)  # (N_t, D)
        K = self.W_K(X)  # (N_t, D)
        V = self.W_V(X)  # (N_t, D)

        # Linear attention: A_t = Q K^T  (Kelly et al. 2025, linear case)
        # Shape: (N_t, N_t) — no softmax applied
        A = Q @ K.T  # Eq. (linear case) Kelly et al. 2025

        A = self.dropout(A)

        # Context vectors: Z = A V,  shape (N_t, D)
        Z = A @ V  # (N_t, D)

        # Output head: predicted returns, shape (N_t,)
        preds: torch.Tensor = self.head(Z).squeeze(-1)
        return preds

    def get_attention_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the learned Q and K weight matrices.

        Returns:
            Tuple ``(W_Q, W_K)`` each of shape ``(embed_dim, d_in)``.
        """
        return self.W_Q.weight.detach(), self.W_K.weight.detach()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict[str, Any]) -> None:
    """Train the linear attention transformer and save the checkpoint.

    Steps
    -----
    1. Load panel data and wrap in a :class:`MonthlyPanelDataset`.
    2. Instantiate :class:`LinearAttentionTransformer`.
    3. Minimise MSE loss on realized excess returns using Adam.
    4. Apply early stopping on validation loss.
    5. Save model checkpoint and Q, K weight matrices.

    Args:
        config: Full config dict loaded from ``config/train.yaml``.
    """
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    df = load_data(config)

    date_col: str = config["data"]["date_col"]
    dates = sorted(df[date_col].unique())
    n_dates = len(dates)

    # Simple chronological train / val split (80 / 20)
    split_idx = int(n_dates * 0.8)
    train_dates = set(dates[:split_idx])
    val_dates = set(dates[split_idx:])

    train_df = df[df[date_col].isin(train_dates)]
    val_df = df[df[date_col].isin(val_dates)]

    train_dataset = MonthlyPanelDataset(train_df, config)
    val_dataset = MonthlyPanelDataset(val_df, config)

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    d_in = len(config["data"]["characteristics"])
    embed_dim: int = config["model"]["embed_dim"]
    dropout: float = config["model"]["dropout"]
    model = LinearAttentionTransformer(d_in, embed_dim, dropout).to(device)
    logger.info("Model: %s", model)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    lr: float = config["training"]["learning_rate"]
    wd: float = config["training"]["weight_decay"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------------------------------
    epochs: int = config["training"]["epochs"]
    patience: int = config["training"]["early_stopping_patience"]
    best_val_loss = float("inf")
    epochs_no_improve = 0

    out_dir = Path(config["output"]["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config["output"]["checkpoint"])

    for epoch in range(1, epochs + 1):
        # -- train --
        model.train()
        train_losses: list[float] = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # -- validate --
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                val_losses.append(criterion(preds, y).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        logger.info(
            "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f",
            epoch,
            epochs,
            train_loss,
            val_loss,
        )

        # -- early stopping --
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("  ↳ New best val_loss=%.6f — checkpoint saved.", best_val_loss)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    # ------------------------------------------------------------------
    # Save Q and K weight matrices separately for decompose.py
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    W_Q, W_K = model.get_attention_weights()
    np.save(out_dir / "W_Q.npy", W_Q.cpu().numpy())
    np.save(out_dir / "W_K.npy", W_K.cpu().numpy())
    logger.info("Saved W_Q and W_K to %s", out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and launch training."""
    parser = argparse.ArgumentParser(
        description="Train the linear attention transformer (Kelly et al. 2025)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train.yaml",
        help="Path to the training YAML config file.",
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

    train(cfg)


if __name__ == "__main__":
    main()
