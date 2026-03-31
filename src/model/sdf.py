"""Stochastic Discount Factor (SDF) model for asset pricing.

This module provides the ``SDFModel`` class, which takes attention-derived
weights and computes the stochastic discount factor used to evaluate
cross-sectional pricing errors in the thesis.

In the Kelly et al. (2025) framework the SDF is written as:
    M_t = 1 - w_t^T r_t
where ``w_t`` are the model-implied portfolio weights and ``r_t`` are excess
returns.
"""

import torch
import torch.nn as nn
from typing import Tuple


class SDFModel(nn.Module):
    """Computes the stochastic discount factor from attention-derived features.

    Applies a learned linear head to per-asset attention-weighted features
    (e.g. ``X`` re-weighted by the rows of the attention matrix ``A``) to
    produce SDF portfolio weights.  The typical calling pattern is:

    .. code-block:: python

        attn   = LinearAttention(d_model=D)
        A      = attn(X)                          # (N, N)
        # Build attention-weighted features: A @ X gives each asset a
        # weighted mixture of other assets' characteristics.
        attn_feats = A @ X                         # (N, D)
        sdf    = SDFModel(d_model=D)
        w      = sdf(X, attn_feats)               # (N,)

    Args:
        d_model: Dimensionality of the attention-derived features (D).

    Attributes:
        head: Linear projection from d_model features to a scalar SDF weight
              per asset.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # Linear SDF head: maps per-asset feature vector to scalar weight
        self.head = nn.Linear(d_model, 1, bias=False)

    def forward(
        self,
        X: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SDF portfolio weight vector.

        Args:
            X: Firm-characteristic matrix of shape (N, D).
            attention_weights: Pre-computed attention-derived features of
                shape (N, D) (e.g., row sums of A multiplied by X).

        Returns:
            Portfolio weight vector ``w_t`` of shape (N,).
        """
        # Project attention-weighted features to scalar weights per asset
        w = self.head(attention_weights).squeeze(-1)  # (N,)
        # Normalise so weights sum to zero (long-short portfolio)
        w = w - w.mean()
        return w

    def pricing_error(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cross-sectional pricing error.

        The SDF moment condition is:
            E[M_t * R_{i,t}] = 0  for all assets i

        Args:
            weights: SDF portfolio weight vector of shape (N,).
            returns: Excess return vector of shape (N,).

        Returns:
            Tuple ``(sdf_value, pricing_error)`` where
            - ``sdf_value`` is a scalar ``M_t = 1 - w^T r``
            - ``pricing_error`` is the cross-sectional mean absolute error
        """
        # M_t = 1 - w_t^T r_t  (SDF moment condition)
        sdf_value = 1.0 - (weights * returns).sum()
        # Cross-sectional pricing error: |E[M_t * R_{i,t}]|
        pricing_error = (sdf_value * returns).abs().mean()
        return sdf_value, pricing_error
