"""Full linear attention transformer for asset pricing.

Implements the complete single-head, single-layer linear attention transformer
described in Kelly, Kuznetsov, Malamud & Xu (NBER WP 33351, 2025).

Architecture
------------
Given a cross-section of N_t firms at time t with P characteristics each::

    X_t ∈ R^{N_t × P}        — firm characteristics matrix

1. Project to query / key / value embeddings (embed_dim = D):

   Q_t = X_t W_Q^T  ∈ R^{N_t × D}
   K_t = X_t W_K^T  ∈ R^{N_t × D}
   V_t = X_t W_V^T  ∈ R^{N_t × D}

2. Compute linear attention (NO softmax) — Kelly et al. 2025, linear case:

   A_t = Q_t K_t^T  ∈ R^{N_t × N_t}
       ≡ X_t (W_Q^T W_K) X_t^T

3. Decompose attention — thesis methodology (H1 / H2):

   A^s_t = (A_t + A_t^T) / 2   symmetric   — factor structure (H1)
   A^a_t = (A_t - A_t^T) / 2   antisymm.   — directional lead-lag (H2)

4. Compute context vectors:

   Z_t = A_t V_t  ∈ R^{N_t × D}

5. Output head — predicted excess returns:

   ŷ_t = Z_t w_o  ∈ R^{N_t}

References
----------
Kelly, Kuznetsov, Malamud & Xu (NBER WP 33351, 2025):
    "Artificial Intelligence Asset Pricing Models"
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.model.attention import decompose_attention


class LinearAttentionTransformer(nn.Module):
    """Single-head, single-layer linear attention transformer for asset pricing.

    Implements the model from Kelly, Kuznetsov, Malamud & Xu (NBER WP 33351,
    2025).  The attention matrix is computed WITHOUT softmax (linear attention):

        A_t = Q_t K_t^T = X_t (W_Q^T W_K) X_t^T      (Kelly et al. 2025)

    The value projection and linear output head follow the standard
    single-layer transformer architecture.

    The attention matrix decomposes into:

        A^s_t = (A_t + A_t^T) / 2  — symmetric,     captures factor structure (H1)
        A^a_t = (A_t - A_t^T) / 2  — antisymmetric, captures directional flows (H2)

    These components are accessible via :meth:`forward_with_components` and
    are used in the H1 / H2 / H3 hypothesis tests.

    Args:
        d_in: Number of input firm characteristics (P).
        embed_dim: Internal embedding dimension (D).
        dropout: Dropout probability applied to the attention weights.
            Set to 0.0 in the paper for full interpretability.

    Attributes:
        W_Q: Query projection, shape (embed_dim, d_in).  No bias.
        W_K: Key projection, shape (embed_dim, d_in).  No bias.
        W_V: Value projection, shape (embed_dim, d_in).  No bias.
        head: Linear output head mapping embed_dim → 1 scalar per asset.
        dropout: Dropout layer (identity when dropout=0.0).

    Example:
        >>> model = LinearAttentionTransformer(d_in=8, embed_dim=32)
        >>> X = torch.randn(200, 8)   # 200 firms, 8 characteristics
        >>> preds = model(X)          # shape (200,) — predicted excess returns
        >>> preds, A, A_sym, A_anti = model.forward_with_components(X)
    """

    def __init__(self, d_in: int, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        # No bias: consistent with the weight-only attention in Kelly et al. 2025
        self.W_Q = nn.Linear(d_in, embed_dim, bias=False)
        self.W_K = nn.Linear(d_in, embed_dim, bias=False)
        self.W_V = nn.Linear(d_in, embed_dim, bias=False)
        # Output head: maps per-asset context vector to a scalar return prediction
        self.head = nn.Linear(embed_dim, 1, bias=True)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Core attention computation
    # ------------------------------------------------------------------

    def compute_attention(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the linear attention matrix A_t = Q_t K_t^T.

        Equivalent to A_t = X_t (W_Q^T W_K) X_t^T   (Kelly et al. 2025).
        No softmax is applied — this is the defining property of the
        linear transformer in the paper.

        Args:
            X: Firm-characteristic matrix of shape (N_t, d_in).

        Returns:
            Attention matrix A_t of shape (N_t, N_t).
        """
        Q = self.W_Q(X)  # (N_t, D)  Q_t = X_t W_Q^T
        K = self.W_K(X)  # (N_t, D)  K_t = X_t W_K^T
        # Linear attention: A_t = Q_t K_t^T  (Kelly et al. 2025, linear case)
        return Q @ K.T  # (N_t, N_t)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute predicted excess returns for a cross-section at time t.

        Args:
            X: Firm-characteristic matrix of shape (N_t, d_in).

        Returns:
            Predicted excess returns of shape (N_t,).
        """
        # Step 1: Linear attention  A_t = Q_t K_t^T  (Kelly et al. 2025)
        A = self.compute_attention(X)  # (N_t, N_t)
        A = self.dropout(A)

        # Step 2: Value projection  V_t = X_t W_V^T
        V = self.W_V(X)  # (N_t, D)

        # Step 3: Context vectors  Z_t = A_t V_t
        Z = A @ V  # (N_t, D)

        # Step 4: Output head  ŷ_t = Z_t w_o  (Kelly et al. 2025)
        preds: torch.Tensor = self.head(Z).squeeze(-1)  # (N_t,)
        return preds

    def forward_with_components(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning predictions and full attention components.

        Useful for hypothesis testing (H1, H2, H3) where attention
        decomposition is required alongside return predictions.

        Args:
            X: Firm-characteristic matrix of shape (N_t, d_in).

        Returns:
            Tuple ``(preds, A, A_sym, A_anti)`` where:

            - ``preds``:  predicted excess returns, shape (N_t,).
            - ``A``:      raw linear attention matrix, shape (N_t, N_t).
            - ``A_sym``:  symmetric component (A + A^T) / 2, shape (N_t, N_t).
              Captures economic similarity — used in H1 regression.
            - ``A_anti``: antisymmetric component (A - A^T) / 2, shape (N_t, N_t).
              Captures directional lead-lag flows — used in H2 regression.
        """
        # Linear attention  A_t = Q_t K_t^T  (Kelly et al. 2025)
        A = self.compute_attention(X)  # (N_t, N_t)

        # Symmetric / antisymmetric decomposition — thesis methodology (H1, H2)
        A_sym, A_anti = decompose_attention(A)  # each (N_t, N_t)

        A_drop = self.dropout(A)
        V = self.W_V(X)   # (N_t, D)
        Z = A_drop @ V    # (N_t, D)   Z_t = A_t V_t
        preds: torch.Tensor = self.head(Z).squeeze(-1)  # (N_t,)
        return preds, A, A_sym, A_anti

    # ------------------------------------------------------------------
    # Weight accessors for interpretability analysis
    # ------------------------------------------------------------------

    def get_combined_weight_matrix(self) -> torch.Tensor:
        """Return the effective combined weight matrix W_eff = W_Q^T W_K.

        This is the single D_in × D_in matrix such that:

            A_t = X_t W_eff X_t^T      (Kelly et al. 2025, single-W form)

        Useful for direct structural analysis of what the model learned.
        A positive-definite W_eff encodes symmetric similarity; a
        skew-symmetric component captures directional lead-lag flows.

        Returns:
            Effective weight matrix of shape (d_in, d_in).  Detached from
            the computation graph.
        """
        # W_eff = W_Q^T W_K  (Kelly et al. 2025, linear case)
        W_eff: torch.Tensor = self.W_Q.weight.T @ self.W_K.weight  # (d_in, d_in)
        return W_eff.detach()

    def get_projection_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the Q and K weight matrices for external analysis.

        Returns:
            Tuple ``(W_Q, W_K)`` each of shape ``(embed_dim, d_in)``.
            Both tensors are detached from the computation graph.
        """
        return self.W_Q.weight.detach(), self.W_K.weight.detach()
