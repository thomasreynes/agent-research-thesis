"""Exact linear transformer model from Kelly, Kuznetsov, Malamud & Xu (2025).

Implements the complete single-layer, single-head linear attention transformer
described in NBER WP 33351, Section 3.  The three-step forward pass is:

    Step 1 — Bilinear attention:
        A_t = X_t W X_t^T                  (Kelly et al. 2025, linear case)
        where W ∈ R^{D × D} is the sole learned interaction matrix.

    Step 2 — Context aggregation:
        Z_t = A_t X_t                       (attention-weighted characteristics)

    Step 3 — SDF output:
        g_t = Z_t b                         (b ∈ R^D is the output projection)

The symmetric / antisymmetric decomposition is available via
:meth:`LinearTransformerModel.decompose_forward` for hypothesis testing (H1–H3):

    A^s_t = (A_t + A_t^T) / 2             — mutual similarity   (H1)
    A^a_t = (A_t - A_t^T) / 2             — directional lead-lag (H2)

W may optionally be parameterised in *factored* form as W = W_Q^T W_K
(``w_rank < d_model``), which matches the standard query / key decomposition
used in transformer literature while keeping the bilinear interpretation intact.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.model.attention import LinearAttention, decompose_attention


class LinearTransformerModel(nn.Module):
    """Exact linear attention transformer from Kelly et al. (NBER WP 33351, 2025).

    Implements the complete three-step algorithm:

        A_t = X_t W X_t^T                  (Kelly et al. 2025, Eq. linear case)
        Z_t = A_t X_t                      (context aggregation)
        g_t = Z_t b                        (SDF output head)

    Args:
        d_model: Number of input characteristics (D).
        w_rank: If ``None`` (default), W is a full D×D matrix.
                If an integer ``r < d_model``, W is parameterised as
                ``W = W_Q^T W_K`` with W_Q, W_K ∈ R^{r × D}, reducing
                parameter count while preserving the bilinear structure.

    Attributes:
        attention: :class:`~src.model.attention.LinearAttention` module
                   holding the learnable weight matrix W (or W_Q, W_K).
        output_head: Linear projection ``b ∈ R^D`` mapping per-asset
                     context vectors Z_t to scalar predicted returns g_t.

    Example::

        >>> model = LinearTransformerModel(d_model=6)
        >>> X = torch.randn(120, 6)          # 120 firms × 6 characteristics
        >>> g = model(X)                     # predicted returns, shape (120,)
        >>> A_s, A_a, g = model.decompose_forward(X)
    """

    def __init__(
        self,
        d_model: int,
        w_rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Step 1: Bilinear attention  A_t = X_t W X_t^T   (Kelly et al. 2025)
        self.attention = LinearAttention(d_model=d_model, w_rank=w_rank)
        # Step 3: Output head  g_t = Z_t b   (Kelly et al. 2025)
        # No bias: forces the model to learn purely attention-driven returns
        self.output_head = nn.Linear(d_model, 1, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute predicted excess returns for one cross-section.

        Runs the three-step algorithm from Kelly et al. (2025):
            1. A_t  = X_t W X_t^T             (bilinear attention)
            2. Z_t  = A_t X_t                 (context aggregation)
            3. g_t  = Z_t b                   (linear output head)

        Args:
            X: Characteristics matrix of shape ``(N_t, D)`` where N_t is the
               number of assets in the cross-section and D is d_model.

        Returns:
            Predicted excess returns of shape ``(N_t,)``.
        """
        # Step 1: A_t = X_t W X_t^T  (Kelly et al. 2025, linear case)
        A: torch.Tensor = self.attention(X)  # (N_t, N_t)

        # Step 2: Z_t = A_t X_t  — attention-weighted characteristic context
        Z: torch.Tensor = A @ X  # (N_t, D)

        # Step 3: g_t = Z_t b  — scalar predicted return per asset
        g: torch.Tensor = self.output_head(Z).squeeze(-1)  # (N_t,)
        return g

    def decompose_forward(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with symmetric / antisymmetric attention decomposition.

        Computes the full attention matrix and decomposes it into:
            A^s_t = (A_t + A_t^T) / 2      — mutual similarity    (H1)
            A^a_t = (A_t - A_t^T) / 2      — directional lead-lag (H2)

        The predicted returns ``g_t`` are computed from the *full* A_t (not
        decomposed), consistent with the trained objective.

        Args:
            X: Characteristics matrix of shape ``(N_t, D)``.

        Returns:
            Tuple ``(A_s, A_a, g)`` where

            - ``A_s``: symmetric attention matrix, shape ``(N_t, N_t)``
            - ``A_a``: antisymmetric attention matrix, shape ``(N_t, N_t)``
            - ``g``:   predicted excess returns, shape ``(N_t,)``
        """
        # Step 1 (Kelly et al. 2025): full bilinear attention
        A: torch.Tensor = self.attention(X)  # (N_t, N_t)

        # Thesis decomposition: A^s and A^a  (H1 / H2 analysis)
        A_s, A_a = decompose_attention(A)

        # Steps 2 & 3: context + output using full A
        Z: torch.Tensor = A @ X  # (N_t, D)
        g: torch.Tensor = self.output_head(Z).squeeze(-1)  # (N_t,)
        return A_s, A_a, g

    def get_weight_matrix(self) -> torch.Tensor:
        """Return the learned interaction weight matrix W.

        If the model uses the factored parameterisation (``w_rank`` was set),
        the effective W is reconstructed as W_Q^T W_K.

        Returns:
            Weight matrix W of shape ``(D, D)``, detached from the computation
            graph, on CPU.
        """
        return self.attention.get_W().detach().cpu()
