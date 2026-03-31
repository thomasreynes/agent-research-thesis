"""Linear attention mechanism for asset pricing.

Implements the linear attention from Kelly, Kuznetsov, Malamud & Xu (NBER WP
33351, 2025) and the symmetric / antisymmetric decomposition used throughout
the thesis.

Mathematical foundations
------------------------
Linear attention:
    A_t = X_t W X_t^T                      (Kelly et al. 2025, eq. linear case)

Decomposition (thesis methodology):
    A^s_t = 1/2 (A_t + A_t^T)  — symmetric:      mutual similarity
    A^a_t = 1/2 (A_t - A_t^T)  — antisymmetric:  directional information flow
"""

import torch
import torch.nn as nn
from typing import Tuple


class LinearAttention(nn.Module):
    """Bilinear attention module: A_t = X_t W X_t^T.

    Computes the linear (non-softmax) attention matrix described in Kelly et
    al. (2025).  The weight matrix ``W`` is the single learnable parameter.

    Args:
        d_model: Dimensionality of the firm-characteristic input vectors (D).

    Attributes:
        W: Learnable weight matrix of shape (d_model, d_model).

    Example:
        >>> attn = LinearAttention(d_model=8)
        >>> X = torch.randn(100, 8)   # 100 firms, D=8 characteristics
        >>> A = attn(X)               # shape (100, 100)
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # W in R^{D x D} — learnable weight matrix (Kelly et al. 2025)
        self.W = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.xavier_uniform_(self.W)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute A_t = X_t W X_t^T.

        Args:
            X: Firm-characteristic matrix of shape (N, D) where N is the
               number of assets at time t and D is the number of
               characteristics.

        Returns:
            Attention matrix A of shape (N, N).
        """
        # A_{ij} = x_i W x_j^T  =>  A = X W X^T   (Kelly et al. 2025)
        return X @ self.W @ X.T


def decompose_attention(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split attention matrix into symmetric and antisymmetric components.

    Implements the thesis decomposition:
        A^s = (A + A^T) / 2   — captures mutual similarity (H1)
        A^a = (A - A^T) / 2   — captures directional lead-lag (H2)

    Args:
        A: Attention matrix of shape (N, N).

    Returns:
        Tuple ``(A_s, A_a)`` where
        - ``A_s`` is the symmetric component  ``(A + A^T) / 2``
        - ``A_a`` is the antisymmetric component ``(A - A^T) / 2``
    """
    # Symmetric component: A^s = 1/2 (A + A^T)  (thesis methodology)
    A_s = 0.5 * (A + A.T)
    # Antisymmetric component: A^a = 1/2 (A - A^T)  (thesis methodology)
    A_a = 0.5 * (A - A.T)
    return A_s, A_a


def compute_sdf_weights(
    A_s: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """Compute SDF portfolio weights from the symmetric attention component.

    Placeholder for the full SDF computation described in the thesis.  The
    intuition is that ``A_s`` encodes cross-sectional similarity, which is
    aggregated to form a diversified pricing kernel.

    Args:
        A_s: Symmetric attention matrix of shape (N, N).
        returns: Cross-sectional return vector of shape (N,).

    Returns:
        SDF weight vector of shape (N,).  Currently returns row-wise softmax
        of ``A_s`` aggregated over assets — replace with trained SDF head.

    Todo:
        Integrate with :class:`src.model.sdf.SDFModel` once trained.
    """
    # TODO: Replace with learned SDF head output
    # Aggregate similarity scores: sum each row of A^s -> shape (N,)
    agg = A_s.sum(dim=1)
    # Normalise to unit sum for interpretability
    weights = agg / (agg.abs().sum() + 1e-8)
    return weights
