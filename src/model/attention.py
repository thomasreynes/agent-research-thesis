"""Linear attention mechanism for asset pricing.

Implements the linear attention from Kelly, Kuznetsov, Malamud & Xu (NBER WP
33351, 2025) and the symmetric / antisymmetric decomposition used throughout
the thesis.

Mathematical foundations
------------------------
Linear attention (exact form):
    A_t = X_t W X_t^T                      (Kelly et al. 2025, eq. linear case)

Factored form (w_rank < d_model):
    W = W_Q^T W_K,  W_Q, W_K ∈ R^{r × D}  (standard Q/K decomposition)
    A_t = (X_t W_Q^T)(X_t W_K^T)^T        (equivalent bilinear attention)

Decomposition (thesis methodology):
    A^s_t = 1/2 (A_t + A_t^T)  — symmetric:      mutual similarity
    A^a_t = 1/2 (A_t - A_t^T)  — antisymmetric:  directional information flow
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class LinearAttention(nn.Module):
    """Bilinear attention module: A_t = X_t W X_t^T.

    Computes the linear (non-softmax) attention matrix described in Kelly et
    al. (2025).  Two parameterisations are supported:

    - **Full rank** (``w_rank=None``): W ∈ R^{D × D} is a single learnable
      matrix.  This is the exact form used in the paper.
    - **Factored** (``w_rank=r``): W is implicitly W = W_Q^T W_K where
      W_Q, W_K ∈ R^{r × D}.  The attention is computed efficiently as
      A = Q K^T = (X W_Q^T)(X W_K^T)^T, which is numerically equivalent
      to X (W_Q^T W_K) X^T but avoids materialising the D×D matrix when r ≪ D.

    Args:
        d_model: Dimensionality of the firm-characteristic input vectors (D).
        w_rank: Rank of the factored parameterisation.  If ``None`` (default),
                the full D×D weight matrix is used.

    Attributes:
        W: Learnable weight matrix of shape ``(d_model, d_model)`` when
           ``w_rank`` is ``None``.
        W_Q: Query factor ``(w_rank, d_model)`` used when ``w_rank`` is set.
        W_K: Key factor ``(w_rank, d_model)`` used when ``w_rank`` is set.

    Example::

        >>> attn = LinearAttention(d_model=8)
        >>> X = torch.randn(100, 8)   # 100 firms, D=8 characteristics
        >>> A = attn(X)               # shape (100, 100)

        >>> # Factored form — equivalent bilinear attention, fewer parameters
        >>> attn_r = LinearAttention(d_model=8, w_rank=4)
        >>> A_r = attn_r(X)           # shape (100, 100)
    """

    def __init__(self, d_model: int, w_rank: Optional[int] = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.w_rank = w_rank

        if w_rank is None:
            # Exact Kelly et al. (2025) form: single D×D weight matrix
            # W in R^{D x D} — learnable interaction matrix (Kelly et al. 2025)
            self.W: nn.Parameter = nn.Parameter(torch.empty(d_model, d_model))
            nn.init.xavier_uniform_(self.W)
            self.W_Q: Optional[nn.Linear] = None
            self.W_K: Optional[nn.Linear] = None
        else:
            # Factored form: W = W_Q^T W_K,  W_Q, W_K in R^{r x D}
            # A_t = Q_t K_t^T = (X W_Q^T)(X W_K^T)^T  (Kelly et al. 2025)
            # Register W as a buffer (not a parameter) to keep the module API
            # consistent regardless of which branch is active.
            self.register_buffer("W", None)
            self.W_Q = nn.Linear(d_model, w_rank, bias=False)
            self.W_K = nn.Linear(d_model, w_rank, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute A_t = X_t W X_t^T (or its factored equivalent).

        Args:
            X: Firm-characteristic matrix of shape ``(N, D)`` where N is the
               number of assets at time t and D equals ``d_model``.

        Returns:
            Attention matrix A of shape ``(N, N)``.
        """
        if self.w_rank is None:
            # Exact form: A_{ij} = x_i W x_j^T  =>  A = X W X^T
            # (Kelly et al. 2025, linear case)
            return X @ self.W @ X.T  # type: ignore[operator]
        else:
            # Factored form: Q = X W_Q^T,  K = X W_K^T
            # A = Q K^T = (X W_Q^T)(X W_K^T)^T  (Kelly et al. 2025)
            Q = self.W_Q(X)  # type: ignore[misc]  # (N, r)
            K = self.W_K(X)  # type: ignore[misc]  # (N, r)
            return Q @ K.T  # (N, N)

    def get_W(self) -> torch.Tensor:
        """Return the effective D×D weight matrix W.

        For the full-rank parameterisation this is just ``self.W``.
        For the factored form it is reconstructed as W_Q^T W_K.

        Returns:
            Weight matrix of shape ``(d_model, d_model)``.
        """
        if self.w_rank is None:
            return self.W  # nn.Parameter — registered in __init__
        # Reconstruct W = W_Q^T W_K  (shape: D×D)
        W_Q = self.W_Q.weight  # type: ignore[union-attr]  # (r, D)
        W_K = self.W_K.weight  # type: ignore[union-attr]  # (r, D)
        return W_Q.T @ W_K  # (D, D)


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
