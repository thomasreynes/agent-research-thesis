"""Unit tests for the linear attention model components.

Tests cover:
- :class:`src.model.attention.LinearAttention`  — bilinear A_t = X_t W X_t^T
- :func:`src.model.attention.decompose_attention` — symmetric/antisymmetric split
- :class:`src.model.sdf.SDFModel`               — SDF weight computation
- :class:`src.model.transformer.LinearAttentionTransformer` — full model

All tests use small synthetic tensors (no real data required).

References
----------
Kelly, Kuznetsov, Malamud & Xu (NBER WP 33351, 2025).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.attention import LinearAttention, decompose_attention
from src.model.sdf import SDFModel
from src.model.transformer import LinearAttentionTransformer


# ---------------------------------------------------------------------------
# LinearAttention tests
# ---------------------------------------------------------------------------

class TestLinearAttention:
    """Tests for the bilinear attention module A_t = X_t W X_t^T."""

    def test_output_shape(self) -> None:
        """A_t should have shape (N, N)."""
        N, D = 10, 8
        attn = LinearAttention(d_model=D)
        X = torch.randn(N, D)
        A = attn(X)
        assert A.shape == (N, N), f"Expected ({N}, {N}), got {A.shape}"

    def test_bilinear_form(self) -> None:
        """A_{ij} = x_i W x_j^T — verify individual entry (Kelly et al. 2025)."""
        N, D = 5, 4
        attn = LinearAttention(d_model=D)
        X = torch.randn(N, D)
        A = attn(X)
        # Manual check: A[0, 1] == x_0 @ W @ x_1
        expected = (X[0] @ attn.W @ X[1]).item()
        assert abs(A[0, 1].item() - expected) < 1e-5

    def test_weight_matrix_shape(self) -> None:
        """W should be square D × D."""
        D = 6
        attn = LinearAttention(d_model=D)
        assert attn.W.shape == (D, D)

    def test_no_nan_output(self) -> None:
        """Forward pass should not produce NaN values."""
        attn = LinearAttention(d_model=8)
        X = torch.randn(20, 8)
        A = attn(X)
        assert not torch.isnan(A).any()

    def test_gradients_flow(self) -> None:
        """Gradients should flow through W during backward pass."""
        attn = LinearAttention(d_model=8)
        X = torch.randn(10, 8)
        A = attn(X)
        loss = A.mean()
        loss.backward()
        assert attn.W.grad is not None


# ---------------------------------------------------------------------------
# decompose_attention tests
# ---------------------------------------------------------------------------

class TestDecomposeAttention:
    """Tests for the symmetric/antisymmetric decomposition.

    Decomposition (thesis methodology):
        A^s = (A + A^T) / 2   — symmetric component
        A^a = (A - A^T) / 2   — antisymmetric component
    """

    def test_symmetric_part_is_symmetric(self) -> None:
        """A_sym should satisfy A_sym == A_sym^T."""
        A = torch.randn(8, 8)
        A_sym, _ = decompose_attention(A)
        assert torch.allclose(A_sym, A_sym.T, atol=1e-6)

    def test_antisymmetric_part_is_antisymmetric(self) -> None:
        """A_anti should satisfy A_anti == -A_anti^T."""
        A = torch.randn(8, 8)
        _, A_anti = decompose_attention(A)
        assert torch.allclose(A_anti, -A_anti.T, atol=1e-6)

    def test_reconstruction(self) -> None:
        """A_sym + A_anti should exactly reconstruct A."""
        A = torch.randn(8, 8)
        A_sym, A_anti = decompose_attention(A)
        assert torch.allclose(A_sym + A_anti, A, atol=1e-6)

    def test_diagonal_of_antisymmetric_is_zero(self) -> None:
        """The diagonal of A_anti must be identically zero."""
        A = torch.randn(8, 8)
        _, A_anti = decompose_attention(A)
        assert torch.allclose(torch.diagonal(A_anti), torch.zeros(8), atol=1e-6)

    def test_output_shapes_match_input(self) -> None:
        """Both components should have the same shape as A."""
        N = 15
        A = torch.randn(N, N)
        A_sym, A_anti = decompose_attention(A)
        assert A_sym.shape == (N, N)
        assert A_anti.shape == (N, N)


# ---------------------------------------------------------------------------
# LinearAttentionTransformer tests
# ---------------------------------------------------------------------------

class TestLinearAttentionTransformer:
    """Tests for the full linear attention transformer (Kelly et al. 2025)."""

    d_in: int = 8
    embed_dim: int = 16
    N: int = 20

    def _model(self, dropout: float = 0.0) -> LinearAttentionTransformer:
        return LinearAttentionTransformer(
            d_in=self.d_in, embed_dim=self.embed_dim, dropout=dropout
        )

    def test_forward_output_shape(self) -> None:
        """Model should output one prediction per asset: shape (N_t,)."""
        model = self._model()
        X = torch.randn(self.N, self.d_in)
        preds = model(X)
        assert preds.shape == (self.N,), f"Expected ({self.N},), got {preds.shape}"

    def test_forward_with_components_shapes(self) -> None:
        """forward_with_components should return (preds, A, A_sym, A_anti)."""
        model = self._model()
        X = torch.randn(self.N, self.d_in)
        preds, A, A_sym, A_anti = model.forward_with_components(X)
        assert preds.shape == (self.N,)
        assert A.shape == (self.N, self.N)
        assert A_sym.shape == (self.N, self.N)
        assert A_anti.shape == (self.N, self.N)

    def test_compute_attention_is_square(self) -> None:
        """Attention matrix should be N_t × N_t for any cross-section size."""
        model = self._model()
        X = torch.randn(15, self.d_in)
        A = model.compute_attention(X)
        assert A.shape == (15, 15)

    def test_components_decompose_A(self) -> None:
        """A_sym + A_anti should equal the raw attention matrix A."""
        model = self._model()
        X = torch.randn(self.N, self.d_in)
        _, A, A_sym, A_anti = model.forward_with_components(X)
        assert torch.allclose(A_sym + A_anti, A, atol=1e-5)

    def test_A_sym_is_symmetric(self) -> None:
        """Symmetric component should satisfy A_sym == A_sym^T."""
        model = self._model()
        X = torch.randn(self.N, self.d_in)
        _, _, A_sym, _ = model.forward_with_components(X)
        assert torch.allclose(A_sym, A_sym.T, atol=1e-5)

    def test_A_anti_is_antisymmetric(self) -> None:
        """Antisymmetric component should satisfy A_anti == -A_anti^T."""
        model = self._model()
        X = torch.randn(self.N, self.d_in)
        _, _, _, A_anti = model.forward_with_components(X)
        assert torch.allclose(A_anti, -A_anti.T, atol=1e-5)

    def test_combined_weight_matrix_shape(self) -> None:
        """Effective weight matrix W_Q^T W_K should have shape (d_in, d_in)."""
        model = self._model()
        W_eff = model.get_combined_weight_matrix()
        assert W_eff.shape == (self.d_in, self.d_in)

    def test_projection_weights_shapes(self) -> None:
        """W_Q and W_K should have shape (embed_dim, d_in)."""
        model = self._model()
        W_Q, W_K = model.get_projection_weights()
        assert W_Q.shape == (self.embed_dim, self.d_in)
        assert W_K.shape == (self.embed_dim, self.d_in)

    def test_no_nan_in_forward(self) -> None:
        """Forward pass should not produce NaN predictions."""
        model = self._model()
        X = torch.randn(self.N, self.d_in)
        preds = model(X)
        assert not torch.isnan(preds).any()

    def test_gradients_flow_through_all_parameters(self) -> None:
        """All model parameters should receive gradients after a backward pass."""
        model = self._model()
        X = torch.randn(self.N, self.d_in)
        y = torch.randn(self.N)
        preds = model(X)
        loss = ((preds - y) ** 2).mean()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter '{name}'"

    def test_different_cross_section_sizes(self) -> None:
        """Model should handle arbitrary N_t at inference time."""
        model = self._model()
        for N in [5, 50, 200]:
            X = torch.randn(N, self.d_in)
            preds = model(X)
            assert preds.shape == (N,), f"Failed for N={N}"

    def test_forward_and_forward_with_components_agree(self) -> None:
        """forward() and forward_with_components() must return identical preds."""
        model = self._model()
        model.eval()
        X = torch.randn(self.N, self.d_in)
        with torch.no_grad():
            preds_simple = model(X)
            preds_full, _, _, _ = model.forward_with_components(X)
        assert torch.allclose(preds_simple, preds_full, atol=1e-6)

    def test_effective_weight_matrix_is_detached(self) -> None:
        """get_combined_weight_matrix() should return a detached tensor."""
        model = self._model()
        W_eff = model.get_combined_weight_matrix()
        assert not W_eff.requires_grad


# ---------------------------------------------------------------------------
# SDFModel tests
# ---------------------------------------------------------------------------

class TestSDFModel:
    """Tests for the Stochastic Discount Factor model."""

    def test_output_shape(self) -> None:
        """SDF should output one weight per asset: shape (N,)."""
        N, D = 10, 8
        sdf = SDFModel(d_model=D)
        X = torch.randn(N, D)
        attn_feats = torch.randn(N, D)
        w = sdf(X, attn_feats)
        assert w.shape == (N,)

    def test_weights_are_zero_mean(self) -> None:
        """SDF portfolio weights must sum to zero (long-short portfolio).

        This is enforced by the mean-centering step in SDFModel.forward.
        """
        N, D = 10, 8
        sdf = SDFModel(d_model=D)
        X = torch.randn(N, D)
        attn_feats = torch.randn(N, D)
        w = sdf(X, attn_feats)
        assert abs(w.mean().item()) < 1e-5, f"Weights mean should be 0, got {w.mean().item():.1e}"

    def test_pricing_error_scalar_outputs(self) -> None:
        """pricing_error() should return two scalars."""
        N, D = 10, 8
        sdf = SDFModel(d_model=D)
        w = torch.randn(N)
        r = torch.randn(N)
        sdf_val, pricing_err = sdf.pricing_error(w, r)
        assert sdf_val.shape == torch.Size([])
        assert pricing_err.shape == torch.Size([])

    def test_sdf_moment_condition(self) -> None:
        """SDF value = 1 - w^T r  (Kelly et al. 2025, SDF moment condition)."""
        N = 5
        w = torch.tensor([0.1, -0.2, 0.05, 0.15, -0.1])
        r = torch.tensor([0.02, -0.01, 0.03, 0.01, 0.00])
        sdf = SDFModel(d_model=8)
        sdf_val, _ = sdf.pricing_error(w, r)
        expected = 1.0 - (w * r).sum().item()
        assert abs(sdf_val.item() - expected) < 1e-6
