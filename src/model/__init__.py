"""Model subpackage for the AIPM Linear Attention thesis.

Exposes the key model classes and functions for attention computation,
the full linear attention transformer, and SDF weight generation.
"""

from src.model.attention import LinearAttention, decompose_attention, compute_sdf_weights
from src.model.sdf import SDFModel
from src.model.transformer import LinearAttentionTransformer

__all__ = [
    "LinearAttention",
    "decompose_attention",
    "compute_sdf_weights",
    "SDFModel",
    "LinearAttentionTransformer",
]
