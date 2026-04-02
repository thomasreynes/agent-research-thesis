"""Model subpackage for the AIPM Linear Attention thesis.

Exposes the key model classes and functions for attention computation
and SDF weight generation.
"""

from src.model.attention import LinearAttention, decompose_attention, compute_sdf_weights
from src.model.sdf import SDFModel
from src.model.transformer import LinearTransformerModel

__all__ = [
    "LinearAttention",
    "decompose_attention",
    "compute_sdf_weights",
    "SDFModel",
    "LinearTransformerModel",
]
