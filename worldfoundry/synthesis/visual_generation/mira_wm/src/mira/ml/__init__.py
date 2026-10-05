"""mira.ml: shared model building blocks for the codec and world model.

Public API:
    ImageConfig — spatial/temporal shape of a video tensor
    init_weights — type-dispatched in-place weight initialisation (pass to `nn.Module.apply`)
    SelfAttention, SelfAttentionConfig — GQA self-attention with QK-norm and optional gating
    AdaptiveLayerNorm, QKLayerNorm, QKRMSNorm — normalisation layers
    apply_rotary_emb, local_causal_mask — RoPE application and local causal masking
"""

from .attention import (
    AdaptiveLayerNorm,
    QKLayerNorm,
    QKRMSNorm,
    SelfAttention,
    SelfAttentionConfig,
    apply_rotary_emb,
    local_causal_mask,
)
from .image_config import ImageConfig
from .init import init_weights

__all__ = [
    "ImageConfig",
    "init_weights",
    "SelfAttention",
    "SelfAttentionConfig",
    "AdaptiveLayerNorm",
    "QKLayerNorm",
    "QKRMSNorm",
    "apply_rotary_emb",
    "local_causal_mask",
]
