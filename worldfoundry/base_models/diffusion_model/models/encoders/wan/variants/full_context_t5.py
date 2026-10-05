"""Matrix-Game-3 T5 compatibility layer on top of shared Wan T5.

:class:`T5EncoderModel` exposes the names Matrix-Game-3
loaders expect while delegating to :mod:`~..model`.

Full-context recipes still emit Wan-style ``context``
tokens.  Not a Gemma or STEP1 encoder.

Use only when that checkpoint layout is required.
"""

from __future__ import annotations

from .. import reference as _shared_t5

GELU = _shared_t5.GELU
T5Attention = _shared_t5.T5Attention
T5CrossAttention = _shared_t5.T5CrossAttention
T5Decoder = _shared_t5.T5Decoder
T5Encoder = _shared_t5.T5Encoder
T5FeedForward = _shared_t5.T5FeedForward
T5LayerNorm = _shared_t5.T5LayerNorm
T5Model = _shared_t5.T5Model
T5RelativeEmbedding = _shared_t5.T5RelativeEmbedding
T5SelfAttention = _shared_t5.T5SelfAttention
fp16_clamp = _shared_t5.fp16_clamp
init_weights = _shared_t5.init_weights
umt5_xxl = _shared_t5.umt5_xxl


class T5EncoderModel(_shared_t5.T5EncoderModel):
    """Encoder model implementation."""
    def __init__(self, *args, **kwargs):
        """Init."""
        kwargs.setdefault("load_with_mmap", True)
        kwargs.setdefault("return_full_context", True)
        super().__init__(*args, **kwargs)


__all__ = [
    "GELU",
    "T5Attention",
    "T5CrossAttention",
    "T5Decoder",
    "T5Encoder",
    "T5EncoderModel",
    "T5FeedForward",
    "T5LayerNorm",
    "T5Model",
    "T5RelativeEmbedding",
    "T5SelfAttention",
    "fp16_clamp",
    "init_weights",
    "umt5_xxl",
]
