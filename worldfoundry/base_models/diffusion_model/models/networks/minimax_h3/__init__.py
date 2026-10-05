"""Checkpoint-compatible network math for the MiniMax H3 audio-video DiT.

``dit.MiniMaxH3DiTModel`` is the packed-token transformer; ``ops`` holds
qk-norm / RoPE / AdaLN fallbacks; ``config`` is the architecture dataclass;
``loading`` reorders grouped-QKV Hub weights into ``[q_all, k_all, v_all]``.

Ported from SGLang's Apache-2.0 ``multimodal_gen`` MiniMax H3 implementation.
This package holds only ``nn.Module`` architecture, tensor-shape contracts, and
forward math (single-GPU PyTorch). Distributed sharding (tensor/sequence
parallel), breakable CUDA graphs, and layerwise offload from the SGLang source
are intentionally omitted; the numerics and the fp32/bf16 parameter split are
preserved exactly.
"""

from .config import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
    MiniMaxH3DiTArchConfig,
)
from .dit import (
    MINIMAX_H3_FP32_BUFFER_NAMES,
    MINIMAX_H3_FP32_PARAM_NAMES,
    MiniMaxH3DiTModel,
    reorder_grouped_qkv_to_qkv,
)

__all__ = [
    "MINIMAX_H3_ADALN_MODALITY_NUM",
    "MINIMAX_H3_FP32_BUFFER_NAMES",
    "MINIMAX_H3_FP32_PARAM_NAMES",
    "MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT",
    "MiniMaxH3DiTArchConfig",
    "MiniMaxH3DiTModel",
    "reorder_grouped_qkv_to_qkv",
]
