"""MAGI-2-preview architecture configuration.

Ported from SandAI's Apache-2.0 ``common/magi2_config.py``. The upstream config
uses pydantic ``BaseModel`` with JSON ``extends`` inheritance and engine/eval
sections; only the architecture hyperparameters relevant to the model math are
retained here as plain dataclasses (the pipeline layer owns eval/sampling).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class MoEConfig:
    """12-head × 256-expert top-6 MoE (routed layers 2..37)."""

    num_experts: int = 256
    top_k: int = 6
    score_func: str = "sigmoid"
    route_norm: bool = True
    route_scale: float = 4.90
    moe_layers: tuple[int, ...] = field(default_factory=lambda: tuple(range(2, 38)))
    expert_intermediate_size: int = 1280
    shared_expert_intermediate_size: int = 1280
    modality_specific_expert_intermediate_size: int = 1280
    num_heads: int = 12
    split_merge_intermediate_size: int | None = None


@dataclass
class AttentionGatingConfig:
    """Per-head scalar sigmoid gate applied after attention (preview + refiner)."""

    enable: bool = True


@dataclass
class AttentionSinksConfig:
    """Learned per-head sink tokens mixed into the softmax normalizer (preview only)."""

    enable: bool = True
    sink_token_num: int = 1


@dataclass
class MHCConfig:
    """Multi-head-channel hyper-connections over ``num_stream`` streams."""

    enable: bool = True
    num_stream: int = 4
    alpha_init: float = 0.01


@dataclass
class Magi2PreviewConfig:
    """Preview DiT: 40 layers, hidden 3072, full MHA (24:24), MoE + MHC + sinks."""

    num_layers: int = 40
    hidden_size: int = 3072
    head_dim: int = 128
    num_query_groups: int = 24
    video_in_channels: int = 48
    audio_in_channels: int = 64
    text_in_channels: int = 5120
    params_dtype: torch.dtype = torch.bfloat16
    intermediate_factor: int = 4
    mm_layers: tuple[int, ...] = field(default_factory=lambda: (0, 1, 38, 39))
    activation_type: str = "swiglu7"
    attn_softcap: float = -1.0
    attn_gating: AttentionGatingConfig = field(default_factory=AttentionGatingConfig)
    attn_sinks: AttentionSinksConfig = field(default_factory=AttentionSinksConfig)
    mhc_config: MHCConfig = field(default_factory=MHCConfig)
    moe_config: MoEConfig = field(default_factory=MoEConfig)

    def __post_init__(self) -> None:
        # Derived head counts: preview is full MHA (num_heads_q == num_heads_kv).
        self.num_heads_q = self.hidden_size // self.head_dim  # 24
        self.num_heads_kv = self.num_query_groups  # 24
        # MHC widens the adapter/embedding dim by the stream count.
        self.virtual_width_factor = self.mhc_config.num_stream if self.mhc_config.enable else 1
        self.adapter_dim = self.hidden_size * self.virtual_width_factor  # 12288


@dataclass
class Magi2RefinerConfig:
    """Refiner DiT: 30 layers, hidden 4096, GQA (32:8), dense (no MoE), no MHC/sinks."""

    num_layers: int = 30
    hidden_size: int = 4096
    head_dim: int = 128
    num_query_groups: int = 8
    video_in_channels: int = 48
    audio_in_channels: int = 64
    text_in_channels: int = 5120
    params_dtype: torch.dtype = torch.bfloat16
    intermediate_factor: int = 4
    mm_layers: tuple[int, ...] = field(default_factory=lambda: (0, 1, 28, 29))
    local_attn_layers: tuple[int, ...] = field(default_factory=lambda: tuple(range(30)))
    enable_attn_gating: bool = True
    activation_type: str = "swiglu7"
    frame_receptive_field: int = 11

    def __post_init__(self) -> None:
        self.num_heads_q = self.hidden_size // self.head_dim  # 32
        self.num_heads_kv = self.num_query_groups  # 8


# Modality token tags (video packed first, then audio, then text).
MAGI2_MODALITY_VIDEO = 0
MAGI2_MODALITY_AUDIO = 1
MAGI2_MODALITY_TEXT = 2


__all__ = [
    "AttentionGatingConfig",
    "AttentionSinksConfig",
    "MAGI2_MODALITY_AUDIO",
    "MAGI2_MODALITY_TEXT",
    "MAGI2_MODALITY_VIDEO",
    "MHCConfig",
    "Magi2PreviewConfig",
    "Magi2RefinerConfig",
    "MoEConfig",
]
