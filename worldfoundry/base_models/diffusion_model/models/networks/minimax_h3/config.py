"""MiniMax H3 DiT architecture configuration.

Ported from SGLang's Apache-2.0 ``configs/models/dits/minimax_h3.py``. The
SGLang ``DiTArchConfig``/``DiTConfig`` base carried FSDP/quantization/parallel
metadata that a single-GPU port does not need; only the architecture
hyperparameters are retained here as a plain dataclass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Packed-token rows are padded to this multiple so varlen kernels stay aligned.
MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT = 64
# AdaLN emits one (shift, scale, gate) triple per (video, audio, text) modality.
MINIMAX_H3_ADALN_MODALITY_NUM = 3


@dataclass
class MiniMaxH3DiTArchConfig:
    """Architecture hyperparameters for the MiniMax H3 packed-token DiT."""

    num_layers: int = 50
    token_refiner_num_layers: int = 2
    hidden_size: int = 5376
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    adaln_out_features: int = 18 * 5376
    final_adaln_out_features: int = 2 * 5376
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)
        if len(self.patch_size) != 3:
            raise ValueError(f"patch_size must have 3 values, got {self.patch_size}.")
        self.num_channels_latents = self.latents_dim

    @property
    def video_patch_output_dim(self) -> int:
        """Video row width = latents_dim * prod(patch_size)."""

        return self.latents_dim * math.prod(self.patch_size)


__all__ = [
    "MINIMAX_H3_ADALN_MODALITY_NUM",
    "MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT",
    "MiniMaxH3DiTArchConfig",
]
