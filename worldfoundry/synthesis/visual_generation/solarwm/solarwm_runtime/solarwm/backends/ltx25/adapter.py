"""Pure packed-document and parameter-free adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...errors import BackendContractError
from .geometry import STABLE_GEOMETRY

FUSED_PROPE_ORDER = (
    "native_ltx_3d_rope",
    "fused_camera_prope_qkv",
    "one_video_self_attention_call",
    "prope_output_transform",
    "native_head_gate_and_output_projection",
)
LORA_SUFFIXES = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)


def lora_target_modules() -> tuple[str, ...]:
    """Return the exact 48-block q/k/v/o + FFN target set (480 modules)."""

    targets = tuple(f"transformer_blocks.{block}.{suffix}" for block in range(48) for suffix in LORA_SUFFIXES)
    if len(targets) != 480 or len(set(targets)) != 480:
        raise BackendContractError("internal LTX LoRA target construction drifted")
    return targets


@dataclass(frozen=True)
class FusedPropeContract:
    """Parameter-free modification restricted to video ``attn1``."""

    attention_modules: int = 48
    parameter_keys_added: int = 0
    state_dict_keys_changed: bool = False
    applies_to: str = "video_self_attention_attn1_only"
    forbidden: tuple[str, ...] = ("caption_cross_attention_attn2", "gemma_connector")
    operation_order: tuple[str, ...] = FUSED_PROPE_ORDER

    def __post_init__(self) -> None:
        expected = (48, 0, False, "video_self_attention_attn1_only", FUSED_PROPE_ORDER)
        observed = (
            self.attention_modules,
            self.parameter_keys_added,
            self.state_dict_keys_changed,
            self.applies_to,
            self.operation_order,
        )
        if observed != expected:
            raise BackendContractError("fused PRoPE adapter contract drifted")


FUSED_PROPE_CONTRACT = FusedPropeContract()


def unpack_video_tokens(tokens: object) -> np.ndarray:
    """Invert patch-size-one F-H-W-major packing."""

    value = np.asarray(tokens)
    if value.ndim != 3 or value.shape[1:] != (
        STABLE_GEOMETRY.video_tokens,
        STABLE_GEOMETRY.latent_channels,
    ):
        raise BackendContractError("video tokens must be [B,7680,128]")
    batch = value.shape[0]
    return value.reshape(
        batch,
        STABLE_GEOMETRY.latent_frames,
        STABLE_GEOMETRY.latent_height,
        STABLE_GEOMETRY.latent_width,
        STABLE_GEOMETRY.latent_channels,
    ).transpose(0, 4, 1, 2, 3)


def verify_parameter_free_state_keys(before: object, after: object) -> None:
    if tuple(before) != tuple(after):
        raise BackendContractError("parameter-free LTX adapter changed checkpoint state keys")


__all__ = [
    "FUSED_PROPE_CONTRACT",
    "FUSED_PROPE_ORDER",
    "FusedPropeContract",
    "lora_target_modules",
    "unpack_video_tokens",
    "verify_parameter_free_state_keys",
]
