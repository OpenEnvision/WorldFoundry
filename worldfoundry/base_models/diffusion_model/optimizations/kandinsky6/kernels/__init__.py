"""Attention kernels. Each backend lives in its own module; this package binds one onto a DiT."""

from __future__ import annotations

from typing import Literal

from torch import nn

from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.base import ConfigModel
from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.kernels.attention_engine import SelfAttentionEngine, resolve_attention_engine

AttentionEngineName = Literal[
    "flash_attention_3",
    "flash_attention_2",
    "sage",
    "sdpa",
    "auto",
]


class AttentionConfig(ConfigModel):
    """Self-attention kernel for the DiT.

    The default asks for FlashAttention 3. ``resolve_attention_engine`` falls
    back to ``auto`` when that package is not installed.
    """

    engine: AttentionEngineName = "flash_attention_3"


def bind_attention(requested: str) -> str:
    """Resolve a config or CLI engine name to one that can be constructed."""
    return resolve_attention_engine(requested)


def rebind_attention(module: nn.Module, engine: str) -> None:
    """Point every attention slot at ``engine``. Padding layers stay on SDPA."""
    for child in module.modules():
        slot = getattr(child, "attn", None)
        if not isinstance(slot, SelfAttentionEngine):
            continue
        child.attn = SelfAttentionEngine("sdpa" if getattr(child, "force_sdpa", False) else engine)


__all__ = (
    "AttentionConfig",
    "AttentionEngineName",
    "SelfAttentionEngine",
    "bind_attention",
    "rebind_attention",
    "resolve_attention_engine",
)
