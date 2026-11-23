"""MiniMax-H3 inference variants."""

from .h3_world import (
    H3WorldActionMaskSpec,
    h3_world_directed_attention_context,
    install_h3_world_directed_attention,
    load_h3_world_adapter,
)
from .pika import load_pika_h3_flex_adapter

__all__ = [
    "H3WorldActionMaskSpec",
    "h3_world_directed_attention_context",
    "install_h3_world_directed_attention",
    "load_h3_world_adapter",
    "load_pika_h3_flex_adapter",
]
