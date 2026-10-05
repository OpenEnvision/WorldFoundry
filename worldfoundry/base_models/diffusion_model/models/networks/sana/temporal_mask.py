"""SANA video temporal attention-mask construction.

Builds the FlexAttention block mask used by windowed / first-frame-visible
softmax layers in :class:`SanaMSVideo`.  The first latent frame attends
globally; later frames see a local temporal window.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch.nn.attention.flex_attention import create_block_mask


@lru_cache
def create_block_mask_cached(score_mod, batch, heads, query_length, key_length, device="cuda", _compile=False):
    """Cache a FlexAttention block mask for immutable sequence geometry."""

    return create_block_mask(
        score_mod,
        batch,
        heads,
        query_length,
        key_length,
        device=device,
        _compile=_compile,
    )


def generate_temporal_head_mask_mod(
    context_length: int = 226,
    prompt_length: int = 226,
    num_frames: int = 13,
    token_per_frame: int = 1350,
    mul: int = 2,
):
    """Build Sana's first-frame-visible local temporal mask modifier."""

    del context_length, prompt_length, num_frames

    def round_to_multiple(index):
        return math.ceil(index / 128) * 128

    def temporal_mask_mod(batch, head, query_index, key_index):
        del batch, head
        temporal_radius = round_to_multiple(mul * token_per_frame)
        temporal_mask = torch.abs(query_index - key_index) <= temporal_radius
        first_frame_mask = key_index < token_per_frame
        return first_frame_mask | temporal_mask

    return temporal_mask_mod


__all__ = ["create_block_mask_cached", "generate_temporal_head_mask_mod"]
