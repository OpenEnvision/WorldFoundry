"""Packed attention sequence metadata for long-context inference.

Variable-length / packed kernels (Flash varlen, jagged SDPA) need
cumulative sequence lengths and a max length, not a padded
``[B, S]`` tensor. This module is the shared ABI. Varlen dispatch and
the CP runtime both consume these dataclasses so a model family does
not invent a second packing layout.

Not this module:
    Building ``cu_seqlens`` from a text mask lives in
    :mod:`.sequence_metadata`. Running packed kernels lives in
    :mod:`.varlen`. These types carry ranges only — they do not
    allocate or launch attention.

Public surface:

- :class:`PackedCoreAttnParams` — self-attention ``cu_seqlens`` + max.
- :class:`PackedCrossAttnParams` — encoder/decoder packed ranges.
- :class:`ModelMetaArgs` — per-forward metadata a DiT block can stash
  (text vs video token counts, packing flags).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────
# Packed range ABI — device tensors plus CPU copies for host-side planners
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PackedCoreAttnParams:
    """Cumulative ranges and maximum lengths for packed self-attention.

    Args:
        q_range: Device-side query range boundaries.
        k_range: Device-side key range boundaries.
        np_q_range: CPU/Numpy query boundaries used by planning code.
        np_k_range: CPU/Numpy key boundaries used by planning code.
        max_seqlen_q: Maximum query segment length.
        max_seqlen_k: Maximum key segment length.
    """

    q_range: torch.Tensor
    k_range: torch.Tensor
    np_q_range: np.ndarray
    np_k_range: np.ndarray
    max_seqlen_q: int
    max_seqlen_k: int


@dataclass(frozen=True)
class PackedCrossAttnParams:
    """Optional cumulative ranges for packed cross-attention.

    Args:
        q_ranges: Query segment ranges.
        kv_ranges: Key/value segment ranges.
        cu_seqlens_q: Cumulative query lengths for varlen kernels.
        cu_seqlens_kv: Cumulative key/value lengths for varlen kernels.
        max_seqlen_q: Maximum query segment length.
        max_seqlen_kv: Maximum key/value segment length.
    """

    q_ranges: torch.Tensor | None = None
    kv_ranges: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_kv: torch.Tensor | None = None
    max_seqlen_q: int | None = None
    max_seqlen_kv: int | None = None


@dataclass(frozen=True)
class ModelMetaArgs:
    """Context-parallel layout metadata carried between pre/post processing.

    The fields describe spatial size, padding and per-rank splits, denoising
    ranges, optional feature-prefix behavior, CUDA-graph mode, and the packed
    self/cross-attention range objects. Model integrations normally construct
    this once per request and pass it unchanged through transformer blocks.
    """

    H: int
    W: int
    cp_pad_size: int
    cp_split_sizes: List[int]
    slice_point: int
    denoising_range_num: int
    range_num: int
    extract_prefix_video_feature: bool
    fwd_extra_1st_chunk: bool
    distill_nearly_clean_chunk: bool
    clip_token_nums: int
    enable_cuda_graph: bool
    core_attn_params: PackedCoreAttnParams
    cross_attn_params: PackedCrossAttnParams


__all__ = ["ModelMetaArgs", "PackedCoreAttnParams", "PackedCrossAttnParams"]
