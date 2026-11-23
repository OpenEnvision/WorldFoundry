# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
# Adapted from apple-aiml-research/ml-fastframes, revision
# a65c59240e01cf1de86197fa771ca09b38cf592a (Apple Sample Code License / Apache-2.0 Wan derivatives).
"""FastFrames causal Wan2.1 DiT, KV refinement network, and pixel decoder variants."""

from __future__ import annotations

from .causal_dit import (
    FASTFRAMES_WAN_CONFIGS,
    FastFramesCausalAttentionBlock,
    FastFramesCausalHead,
    FastFramesCausalWanModel,
    FastFramesCrossAttention,
    FastFramesFastOps,
    FastFramesSelfAttention,
    FastFramesWanConfig,
    allocate_fastframes_kv_cache,
    causal_rope_apply,
    fastframes_attention,
    fastframes_cross_attention,
    rope_real,
    set_fastframes_cross_attention_backend,
)
from .kv_refinement import (
    GroupedKVRepairHeads,
    GroupGEMM,
    KVRefinementNetwork,
    Modulation,
    StemBlock,
    TimestepEmbedder,
)
from .pixel_decoder import (
    BridgedPixelDecoder,
    PixelDecoder,
    align_frames,
    unpatchify_tokens,
)

__all__ = [
    "FASTFRAMES_WAN_CONFIGS",
    "BridgedPixelDecoder",
    "FastFramesCausalAttentionBlock",
    "FastFramesCausalHead",
    "FastFramesCausalWanModel",
    "FastFramesCrossAttention",
    "FastFramesFastOps",
    "FastFramesSelfAttention",
    "FastFramesWanConfig",
    "GroupGEMM",
    "GroupedKVRepairHeads",
    "KVRefinementNetwork",
    "Modulation",
    "PixelDecoder",
    "StemBlock",
    "TimestepEmbedder",
    "align_frames",
    "allocate_fastframes_kv_cache",
    "causal_rope_apply",
    "fastframes_attention",
    "fastframes_cross_attention",
    "rope_real",
    "set_fastframes_cross_attention_backend",
    "unpatchify_tokens",
]
