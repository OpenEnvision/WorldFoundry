"""Internal MiniMax H3 pipeline building blocks (ported from SGLang Apache-2.0).

Pure-tensor implementations of the packed-sequence layout, token
patchify/unpatchify, task profiles, and the CFG-distilled coupled video+audio
denoise loop. SGLang's ``Req``/``OutputBatch``/``Stage`` framework and its
distributed (Ulysses) coupling are stripped; every function here is a plain
single-GPU callable the bespoke ``NativeMiniMaxH3Pipeline`` orchestrates.
"""

from __future__ import annotations

from .denoise_loop import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_AUDIO_ROW_WIDTH,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
    MINIMAX_H3_VIDEO_ROW_WIDTH,
    MiniMaxH3DenoiseBranch,
    minimax_h3_denoise_loop,
)
from .packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from .packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from .task_profiles import (
    MINIMAX_H3_TASK_FL2VA,
    MINIMAX_H3_TASK_REF2VA,
    MINIMAX_H3_TASK_T2VA,
    minimax_h3_task_profile,
    partition_for_task,
)

__all__ = [
    "MINIMAX_H3_AUDIO_REF_COND_TIMESTEP",
    "MINIMAX_H3_AUDIO_ROW_WIDTH",
    "MINIMAX_H3_IMGVID_COND_TIMESTEP",
    "MINIMAX_H3_TASK_FL2VA",
    "MINIMAX_H3_TASK_REF2VA",
    "MINIMAX_H3_TASK_T2VA",
    "MINIMAX_H3_VIDEO_ROW_WIDTH",
    "MiniMaxH3DenoiseBranch",
    "minimax_h3_denoise_loop",
    "minimax_h3_packed_sequence",
    "minimax_h3_packed_sequence_ref2va_blocks",
    "minimax_h3_patchify_video_latent",
    "minimax_h3_task_profile",
    "minimax_h3_unpack_audio_tokens",
    "minimax_h3_unpatchify_video_tokens",
    "partition_for_task",
]
