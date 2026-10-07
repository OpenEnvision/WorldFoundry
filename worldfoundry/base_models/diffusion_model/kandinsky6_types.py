from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import Tensor
from typing import TypedDict


class TextEmbeds(TypedDict):
    """Output of ``TextEmbedder.encode()`` in native batch-dimension format."""
    text_embeds: Tensor   # (B,S,text_dim) or (S,text_dim) for one input
    pooled_embed: Tensor  # (B,clip_dim)


@dataclass
class LatentBundle:
    """Latent state flowing through the denoising loop.

    Native inference uses ``(B,T,H,W,C)`` / ``(B,A,D)`` batch-dimension
    tensors.  The packed ``(sum_T,H,W,C)`` / ``(sum_A,D)`` layout remains
    supported for low-level and export callers, with ``cu_seqlens`` marking
    boundaries. Audio is None for T2V modes.
    """
    video: Tensor | None             # (B,T,H,W,C) or packed (sum_T,H,W,C)
    audio: Tensor | None             # (B,A,D) or packed (sum_A,D); None for T2V
    video_cu_seqlens: Tensor | None  # (bs+1,) int32
    audio_cu_seqlens: Tensor | None  # (bs+1,) int32; None for T2V


@dataclass
class Kandinsky6PipelineOutput:
    frames: Tensor                        # (bs, 3, T, H, W) uint8
    audio: list[np.ndarray] | None = None # list[bs] of (samples,) int16; None for T2V
    path: str | list[str] | None = None   # set when save_path was passed to the pipeline
    prompts: list[str] | None = None      # captions encoded for this clip, after expansion
    latents: LatentBundle | None = None
