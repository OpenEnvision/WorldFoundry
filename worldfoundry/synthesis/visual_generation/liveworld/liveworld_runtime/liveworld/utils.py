"""Inference-only compatibility helpers for the LiveWorld runtime.

LiveWorld's upstream utility module also bundled training losses, EMA/FSDP
checkpointing, dataset workers, and experiment logging.  The in-tree runtime
keeps only the symbols consumed by inference and delegates shared Wan/LoRA and
media functionality to WorldFoundry's canonical modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from worldfoundry.core.model_loading import GeneralLoRALoader
from worldfoundry.core.utils.tensors.torch import set_seed_everywhere as set_seed
from worldfoundry.core.nn.diffusion.schedulers import (
    FlowMatchScheduler,
    SchedulerInterface,
)
from worldfoundry.core.io import save_video_h264 as _save_video_h264


def save_video_h264(
    path: str | Path,
    frames: np.ndarray | torch.Tensor | Iterable[np.ndarray],
    fps: float = 16.0,
) -> None:
    """Write inference frames as an H.264 MP4 through shared video I/O."""

    if isinstance(frames, torch.Tensor):
        frame_count = int(frames.shape[0])
    elif isinstance(frames, np.ndarray):
        frame_count = int(frames.shape[0])
    else:
        frames = list(frames)
        frame_count = len(frames)
    if frame_count == 0:
        return

    _save_video_h264(frames, path, fps=fps)


__all__ = [
    "FlowMatchScheduler",
    "GeneralLoRALoader",
    "SchedulerInterface",
    "save_video_h264",
    "set_seed",
]
