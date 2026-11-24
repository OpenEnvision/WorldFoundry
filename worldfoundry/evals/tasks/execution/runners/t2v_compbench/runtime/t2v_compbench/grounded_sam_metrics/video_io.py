from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def write_video(output_path: str | Path, video_tensor: torch.Tensor, fps: int, **_: object) -> None:
    """Write an RGB uint8 video tensor with OpenCV.

    torchvision.io.write_video is not available in some current torchvision builds,
    so T2V-CompBench uses this small inference-only fallback.
    """
    frames = video_tensor.detach().cpu().numpy()
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected video tensor with shape [T,H,W,3], got {frames.shape}")
    frames = np.clip(frames, 0, 255).astype(np.uint8)
    height, width = frames.shape[1:3]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {output_path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
