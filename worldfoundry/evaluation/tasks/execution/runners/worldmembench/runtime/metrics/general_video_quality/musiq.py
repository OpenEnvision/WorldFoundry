# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Fixed MUSIQ-SPAQ backend with VBench-compatible longer preprocessing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import EvaluationError


def resize_longer_rgb(rgb: np.ndarray, max_side: int = 512) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got shape {rgb.shape}")
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return rgb
    import cv2

    scale = max_side / float(longest)
    resized_height = max(1, int(scale * height))
    resized_width = max(1, int(scale * width))
    return cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )


class MusiqSpaqBackend:
    def __init__(
        self,
        *,
        checkpoint: str | Path,
        device: str,
        batch_size: int = 8,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise EvaluationError(f"MUSIQ-SPAQ checkpoint not found: {self.checkpoint}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.device = device
        self.batch_size = batch_size
        try:
            from pyiqa.archs.musiq_arch import MUSIQ
        except ImportError as error:
            raise EvaluationError(
                "imaging_quality requires pyiqa; install the WorldMemBench metric requirements"
            ) from error
        self.model = MUSIQ(pretrained_model_path=str(self.checkpoint))
        self.model.to(device)
        self.model.eval()

    def score(self, frames: list[np.ndarray]) -> list[float]:
        if not frames:
            return []
        import torch

        outputs: list[float] = []
        for start in range(0, len(frames), self.batch_size):
            resized = [
                np.ascontiguousarray(resize_longer_rgb(frame)) for frame in frames[start : start + self.batch_size]
            ]
            tensor = (
                torch.from_numpy(np.stack(resized))
                .permute(0, 3, 1, 2)
                .contiguous()
                .to(device=self.device, dtype=torch.float32)
                / 255.0
            )
            with torch.inference_mode():
                raw: Any = self.model(tensor)
            values = raw.detach().float().cpu().reshape(-1).tolist()
            outputs.extend(float(value) for value in values)
        if len(outputs) != len(frames):
            raise EvaluationError(f"MUSIQ returned {len(outputs)} scores for {len(frames)} frames")
        if not all(math.isfinite(value) for value in outputs):
            raise EvaluationError("MUSIQ returned a non-finite score")
        return outputs
