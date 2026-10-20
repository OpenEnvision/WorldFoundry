# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""OpenAI CLIP + LAION aesthetic predictor backend."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import EvaluationError

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_clip_rgb(
    frames: list[np.ndarray],
    image_size: int = 224,
) -> Any:
    """Apply VBench's tensor CLIP transform to RGB uint8 frames."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if not frames:
        import torch

        return torch.empty((0, 3, image_size, image_size), dtype=torch.float32)

    import torch
    from torchvision.transforms import CenterCrop, InterpolationMode, Resize

    resize = Resize(
        image_size,
        interpolation=InterpolationMode.BICUBIC,
        antialias=False,
    )
    center_crop = CenterCrop(image_size)
    tensors = []
    for frame in frames:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB image, got shape {frame.shape}")
        tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).to(dtype=torch.float32)
        tensors.append(center_crop(resize(tensor)))

    batch = torch.stack(tensors).to(dtype=torch.float32).div_(255.0)
    mean = torch.tensor(CLIP_MEAN, dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=batch.dtype).view(1, 3, 1, 1)
    return (batch - mean) / std


class LaionAestheticBackend:
    """Fixed VBench-compatible ViT-L/14 aesthetic scoring backend."""

    def __init__(
        self,
        *,
        clip_checkpoint: str | Path,
        aesthetic_checkpoint: str | Path,
        device: str,
        batch_size: int = 32,
    ) -> None:
        self.clip_checkpoint = Path(clip_checkpoint).expanduser().resolve()
        self.aesthetic_checkpoint = Path(aesthetic_checkpoint).expanduser().resolve()
        if not self.clip_checkpoint.is_file():
            raise EvaluationError(f"clip checkpoint not found: {self.clip_checkpoint}")
        if not self.aesthetic_checkpoint.is_file():
            raise EvaluationError(f"aesthetic checkpoint not found: {self.aesthetic_checkpoint}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.device = device
        self.batch_size = batch_size
        try:
            from worldfoundry.base_models.perception_core.general_perception import openai_clip as clip
            from worldfoundry.evaluation.tasks.metrics._shared.aesthetic import load_laion_aesthetic_linear_head
        except ImportError as error:
            raise EvaluationError(
                "aesthetic_quality requires torch and torchvision; install the WorldMemBench metric requirements"
            ) from error

        self.clip_model, _ = clip.load(
            str(self.clip_checkpoint),
            device=device,
            jit=False,
        )
        self.clip_model.eval()

        self.aesthetic_model = load_laion_aesthetic_linear_head(self.aesthetic_checkpoint).to(device)

    def score(self, frames: list[np.ndarray]) -> list[float]:
        if not frames:
            return []

        import torch
        import torch.nn.functional as functional

        outputs: list[float] = []
        for start in range(0, len(frames), self.batch_size):
            batch = preprocess_clip_rgb(frames[start : start + self.batch_size]).to(self.device)
            with torch.inference_mode():
                features: Any = self.clip_model.encode_image(batch)
                features = functional.normalize(
                    features.to(torch.float32),
                    dim=-1,
                    p=2,
                )
                raw: Any = self.aesthetic_model(features).squeeze(dim=-1)
            outputs.extend(float(value) for value in raw.detach().float().cpu().reshape(-1).tolist())

        if len(outputs) != len(frames):
            raise EvaluationError(f"LAION aesthetic backend returned {len(outputs)} scores for {len(frames)} frames")
        if not all(math.isfinite(value) for value in outputs):
            raise EvaluationError("LAION aesthetic backend returned a non-finite score")
        return outputs
