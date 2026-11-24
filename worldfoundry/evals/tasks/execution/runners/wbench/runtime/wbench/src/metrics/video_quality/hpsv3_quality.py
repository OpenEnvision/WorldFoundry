"""HPSv3 human preference score — Qwen2-VL-7B based preference model."""

import os
from typing import Any

import numpy as np
import torch
from PIL import Image

from worldfoundry.base_models.perception_core.video_quality import hpsv3 as hpsv3_base

from ..base import BaseMetric


class HPSv3QualityMetric(BaseMetric):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.inferencer = hpsv3_base.load_inferencer(device=device)

    @property
    def name(self):
        return "hpsv3_quality"

    def compute(self, frames: list[Image.Image], **kwargs) -> dict[str, Any]:
        tmp_paths = []
        for i, frame in enumerate(frames):
            p = f"/tmp/_hpsv3_{os.getpid()}_{i}.png"
            frame.save(p)
            tmp_paths.append(p)
        try:
            prompts = [""] * len(tmp_paths)
            with torch.no_grad():
                rewards = self.inferencer.reward(tmp_paths, prompts)
            raw_scores = [rewards[i][0].item() for i in range(len(tmp_paths))]
        finally:
            for p in tmp_paths:
                if os.path.exists(p):
                    os.remove(p)
        return {
            f"{self.name}_score": float(np.mean(raw_scores)),
            f"{self.name}_raw_scores": raw_scores,
        }
