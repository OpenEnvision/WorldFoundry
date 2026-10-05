"""HPSv3 human preference score — Qwen2-VL-7B based preference model."""
import os
import tempfile
from contextlib import contextmanager
from typing import Dict, Any, List

import numpy as np
import torch
from PIL import Image

from ..base import BaseMetric

class HPSv3QualityMetric(BaseMetric):
    def __init__(self, device="cuda"):
        super().__init__(device)
        from worldfoundry.base_models.perception_core.video_quality.hpsv3 import load_inferencer

        self.inferencer = load_inferencer(device=device)

    @property
    def name(self):
        return "hpsv3_quality"

    def compute(self, frames: List[Image.Image], **kwargs) -> Dict[str, Any]:
        with self.prepare_cpu(frames) as prepared:
            return self.compute_prepared(prepared)

    @contextmanager
    def prepare_cpu(self, frames: List[Image.Image], *, executor=None):
        with tempfile.TemporaryDirectory(prefix=f"hpsv3_{os.getpid()}_") as tmp_dir:
            tmp_paths = [os.path.join(tmp_dir, f"frame_{i}.png") for i in range(len(frames))]

            def save_frame(item):
                i, frame = item
                p = os.path.join(tmp_dir, f"frame_{i}.png")
                frame.save(p)
                return p

            items = list(enumerate(frames))
            if executor is not None:
                list(executor.map(save_frame, items))
            else:
                for item in items:
                    save_frame(item)
            prompts = [""] * len(tmp_paths)
            yield (tmp_paths, prompts)

    def compute_prepared(self, batch) -> Dict[str, Any]:
        with torch.no_grad():
            rewards = self.inferencer.reward(*batch)
        raw_scores = [rewards[i][0].item() for i in range(len(rewards))]
        return {
            f"{self.name}_score": float(np.mean(raw_scores)),
            f"{self.name}_raw_scores": raw_scores,
        }
