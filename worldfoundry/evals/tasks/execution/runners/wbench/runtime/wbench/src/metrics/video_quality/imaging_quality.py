"""Imaging quality metric — the official MUSIQ-KonIQ variant via PyIQA."""

import numpy as np
import torch
import torchvision.transforms as T

from ..base import BaseMetric
from ..weight_utils import wbench_asset_path


def _vbench_transform(img_tensor, max_edge=512):
    """VBench-aligned: scale long edge to max_edge, normalize to [0,1]."""
    _, h, w = img_tensor.shape
    if max(h, w) > max_edge:
        scale = max_edge / max(h, w)
        new_h, new_w = int(scale * h), int(scale * w)
        img_tensor = T.Resize(size=(new_h, new_w), antialias=False)(img_tensor)
    return img_tensor.float() / 255.0


class ImagingQualityMetric(BaseMetric):
    def __init__(self, device="cuda"):
        super().__init__(device)
        import pyiqa

        self.model = pyiqa.create_metric(
            "musiq",
            device=self.device,
            pretrained_model_path=str(wbench_asset_path("wbench_musiq_koniq_checkpoint")),
        )

    @property
    def name(self):
        return "imaging_quality"

    def compute(self, frames, first_frame=None, prompt=None, **kwargs):
        scores = []
        to_tensor = T.PILToTensor()
        for frame in frames:
            img_tensor = to_tensor(frame)
            img_tensor = _vbench_transform(img_tensor)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                score = self.model(img_tensor).item()
                scores.append(score / 100.0)
        return {f"{self.name}_score": float(np.mean(scores))}
