"""Aesthetic quality metric — LAION aesthetic predictor on CLIP ViT-L/14 features."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from worldfoundry.base_models.perception_core.general_perception import openai_clip
from worldfoundry.evaluation.tasks.metrics._shared.aesthetic import (
    load_laion_aesthetic_linear_head,
)

from ..base import BaseMetric
from ..weight_utils import wbench_asset_path


class AestheticQualityMetric(BaseMetric):
    def __init__(self, device="cuda"):
        super().__init__(device)
        clip_path = str(wbench_asset_path("wbench_clip_vit_l14_checkpoint"))
        self.clip_model, self.preprocess = openai_clip.load(clip_path, device=self.device)
        self.aesthetic_model = self._get_aesthetic_model()

    @property
    def name(self):
        return "aesthetic_quality"

    def _get_aesthetic_model(self):
        checkpoint = wbench_asset_path("wbench_aesthetic_linear_checkpoint")
        return load_laion_aesthetic_linear_head(checkpoint).to(self.device)

    def compute(self, frames, first_frame=None, prompt=None, **kwargs):
        scores = []
        for frame in frames:
            img = self.preprocess(frame).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feats = self.clip_model.encode_image(img).to(torch.float32)
                feats = F.normalize(feats, dim=-1, p=2)
                score = self.aesthetic_model(feats).item()
                scores.append(score / 10.0)
        return {f"{self.name}_score": float(np.mean(scores))}
