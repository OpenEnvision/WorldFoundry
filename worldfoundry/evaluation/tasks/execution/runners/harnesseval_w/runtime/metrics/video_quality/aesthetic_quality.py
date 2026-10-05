"""Aesthetic quality metric — LAION aesthetic predictor on CLIP ViT-L/14 features."""
from contextlib import contextmanager

from worldfoundry.base_models.perception_core.general_perception.openai_clip_runtime import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseMetric
from ..weight_utils import get_weights_dir, get_weight_file


_GPU_BATCH_SIZE = 16


class AestheticQualityMetric(BaseMetric):
    def __init__(self, device="cuda"):
        super().__init__(device)
        clip_dir = get_weights_dir("clip")
        self.clip_model, self.preprocess = clip.load(get_weight_file("clip", "ViT-L-14.pt"), device=self.device, download_root=clip_dir)
        self.aesthetic_model = self._get_aesthetic_model()

    @property
    def name(self):
        return "aesthetic_quality"

    def _get_aesthetic_model(self):
        path_to_model = get_weight_file("aesthetic", "sa_0_4_vit_l_14_linear.pth")
        model = nn.Linear(768, 1)
        state_dict = torch.load(path_to_model, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        return model

    def compute(self, frames, first_frame=None, prompt=None, **kwargs):
        with self.prepare_cpu(frames) as prepared:
            return self.compute_prepared(prepared)

    @contextmanager
    def prepare_cpu(self, frames, *, executor=None):
        prepared = (
            list(executor.map(self.preprocess, frames))
            if executor is not None
            else [self.preprocess(frame) for frame in frames]
        )
        yield prepared

    def compute_prepared(self, prepared):
        scores = []
        for start in range(0, len(prepared), _GPU_BATCH_SIZE):
            images = torch.stack(prepared[start : start + _GPU_BATCH_SIZE]).to(
                self.device
            )
            with torch.no_grad():
                feats = self.clip_model.encode_image(images).to(torch.float32)
                feats = F.normalize(feats, dim=-1, p=2)
                batch_scores = self.aesthetic_model(feats).reshape(-1)
            scores.extend((batch_scores / 10.0).cpu().tolist())
        return {f"{self.name}_score": float(np.mean(scores))}
