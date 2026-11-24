"""
Subject consistency — DINOv2 + CLIP with mask-cropped regions.

Computes identity consistency of tracked subjects across frames:
- DINOv2_adj: Adjacent frame cosine similarity on masked crops
- CLIP_first: First-frame anchored cosine similarity on masked crops
- Final score: (DINOv2_adj + CLIP_first) / 2
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from worldfoundry.base_models.perception_core.general_perception.dino_embeddings import load_dinov2_base_feature_model

from ..base import BaseMetric
from ..weight_utils import clip_vit_b16_model_dir

METRIC_NAME = "subject_consistency"


class SubjectConsistencyMetric(BaseMetric):
    def __init__(self, device="cuda", batch_size=16):
        super().__init__(device)
        self.batch_size = batch_size
        self._dinov2 = None
        self._clip_model = None
        self._clip_processor = None
        self._dinov2_transform = T.Compose(
            [
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @property
    def name(self):
        return "subject_consistency"

    def _load_dinov2(self):
        if self._dinov2 is None:
            self._dinov2 = load_dinov2_base_feature_model(device=self.device)
        return self._dinov2

    def _load_clip(self):
        if self._clip_model is None:
            from transformers import CLIPModel, CLIPProcessor

            model_name = str(clip_vit_b16_model_dir())
            self._clip_model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
            self._clip_processor = CLIPProcessor.from_pretrained(model_name)
        return self._clip_model, self._clip_processor

    @torch.no_grad()
    def _extract_dinov2_features(self, images: list[Image.Image]) -> torch.Tensor:
        model = self._load_dinov2()
        all_feats = []
        for i in range(0, len(images), self.batch_size):
            batch = torch.stack([self._dinov2_transform(img) for img in images[i : i + self.batch_size]]).to(
                self.device
            )
            feats = model(batch)
            all_feats.append(F.normalize(feats, dim=-1, p=2))
        return torch.cat(all_feats, dim=0)

    @torch.no_grad()
    def _extract_clip_features(self, images: list[Image.Image]) -> torch.Tensor:
        model, processor = self._load_clip()
        all_feats = []
        for i in range(0, len(images), self.batch_size):
            inputs = processor(images=images[i : i + self.batch_size], return_tensors="pt").to(self.device)
            feats = model.get_image_features(**inputs)
            all_feats.append(F.normalize(feats, dim=-1, p=2))
        return torch.cat(all_feats, dim=0)

    @staticmethod
    def masked_crop(frame: Image.Image, mask: np.ndarray, padding: int = 10) -> Image.Image:
        """Crop frame to mask bounding box with gray background."""
        frame_np = np.array(frame)
        if mask.shape != frame_np.shape[:2]:
            mask = (
                np.array(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        (frame_np.shape[1], frame_np.shape[0]), Image.NEAREST
                    )
                )
                > 127
            )
        result = np.full_like(frame_np, 128)
        result[mask] = frame_np[mask]
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return frame
        y1, y2 = max(0, ys.min() - padding), min(frame_np.shape[0], ys.max() + padding)
        x1, x2 = max(0, xs.min() - padding), min(frame_np.shape[1], xs.max() + padding)
        return Image.fromarray(result[y1:y2, x1:x2])

    def compute(self, frames: list[Image.Image], **kwargs) -> dict[str, Any]:
        """Compute subject consistency score from masked frame crops."""
        if len(frames) < 2:
            return {f"{self.name}_score": 1.0, "skipped": True}

        masks = kwargs.get("masks")
        frame_indices = kwargs.get("frame_indices")

        if masks is not None and frame_indices is not None:
            masked_frames = []
            for frame, fid in zip(frames, frame_indices, strict=False):
                if fid in masks and masks[fid].sum() >= 10:
                    masked_frames.append(self.masked_crop(frame, masks[fid]))
                else:
                    masked_frames.append(frame)
            frames = masked_frames

        if len(frames) < 2:
            return {f"{self.name}_score": 1.0, "skipped": True}

        dinov2_feats = self._extract_dinov2_features(frames)
        clip_feats = self._extract_clip_features(frames)

        first_clip = clip_feats[0:1]
        scores = []
        for i in range(1, len(dinov2_feats)):
            dinov2_adj = max(0.0, F.cosine_similarity(dinov2_feats[i - 1 : i], dinov2_feats[i : i + 1]).item())
            clip_first = max(0.0, F.cosine_similarity(first_clip, clip_feats[i : i + 1]).item())
            scores.append((dinov2_adj + clip_first) / 2)

        return {
            f"{self.name}_score": float(np.mean(scores)),
            "dinov2_adj_mean": float(
                np.mean(
                    [
                        max(0.0, F.cosine_similarity(dinov2_feats[i - 1 : i], dinov2_feats[i : i + 1]).item())
                        for i in range(1, len(dinov2_feats))
                    ]
                )
            ),
            "clip_first_mean": float(
                np.mean(
                    [
                        max(0.0, F.cosine_similarity(first_clip, clip_feats[i : i + 1]).item())
                        for i in range(1, len(clip_feats))
                    ]
                )
            ),
        }
