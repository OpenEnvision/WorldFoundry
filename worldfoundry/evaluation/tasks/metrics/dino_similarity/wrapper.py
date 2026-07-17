"""WorldFoundry facade for DINO image similarity."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from worldfoundry.evaluation.tasks.metrics._shared.images import ImageInput, load_rgb_image
from worldfoundry.evaluation.tasks.metrics._shared.perceptual import resolve_device

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


def _preprocess(image: Image.Image) -> torch.Tensor:
    transform = T.Compose(
        [
            T.Resize(518),
            T.CenterCrop(518),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def _load_dino_model(device: torch.device):
    from worldfoundry.base_models.perception_core.general_perception.dino_embeddings import (
        load_dinov2_base_feature_model,
    )

    return load_dinov2_base_feature_model(device=device)


@torch.no_grad()
def compute_dino_similarity(
    reference: ImageInput,
    generated: ImageInput,
    *,
    device: str | None = None,
) -> float:
    """Compute cosine similarity between DINOv2 embeddings (higher is better)."""
    device_t = resolve_device(device)
    model = _load_dino_model(device_t)
    ref = _preprocess(load_rgb_image(reference)).to(device_t)
    gen = _preprocess(load_rgb_image(generated)).to(device_t)
    ref_feat = model(ref)
    gen_feat = model(gen)
    ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
    gen_feat = gen_feat / gen_feat.norm(dim=-1, keepdim=True)
    return float(F.cosine_similarity(ref_feat, gen_feat).item())


__all__ = [
    "compute_dino_similarity",
    "package_root",
]
