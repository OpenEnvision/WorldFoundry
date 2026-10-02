"""Batched DINO temporal-consistency metric used by MiraBench."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from worldfoundry.core.media.codecs.video import list_numbered_frame_paths
from worldfoundry.core.execution.inference_runtime import (
    adaptive_batched_inference,
    resolve_inference_batch_size,
)
from worldfoundry.core.utils.tensors.torch import temporal_feature_consistency


def dino_transform_Image(n_px):
    return Compose(
        [
            Resize(size=n_px),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def _load_frame(path: Path, image_transform):
    with Image.open(path) as image:
        return image_transform(image.convert("RGB"))


def EvaluateTemporalDinoConsistency(dino_model, store_image_folder, device, batch_size=16):
    """Evaluate frame consistency with batched DINO inference and one GPU sync."""

    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    frame_paths = list_numbered_frame_paths(store_image_folder)
    if len(frame_paths) < 2:
        raise ValueError("temporal DINO consistency requires at least two frames")
    image_transform = dino_transform_Image(224)
    images = torch.stack([_load_frame(path, image_transform) for path in frame_paths])
    if torch.device(device).type == "cuda":
        images = images.pin_memory()

    resolved_batch_size = resolve_inference_batch_size(
        int(batch_size),
        device=device,
        scope="mirabench_dino",
    )
    features = adaptive_batched_inference(
        images,
        dino_model,
        batch_size=resolved_batch_size,
        device=device,
        pad_to_batch_size=True,
        scope="mirabench_dino",
    )
    features = torch.nn.functional.normalize(features, dim=-1, p=2)
    return float(temporal_feature_consistency(features).item())
