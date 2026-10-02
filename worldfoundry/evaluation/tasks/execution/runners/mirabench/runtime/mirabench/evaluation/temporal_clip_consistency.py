"""Batched CLIP temporal-consistency metric used by MiraBench."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from worldfoundry.core.media.codecs.video import list_numbered_frame_paths
from worldfoundry.core.execution.inference_runtime import (
    adaptive_batched_inference,
    resolve_inference_batch_size,
)
from worldfoundry.core.utils.tensors.torch import temporal_feature_consistency


def _load_frame(path: Path, preprocess):
    with Image.open(path) as image:
        return preprocess(image.convert("RGB"))


def EvaluateTemporalClipConsistency(clip_model, preprocess, store_image_folder, device, batch_size=32):
    """Evaluate CLIP frame consistency in bounded batches without per-frame syncs."""

    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    frame_paths = list_numbered_frame_paths(store_image_folder)
    if len(frame_paths) < 2:
        raise ValueError("temporal CLIP consistency requires at least two frames")
    images = torch.stack([_load_frame(path, preprocess) for path in frame_paths])
    if torch.device(device).type == "cuda":
        images = images.pin_memory()

    resolved_batch_size = resolve_inference_batch_size(
        int(batch_size),
        device=device,
        scope="mirabench_clip",
    )
    features = adaptive_batched_inference(
        images,
        clip_model.encode_image,
        batch_size=resolved_batch_size,
        device=device,
        pad_to_batch_size=True,
        scope="mirabench_clip",
    )
    features = torch.nn.functional.normalize(features, dim=-1, p=2)
    return float(temporal_feature_consistency(features).item())
