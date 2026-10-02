"""Batched MUSIQ imaging-quality metric used by MiraBench."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from pyiqa.archs.musiq_arch import MUSIQ
from torchvision import transforms

from worldfoundry.core.media.codecs.video import list_numbered_frame_paths
from worldfoundry.core.execution.inference_runtime import (
    adaptive_batched_inference,
    resolve_inference_batch_size,
)


def transform(images, preprocess_mode="shorter"):
    if preprocess_mode.startswith("shorter"):
        _, _, height, width = images.size()
        if min(height, width) > 512:
            scale = 512.0 / min(height, width)
            images = transforms.Resize(size=(int(scale * height), int(scale * width)))(images)
            if preprocess_mode == "shorter_centercrop":
                images = transforms.CenterCrop(512)(images)
    elif preprocess_mode == "longer":
        _, _, height, width = images.size()
        if max(height, width) > 512:
            scale = 512.0 / max(height, width)
            images = transforms.Resize(size=(int(scale * height), int(scale * width)))(images)
    elif preprocess_mode == "None":
        return images / 255.0
    else:
        raise ValueError("Please recheck imaging_quality_mode")
    return images / 255.0


def EvaluateImagingQuality(imaging_quality_model, store_image_folder, device, batch_size=8):
    """Evaluate MUSIQ in bounded batches instead of synchronizing every frame."""

    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    frame_paths = list_numbered_frame_paths(store_image_folder)
    if not frame_paths:
        raise ValueError("imaging quality requires at least one frame")
    arrays = []
    for path in frame_paths:
        with Image.open(path) as image:
            arrays.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    images = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float()
    images = transform(images, "longer")
    if torch.device(device).type == "cuda":
        images = images.pin_memory()

    resolved_batch_size = resolve_inference_batch_size(
        int(batch_size),
        device=device,
        scope="mirabench_musiq",
    )
    scores = adaptive_batched_inference(
        images,
        imaging_quality_model,
        batch_size=resolved_batch_size,
        device=device,
        pad_to_batch_size=True,
        scope="mirabench_musiq",
    )
    return float(scores.reshape(-1).mean().item())


__all__ = ["EvaluateImagingQuality", "MUSIQ", "transform"]
