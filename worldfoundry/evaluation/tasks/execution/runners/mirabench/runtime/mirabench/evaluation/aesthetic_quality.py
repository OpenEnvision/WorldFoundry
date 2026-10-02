"""Batched LAION aesthetic-quality metric used by MiraBench."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize

from worldfoundry.core.media.codecs.video import list_numbered_frame_paths
from worldfoundry.core.execution.inference_runtime import (
    adaptive_batched_inference,
    resolve_inference_batch_size,
)
from worldfoundry.evaluation.tasks.metrics._shared.aesthetic import (
    load_laion_aesthetic_linear_head,
)

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def get_aesthetic_model(cache_folder):
    """Load the LAION aesthetic linear head (delegates to the shared in-tree loader)."""

    candidate = Path(cache_folder) / "aesthetic_model" / "sa_0_4_vit_l_14_linear.pth"
    return load_laion_aesthetic_linear_head(candidate if candidate.is_file() else None)


def clip_transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            transforms.Lambda(lambda value: value.float().div(255.0)),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


def _load_frame(path: Path, preprocess):
    with Image.open(path) as image:
        return preprocess(image.convert("RGB"))


def EvaluateLaionAesthetic(
    aesthetic_model,
    clip_model,
    preprocess,
    store_image_folder,
    device,
    batch_size=16,
):
    """Evaluate frames in bounded CLIP batches and synchronize only once."""

    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    frame_paths = list_numbered_frame_paths(store_image_folder)
    if not frame_paths:
        raise ValueError("aesthetic quality requires at least one frame")
    images = torch.stack([_load_frame(path, preprocess) for path in frame_paths])
    if torch.device(device).type == "cuda":
        images = images.pin_memory()

    aesthetic_model.eval().to(dtype=clip_model.dtype)
    clip_model.eval()
    resolved_batch_size = resolve_inference_batch_size(
        int(batch_size),
        device=device,
        scope="mirabench_aesthetic",
    )

    def score_batch(batch):
        features = torch.nn.functional.normalize(clip_model.encode_image(batch), dim=-1, p=2)
        return aesthetic_model(features).reshape(-1)

    scores = adaptive_batched_inference(
        images,
        score_batch,
        batch_size=resolved_batch_size,
        device=device,
        pad_to_batch_size=True,
        scope="mirabench_aesthetic",
    )
    return float(scores.mean().item())


__all__ = ["EvaluateLaionAesthetic", "clip_transform", "get_aesthetic_model"]
