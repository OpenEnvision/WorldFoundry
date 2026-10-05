"""LAION aesthetic score over CLIP ViT-L/14 image embeddings."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.evaluation.tasks.metrics._shared.aesthetic import load_laion_aesthetic_linear_head
from worldfoundry.evaluation.tasks.metrics._shared.clip_embed import encode_clip_images
from worldfoundry.evaluation.tasks.metrics._shared.images import ImageInput

AESTHETIC_CLIP_MODEL = "openai:ViT-L-14"


def compute_laion_aesthetic(
    images: list[ImageInput],
    *,
    checkpoint: str | Path | None = None,
    device: str | None = None,
) -> float:
    """Mean LAION aesthetic score of ``images`` (raw ~0-10 scale, higher is better).

    Encodes frames with CLIP ViT-L/14, L2-normalizes the features, and applies
    the shared linear head. Raises ``FileNotFoundError`` when the head
    checkpoint is not staged; there is no fallback score.
    """
    import torch

    if not images:
        raise ValueError("laion_aesthetic requires at least one image")
    features = encode_clip_images(images, model=AESTHETIC_CLIP_MODEL, device=device)
    head = load_laion_aesthetic_linear_head(checkpoint)
    with torch.no_grad():
        scores = head(torch.as_tensor(features, dtype=torch.float32)).reshape(-1)
    return float(scores.mean().item())


__all__ = ["AESTHETIC_CLIP_MODEL", "compute_laion_aesthetic"]
