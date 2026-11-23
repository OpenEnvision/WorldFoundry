"""Manipulation Direction (MD) from Sensors 2023 / ICIP 2022."""

from __future__ import annotations

from typing import Any

import numpy as np

from worldfoundry.evals.tasks.metrics._shared.clip_embed import (
    encode_clip_images,
    encode_clip_texts,
)


def compute_manipulation_direction(
    delta_image: np.ndarray,
    delta_text: np.ndarray,
    *,
    eps: float = 1e-8,
) -> float:
    """MD score from image/text change vectors (Eq. 1, Watanabe et al.)."""
    d_i = np.asarray(delta_image, dtype=np.float64).reshape(-1)
    d_t = np.asarray(delta_text, dtype=np.float64).reshape(-1)
    norm_i = np.linalg.norm(d_i)
    norm_t = np.linalg.norm(d_t)
    if norm_i <= eps or norm_t <= eps:
        return 0.0
    return float(np.dot(d_i, d_t) / (norm_i * norm_t))


def compute_manipulation_direction_from_embeddings(
    embed_input: np.ndarray,
    embed_manipulated: np.ndarray,
    embed_text_original: np.ndarray,
    embed_text_replaced: np.ndarray,
) -> float:
    """Compute MD from CLIP embeddings of I, I', T, T'."""
    delta_image = np.asarray(embed_manipulated, dtype=np.float64) - np.asarray(embed_input, dtype=np.float64)
    delta_text = np.asarray(embed_text_replaced, dtype=np.float64) - np.asarray(embed_text_original, dtype=np.float64)
    return compute_manipulation_direction(delta_image, delta_text)


def compute_manipulation_direction_from_pairs(
    image_input: Any,
    image_manipulated: Any,
    text_original: str,
    text_replaced: str,
    *,
    model: str = "openai:ViT-B-32",
    device: str | None = None,
) -> float:
    """Compute MD from input/manipulated images and original/replaced prompts."""
    image_feats = encode_clip_images(
        [image_input, image_manipulated],
        model=model,
        device=device,
    )
    text_feats = encode_clip_texts(
        [text_original, text_replaced],
        model=model,
        device=device,
    )
    return compute_manipulation_direction_from_embeddings(
        image_feats[0],
        image_feats[1],
        text_feats[0],
        text_feats[1],
    )


def compute_manipulation_direction_batch(
    pairs: list[dict[str, Any]],
    *,
    model: str = "openai:ViT-B-32",
    device: str | None = None,
) -> dict[str, float | list[float]]:
    """Compute mean MD over multiple manipulation pairs."""
    if not pairs:
        raise ValueError("pairs must be non-empty")
    scores = [
        compute_manipulation_direction_from_pairs(
            item["image_input"],
            item["image_manipulated"],
            item["text_original"],
            item["text_replaced"],
            model=model,
            device=device,
        )
        for item in pairs
    ]
    return {
        "md_mean": float(np.mean(scores)),
        "md_scores": scores,
    }


__all__ = [
    "compute_manipulation_direction",
    "compute_manipulation_direction_batch",
    "compute_manipulation_direction_from_embeddings",
    "compute_manipulation_direction_from_pairs",
]
