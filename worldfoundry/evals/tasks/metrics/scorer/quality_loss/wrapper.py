"""Quality Loss from IEEE Access 2023 prompt-optimization paper."""

from __future__ import annotations

from typing import Any

import numpy as np

from worldfoundry.evals.tasks.metrics._shared.clip_embed import clip_image_text_cosine


def compute_text_presence_probability(
    has_text_flags: list[bool] | np.ndarray,
) -> float:
    """Probability of characters/text appearing in generated images (PC)."""
    flags = np.asarray(has_text_flags, dtype=np.float64)
    if flags.size == 0:
        raise ValueError("has_text_flags must be non-empty")
    return float(np.mean(flags))


def compute_quality_loss(
    clip_score: float,
    text_presence_probability: float,
) -> float:
    """Quality Loss = CLIPScore × PC (IEEE Access 2023.3348778)."""
    return float(clip_score * text_presence_probability)


def compute_quality_loss_from_batch(
    clip_scores: list[float] | np.ndarray,
    has_text_flags: list[bool] | np.ndarray,
) -> dict[str, float]:
    """Aggregate Quality Loss over a generated image batch."""
    scores = np.asarray(clip_scores, dtype=np.float64)
    flags = np.asarray(has_text_flags, dtype=np.float64)
    if scores.shape != flags.shape:
        raise ValueError("clip_scores and has_text_flags must have the same length")
    if scores.size == 0:
        raise ValueError("batch must be non-empty")
    per_sample = scores * flags
    return {
        "quality_loss_mean": float(np.mean(per_sample)),
        "clip_score_mean": float(np.mean(scores)),
        "text_presence_probability": float(np.mean(flags)),
        "quality_loss_scores": per_sample.tolist(),
    }


def compute_quality_loss_for_pair(
    image: Any,
    prompt: str,
    *,
    has_text: bool,
    model: str = "openai:ViT-B-32",
    device: str | None = None,
) -> dict[str, float]:
    """Compute Quality Loss for one image/prompt pair."""
    clip_score = clip_image_text_cosine(image, prompt, model=model, device=device)
    pc = 1.0 if has_text else 0.0
    return {
        "quality_loss": compute_quality_loss(clip_score, pc),
        "clip_score": clip_score,
        "text_presence_probability": pc,
    }


__all__ = [
    "compute_quality_loss",
    "compute_quality_loss_for_pair",
    "compute_quality_loss_from_batch",
    "compute_text_presence_probability",
]
