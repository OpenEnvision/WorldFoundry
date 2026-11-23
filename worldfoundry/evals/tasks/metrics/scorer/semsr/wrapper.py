"""Semantic Shift Rate (SemSR) from arXiv:2402.07562."""

from __future__ import annotations

from typing import Any

import numpy as np

from worldfoundry.evals.tasks.metrics._shared.clip_embed import (
    clip_image_text_cosine,
    cosine_similarity_vectors,
)


def compute_sem_shift(
    sim_ust: float,
    sim_ori: float,
) -> float:
    """Raw semantic shift: Sim_sem(I_ust) - Sim_sem(I_ori)."""
    return float(sim_ust - sim_ori)


def compute_semsr(
    sim_ust: float,
    sim_ori: float,
    sim_tar: float,
    *,
    eps: float = 1e-8,
) -> float:
    """Normalized Semantic Shift Rate (Eq. 9 in arXiv:2402.07562)."""
    denom = sim_tar - sim_ori
    if abs(denom) <= eps:
        raise ValueError("sim_tar and sim_ori are too close for SemSR normalization")
    return float((sim_ust - sim_ori) / denom)


def compute_semsr_from_similarities(
    similarities: dict[str, float],
    *,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute SemSR from precomputed CLIP similarities to the target semantic text."""
    required = ("ust", "ori", "tar")
    missing = [key for key in required if key not in similarities]
    if missing:
        raise ValueError(f"missing similarity keys: {missing}")
    sim_ust = float(similarities["ust"])
    sim_ori = float(similarities["ori"])
    sim_tar = float(similarities["tar"])
    return {
        "semsr": compute_semsr(sim_ust, sim_ori, sim_tar, eps=eps),
        "sem_shift": compute_sem_shift(sim_ust, sim_ori),
        "sim_ust": sim_ust,
        "sim_ori": sim_ori,
        "sim_tar": sim_tar,
    }


def compute_semsr_from_images(
    image_ust: Any,
    image_ori: Any,
    image_tar: Any,
    semantic_text: str,
    *,
    model: str = "openai:ViT-B-32",
    device: str | None = None,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute SemSR from three generated images and one fixed semantic sentence."""
    sim_ust = clip_image_text_cosine(image_ust, semantic_text, model=model, device=device)
    sim_ori = clip_image_text_cosine(image_ori, semantic_text, model=model, device=device)
    sim_tar = clip_image_text_cosine(image_tar, semantic_text, model=model, device=device)
    return compute_semsr_from_similarities(
        {"ust": sim_ust, "ori": sim_ori, "tar": sim_tar},
        eps=eps,
    )


def compute_semsr_from_embeddings(
    embed_ust: np.ndarray,
    embed_ori: np.ndarray,
    embed_tar: np.ndarray,
    embed_sem: np.ndarray,
    *,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute SemSR from precomputed CLIP image/text embeddings."""
    sim_ust = cosine_similarity_vectors(embed_ust, embed_sem)
    sim_ori = cosine_similarity_vectors(embed_ori, embed_sem)
    sim_tar = cosine_similarity_vectors(embed_tar, embed_sem)
    return compute_semsr_from_similarities(
        {"ust": sim_ust, "ori": sim_ori, "tar": sim_tar},
        eps=eps,
    )


__all__ = [
    "compute_sem_shift",
    "compute_semsr",
    "compute_semsr_from_embeddings",
    "compute_semsr_from_images",
    "compute_semsr_from_similarities",
]
