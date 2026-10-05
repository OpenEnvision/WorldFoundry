"""Accelerator-aware feature extraction shared by inference evaluators.

Responsibility
    Encode a batched image tensor through a caller-supplied encoder with OOM
    backoff, then (optionally) score pairwise cosine consistency.

Boundaries
    Eval-only. Not part of the denoise loop, not a model preprocess step, and
    not a CUDA Graph runner — batching is :func:`adaptive_batched_inference`.

Public surface
    :func:`batched_image_features`, :func:`mean_pairwise_cosine_distance`.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────
# Eval-only encoders — OOM-safe batching; encoder must be shape-stable
# ──────────────────────────────────────────────────────────────────────────

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from worldfoundry.core.execution.inference_runtime import adaptive_batched_inference, resolve_inference_batch_size


def batched_image_features(
    images: torch.Tensor,
    encoder: Callable[[torch.Tensor], torch.Tensor],
    *,
    device: torch.device | str | int,
    batch_size: int | None = None,
    default_batch_size: int = 32,
    scope: str = "image_features",
    normalize: bool = False,
    output_device: torch.device | str | int | None = "cpu",
    batch_size_buckets: Sequence[int] | None = None,
) -> torch.Tensor:
    """Encode images with OOM backoff and shape-stable tail padding."""

    if not isinstance(images, torch.Tensor) or images.ndim < 2 or len(images) == 0:
        raise ValueError("images must be a non-empty batched tensor")
    if not callable(encoder):
        raise TypeError("encoder must be callable")
    resolved_batch_size = (
        int(batch_size)
        if batch_size is not None
        else resolve_inference_batch_size(
            default_batch_size,
            device=device,
            scope=scope,
            maximum=len(images),
        )
    )

    # Pass the stable module/bound method directly. Wrapping it in a local
    # closure would prevent the compile wrapper from being retained across
    # videos and cause repeated Inductor compilation in throughput mode.
    features = adaptive_batched_inference(
        images,
        encoder,
        batch_size=resolved_batch_size,
        device=device,
        output_device=None,
        batch_size_buckets=batch_size_buckets,
        pad_to_batch_size=True,
        scope=scope,
    )
    if normalize:
        features = F.normalize(features, dim=-1, p=2)
    if output_device is not None:
        features = features.to(output_device)
    return features


def mean_pairwise_cosine_distance(features: Sequence[torch.Tensor]) -> float:
    """Return the mean cosine distance across all unordered feature pairs."""

    if len(features) < 2:
        return 0.0
    vectors = torch.stack([feature.flatten() for feature in features], dim=0)
    pair_indices = torch.triu_indices(
        len(vectors),
        len(vectors),
        offset=1,
        device=vectors.device,
    )
    similarities = F.cosine_similarity(
        vectors.index_select(0, pair_indices[0]),
        vectors.index_select(0, pair_indices[1]),
        dim=1,
        eps=1e-8,
    )
    return float((1.0 - similarities).mean().item())


__all__ = ["batched_image_features", "mean_pairwise_cosine_distance"]
