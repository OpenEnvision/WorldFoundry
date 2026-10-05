"""Fréchet Inception Distance (FID) and FID-family variants.

Canonical integration for torch-fidelity FID, CLIP-FID, SwAV-FID, video-frame FID,
and SceneFID (object-crop protocol). Registry aliases cover variant ids; use
``feature_extractor`` or the deprecated ``compute_*`` helpers for each variant.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

compute_distribution_metrics = lazy_export(f"{__name__}.compute", "compute_distribution_metrics", owner=__name__)
compute_fid = lazy_export(f"{__name__}.compute", "compute_fid", owner=__name__)
compute_paired_fid_kid = lazy_export(f"{__name__}.compute", "compute_paired_fid_kid", owner=__name__)
summarize_distribution_metrics = lazy_export(f"{__name__}.compute", "summarize_distribution_metrics", owner=__name__)
compute_scene_fid = lazy_export(f"{__name__}.scene", "compute_scene_fid", owner=__name__)
extract_object_crops = lazy_export(f"{__name__}.scene", "extract_object_crops", owner=__name__)
compute_swav_fid = lazy_export(f"{__name__}.swav", "compute_swav_fid", owner=__name__)

_FID_TAGS = ("distribution", "image_generation", "fid_family")

METRIC_ID = "fid"
ALIASES = (
    "frechet-inception-distance",
    "clip-fid",
    "clip_fid",
    "fid-vid",
    "fid_vid",
    "fvid",
    "swav-fid",
    "swav_fid",
    "scene-fid",
    "scene_fid",
    "object-crop-fid",
    "object_crop_fid",
)
HIGHER_IS_BETTER = False
FAMILY = "distribution"
TAGS = _FID_TAGS

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description=(
        "Fréchet distance between reference and generated image sets. "
        "Variants: inception (default), CLIP (feature_extractor=clip-vit-*), "
        "SwAV (feature_extractor=swav-resnet50), video frames (frame dirs), "
        "SceneFID (compute_scene_fid / object-crop protocol)."
    ),
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)

compute = compute_fid


def compute_clip_fid(
    reference: str | Path | Sequence[str | Path],
    generated: str | Path | Sequence[str | Path],
    *,
    batch_size: int = 64,
    cuda: bool = True,
    clip_model: str = "clip-vit-b-32",
    **kwargs: Any,
) -> float:
    """Deprecated alias — use :func:`compute_fid` with ``feature_extractor=clip_model``."""
    return compute_fid(
        reference,
        generated,
        batch_size=batch_size,
        cuda=cuda,
        feature_extractor=clip_model,
        **kwargs,
    )


def compute_fid_vid(
    reference_frames_dir: str | Path,
    generated_frames_dir: str | Path,
    *,
    batch_size: int = 64,
    cuda: bool = True,
    **kwargs: Any,
) -> float:
    """Deprecated alias — FID on extracted video frame directories."""
    return compute_fid(
        reference_frames_dir,
        generated_frames_dir,
        batch_size=batch_size,
        cuda=cuda,
        **kwargs,
    )


__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_clip_fid",
    "compute_distribution_metrics",
    "compute_fid",
    "compute_fid_vid",
    "compute_paired_fid_kid",
    "compute_scene_fid",
    "compute_swav_fid",
    "extract_object_crops",
    "summarize_distribution_metrics",
]
