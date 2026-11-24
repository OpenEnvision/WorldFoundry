"""Object-wise consistency from arXiv:2304.13427 (location-aware T2I)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np

from worldfoundry.evals.tasks.metrics._shared.bbox import bbox_iou


class _ConsistencyResult(TypedDict):
    object_wise_iou_mean: float
    object_wise_success_rate: float
    object_wise_ious: list[float]


class _ConsistencyBatchResult(TypedDict):
    object_wise_iou_mean: float
    object_wise_success_rate: float
    per_image: list[_ConsistencyResult]


def _normalize_label(label: str) -> str:
    return str(label).strip().lower()


def compute_object_wise_iou(
    guidance_boxes: Sequence[tuple[Sequence[float], str]],
    detections: Sequence[tuple[Sequence[float], str]],
) -> list[float]:
    """Maximum IoU per guidance object (YOLOR protocol in arXiv:2304.13427)."""
    ious: list[float] = []
    for box, label in guidance_boxes:
        target = _normalize_label(label)
        best = 0.0
        for det_box, det_label in detections:
            if _normalize_label(det_label) != target:
                continue
            best = max(best, bbox_iou(box, det_box))
        ious.append(best)
    return ious


def compute_object_wise_consistency(
    guidance_boxes: Sequence[tuple[Sequence[float], str]],
    detections: Sequence[tuple[Sequence[float], str]],
    *,
    iou_threshold: float = 0.5,
) -> _ConsistencyResult:
    """Compute mean IoU and success rate R_suc for object-wise guidance."""
    if not guidance_boxes:
        raise ValueError("guidance_boxes must be non-empty")
    ious = compute_object_wise_iou(guidance_boxes, detections)
    successes = [1.0 if value > iou_threshold else 0.0 for value in ious]
    return {
        "object_wise_iou_mean": float(np.mean(ious)),
        "object_wise_success_rate": float(np.mean(successes)),
        "object_wise_ious": ious,
    }


def compute_object_wise_consistency_batch(
    samples: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> _ConsistencyBatchResult:
    """Aggregate object-wise consistency over multiple generated images."""
    if not samples:
        raise ValueError("samples must be non-empty")
    per_image = [
        compute_object_wise_consistency(
            item["guidance_boxes"],
            item["detections"],
            iou_threshold=iou_threshold,
        )
        for item in samples
    ]
    return {
        "object_wise_iou_mean": float(np.mean([row["object_wise_iou_mean"] for row in per_image])),
        "object_wise_success_rate": float(np.mean([row["object_wise_success_rate"] for row in per_image])),
        "per_image": per_image,
    }


__all__ = [
    "compute_object_wise_consistency",
    "compute_object_wise_consistency_batch",
    "compute_object_wise_iou",
]
