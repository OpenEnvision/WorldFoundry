# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Canonical silhouette comparison for returned object shapes."""

from __future__ import annotations

import math

import numpy as np

from .schemas import ObjectGeometryResult, ObjectTrackAnalysis

DEFAULT_CANONICAL_SIZE = 128
DEFAULT_CANONICAL_PADDING = 8


def canonicalize_object_mask(
    mask: np.ndarray,
    *,
    canvas_size: int = DEFAULT_CANONICAL_SIZE,
    padding: int = DEFAULT_CANONICAL_PADDING,
) -> np.ndarray | None:
    """Center a binary mask on a square canvas using uniform-only scaling.

    Translation and global image scale are removed, while aspect ratio and
    local silhouette deformation are preserved.
    """

    if mask.ndim != 2:
        raise ValueError("object shape mask must be 2-D")
    if canvas_size <= 0:
        raise ValueError("canonical canvas size must be positive")
    if padding < 0 or 2 * padding >= canvas_size:
        raise ValueError("canonical padding must leave a non-empty canvas interior")

    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return None
    y_coordinates, x_coordinates = np.where(binary)
    crop = binary[
        int(y_coordinates.min()) : int(y_coordinates.max()) + 1,
        int(x_coordinates.min()) : int(x_coordinates.max()) + 1,
    ]
    crop_height, crop_width = crop.shape
    usable_size = canvas_size - 2 * padding
    scale = usable_size / float(max(crop_height, crop_width))
    target_height = max(1, min(usable_size, int(round(crop_height * scale))))
    target_width = max(1, min(usable_size, int(round(crop_width * scale))))

    source_y = np.minimum(
        (np.arange(target_height) * crop_height // target_height),
        crop_height - 1,
    )
    source_x = np.minimum(
        (np.arange(target_width) * crop_width // target_width),
        crop_width - 1,
    )
    resized = crop[np.ix_(source_y, source_x)]
    canvas = np.zeros((canvas_size, canvas_size), dtype=bool)
    top = (canvas_size - target_height) // 2
    left = (canvas_size - target_width) // 2
    canvas[top : top + target_height, left : left + target_width] = resized
    return canvas


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("canonical masks must have the same shape")
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(first, second).sum()
    return float(intersection / union)


def analyze_object_geometry(
    track: ObjectTrackAnalysis,
    *,
    canonical_size: int = DEFAULT_CANONICAL_SIZE,
    canonical_padding: int = DEFAULT_CANONICAL_PADDING,
) -> ObjectGeometryResult:
    phase = track.phase
    frame_scores = {index: 0.0 for index in phase.return_indices}
    reference_indices: list[int] = []
    reference_masks: list[np.ndarray] = []
    reference_areas: list[float] = []
    for frame_index in phase.visible_indices:
        observation = track.observations[frame_index]
        if not observation.present or observation.mask is None or not observation.mask_valid:
            continue
        canonical = canonicalize_object_mask(
            observation.mask,
            canvas_size=canonical_size,
            padding=canonical_padding,
        )
        if canonical is None:
            continue
        reference_indices.append(frame_index)
        reference_masks.append(canonical)
        reference_areas.append(float(np.asarray(observation.mask, dtype=bool).sum()))

    returned_masks: dict[int, np.ndarray] = {}
    area_drifts_by_frame: dict[int, float] = {}
    for frame_index in phase.return_indices:
        observation = track.observations[frame_index]
        if not observation.present or observation.mask is None or not observation.mask_valid:
            continue
        canonical_candidate = canonicalize_object_mask(
            observation.mask,
            canvas_size=canonical_size,
            padding=canonical_padding,
        )
        if canonical_candidate is None:
            continue
        returned_masks[frame_index] = canonical_candidate
        if not reference_masks:
            continue
        similarities = [mask_iou(reference, canonical_candidate) for reference in reference_masks]
        best_reference = int(np.argmax(similarities))
        score = similarities[best_reference]
        candidate_area = float(np.asarray(observation.mask, dtype=bool).sum())
        area_ratio = max(
            candidate_area / reference_areas[best_reference],
            np.finfo(float).tiny,
        )
        frame_scores[frame_index] = score
        area_drifts_by_frame[frame_index] = abs(math.log(area_ratio))

    return_frame_count = len(phase.return_indices)
    n_return_masks = len(returned_masks)
    similarity = (
        sum(frame_scores[index] for index in returned_masks) / n_return_masks
        if n_return_masks and reference_masks and phase.is_scorable
        else 0.0
    )
    coverage = n_return_masks / return_frame_count if return_frame_count else 0.0
    all_area_drifts = list(area_drifts_by_frame.values())
    area_ratio_drift = float(np.median(all_area_drifts)) if all_area_drifts else 0.0
    revisit_scores: list[float] = []
    revisit_coverages: list[float] = []
    n_revisit_masks: list[int] = []
    revisit_area_ratio_drifts: list[float] = []
    for revisit_indices in phase.revisit_indices:
        valid_indices = [index for index in revisit_indices if index in returned_masks]
        n_masks = len(valid_indices)
        n_revisit_masks.append(n_masks)
        revisit_coverages.append(n_masks / len(revisit_indices) if revisit_indices else 0.0)
        revisit_scores.append(
            (
                sum(frame_scores[index] for index in valid_indices) / n_masks
                if n_masks and reference_masks and phase.is_scorable
                else 0.0
            )
        )
        revisit_drifts = [area_drifts_by_frame[index] for index in valid_indices if index in area_drifts_by_frame]
        revisit_area_ratio_drifts.append(float(np.median(revisit_drifts)) if revisit_drifts else 0.0)
    return ObjectGeometryResult(
        phase=phase,
        reference_frame_indices=tuple(reference_indices),
        frame_scores=frame_scores,
        similarity=similarity,
        geometry_coverage=coverage,
        area_ratio_drift=area_ratio_drift,
        n_reference_masks=len(reference_masks),
        n_return_masks=n_return_masks,
        revisit_scores=tuple(revisit_scores),
        revisit_coverages=tuple(revisit_coverages),
        n_revisit_masks=tuple(n_revisit_masks),
        revisit_area_ratio_drifts=tuple(revisit_area_ratio_drifts),
    )


__all__ = [
    "DEFAULT_CANONICAL_PADDING",
    "DEFAULT_CANONICAL_SIZE",
    "analyze_object_geometry",
    "canonicalize_object_mask",
    "mask_iou",
]
