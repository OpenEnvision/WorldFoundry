# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Reference DA3 return selection and spherical nearest-surface reconstruction."""

from dataclasses import dataclass

import numpy as np


def rotation_distance_deg(first, second) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


@dataclass(frozen=True)
class ReturnCandidate:
    slot_id: int
    axis: str
    return_index: int
    rotation_error_deg: float
    returned: bool


def select_returns(
    poses, protocol, *, rotation_threshold_deg=30.0, translation_threshold=0.15, return_buffer_frames=30
):
    if rotation_threshold_deg <= 0 or return_buffer_frames < 0:
        raise ValueError("rotation threshold must be positive and return buffer non-negative")
    results = []
    reference = poses[0]
    for slot in protocol.slots:
        window = protocol.phases[slot.return_phase]
        # Extend only to the right; a left extension can steal the previous return.
        last = min(window.end + return_buffer_frames, len(poses) - 1)
        candidates = np.arange(window.start, last + 1, dtype=np.int64)
        rotations = np.asarray(
            [rotation_distance_deg(reference[:3, :3], poses[i, :3, :3]) for i in candidates], dtype=np.float64
        )
        translation_gated = slot.axis in {"lateral", "longitudinal", "translation"}
        if translation_gated:
            positions = poses[:, :3, 3]
            span = float(2.0 * np.linalg.norm(positions - positions.mean(axis=0), axis=1).max())
            translations = np.asarray(
                [np.linalg.norm(reference[:3, 3] - poses[i, :3, 3]) / max(span, 1e-8) for i in candidates],
                dtype=np.float64,
            )
            best = int(np.argmin(rotations / rotation_threshold_deg + translations / max(translation_threshold, 1e-8)))
        else:
            best = int(np.argmin(rotations))
        error = float(rotations[best])
        returned = error <= rotation_threshold_deg and (
            not translation_gated or float(translations[best]) <= translation_threshold
        )
        results.append(ReturnCandidate(slot.slot_id, slot.axis, int(candidates[best]), error, returned))
    return tuple(results)


def departure_evidence(poses, window, threshold_deg):
    errors = [rotation_distance_deg(poses[0, :3, :3], poses[i, :3, :3]) for i in window.indices]
    best = int(np.argmax(errors))
    maximum = float(errors[best])
    return {
        "departed": maximum >= threshold_deg,
        "departure_peak_index": window.start + best,
        "departure_rotation_deg": maximum,
    }


def window_indices(available, start, end, max_count=8):
    values = np.asarray(available, dtype=np.int64)
    values = values[(values >= start) & (values <= end)]
    if len(values) > max_count:
        values = values[np.asarray(np.round(np.linspace(0, len(values) - 1, max_count)), dtype=np.int64)]
    return tuple(int(value) for value in values)


def spherical_range_field(
    geometry,
    indices,
    anchor_rotation,
    *,
    semantic_masks=None,
    angular_width=128,
    confidence_quantile=0.5,
    near_depth_percentile=5.0,
    far_depth_percentile=85.0,
    max_points_per_frame=5000,
):
    if angular_width < 8 or angular_width % 2:
        raise ValueError("angular_width must be an even integer >= 8")
    if not 0 <= confidence_quantile < 1 or not 0 <= near_depth_percentile < far_depth_percentile <= 100:
        raise ValueError("invalid confidence or depth quantiles")
    if max_points_per_frame < 1:
        raise ValueError("max_points_per_frame must be positive")
    height = angular_width // 2
    flat = np.full(angular_width * height, np.inf, dtype=np.float64)
    by_index = geometry.index_to_position
    anchor_inverse = np.asarray(anchor_rotation, dtype=np.float64).T
    for index in indices:
        position = by_index[index]
        depth = np.asarray(geometry.depths[position], dtype=np.float64)
        confidence = np.asarray(geometry.confidence[position], dtype=np.float64)
        valid = np.isfinite(depth) & np.isfinite(confidence) & (depth > 0)
        if semantic_masks is not None:
            valid &= semantic_masks[index]
        if not valid.any():
            continue
        threshold = float(np.quantile(confidence[valid], confidence_quantile))
        valid &= confidence >= threshold
        low, high = np.percentile(depth[valid], [near_depth_percentile, far_depth_percentile])
        valid &= (depth >= low) & (depth <= high)
        ys, xs = np.where(valid)
        if not len(xs):
            continue
        if len(xs) > max_points_per_frame:
            selected = np.asarray(np.round(np.linspace(0, len(xs) - 1, max_points_per_frame)), dtype=np.int64)
            ys, xs = ys[selected], xs[selected]
        pixels = np.stack([xs.astype(np.float64), ys.astype(np.float64), np.ones(len(xs))], axis=0)
        rays = np.linalg.inv(geometry.intrinsics[position]) @ pixels
        ranges = np.linalg.norm(rays, axis=0) * depth[ys, xs]
        directions = anchor_inverse @ geometry.c2w[position, :3, :3] @ rays
        directions /= np.maximum(np.linalg.norm(directions, axis=0), 1e-12)
        azimuth = np.arctan2(directions[0], directions[2])
        elevation = np.arcsin(np.clip(directions[1], -1.0, 1.0))
        x = np.floor((azimuth + np.pi) * angular_width / (2.0 * np.pi)).astype(np.int64)
        y = np.floor((elevation + np.pi / 2.0) * height / np.pi).astype(np.int64)
        bins = np.clip(y, 0, height - 1) * angular_width + np.clip(x, 0, angular_width - 1)
        np.minimum.at(flat, bins, np.log(np.maximum(ranges, 1e-12)))
    values = flat.reshape(height, angular_width)
    values[~np.isfinite(values)] = np.nan
    return values


def compare_range_fields(reference, revisit):
    ref_valid, rev_valid = np.isfinite(reference), np.isfinite(revisit)
    common = ref_valid & rev_valid
    union_bins, common_bins = int((ref_valid | rev_valid).sum()), int(common.sum())
    if not union_bins or not common_bins:
        return {"reconstruction_loss": 1.0, "common_bins": common_bins, "union_bins": union_bins}
    offset = float(np.median(reference[common] - revisit[common]))
    residual = np.abs(reference[common] - (revisit[common] + offset))
    similarity = float(np.exp(-residual).sum() / union_bins)
    similarity = float(np.clip(similarity, 0.0, 1.0))
    return {"reconstruction_loss": 1.0 - similarity, "common_bins": common_bins, "union_bins": union_bins}
