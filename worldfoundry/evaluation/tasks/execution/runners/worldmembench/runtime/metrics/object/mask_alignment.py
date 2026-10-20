# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Align a source-frame object mask to a generated video's first frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskAlignment:
    method: str
    score: float
    mask: np.ndarray


def _resize_pair(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    return resized_image, resized_mask


def _center_crop(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    destination_aspect: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    source_aspect = width / float(height)
    if source_aspect > destination_aspect:
        crop_width = max(1, min(width, int(round(height * destination_aspect))))
        left = (width - crop_width) // 2
        return (
            image[:, left : left + crop_width],
            mask[:, left : left + crop_width],
        )
    crop_height = max(1, min(height, int(round(width / destination_aspect))))
    top = (height - crop_height) // 2
    return (
        image[top : top + crop_height],
        mask[top : top + crop_height],
    )


def _letterbox_pair(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_height, source_width = image.shape[:2]
    scale = min(width / float(source_width), height / float(source_height))
    scaled_width = max(1, int(round(source_width * scale)))
    scaled_height = max(1, int(round(source_height * scale)))
    scaled_image, scaled_mask = _resize_pair(
        image,
        mask,
        width=scaled_width,
        height=scaled_height,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    mask_canvas = np.zeros((height, width), dtype=bool)
    left = (width - scaled_width) // 2
    top = (height - scaled_height) // 2
    canvas[top : top + scaled_height, left : left + scaled_width] = scaled_image
    mask_canvas[top : top + scaled_height, left : left + scaled_width] = scaled_mask
    return canvas, mask_canvas


def _alignment_score(candidate: np.ndarray, generated: np.ndarray) -> float:
    difference = np.abs(candidate.astype(np.float32) - generated.astype(np.float32))
    return float(np.clip(1.0 - difference.mean() / 255.0, 0.0, 1.0))


def align_mask_to_generated_frame(
    condition_rgb: np.ndarray,
    source_mask: np.ndarray,
    generated_frame_rgb: np.ndarray,
) -> MaskAlignment:
    if condition_rgb.ndim != 3 or condition_rgb.shape[2] != 3:
        raise ValueError("condition image must be an RGB image")
    if source_mask.shape != condition_rgb.shape[:2]:
        raise ValueError("target object mask must match the condition image size")
    if generated_frame_rgb.ndim != 3 or generated_frame_rgb.shape[2] != 3:
        raise ValueError("generated first frame must be an RGB image")

    height, width = generated_frame_rgb.shape[:2]
    resized = _resize_pair(condition_rgb, source_mask, width=width, height=height)
    cropped_source = _center_crop(
        condition_rgb,
        source_mask,
        destination_aspect=width / float(height),
    )
    center_cropped = _resize_pair(
        *cropped_source,
        width=width,
        height=height,
    )
    letterboxed = _letterbox_pair(
        condition_rgb,
        source_mask,
        width=width,
        height=height,
    )
    candidates = (
        ("resize", resized),
        ("center_crop_resize", center_cropped),
        ("letterbox_resize", letterboxed),
    )
    method, (_, aligned_mask), score = max(
        ((name, pair, _alignment_score(pair[0], generated_frame_rgb)) for name, pair in candidates),
        key=lambda value: value[2],
    )
    return MaskAlignment(method=method, score=score, mask=aligned_mask)
