# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# Adapted from Diffusers v0.40.0 modular_pipelines/minimax_h3/encoders.py.
"""Tokenize the released H3 keyframe/reference presentation without a chat template."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .media import ReferenceMedia


def sample_video_frames(frames: np.ndarray, temporal_patch: int) -> tuple[np.ndarray, list[float]]:
    """Sample a normalized reference at 2 fps; label temporal groups with their mean time."""
    indices = list(range(0, len(frames), 12))
    if len(indices) < temporal_patch:
        raise ValueError(f"HyperFlow video references need at least {(temporal_patch - 1) * 12 + 1} frames at 24 fps.")
    timestamps = [index / 2 for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % temporal_patch)
    grouped = [
        (timestamps[index] + timestamps[index + temporal_patch - 1]) / 2
        for index in range(0, len(timestamps), temporal_patch)
    ]
    return np.stack([frames[index] for index in indices]), grouped


def prepare_presentation(
    tokenizer: Any, processor: Any, prompt: str, references: list[ReferenceMedia]
) -> tuple[list[int], torch.Tensor, dict[str, torch.Tensor]]:
    """Batch vision features by modality while preserving request-order labels and tags."""
    merge = processor.image_processor.merge_size**2
    vision: dict[str, torch.Tensor] = {}
    images = [reference.image for reference in references if reference.kind == "image"]
    image_counts = []
    if images:
        features = processor.image_processor(images=images, return_tensors="pt")
        vision.update(pixel_values=features["pixel_values"], image_grid_thw=features["image_grid_thw"])
        image_counts = [int(grid.prod()) // merge for grid in features["image_grid_thw"]]
    videos = [reference.frames for reference in references if reference.kind == "video"]
    video_counts, video_times = [], []
    if videos:
        sampled = [sample_video_frames(frames, processor.video_processor.temporal_patch_size) for frames in videos]
        video_times = [timestamps for _, timestamps in sampled]
        features = processor.video_processor(
            videos=[frames for frames, _ in sampled], do_sample_frames=False, return_tensors="pt"
        )
        vision.update(pixel_values_videos=features["pixel_values_videos"], video_grid_thw=features["video_grid_thw"])
        video_counts = [int(grid[1]) * int(grid[2]) // merge for grid in features["video_grid_thw"]]
        for timestamps, grid in zip(video_times, features["video_grid_thw"], strict=True):
            if int(grid[0]) != len(timestamps):
                raise ValueError("HyperFlow video presentation and Qwen temporal groups disagree.")
    ids, tags = [], []

    def emit_text(value: str) -> None:
        tokens = tokenizer(value, add_special_tokens=False)["input_ids"]
        ids.extend(tokens)
        tags.extend([1] * len(tokens))

    def emit_vision(pad: str, count: int) -> None:
        tokens = (
            [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [tokenizer.convert_tokens_to_ids(pad)] * count
            + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        )
        ids.extend(tokens)
        tags.extend([0] * len(tokens))

    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            emit_text(f"<Audio {counts['audio']}>: ")
        if reference.kind == "image":
            counts["image"] += 1
            emit_text(f"<Picture {counts['image']}>: ")
            emit_vision("<|image_pad|>", image_counts[counts["image"] - 1])
        elif reference.kind == "video":
            counts["video"] += 1
            emit_text(f"<Video {counts['video']}>: ")
            for timestamp in video_times[counts["video"] - 1]:
                emit_text(f"<{timestamp:.1f} seconds>")
                emit_vision("<|video_pad|>", video_counts[counts["video"] - 1])
    emit_text(prompt)
    return ids, torch.tensor(tags, dtype=torch.long), vision
