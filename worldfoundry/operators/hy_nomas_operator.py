# Copyright 2026 The Hy-Nomas Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical camera trajectory adaptation for Hy-Nomas realtime controls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def realtime_camera(
    interactions: list[str],
    frames: int,
    *,
    pose: Any = None,
    segments: list[Mapping[str, Any]] | None = None,
    fps: int = 24,
) -> tuple[torch.Tensor, Any]:
    """Sample held keys and reuse the shared OpenCV camera-to-world integrator."""
    import numpy as np

    from worldfoundry.core.geometry.trajectory import rollout_wasd_camera_actions

    tokens = dict(
        zip(
            ("forward", "backward", "left", "right", "camera_up", "camera_down", "camera_l", "camera_r"),
            "wsad ikjl".replace(" ", ""),
            strict=False,
        )
    )

    def keys(values: Any) -> str:
        return "".join(tokens.get(value, value if value in "wasdijkl" else "") for value in values)

    held = [keys(interactions)] * frames
    if segments:
        rows, end = [], 0.0
        for segment in segments:
            end += float(segment.get("duration", 0))
            rows.append((end, keys(segment.get("keys", ()))))
        held = [next((value for end, value in rows if (i + 0.5) / fps < end), "") for i in range(frames)]
    local = rollout_wasd_camera_actions(held, num_frames=frames + 1)[1:]
    anchor = np.eye(4, dtype=np.float32) if pose is None else pose
    poses = anchor[None] @ local
    return torch.from_numpy(poses).unsqueeze(0), poses[-1].copy()
