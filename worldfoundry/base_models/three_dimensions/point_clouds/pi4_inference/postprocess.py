# SPDX-FileCopyrightText: 2026 Physical Intelligence Team, Shanghai AI Laboratory
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Non-mutating display-confidence postprocessing for Pi4 geometry outputs.

Adapted from ``yyfz/Pi4`` revision ``32dfbea9ce6268ea17446208a804710df9475c32``
(``pi4/postprocess.py``). Upstream code and weights are licensed under CC BY-NC 4.0.
"""

from __future__ import annotations

import torch

from .geometry import depth_normal_edge


def geometry_confidence(points: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
    """Copy geometry confidence, zeroing invalid pixels and depth-normal edges.

    Args:
        points: Camera-space pointmap tensor of shape ``[H, W, 3]``.
        confidence: Geometry confidence probability tensor of shape ``[H, W]``.

    Returns:
        Filtered confidence tensor of shape ``[H, W]``.
    """
    valid = torch.isfinite(points).all(dim=-1) & (points[..., 2] > 0) & torch.isfinite(confidence) & (confidence >= 0)
    edges = depth_normal_edge(points, valid)
    return confidence.masked_fill(~valid | edges, 0)
