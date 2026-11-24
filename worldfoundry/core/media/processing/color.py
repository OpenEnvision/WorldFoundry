# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""Torch BT.709 limited-range RGB-to-I420 conversion.

Derived from Lightricks/LTX-2 at a95ab856 (LTX-2.x Community License;
see THIRD-PARTY-NOTICES). Conversion retains the input arithmetic dtype,
2x2 chroma averaging and truncation to 8-bit code values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def rgb_to_i420_bt709(frames: torch.Tensor) -> torch.Tensor:
    """Convert RGB ``[0,1]`` ``(*,3,H,W)`` to uint8 ``(*,H*3/2,W)`` I420.

    Height and width must be even. Matrix multiplies are frame-chunked to
    keep their flattened pixel dimension within cuBLAS's int32 limit.
    """
    import torch

    if frames.ndim < 3 or frames.shape[-3] != 3:
        raise ValueError("BT.709 conversion requires RGB shaped (*,3,H,W)")
    height, width = frames.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError("I420 conversion requires even height and width")
    matrix = torch.tensor(
        [[0.2126, 0.7152, 0.0722], [-0.1146, -0.3854, 0.5], [0.5, -0.4542, -0.0458]],
        dtype=torch.float32,
    ).to(device=frames.device, dtype=frames.dtype)
    flattened = frames.reshape(-1, 3, height, width)
    limit = (2**31 - 1) // height // width
    if limit < 1:
        raise ValueError("one RGB frame exceeds the cuBLAS int32 GEMM limit")
    chunk_size = max(1, limit - 1)
    chunks = []
    for start in range(0, flattened.shape[0], chunk_size):
        pixels = flattened[start : start + chunk_size].movedim(-3, -1)
        yuv = (pixels.flatten(-3, -2) @ matrix.T).unflatten(-2, (height, width)).movedim(-1, -3)
        y = yuv[:, :1]
        uv = torch.nn.functional.avg_pool2d(yuv[:, 1:3].contiguous(), kernel_size=2, stride=2)
        y.mul_(219).add_(16)
        uv.mul_(224).add_(128)
        packed = torch.cat([y[:, 0], uv.reshape(-1, height // 2, width)], dim=-2)
        chunks.append(packed.clamp_(0, 255).to(torch.uint8))
    return torch.cat(chunks, dim=0).reshape(*frames.shape[:-3], height * 3 // 2, width)
