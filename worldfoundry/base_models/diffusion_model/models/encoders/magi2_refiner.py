# SPDX-FileCopyrightText: Copyright (c) 2025 SandAI
# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""MAGI-2 refiner token packing from SandAI MAGI-2-preview.

Adapted from f68a0f9bbccbea56e0177a9bd912abc3b18ffe61,
inference/pipeline/refiner_data_proxy.py.

The released video-only refiner uses 8×4×4 token blocks with a radius-two
spatiotemporal window. Text queries attend globally; every video query sees
all text tokens. Coordinates retain the unpermuted latent grid positions.
"""

from __future__ import annotations

from itertools import product

import torch
from einops import rearrange

from ..networks.magi2.preview_layers import VarlenHandler, get_coords
from ..networks.magi2.refiner_dit import LocalAttnHandler


class Magi2RefinerDataProxy:
    """Pack one ``[1,48,T,H,W]`` latent and ``[1,S,D]`` text tensor."""

    def __init__(self, shape: tuple[int, int, int], *, device: torch.device) -> None:
        self.shape = shape
        frames, height, width = shape
        sizes = (8, 4, 4)
        grid = tuple((length + size - 1) // size for length, size in zip(shape, sizes))
        indices = torch.arange(frames * height * width, device=device).reshape(shape)
        blocks = []
        bounds = [0]
        for t, h, w in product(*(range(length) for length in grid)):
            block = indices[t * 8 : (t + 1) * 8, h * 4 : (h + 1) * 4, w * 4 : (w + 1) * 4].flatten()
            blocks.append(block)
            bounds.append(bounds[-1] + block.numel())
        self.order = torch.cat(blocks)
        self.restore = torch.argsort(self.order)
        self.video_tokens = bounds[-1]
        q_ranges: list[list[int]] = []
        k_ranges: list[list[int]] = []
        for index, coords in enumerate(product(*(range(length) for length in grid))):
            neighbours = (
                range(max(0, coordinate - 2), min(length, coordinate + 3)) for coordinate, length in zip(coords, grid)
            )
            for t, h, w in product(*neighbours):
                neighbour = (t * grid[1] + h) * grid[2] + w
                q_ranges.append(bounds[index : index + 2])
                k_ranges.append(bounds[neighbour : neighbour + 2])
        self.q_ranges = torch.tensor(q_ranges, device=device, dtype=torch.int32)
        self.k_ranges = torch.tensor(k_ranges, device=device, dtype=torch.int32)
        self.video_coords = get_coords(shape, shape, device=device, dtype=torch.float32)[self.order]

    def build_model_inputs(self, latent: torch.Tensor, context: torch.Tensor) -> dict[str, object]:
        """Return the released block-ordered video/text stream and attention ranges."""
        if latent.shape[0] != 1 or tuple(latent.shape[2:]) != self.shape or context.shape[0] != 1:
            raise ValueError("MAGI-2 refiner packing requires one sample with its configured latent geometry")
        video = rearrange(latent.float(), "1 c t h w -> (t h w) c")[self.order]
        text = context[0].float()
        width = max(video.shape[1], text.shape[1])
        x = torch.cat(
            (
                torch.nn.functional.pad(video, (0, width - video.shape[1])),
                torch.nn.functional.pad(text, (0, width - text.shape[1])),
            )
        )
        text_count = text.shape[0]
        total = self.video_tokens + text_count
        text_coords = get_coords(
            (text_count, 1, 1), (1, 1, 1), offset_thw=(-text_count, 0, 0), device=x.device, dtype=torch.float32
        )
        q_ranges = torch.cat(
            (self.q_ranges, self.q_ranges.new_tensor([[0, self.video_tokens], [self.video_tokens, total]]))
        )
        k_ranges = torch.cat((self.k_ranges, self.k_ranges.new_tensor([[self.video_tokens, total], [0, total]])))
        cu = torch.tensor([0, total], device=x.device, dtype=torch.int32)
        return {
            "x": x,
            "coords_mapping": torch.cat((self.video_coords, text_coords)),
            "modality_mapping": torch.cat(
                (
                    torch.zeros(self.video_tokens, device=x.device, dtype=torch.long),
                    torch.full((text_count,), 2, device=x.device, dtype=torch.long),
                )
            ),
            "varlen_handler": VarlenHandler(cu, cu, total, total),
            "local_attn_handler": LocalAttnHandler(
                q_ranges, k_ranges, total, total, torch.zeros(len(q_ranges), device=x.device, dtype=torch.int32)
            ),
        }

    def unpack_output(self, output: torch.Tensor) -> torch.Tensor:
        """Restore the original ``[1,48,T,H,W]`` video latent token order."""
        frames, height, width = self.shape
        return rearrange(
            output[: self.video_tokens, :48][self.restore], "(t h w) c -> 1 c t h w", t=frames, h=height, w=width
        ).contiguous()
