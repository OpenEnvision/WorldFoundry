# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""DreamDojo student's zero-padded image/history windows using the shared Wan VAE."""

from __future__ import annotations

import numpy as np
import torch

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest, LatentInitialization
from .component import Cosmos25VideoCodec, build_cosmos25_video_codec


class DreamDojoStudentCodec(Cosmos25VideoCodec):
    """Encode 1 or 9 observed pixels into a 13-frame window without repeating its tail."""

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LatentInitialization:
        frames = request.inputs.get("conditioning_frames")
        if not isinstance(frames, torch.Tensor) or frames.dtype != torch.uint8:
            raise ValueError("DreamDojo student conditioning_frames must be uint8 [1,3,1|9,H,W]")
        if frames.ndim != 5 or tuple(frames.shape[:2]) != (1, 3) or frames.shape[2] not in {1, 9}:
            raise ValueError("DreamDojo student requires one initial frame or nine continuation frames")
        if tuple(frames.shape[-2:]) != (request.height, request.width) or request.num_frames != 13:
            raise ValueError("DreamDojo student requires a 13-frame window matching the configured resolution")
        count = (frames.shape[2] - 1) // 4 + 1
        pixels = torch.cat(
            (frames, frames.new_zeros((1, 3, 13 - frames.shape[2], request.height, request.width))), dim=2
        )
        pixels = pixels.to(device=device, dtype=torch.float32).div(127.5).sub(1).to(dtype)
        observation = self.encode(pixels).to(device=device, dtype=dtype)
        shape = tuple(observation.shape)
        noise = torch.from_numpy(
            np.random.RandomState(generator.initial_seed() & 0xFFFFFFFF).standard_normal(shape).astype(np.float32)
        ).to(device=device, dtype=dtype)
        return LatentInitialization(
            latents=noise,
            conditioning={
                "condition_latents": observation,
                "start_index": count,
                "condition_mask": torch.zeros((1, 1, shape[2], shape[3], shape[4]), device=device, dtype=dtype),
                "padding_mask": torch.zeros((1, 1, request.height, request.width), device=device, dtype=dtype),
            },
        )


def build_dreamdojo_student_codec(context: ComponentBuildContext) -> DreamDojoStudentCodec:
    """Reuse canonical Wan loading and decoding with student window preparation."""
    codec = build_cosmos25_video_codec(context)
    return DreamDojoStudentCodec(
        codec.vae,
        device=codec.device,
        dtype=codec.dtype,
        tiled=codec.tiled,
        tile_size=codec.tile_size,
        tile_stride=codec.tile_stride,
    )
