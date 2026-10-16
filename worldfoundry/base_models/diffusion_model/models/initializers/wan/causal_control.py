# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Wan2.2 causal control-video and independent KV-prefix initialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest, LatentEncoder, LatentInitialization
from ....loaders import NativeCheckpointResolver
from ...encoders.wan.control_video import resize_control_frame


class WanCausalControlInitializer:
    """Encode low-poly RGB and a separate identity prefix for offline rollout."""

    def __init__(self, sink_image: str | Path) -> None:
        self.sink_image = Path(sink_image)

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LatentInitialization:
        del request, generator, device, dtype
        raise RuntimeError("causal control-video initialization requires a latent_encoder binding")

    @torch.no_grad()
    def initialize_with_encoder(
        self,
        request: DiffusionRequest,
        *,
        latent_encoder: LatentEncoder,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LatentInitialization:
        from PIL import Image, ImageOps

        del generator
        pixels = request.inputs.get("video")
        if not isinstance(pixels, Tensor) or pixels.ndim != 5 or tuple(pixels.shape[:2]) != (1, 3):
            raise ValueError("causal control-video initialization requires signed BCTHW RGB")
        if request.batch_size != 1 or pixels.shape[2] != request.num_frames:
            raise ValueError("causal control-video input must match one request's frame count")
        if (request.height, request.width) != tuple(pixels.shape[-2:]):
            raise ValueError("causal control-video dimensions must match the request")
        if request.sampling.guidance_scale != 1.0:
            raise ValueError("Matrix-Game 4 released inference requires guidance_scale=1")
        latent_frames = 1 + (request.num_frames + 2) // 4
        if latent_frames > 8192:
            raise ValueError("Matrix-Game 4 supports at most 8192 generated latent chunks")
        padded_frames = (latent_frames - 1) * 4 + 1
        if padded_frames > pixels.shape[2]:
            pixels = torch.cat(
                (pixels, pixels[:, :, -1:].expand(-1, -1, padded_frames - pixels.shape[2], -1, -1)),
                dim=2,
            )
        codec_dtype = getattr(latent_encoder, "dtype", dtype)
        with Image.open(self.sink_image) as image:
            image = resize_control_frame(ImageOps.exif_transpose(image), height=request.height, width=request.width)
            array = np.array(image, dtype=np.uint8)
        sink_pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).float()
        sink_pixels = sink_pixels * (2.0 / 255.0) - 1.0
        # Each canonical encode starts with fresh CNN history. The identity
        # image must never seed the low-poly encoder or the output decoder.
        sink = latent_encoder.encode(sink_pixels.to(device=device, dtype=codec_dtype)).to(dtype=dtype)
        control = latent_encoder.encode(pixels.to(device=device, dtype=codec_dtype)).to(dtype=dtype)
        expected = (1, 48, latent_frames, request.height // 16, request.width // 16)
        if tuple(control.shape) != expected or tuple(sink.shape) != (1, 48, 1, *expected[-2:]):
            raise ValueError("causal control-video codec must produce normalized Wan2.2 48-channel latents")
        # Match the release's CPU FP32 seed+1 initial-noise stream, independently
        # of the runtime-device seed used for intermediate re-noising.
        initial_rng = torch.Generator("cpu").manual_seed(request.sampling.seed + 1)
        noise = (
            torch.cat(
                [torch.randn((1, 1, 48, *expected[-2:]), generator=initial_rng) for _ in range(latent_frames)],
                dim=1,
            )
            .permute(0, 2, 1, 3, 4)
            .to(device=device, dtype=dtype)
        )
        return LatentInitialization(
            latents=noise,
            conditioning={
                "control_latents": control,
                "kv_prefix_latents": sink,
                "output_num_frames": request.num_frames,
            },
            artifacts={"padded_frames": padded_frames - request.num_frames},
        )


def build_wan_causal_control_initializer(context: ComponentBuildContext) -> WanCausalControlInitializer:
    """Bind the declared sink image without discovering sibling assets."""
    checkpoint = NativeCheckpointResolver().materialize(context.require_checkpoint("sink"))
    return WanCausalControlInitializer(checkpoint.paths[0])
