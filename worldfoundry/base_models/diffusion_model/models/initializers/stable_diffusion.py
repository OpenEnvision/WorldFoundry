# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Native SD inpainting and Marigold observation/noise preparation."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from ...components import ComponentBuildContext
from ...contracts import DiffusionRequest, LatentEncoder, LatentInitialization
from ..autoencoders.stable_diffusion import SDImageCodec


class SDImageInitializer:
    """Encode a signed RGB observation through the recipe's shared VAE."""

    def __init__(self, *, mode: str) -> None:
        self.mode = mode

    def initialize(
        self, request: DiffusionRequest, *, generator: torch.Generator, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Reject execution without the declared encoder binding."""
        del request, generator, device, dtype
        raise RuntimeError("SD image initialization requires its latent_encoder binding")

    def initialize_with_encoder(
        self,
        request: DiffusionRequest,
        *,
        latent_encoder: LatentEncoder,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LatentInitialization:
        """Preserve released RNG ordering, VAE posterior, and mask semantics."""
        pixels = request.inputs["image"]
        if not isinstance(pixels, torch.Tensor) or pixels.shape != (
            request.batch_size,
            3,
            request.height,
            request.width,
        ):
            raise ValueError("SD image must be signed [B,3,height,width] pixels")
        pixels = pixels.to(device=device, dtype=dtype).contiguous()
        if self.mode == "inpaint":
            mask = request.inputs["mask"]
            if not isinstance(mask, torch.Tensor) or mask.shape != (
                request.batch_size,
                1,
                request.height,
                request.width,
            ):
                raise ValueError("SD inpaint mask must be [B,1,height,width]")
            mask = mask.to(pixels)
            # Upstream draws initial noise before sampling the masked-image VAE.
            noise = torch.randn(
                (request.batch_size, 4, request.height // 8, request.width // 8),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            if not isinstance(latent_encoder, SDImageCodec):
                raise TypeError("SD inpainting requires the sampled-posterior SD codec")
            masked_pixels = (pixels * (mask < 0.5)).contiguous(memory_format=torch.channels_last)
            masked = latent_encoder.encode_sample(masked_pixels, generator)
            return LatentInitialization(
                noise,
                {
                    "mask": functional.interpolate(mask, masked.shape[-2:], mode="nearest"),
                    "masked_image_latents": masked,
                },
            )
        resolution = request.inputs.get("processing_resolution", 0)
        if not isinstance(resolution, int) or resolution < 0:
            raise ValueError("processing_resolution must be a nonnegative integer")
        if resolution:
            ratio = resolution / max(request.height, request.width)
            pixels = functional.interpolate(
                pixels,
                (int(request.height * ratio), int(request.width * ratio)),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        rgb_latents = latent_encoder.encode(pixels)
        # Marigold's WonderWorld path deliberately uses unit noise, not sigma-max
        # noise, and does not scale the concatenated U-Net input for Euler.
        noise = torch.randn(rgb_latents.shape, generator=generator, device=device, dtype=dtype)
        return LatentInitialization(noise, {"rgb_latents": rgb_latents})


def build_sd_image_initializer(context: ComponentBuildContext) -> SDImageInitializer:
    """Bind the observation preparation mode declared by the recipe."""
    return SDImageInitializer(mode=str(context.component_options["mode"]))
