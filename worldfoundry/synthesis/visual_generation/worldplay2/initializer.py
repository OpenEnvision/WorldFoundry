"""WorldPlay2's paired HR/LR I2V conditions reuse the Wan initializer."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from worldfoundry.base_models.diffusion_model.contracts import LatentInitialization
from worldfoundry.base_models.diffusion_model.models.initializers.wan.component import (
    WanImageToVideoLatentInitializer,
    encode_wan_image_condition,
)


class WorldPlay2LatentInitializer(WanImageToVideoLatentInitializer):
    @torch.no_grad()
    def initialize_with_encoder(self, request, *, latent_encoder, generator, device, dtype):
        if request.height % 64 or request.width % 64:
            raise ValueError("WorldPlay2 height and width must be divisible by 64 for compressed memory")
        initialized = super().initialize_with_encoder(
            request, latent_encoder=latent_encoder, generator=generator, device=device, dtype=dtype,
        )
        frames = initialized.latents.shape[2]
        if frames % 2:
            raise ValueError("WorldPlay2 requires an even number of latent frames")
        reference = initialized.conditioning["reference_pixels"]
        low_resolution = F.interpolate(reference[:, :, 0].cpu(),
                                       size=(request.height // 4, request.width // 4), mode="bilinear")
        low_condition = encode_wan_image_condition(
            low_resolution.unsqueeze(2), num_frames=(frames // 2 - 1) * 4 + 1,
            latent_encoder=latent_encoder, device=device, dtype=initialized.latents.dtype,
        )
        return LatentInitialization(
            initialized.latents,
            {**initialized.conditioning, "low_resolution_condition": low_condition},
        )


def build_worldplay2_latent_initializer(context):
    del context
    return WorldPlay2LatentInitializer()
