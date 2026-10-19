# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared SD2 VAE, with explicit RGB, Marigold depth and normal decoding."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional

from ...components import ComponentBuildContext
from ...contracts import DiffusionRequest
from ...loaders import MaterializedCheckpoint, ModuleLoadSpec, NativeModuleLoader, checkpoint_json_config


def vae_config(checkpoint: MaterializedCheckpoint) -> Mapping[str, object]:
    """Read only checkpoint-visible VAE constructor options."""
    return {
        key: value
        for key, value in checkpoint_json_config(checkpoint, "vae/config.json").items()
        if not key.startswith("_")
    }


class SDImageCodec:
    """One VAE instance serving both native encoder and decoder roles.

    Encoder pixels are signed BCHW. RGB output is unit-range; depth uses
    WonderWorld's normalized-depth / 200 + 0.001 scene units. Normal output
    uses OpenGL axes (+X right, +Y up, +Z toward the observer) and the
    released WonderWorld decoder's observer-facing Z convention.
    """

    def __init__(self, vae: torch.nn.Module, *, mode: str, dtype: torch.dtype) -> None:
        if mode not in {"rgb", "depth", "normals"}:
            raise ValueError("SD codec mode must be rgb, depth or normals")
        self.vae = vae
        self.mode = mode
        self.dtype = dtype
        self.scale = float(vae.config.scaling_factor)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode the posterior mean; inpainting sampling is explicit below."""
        return self.vae.encode(images.to(dtype=self.dtype)).latent_dist.mode() * self.scale

    def encode_sample(self, images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Sample the masked-image posterior using the request's generator."""
        return self.vae.encode(images.to(dtype=self.dtype)).latent_dist.sample(generator=generator) * self.scale

    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        """Decode without disabling autograd needed by depth guidance."""
        prediction = self.vae.decode(latents.to(dtype=self.dtype) / self.scale, return_dict=False)[0]
        if self.mode == "depth":
            normalized = (prediction.mean(dim=1, keepdim=True).clamp(-1, 1) + 1) / 2
            return normalized.float() / 200 + 0.001
        prediction = prediction.clamp(-1, 1)
        if self.mode == "normals":
            prediction = torch.cat((prediction[:, :2], prediction[:, 2:3].clamp_min(0)), dim=1)
            prediction = functional.normalize(prediction, dim=1, eps=1e-6)
            if prediction.shape[-2:] != (request.height, request.width):
                prediction = functional.interpolate(
                    prediction, (request.height, request.width), mode="bilinear", align_corners=False
                )
            return functional.normalize(prediction, dim=1, eps=1e-6)
        return prediction / 2 + 0.5

    def decode_for_guidance(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        """Keep the checkpoint dtype through Marigold's guided scene scaling."""
        del request
        prediction = self.vae.decode(latents.to(dtype=self.dtype) / self.scale, return_dict=False)[0]
        return (prediction.mean(dim=1, keepdim=True).clamp(-1, 1) + 1) / 2 / 200 + 0.001


def build_sd_image_codec(context: ComponentBuildContext) -> SDImageCodec:
    """Load the checkpoint VAE once for both recipe bindings."""
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

    vae = NativeModuleLoader().load(
        ModuleLoadSpec(module_class=AutoencoderKL, config_resolver=vae_config),
        context.require_checkpoint(),
        context.policy,
    )
    vae.requires_grad_(False)
    return SDImageCodec(vae, mode=str(context.component_options["mode"]), dtype=context.policy.dtype)
