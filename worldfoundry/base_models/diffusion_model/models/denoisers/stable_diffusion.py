# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-shaped SD2 U-Net components for inpainting and Marigold.

Only Diffusers' reusable network primitive is used; execution belongs to the
native assembler and runner. Marigold concatenates RGB before noisy prediction
latents; SD inpainting concatenates noise, mask, then masked-image latents.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch

from ...components import ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import MaterializedCheckpoint, ModuleLoadSpec, NativeModuleLoader, checkpoint_json_config


def unet_config(checkpoint: MaterializedCheckpoint) -> Mapping[str, object]:
    """Read the U-Net topology without Diffusers serialization bookkeeping."""
    return {
        key: value
        for key, value in checkpoint_json_config(checkpoint, "unet/config.json").items()
        if not key.startswith("_")
    }


class SDUNetDenoiser:
    """Adapt an SD2 U-Net to native single-image denoising contracts."""

    def __init__(self, unet: torch.nn.Module, *, mode: str, dtype: torch.dtype) -> None:
        if mode not in {"inpaint", "prediction"}:
            raise ValueError("SD U-Net mode must be inpaint or prediction")
        self.unet = unet
        self.mode = mode
        self.dtype = dtype

    def batch_cfg_conditioning(
        self, positive: Mapping[str, object], negative: Mapping[str, object]
    ) -> Mapping[str, object]:
        """Preserve SD2's negative-then-positive U-Net batch arithmetic."""
        first = cast(Mapping[str, torch.Tensor], negative)
        second = cast(Mapping[str, torch.Tensor], positive)
        names = ("context", "mask", "masked_image_latents") if self.mode == "inpaint" else ("context", "rgb_latents")
        return {name: torch.cat((first[name], second[name]), dim=0) for name in names}

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        """Predict four latent channels using explicitly supplied conditions."""
        values = cast(Mapping[str, torch.Tensor], model_input.conditioning)
        noisy = model_input.latents.to(dtype=self.dtype)
        if self.mode == "inpaint":
            sample = torch.cat((noisy, values["mask"].to(noisy), values["masked_image_latents"].to(noisy)), dim=1)
        else:
            sample = torch.cat((values["rgb_latents"].to(noisy), noisy), dim=1)
        prediction = self.unet(
            sample, model_input.timestep, encoder_hidden_states=values["context"].to(noisy), return_dict=False
        )[0]
        return DenoiserOutput(sample=prediction)


def build_sd_unet_denoiser(context: ComponentBuildContext) -> SDUNetDenoiser:
    """Restore one network through the canonical native module loader."""
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

    unet = NativeModuleLoader().load(
        ModuleLoadSpec(module_class=UNet2DConditionModel, config_resolver=unet_config),
        context.require_checkpoint(),
        context.policy,
    )
    unet.requires_grad_(False)
    return SDUNetDenoiser(unet, mode=str(context.component_options["mode"]), dtype=context.policy.dtype)
