"""Native T2V-Turbo denoiser built from the shared LVDM 3D UNet.

Implements :class:`~...contracts.Denoiser` for the VideoCrafter-derived
turbo recipe.  The runner supplies ``context`` and ``timestep_cond``;
missing either raises :exc:`TypeError`.  Weights load through
:class:`~...loaders.module.NativeModuleLoader`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from ...components import ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeCheckpointResolver, NativeModuleLoader
from ..networks.lvdm.unet3d import UNetModel

T2V_TURBO_UNET_CONFIG = {
    "in_channels": 4,
    "out_channels": 4,
    "model_channels": 320,
    "attention_resolutions": (4, 2, 1),
    "num_res_blocks": 2,
    "channel_mult": (1, 2, 4, 4),
    "num_head_channels": 64,
    "transformer_depth": 1,
    "context_dim": 1024,
    "use_linear": True,
    "use_checkpoint": False,
    "temporal_conv": True,
    "temporal_attention": True,
    "temporal_selfatt_only": True,
    "use_relative_position": False,
    "use_causal_attention": False,
    "temporal_length": 16,
    "addition_attention": True,
    "fps_cond": True,
    "time_cond_proj_dim": 256,
}


def t2v_turbo_guidance_projection(reference: torch.Tensor) -> torch.Tensor:
    """Return the frozen guidance projection used when VideoCrafter has no such key."""

    generator = torch.Generator(device="cpu").manual_seed(0)
    bound = 1.0 / math.sqrt(256.0)
    value = torch.empty((320, 256), dtype=torch.float32).uniform_(
        -bound,
        bound,
        generator=generator,
    )
    return value.to(dtype=reference.dtype, device=reference.device)


def _unet_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    prefix = "model.diffusion_model."
    converted = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not converted:
        raise KeyError("VideoCrafter checkpoint contains no model.diffusion_model parameters")
    if "time_cond_proj.weight" not in converted:
        reference = next(value for value in converted.values() if isinstance(value, torch.Tensor))
        converted["time_cond_proj.weight"] = t2v_turbo_guidance_projection(reference)
    return converted


class T2VTurboDenoiser:
    """LVDM UNet adapter that consumes ``context`` and LCM ``timestep_cond``."""
    def __init__(self, model: UNetModel) -> None:
        self.model = model

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        context = model_input.conditioning.get("context")
        timestep_cond = model_input.conditioning.get("timestep_cond")
        if not isinstance(context, torch.Tensor) or not isinstance(timestep_cond, torch.Tensor):
            raise TypeError("T2V-Turbo requires tensor context and timestep_cond values")
        timestep = model_input.timestep.to(device=model_input.latents.device, dtype=torch.long).reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(model_input.latents.shape[0])
        fps = int(model_input.conditioning.get("fps", 16))
        sample = self.model(
            model_input.latents,
            timestep,
            context=context.to(device=model_input.latents.device, dtype=model_input.latents.dtype),
            fps=fps,
            timestep_cond=timestep_cond.to(device=model_input.latents.device),
        )
        return DenoiserOutput(sample=sample)


def build_t2v_turbo_denoiser(context: ComponentBuildContext) -> T2VTurboDenoiser:
    from worldfoundry.core.model_loading import load_torch_checkpoint, merge_ordered_lora_

    def merge_turbo_lora(model: torch.nn.Module) -> None:
        materialized = NativeCheckpointResolver().materialize(context.require_checkpoint("lora"))
        lora = load_torch_checkpoint(materialized.paths[0], map_location="cpu")
        if not isinstance(lora, (list, tuple)):
            raise TypeError("T2V-Turbo LoRA checkpoint must be an ordered tensor sequence")
        merge_ordered_lora_(
            model,
            lora,
            ancestor_class_names=("UNetModel",),
        )

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=UNetModel,
            config=T2V_TURBO_UNET_CONFIG,
            state_dict_converter=_unet_state_dict,
            post_load_hook=merge_turbo_lora,
        ),
        context.require_checkpoint("base"),
        context.policy,
    )
    if not isinstance(model, UNetModel):
        raise TypeError(f"expected UNetModel, got {type(model).__name__}")
    return T2VTurboDenoiser(model)


__all__ = [
    "T2VTurboDenoiser",
    "T2V_TURBO_UNET_CONFIG",
    "build_t2v_turbo_denoiser",
    "t2v_turbo_guidance_projection",
]
