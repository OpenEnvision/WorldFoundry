"""Native denoiser adapter for Wan2.1 VACE.

VACE adds a control context stream alongside text.  This wrapper implements
the runner ``Denoiser`` Protocol and loads official VACE weights through
:class:`~...loaders.module.NativeModuleLoader`.
"""

from __future__ import annotations

import torch

from ...components import ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader
from ..networks.wan.model import RMSNorm
from ..networks.wan.vace import VaceWanModel


WAN21_VACE_14B_CONFIG = {
    "vace_layers": (0, 5, 10, 15, 20, 25, 30, 35),
    "vace_in_dim": 96,
    "patch_size": (1, 2, 2),
    "in_dim": 16,
    "dim": 5120,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 16,
    "num_heads": 40,
    "num_layers": 40,
    "eps": 1e-6,
}


class WanVaceDenoiser:
    """Expose the VACE transformer through the shared denoiser contract.

    Requires tensor ``context`` and ``vace_context``.  Missing either raises
    :exc:`TypeError`.  Optional ``vace_context_scale`` defaults to 1.0.
    """

    def __init__(self, model: VaceWanModel) -> None:
        self.model = model

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        context = model_input.conditioning.get("context")
        vace_context = model_input.conditioning.get("vace_context")
        if not isinstance(context, torch.Tensor):
            raise TypeError("Wan VACE denoising requires tensor text context")
        if not isinstance(vace_context, torch.Tensor):
            raise TypeError("Wan VACE denoising requires tensor vace_context")
        latents = model_input.latents
        try:
            model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            model_dtype = latents.dtype
        model_latents = latents.to(dtype=model_dtype)
        sample = self.model(
            x=model_latents,
            timestep=model_input.timestep.to(device=latents.device, dtype=model_dtype),
            context=context.to(device=latents.device, dtype=model_dtype),
            vace_context=vace_context.to(device=latents.device, dtype=model_dtype),
            vace_context_scale=float(model_input.conditioning.get("vace_context_scale", 1.0)),
        )
        return DenoiserOutput(sample=sample.to(dtype=latents.dtype))


def build_wan21_vace_14b_denoiser(context: ComponentBuildContext) -> WanVaceDenoiser:
    """Load official VACE weights through the framework-owned native loader."""

    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=VaceWanModel,
            config=WAN21_VACE_14B_CONFIG,
            vram_module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            layer_container="blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, VaceWanModel):
        raise TypeError(f"expected VaceWanModel, got {type(model).__name__}")
    return WanVaceDenoiser(model)


__all__ = [
    "WAN21_VACE_14B_CONFIG",
    "WanVaceDenoiser",
    "build_wan21_vace_14b_denoiser",
]
