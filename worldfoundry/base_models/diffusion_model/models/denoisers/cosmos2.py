"""Native Cosmos Predict2 Video2World denoiser roles.

Implements :class:`~...contracts.Denoiser` for 2B / 14B Video2World recipes.
The runner calls ``__call__(DenoiserInput) -> DenoiserOutput``.  The inner
network is the shared Predict 2.5 DiT
(:class:`~..networks.cosmos2p5.model.Cosmos25Transformer3DModel`) with
Predict2 channel / RoPE configs and the official Predict 2.5 key remap.

``DenoiserInput.conditioning`` mapping
--------------------------------------
- ``context``: text embeddings.
- ``condition_latents`` / ``condition_mask`` / ``condition_indicator``:
  Video2World frozen-frame tensors.  Masked regions stay on the condition
  latent; unmasked regions receive rectified-flow preconditioning.
- ``padding_mask`` (optional tensor).
- ``fps`` (optional float, default 16).

Rectified-flow coefficients: ``c_in = c_skip = 1/(σ+1)``,
``c_out = -σ/(σ+1)``.  Conditioned frames use ``sigma_conditional``
(default ``1e-4``) so their effective timestep stays near zero.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ...components import ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader
from ..networks.cosmos2p5.model import Cosmos25Transformer3DModel
from .cosmos2p5 import convert_cosmos25_state_dict

COSMOS2_2B_CONFIG = {
    "in_channels": 17,
    "out_channels": 16,
    "num_attention_heads": 16,
    "attention_head_dim": 128,
    "num_layers": 28,
    "text_in_channels": 1024,
    "text_embed_dim": 1024,
    "use_crossattn_projection": False,
    "rope_scale": (1.0, 3.0, 3.0),
    "rope_enable_fps_modulation": False,
}
COSMOS2_14B_CONFIG = {
    **COSMOS2_2B_CONFIG,
    "num_attention_heads": 40,
    "num_layers": 36,
    "rope_scale": (0.8333333333333334, 2.0, 2.0),
}


class Cosmos2Denoiser:
    """Rectified-flow preconditioning around the shared Predict2 DiT math."""

    def __init__(self, model: Cosmos25Transformer3DModel, *, sigma_conditional: float = 0.0001) -> None:
        self.model = model
        self.sigma_conditional = float(sigma_conditional)

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        """Evaluate one CFG branch and return clean-x0 style ``sample``."""
        values = model_input.conditioning
        context = values.get("context")
        conditions = tuple(
            values.get(name) for name in ("condition_latents", "condition_mask", "condition_indicator")
        )
        if not isinstance(context, torch.Tensor) or not all(isinstance(value, torch.Tensor) for value in conditions):
            raise TypeError("Cosmos Predict2 Video2World conditioning is incomplete")
        latents = model_input.latents
        model_dtype = next(self.model.parameters(), latents).dtype
        context = context.to(device=latents.device, dtype=model_dtype)
        condition_latents, condition_mask, condition_indicator = (
            value.to(device=latents.device, dtype=latents.dtype) for value in conditions
        )
        sigma = model_input.timestep.to(device=latents.device, dtype=latents.dtype).reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(latents.shape[0])
        elif sigma.numel() != latents.shape[0]:
            raise ValueError("Cosmos Predict2 sigma must be scalar or match the latent batch")
        sigma = sigma.reshape(-1, 1, 1, 1, 1)
        c_in = 1.0 / (sigma + 1.0)
        c_skip = c_in
        c_out = -sigma / (sigma + 1.0)
        network_input = latents * c_in
        network_input = condition_mask * condition_latents + (1.0 - condition_mask) * network_input

        noise_timestep = sigma / (sigma + 1.0)
        condition_sigma = latents.new_tensor(self.sigma_conditional)
        condition_timestep = condition_sigma / (condition_sigma + 1.0)
        timestep = condition_indicator[:, 0, :, 0, 0] * condition_timestep
        timestep = timestep + (1.0 - condition_indicator[:, 0, :, 0, 0]) * noise_timestep[:, 0, 0, 0, 0, None]
        padding_mask = values.get("padding_mask")
        prediction = self.model(
            network_input.to(model_dtype),
            timestep.to(model_dtype),
            context,
            fps=float(values.get("fps", 16.0)),
            condition_mask=condition_mask,
            padding_mask=padding_mask if isinstance(padding_mask, torch.Tensor) else None,
        )
        clean = c_skip * latents + c_out * prediction.float()
        clean = condition_mask * condition_latents + (1.0 - condition_mask) * clean
        return DenoiserOutput(sample=clean)


def _build(context: ComponentBuildContext, config: Mapping[str, object]) -> Cosmos2Denoiser:
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=Cosmos25Transformer3DModel,
            config=config,
            state_dict_converter=convert_cosmos25_state_dict,
            vram_module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.LayerNorm: AutoWrappedModule,
                torch.nn.RMSNorm: AutoWrappedModule,
            },
            layer_container="transformer_blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, Cosmos25Transformer3DModel):
        raise TypeError(f"expected Cosmos25Transformer3DModel, got {type(model).__name__}")
    return Cosmos2Denoiser(
        model,
        sigma_conditional=float(context.component_options.get("sigma_conditional", 0.0001)),
    )


def build_cosmos2_2b_denoiser(context: ComponentBuildContext) -> Cosmos2Denoiser:
    """Load the 2B Video2World Predict2 denoiser from ``weights``."""
    return _build(context, COSMOS2_2B_CONFIG)


def build_cosmos2_14b_denoiser(context: ComponentBuildContext) -> Cosmos2Denoiser:
    """Load the 14B Video2World Predict2 denoiser from ``weights``."""
    return _build(context, COSMOS2_14B_CONFIG)


__all__ = [
    "COSMOS2_2B_CONFIG",
    "COSMOS2_14B_CONFIG",
    "Cosmos2Denoiser",
    "build_cosmos2_2b_denoiser",
    "build_cosmos2_14b_denoiser",
]
