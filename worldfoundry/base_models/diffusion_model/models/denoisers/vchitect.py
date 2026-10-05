"""Native Vchitect-2B denoiser adapter.

Implements :class:`~...contracts.Denoiser`.  The runner calls
``__call__(DenoiserInput) -> DenoiserOutput``.  Conditioning is SD3-style:
``prompt_embeds`` (joint attention tokens) and ``pooled_prompt_embeds``
from :mod:`..encoders.vchitect`.  ``DenoiserInput.latents`` are 16-channel
SD3 latents; the transformer predicts the flow/velocity residual returned
as ``DenoiserOutput.sample``.
"""

from __future__ import annotations

import torch

from ...components import ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader
from ..networks.vchitect import VchitectXLTransformerModel


VCHITECT_2B_CONFIG = {
    "sample_size": 128,
    "patch_size": 2,
    "in_channels": 16,
    "num_layers": 24,
    "attention_head_dim": 64,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "caption_projection_dim": 1536,
    "pooled_projection_dim": 2048,
    "out_channels": 16,
    "pos_embed_max_size": 192,
}


class VchitectDenoiser:
    """Vchitect-XL transformer adapter for the standard runner loop."""
    def __init__(self, model: VchitectXLTransformerModel) -> None:
        self.model = model

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        """Map ``DenoiserInput`` onto :class:`VchitectXLTransformerModel`."""
        prompt = model_input.conditioning.get("prompt_embeds")
        pooled = model_input.conditioning.get("pooled_prompt_embeds")
        if not isinstance(prompt, torch.Tensor) or not isinstance(pooled, torch.Tensor):
            raise TypeError("Vchitect requires prompt_embeds and pooled_prompt_embeds tensors")
        timestep = model_input.timestep.to(device=model_input.latents.device).reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(model_input.latents.shape[0])
        # The released model uses bfloat16 weights under autocast. Its first
        # latent is float32 and its text sequence/pooled vectors have different
        # dtypes; coercing everything to the latent dtype changes the trajectory.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=model_input.latents.is_cuda):
            sample = self.model(
                model_input.latents,
                encoder_hidden_states=prompt.to(device=model_input.latents.device),
                pooled_projections=pooled.to(device=model_input.latents.device),
                timestep=timestep,
            )
        return DenoiserOutput(sample=sample)


def build_vchitect_denoiser(context: ComponentBuildContext) -> VchitectDenoiser:
    """Load the 2B Vchitect-XL transformer from the ``weights`` checkpoint."""
    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=VchitectXLTransformerModel,
            config=VCHITECT_2B_CONFIG,
            layer_container="transformer_blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, VchitectXLTransformerModel):
        raise TypeError(f"expected VchitectXLTransformerModel, got {type(model).__name__}")
    return VchitectDenoiser(model)


__all__ = ["VCHITECT_2B_CONFIG", "VchitectDenoiser", "build_vchitect_denoiser"]
