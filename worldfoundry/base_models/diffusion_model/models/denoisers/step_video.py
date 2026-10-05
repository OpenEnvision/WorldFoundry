"""Native StepVideo denoiser adapter.

Implements :class:`~...contracts.Denoiser`.  Requires ``prompt_embeds``,
``clip_embeds``, and ``attention_mask`` tensors.  Pairs with
:class:`~...schedulers.step_video.StepVideoFlowScheduler`.
"""

from __future__ import annotations

import torch

from ...components import ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader
from ..networks.step_video import StepVideoModel


STEP_VIDEO_CONFIG = {
    "num_attention_heads": 48,
    "attention_head_dim": 128,
    "in_channels": 64,
    "out_channels": 64,
    "num_layers": 48,
    "dropout": 0.0,
    "patch_size": 1,
    "norm_type": "ada_norm_single",
    "norm_elementwise_affine": False,
    "norm_eps": 1e-6,
    "use_additional_conditions": False,
    "caption_channels": (6144, 1024),
    "attention_type": "torch",
}


class StepVideoDenoiser:
    """StepVideo DiT adapter that forwards dual text/CLIP embeddings."""
    def __init__(self, model: StepVideoModel) -> None:
        self.model = model

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        prompt = model_input.conditioning.get("prompt_embeds")
        clip = model_input.conditioning.get("clip_embeds")
        mask = model_input.conditioning.get("attention_mask")
        if not all(isinstance(value, torch.Tensor) for value in (prompt, clip, mask)):
            raise TypeError("StepVideo requires prompt, CLIP, and attention-mask tensors")
        timestep = model_input.timestep.to(model_input.latents).reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(model_input.latents.shape[0])
        sample = self.model(
            model_input.latents,
            encoder_hidden_states=prompt.to(model_input.latents),
            encoder_hidden_states_2=clip.to(model_input.latents),
            encoder_attention_mask=mask.to(device=model_input.latents.device),
            timestep=timestep,
            fps=torch.full_like(timestep, float(model_input.conditioning.get("fps", 25))),
            return_dict=False,
        )
        return DenoiserOutput(sample=sample)


def build_step_video_denoiser(context: ComponentBuildContext) -> StepVideoDenoiser:
    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=StepVideoModel,
            config=STEP_VIDEO_CONFIG,
            layer_container="transformer_blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, StepVideoModel):
        raise TypeError(f"expected StepVideoModel, got {type(model).__name__}")
    return StepVideoDenoiser(model)


__all__ = ["STEP_VIDEO_CONFIG", "StepVideoDenoiser", "build_step_video_denoiser"]
