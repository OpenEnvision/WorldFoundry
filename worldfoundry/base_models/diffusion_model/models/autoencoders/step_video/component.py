"""StepVideo :class:`~...contracts.LatentDecoder`.

Wraps the native 64-channel :class:`~.model.AutoencoderKL`.  The VAE
returns ``[B, F, C, H, W]``; this adapter permutes to BCTHW and clamps
``[-1, 1]``.  ``return_latent`` skips decode.  Encode is not on this
role (T2V draws noise with 17-frame packing).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest
from ....loaders import ModuleLoadSpec, NativeModuleLoader
from ....optimizations import OffloadPolicy
from .model import AutoencoderKL


def _vae_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    return {
        key.replace("decoder.conv_out.", "decoder.conv_out.conv."): value
        for key, value in state_dict.items()
    }


class StepVideoDecoder:
    """Permute StepVideo VAE output to the runner's BCTHW layout."""

    def __init__(self, vae: AutoencoderKL) -> None:
        self.vae = vae

    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        """Decode packed latents to BCTHW pixels."""
        if bool(request.inputs.get("return_latent", False)):
            return latents
        video = self.vae.decode(latents)
        if video.ndim != 5:
            raise ValueError("StepVideo VAE must return [B,F,C,H,W]")
        return video.permute(0, 2, 1, 3, 4).clamp(-1.0, 1.0)


def build_step_video_decoder(context: ComponentBuildContext) -> StepVideoDecoder:
    """Load the StepVideo VAE and remap ``decoder.conv_out`` keys."""
    # StepVideo's VAE has no block container for asynchronous layer offload.
    # Keep this small decoder resident while the much larger DiT uses the
    # caller's block-offload policy.
    decoder_policy = replace(context.policy, offload=OffloadPolicy())
    vae = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=AutoencoderKL,
            config={"z_channels": 64, "version": 2, "world_size": 1},
            state_dict_converter=_vae_state_dict,
            layer_container="decoder",
        ),
        context.require_checkpoint("weights"),
        decoder_policy,
    )
    if not isinstance(vae, AutoencoderKL):
        raise TypeError(f"expected StepVideo AutoencoderKL, got {type(vae).__name__}")
    return StepVideoDecoder(vae)


__all__ = ["StepVideoDecoder", "build_step_video_decoder"]
