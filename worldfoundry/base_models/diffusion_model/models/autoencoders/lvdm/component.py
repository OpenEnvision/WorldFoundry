"""LVDM / VideoCrafter frame-wise :class:`~...contracts.LatentDecoder`.

Decodes each latent frame independently through an SD AutoencoderKL
(``scale_factor=0.18215``).  Input is BCTHW; there is no temporal VAE.
Weights come from the VideoCrafter ``first_stage_model.*`` prefix.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest
from ....loaders import ModuleLoadSpec, NativeModuleLoader
from .model import AutoencoderKL


LVDM_AUTOENCODER_CONFIG = {
    "embed_dim": 4,
    "monitor": "val/rec_loss",
    "ddconfig": {
        "double_z": True,
        "z_channels": 4,
        "resolution": 512,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": (1, 2, 4, 4),
        "num_res_blocks": 2,
        "attn_resolutions": (),
        "dropout": 0.0,
    },
    "lossconfig": {"target": "torch.nn.Identity"},
}


def _autoencoder_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    prefix = "first_stage_model."
    converted = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not converted:
        raise KeyError("VideoCrafter checkpoint contains no first_stage_model parameters")
    return converted


class LVDMVideoDecoder:
    """Per-frame SD-VAE decode of BCTHW latents."""

    def __init__(self, model: AutoencoderKL, *, scale_factor: float = 0.18215) -> None:
        self.model = model
        self.scale_factor = float(scale_factor)

    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        """Decode each time slice independently and concat along T."""
        if bool(request.inputs.get("return_latent", False)):
            return latents
        if latents.ndim != 5:
            raise ValueError("LVDM decoder expects [B,C,T,H,W] latents")
        frames = [
            self.model.decode(latents[:, :, index] / self.scale_factor).unsqueeze(2)
            for index in range(latents.shape[2])
        ]
        return torch.cat(frames, dim=2).clamp_(-1.0, 1.0)


def build_lvdm_video_decoder(context: ComponentBuildContext) -> LVDMVideoDecoder:
    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=AutoencoderKL,
            config=LVDM_AUTOENCODER_CONFIG,
            state_dict_converter=_autoencoder_state_dict,
            layer_container="decoder",
        ),
        context.require_checkpoint("base"),
        context.policy,
    )
    if not isinstance(model, AutoencoderKL):
        raise TypeError(f"expected AutoencoderKL, got {type(model).__name__}")
    return LVDMVideoDecoder(
        model,
        scale_factor=float(context.component_options.get("scale_factor", 0.18215)),
    )


__all__ = [
    "LVDM_AUTOENCODER_CONFIG",
    "LVDMVideoDecoder",
    "build_lvdm_video_decoder",
]
