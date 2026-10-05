"""SD3-family per-frame :class:`~...contracts.LatentDecoder`.

Decodes ``[B, F, C, H, W]`` latents through a 2-D VAE decoder
(``NativeVAE2DDecoder``), applying the official scale/shift
(``1.5305`` / ``0.0609``).  Output is BCTHW in ``[-1, 1]``.  Used by
Vchitect and other SD3-latent video recipes.  Encode is not implemented
here (T2V noise is drawn directly).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from worldfoundry.core.nn.latent.vae2d import NativeVAE2DDecoder

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest
from ....loaders import ModuleLoadSpec, NativeModuleLoader


def _decoder_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    prefix = "decoder."
    converted = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
    if not converted:
        raise KeyError("SD3 VAE checkpoint contains no decoder parameters")
    return converted


class SD3FrameDecoder:
    """Fold time into batch, decode each frame, restore BCTHW."""

    def __init__(
        self,
        decoder: NativeVAE2DDecoder,
        *,
        scaling_factor: float = 1.5305,
        shift_factor: float = 0.0609,
    ) -> None:
        self.decoder = decoder
        self.scaling_factor = float(scaling_factor)
        self.shift_factor = float(shift_factor)

    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        """Decode ``[B,F,C,H,W]`` latents to BCTHW pixels."""
        if bool(request.inputs.get("return_latent", False)):
            return latents
        if latents.ndim != 5:
            raise ValueError("SD3 frame decoder expects [B,F,C,H,W] latents")
        batch, frames = latents.shape[:2]
        values = latents.reshape(batch * frames, *latents.shape[2:])
        values = values / self.scaling_factor + self.shift_factor
        decoded = self.decoder(values).clamp_(-1.0, 1.0)
        return decoded.reshape(batch, frames, *decoded.shape[1:]).permute(0, 2, 1, 3, 4)


def build_sd3_frame_decoder(context: ComponentBuildContext) -> SD3FrameDecoder:
    """Load decoder-only weights from the ``weights`` checkpoint."""
    decoder = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=NativeVAE2DDecoder,
            config={
                "latent_channels": int(context.component_options.get("latent_channels", 16)),
                "out_channels": int(context.component_options.get("out_channels", 3)),
                "block_out_channels": tuple(context.component_options.get("block_out_channels", (128, 256, 512, 512))),
                "norm_num_groups": int(context.component_options.get("norm_num_groups", 32)),
                "eps": float(context.component_options.get("eps", 1e-6)),
            },
            state_dict_converter=_decoder_state_dict,
            layer_container="up_blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(decoder, NativeVAE2DDecoder):
        raise TypeError(f"expected NativeVAE2DDecoder, got {type(decoder).__name__}")
    return SD3FrameDecoder(
        decoder,
        scaling_factor=float(context.component_options.get("scaling_factor", 1.5305)),
        shift_factor=float(context.component_options.get("shift_factor", 0.0609)),
    )


__all__ = ["SD3FrameDecoder", "build_sd3_frame_decoder"]
