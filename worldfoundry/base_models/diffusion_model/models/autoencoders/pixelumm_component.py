"""Native PixelUMM decoder role and checkpoint selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import torch

from ...components import ComponentBuildContext
from ...contracts import DiffusionRequest
from ...loaders import ModuleLoadSpec, NativeModuleLoader
from ...optimizations import OffloadPolicy, QuantizationPolicy, parse_quantization_policy
from .pixelumm import PixelUMMDecoder


def pixelumm_decoder_weights(state: Mapping[str, Any]) -> dict[str, Any]:
    """The inference slice needs only decoder weights, not the unused encoder."""
    unexpected = [key for key in state if not key.startswith(("decoder.", "encoder."))]
    if unexpected:
        raise ValueError(f"unexpected PixelUMM parameters: {unexpected[:5]}")
    return {name: value for name, value in state.items() if name.startswith("decoder.")}


class PixelUMMLatentDecoder:
    """Decode native BCTHW latents and crop padded output to the public request."""

    def __init__(self, model: PixelUMMDecoder, *, low_memory: bool = False) -> None:
        self.model = model
        self.low_memory = low_memory

    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        pixels = cast(tuple[torch.Tensor], self.model.decode(latents, return_dict=False))[0]
        return pixels[:, :, : request.num_frames, : request.height, : request.width]


def build_pixelumm_decoder(context: ComponentBuildContext) -> PixelUMMLatentDecoder:
    low_memory = bool(context.component_options.get("low_memory", False))
    policy = replace(
        context.policy,
        compile=False,
        offload=OffloadPolicy(),
        device="cpu" if low_memory else context.policy.device,
        quantization=(
            parse_quantization_policy({"mode": "fp8", "keep_dense_fallback": False})
            if low_memory
            else QuantizationPolicy()
        ),
    )
    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=PixelUMMDecoder,
            config=dict(cast(Mapping[str, Any], context.component_options.get("config", {}))),
            state_dict_converter=pixelumm_decoder_weights,
        ),
        context.require_checkpoint(),
        policy,
    )
    return PixelUMMLatentDecoder(cast(PixelUMMDecoder, model), low_memory=low_memory)
