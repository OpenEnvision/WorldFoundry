"""Fun-Control / Video-X adapter around the canonical Wan 2.1 VAE.

:class:`AutoencoderKLWan` wraps the Wan 2.1 causal CNN via
:class:`~..adapter.WanAutoencoderAdapterMixin` so Fun-Control
and Video-X loaders see a Diffusers-like API.

Latents remain 16-channel BCTHW, 8× spatial / 4× temporal,
official Wan 2.1 mean/std.  No new residual math.

Prefer :class:`~..component.WanVideoDecoder` for native
recipes.
"""

from __future__ import annotations

import inspect
import re

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.adapter import (
    WanAutoencoderAdapterMixin,
    WanAutoencoderConfig,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.resident import (
    WAN_2P1_VAE_MEAN,
    WAN_2P1_VAE_STD,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.camera_21 import (
    _video_vae,
)
from worldfoundry.core.model_loading import load_model


_RESIDUAL_LAYER_MAP = {
    "conv1": "residual.2",
    "conv2": "residual.6",
    "norm1": "residual.0",
    "norm2": "residual.3",
    "conv_shortcut": "shortcut",
}


def _convert_diffusers_vae_key(key: str) -> str | None:
    if not key.startswith("model."):
        key = f"model.{key}"
    key = key.replace(".quant_conv.", ".conv1.")
    key = key.replace(".post_quant_conv.", ".conv2.")

    match = re.match(
        r"(.*)\.(encoder|decoder)\.mid_block\.attentions\.\d+\."
        r"(norm|to_qkv|proj)\.(.*)",
        key,
    )
    if match:
        prefix, side, layer, parameter = match.groups()
        return f"{prefix}.{side}.middle.1.{layer}.{parameter}"

    match = re.match(
        r"(.*)\.encoder\.down_blocks\.(\d+)\."
        r"(conv1|conv2|norm1|norm2|conv_shortcut|resample\.\d+|time_conv)\.(.*)",
        key,
    )
    if match:
        prefix, block, layer, parameter = match.groups()
        layer = _RESIDUAL_LAYER_MAP.get(layer, layer)
        return f"{prefix}.encoder.downsamples.{block}.{layer}.{parameter}"

    match = re.match(
        r"(.*)\.decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\."
        r"(conv1|conv2|norm1|norm2|conv_shortcut)\.(.*)",
        key,
    )
    if match:
        prefix, block, residual, layer, parameter = match.groups()
        output_index = {0: 0, 1: 4, 2: 8, 3: 12}[int(block)] + int(residual)
        layer = _RESIDUAL_LAYER_MAP[layer]
        return f"{prefix}.decoder.upsamples.{output_index}.{layer}.{parameter}"

    match = re.match(
        r"(.*)\.decoder\.up_blocks\.(\d+)\.upsamplers\.0\."
        r"(resample\.\d+|time_conv)\.(.*)",
        key,
    )
    if match:
        prefix, block, layer, parameter = match.groups()
        output_index = {0: 3, 1: 7, 2: 11}.get(int(block))
        if output_index is None:
            return None
        return f"{prefix}.decoder.upsamples.{output_index}.{layer}.{parameter}"

    match = re.match(
        r"(.*)\.(encoder|decoder)\.mid_block\.resnets\.(\d+)\."
        r"(conv1|conv2|norm1|norm2)\.(.*)",
        key,
    )
    if match:
        prefix, side, residual, layer, parameter = match.groups()
        layer = _RESIDUAL_LAYER_MAP[layer]
        return (
            f"{prefix}.{side}.middle.{int(residual) * 2}.{layer}.{parameter}"
        )

    key = key.replace(".encoder.norm_out.", ".encoder.head.0.")
    key = key.replace(".decoder.norm_out.", ".decoder.head.0.")
    key = key.replace(".encoder.conv_in.", ".encoder.conv1.")
    key = key.replace(".decoder.conv_in.", ".decoder.conv1.")
    key = key.replace(".encoder.conv_out.", ".encoder.head.2.")
    key = key.replace(".decoder.conv_out.", ".decoder.head.2.")
    return key


def _convert_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    for key, value in state.items():
        converted_key = _convert_diffusers_vae_key(key)
        if converted_key is not None:
            converted[converted_key] = value
    return converted


class AutoencoderKLWan(WanAutoencoderAdapterMixin, nn.Module):
    """Fun-Control / Video-X wrapper around the Wan 2.1 16-channel VAE."""
    def __init__(
        self,
        latent_channels: int = 16,
        temporal_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
    ) -> None:
        super().__init__()
        self.config = WanAutoencoderConfig(
            latent_channels=int(latent_channels),
            temporal_compression_ratio=int(temporal_compression_ratio),
            spatial_compression_ratio=int(spatial_compression_ratio),
        )
        self.register_buffer(
            "latent_mean", torch.tensor(WAN_2P1_VAE_MEAN), persistent=False
        )
        self.register_buffer(
            "latent_inverse_std",
            1.0 / torch.tensor(WAN_2P1_VAE_STD),
            persistent=False,
        )
        self.model = _video_vae(pretrained_path=None, z_dim=latent_channels)

    @property
    def scale(self) -> list[torch.Tensor]:
        return [self.latent_mean, self.latent_inverse_std]
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        additional_kwargs: dict | None = None,
    ) -> "AutoencoderKLWan":
        parameters = set(inspect.signature(cls.__init__).parameters) - {"self", "cls"}
        config = {
            key: value
            for key, value in (additional_kwargs or {}).items()
            if key in parameters
        }
        return load_model(
            cls,
            pretrained_model_path,
            config=config,
            torch_dtype=torch.float32,
            device="cpu",
            state_dict_converter=_convert_state_dict,
        )


__all__ = ["AutoencoderKLWan"]
