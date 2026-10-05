"""HunyuanVideo / 1.5 :class:`~...contracts.LatentEncoder` + :class:`~...contracts.LatentDecoder`.

:class:`HunyuanVideoOriginalCodec` wraps the causal 3-D VAE (mode ×
scaling_factor; decode returns ``[0, 1]``).  :class:`HunyuanVideo15Codec`
wraps the 1.5 Conv3D VAE.  Both honor ``return_latent`` and optional
spatial tiling.  Shared by I2V initialization and final decode.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest
from ....loaders import ModuleLoadSpec, NativeModuleLoader, checkpoint_json_config
from .causal3d import HunyuanVideoCausal3DAutoencoder
from .h15 import AutoencoderKLConv3D, RMS_norm


def _original_vae_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    if any(name.startswith("vae.") for name in state_dict):
        return {name.removeprefix("vae."): value for name, value in state_dict.items() if name.startswith("vae.")}
    return state_dict


class HunyuanVideoOriginalCodec:
    """Shared original-release VAE encoder/decoder without a model-owned runtime."""

    def __init__(
        self,
        vae: HunyuanVideoCausal3DAutoencoder,
        *,
        tiled: bool = True,
    ) -> None:
        self.vae = vae
        self.tiled = bool(tiled)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"HunyuanVideo encoder expects BCTHW pixels, got {tuple(images.shape)}")
        self.vae.enable_tiling(self.tiled)
        posterior = self.vae.encode(images).latent_dist
        return posterior.mode() * float(self.vae.config.scaling_factor)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        if bool(request.inputs.get("return_latent", False)):
            return latents
        self.vae.enable_tiling(self.tiled)
        video = self.vae.decode(latents / float(self.vae.config.scaling_factor)).sample
        return (video.float() / 2.0 + 0.5).clamp_(0.0, 1.0)


class HunyuanVideo15Codec:
    """HunyuanVideo 1.5 VAE shared by I2V initialization and decoding."""

    def __init__(self, vae: AutoencoderKLConv3D, *, tiled: bool = True) -> None:
        self.vae = vae
        self.tiled = bool(tiled)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"HunyuanVideo 1.5 encoder expects BCTHW pixels, got {tuple(images.shape)}")
        self.vae.enable_tiling(self.tiled)
        compute_dtype = next(self.vae.parameters()).dtype
        with torch.autocast(
            device_type=images.device.type,
            dtype=compute_dtype,
            enabled=compute_dtype in {torch.float16, torch.bfloat16},
        ):
            posterior = self.vae.encode(images).latent_dist
        return posterior.mode() * float(self.vae.scaling_factor)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        if bool(request.inputs.get("return_latent", False)):
            return latents
        self.vae.enable_tiling(self.tiled)
        compute_dtype = next(self.vae.parameters()).dtype
        with torch.autocast(
            device_type=latents.device.type,
            dtype=compute_dtype,
            enabled=compute_dtype in {torch.float16, torch.bfloat16},
        ):
            video = self.vae.decode(latents / float(self.vae.scaling_factor)).sample
        return (video.float() / 2.0 + 0.5).clamp_(0.0, 1.0)


def build_hunyuan_video_original_codec(context: ComponentBuildContext) -> HunyuanVideoOriginalCodec:
    checkpoint = context.require_checkpoint("weights")
    config_path = str(context.component_options.get("config_path", "config.json"))

    def resolve_config(artifact) -> Mapping[str, object]:
        config = dict(checkpoint_json_config(artifact, config_path))
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)
        return config

    vae = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=HunyuanVideoCausal3DAutoencoder,
            config_resolver=resolve_config,
            state_dict_converter=_original_vae_state_dict,
            layer_container="decoder.up_blocks",
        ),
        checkpoint,
        context.policy,
    )
    if not isinstance(vae, HunyuanVideoCausal3DAutoencoder):
        raise TypeError("native loader returned an unexpected HunyuanVideo VAE module")
    return HunyuanVideoOriginalCodec(
        vae,
        tiled=bool(context.component_options.get("tiled", True)),
    )


def _h15_vae_config(checkpoint) -> Mapping[str, object]:
    config = dict(checkpoint_json_config(checkpoint, "vae/config.json"))
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)
    return config


def build_hunyuan_video15_codec(context: ComponentBuildContext) -> HunyuanVideo15Codec:
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    vae = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=AutoencoderKLConv3D,
            config_resolver=_h15_vae_config,
            vram_module_map={
                torch.nn.Conv2d: AutoWrappedModule,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.GroupNorm: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
            },
            # Decoder.up entries are structural containers: decode calls their
            # block/upsample children directly, so hooks on the containers
            # never run. Use the declared leaf-module offload wrappers.
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(vae, AutoencoderKLConv3D):
        raise TypeError(f"expected AutoencoderKLConv3D, got {type(vae).__name__}")
    return HunyuanVideo15Codec(
        vae,
        tiled=bool(context.component_options.get("tiled", True)),
    )


__all__ = [
    "HunyuanVideo15Codec",
    "HunyuanVideoOriginalCodec",
    "build_hunyuan_video15_codec",
    "build_hunyuan_video_original_codec",
]
