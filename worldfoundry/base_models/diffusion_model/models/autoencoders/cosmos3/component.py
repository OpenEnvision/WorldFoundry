"""Cosmos3 multimodal :class:`~...runners.MultiModalLatentDecoder`.

Video uses the shared Wan2.2 48-channel :class:`WanVideoVAE38`.  Sound
uses :class:`~.audio.Cosmos3AVAEAudioDecoder`.  :class:`Cosmos3MediaDecoder`
implements ``decode(states, request)`` for the joint runner.  Loaders
remap official Diffusers Wan2.2 keys onto the native VAE38.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest, ModalityState
from ....loaders import CheckpointSpec, ModuleLoadSpec, NativeModuleLoader, checkpoint_json_config
from ....optimizations import RuntimePolicy
from ....runners import MultiModalLatentDecoder
from ..wan import convert_diffusers_wan22_vae_state_dict
from ..wan.component import _resolve_vae_decode_autocast
from ..wan.model import CausalConv3d, RMS_norm, Upsample, WanVideoVAE38
from .audio import Cosmos3AVAEAudioDecoder


def load_cosmos3_video_vae_checkpoint(
    checkpoint: CheckpointSpec,
    policy: RuntimePolicy,
) -> WanVideoVAE38:
    """Load the official Wan2.2-layout Cosmos3 VAE into the shared VAE38."""

    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    module = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=WanVideoVAE38,
            state_dict_converter=convert_diffusers_wan22_vae_state_dict,
            vram_module_map={
                torch.nn.Conv2d: AutoWrappedModule,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.Linear: AutoWrappedLinear,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
            },
        ),
        checkpoint,
        policy,
    )
    if not isinstance(module, WanVideoVAE38):
        raise TypeError(f"expected WanVideoVAE38, got {type(module).__name__}")
    module.decode_autocast_dtype = _resolve_vae_decode_autocast(policy)
    return module


def load_cosmos3_video_vae(context: ComponentBuildContext) -> WanVideoVAE38:
    """Load the recipe-bound Wan2.2 VAE38 from the ``weights`` checkpoint."""
    return load_cosmos3_video_vae_checkpoint(context.require_checkpoint("weights"), context.policy)


def load_cosmos3_audio_decoder(context: ComponentBuildContext) -> Cosmos3AVAEAudioDecoder:
    from worldfoundry.core.vram import AutoWrappedModule

    module = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=Cosmos3AVAEAudioDecoder,
            config_resolver=lambda checkpoint: checkpoint_json_config(checkpoint, "sound_tokenizer/config.json"),
            state_dict_converter=lambda state_dict: {
                key: value
                for key, value in state_dict.items()
                if key.startswith("decoder.")
            },
            vram_module_map={
                torch.nn.Conv1d: AutoWrappedModule,
                torch.nn.ConvTranspose1d: AutoWrappedModule,
            },
        ),
        context.require_checkpoint("sound"),
        context.policy,
    )
    if not isinstance(module, Cosmos3AVAEAudioDecoder):
        raise TypeError(f"expected Cosmos3AVAEAudioDecoder, got {type(module).__name__}")
    return module


class Cosmos3MediaDecoder(MultiModalLatentDecoder):
    """Decode Cosmos3 video states and pass research modalities through explicitly."""

    def __init__(
        self,
        vae: WanVideoVAE38,
        audio: Cosmos3AVAEAudioDecoder,
        *,
        device: torch.device,
        tiled: bool,
        tile_size: tuple[int, int],
        tile_stride: tuple[int, int],
    ) -> None:
        self.vae = vae
        self.audio = audio
        self.device = device
        self.tiled = tiled
        self.tile_size = tile_size
        self.tile_stride = tile_stride

    def decode_modalities(
        self,
        states: Mapping[str, ModalityState],
        request: DiffusionRequest,
    ) -> Mapping[str, object]:
        """Decode video (and optional sound / action) from joint modality states."""
        video_latent = states["video"].latent
        if bool(request.inputs.get("return_latent", False)):
            video = video_latent
        else:
            video = self.vae.decode(
                video_latent,
                self.device,
                tiled=self.tiled,
                tile_size=self.tile_size,
                tile_stride=self.tile_stride,
            )
        artifacts: dict[str, object] = {"video": video}
        if "sound" in states:
            artifacts["sound"] = self.audio.decode(states["sound"].latent)
            artifacts["audio_sampling_rate"] = self.audio.sampling_rate
        if "action" in states:
            action = states["action"].latent
            raw_dim = request.inputs.get("raw_action_dim")
            artifacts["action"] = action[:, : int(raw_dim)] if raw_dim is not None else action
        return artifacts


def build_cosmos3_media_decoder(context: ComponentBuildContext) -> Cosmos3MediaDecoder:
    tile_size = tuple(int(value) for value in context.component_options.get("tile_size", (34, 34)))
    tile_stride = tuple(int(value) for value in context.component_options.get("tile_stride", (18, 16)))
    if len(tile_size) != 2 or len(tile_stride) != 2:
        raise ValueError("Cosmos3 VAE tile_size and tile_stride must contain two values")
    return Cosmos3MediaDecoder(
        load_cosmos3_video_vae(context),
        load_cosmos3_audio_decoder(context),
        device=context.policy.device,
        tiled=bool(context.component_options.get("tiled", False)),
        tile_size=(tile_size[0], tile_size[1]),
        tile_stride=(tile_stride[0], tile_stride[1]),
    )


__all__ = [
    "Cosmos3MediaDecoder",
    "build_cosmos3_media_decoder",
    "load_cosmos3_video_vae",
    "load_cosmos3_video_vae_checkpoint",
    "load_cosmos3_audio_decoder",
]
