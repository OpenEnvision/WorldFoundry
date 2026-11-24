"""LTX :class:`~...contracts.LatentProcessor` between multi-stage passes.

Implements ``process(Mapping[str, ModalityState], DiffusionRequest) ->
Mapping[str, Tensor]``.  The joint runner calls this after stage 1 so
stage 2 can re-initialize from dense (unpatchified, spatially upsampled)
video latents.  Audio is unpatchified but not spatially scaled.

Pipeline: unpatchify → per-channel un-normalize → Conv3d upsampler →
normalize → optional AdaIN toward the pre-upsample latent.  Weights load
through :class:`~...loaders.module.NativeModuleLoader` (``weights`` +
``statistics`` checkpoints).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial

import torch

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest, ModalityState
from ....loaders import ModuleLoadSpec, NativeModuleLoader, safetensors_json_metadata
from ....optimizations import OffloadPolicy, QuantizationPolicy
from ...autoencoders.ltx.video.ops import PerChannelStatistics
from ...representations.ltx.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ...representations.ltx.types import AudioLatentShape, VideoLatentShape, VideoPixelShape
from .model_configurator import LatentUpsamplerConfigurator


class LTXLatentUpsamplerModule(torch.nn.Module):
    """Thin ``nn.Module`` wrapper so the native loader can own the CNN."""

    def __init__(self, checkpoint_config: Mapping[str, object]) -> None:
        super().__init__()
        self.upsampler = LatentUpsamplerConfigurator.from_config(dict(checkpoint_config))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Spatially upsample an un-normalized BCTHW video latent."""
        return self.upsampler(latent)


class LTXVideoLatentStatistics(torch.nn.Module):
    """Per-channel mean/std used before and after the spatial upsampler."""

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.statistics = PerChannelStatistics(latent_channels=latent_channels)


def convert_ltx_upsampler_state_dict(
    state_dict: Mapping[str, object],
    *,
    lazy: bool = False,
) -> Mapping[str, object]:
    """Prefix official upsampler keys with ``upsampler.`` for the wrapper module."""
    if lazy:
        from worldfoundry.core.model_loading.checkpoints.transforms import TransformedStateDict

        return TransformedStateDict(state_dict, {f"upsampler.{key}": key for key in state_dict})
    return {f"upsampler.{key}": value for key, value in state_dict.items()}


def convert_ltx_video_statistics_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    return {
        f"statistics.{key.rsplit('.', 1)[-1]}": state_dict[key]
        for key in state_dict
        if key.startswith(("vae.per_channel_statistics.", "per_channel_statistics."))
        and key.rsplit(".", 1)[-1] in {"mean-of-means", "std-of-means"}
    }


class LTXSpatialLatentProcessor:
    """Unpatchify stage one, spatially upsample video, and carry audio forward."""

    def __init__(
        self,
        upsampler: LTXLatentUpsamplerModule,
        statistics: LTXVideoLatentStatistics,
        *,
        compute_dtype: torch.dtype,
        first_stage_scale: float | None = None,
        adain_factor: float = 0.0,
        split_checkpoint: bool = False,
        strip_reference_tokens: bool = False,
    ) -> None:
        self.upsampler = upsampler
        self.statistics = statistics
        self.compute_dtype = compute_dtype
        if first_stage_scale is not None and not 0.0 < float(first_stage_scale) <= 1.0:
            raise ValueError("LTX first-stage scale must be in (0, 1]")
        self.first_stage_scale = None if first_stage_scale is None else float(first_stage_scale)
        self.adain_factor = float(adain_factor)
        self.split_checkpoint = split_checkpoint
        self.strip_reference_tokens = strip_reference_tokens

    @staticmethod
    def _round_down_to_vae(value: int) -> int:
        result = int(value) - int(value) % 32
        if result < 32:
            raise ValueError("LTX stage dimensions must be at least 32 pixels")
        return result

    def _stage_one_size(self, request: DiffusionRequest) -> tuple[int, int]:
        if self.first_stage_scale is None:
            return request.height // 2, request.width // 2
        return (
            self._round_down_to_vae(int(request.height * self.first_stage_scale)),
            self._round_down_to_vae(int(request.width * self.first_stage_scale)),
        )

    def _adain(
        self,
        latent: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if self.adain_factor == 0.0:
            return latent
        reduce_dims = tuple(range(2, latent.ndim))
        reference_std, reference_mean = torch.std_mean(reference.float(), dim=reduce_dims, keepdim=True)
        latent_std, latent_mean = torch.std_mean(latent.float(), dim=reduce_dims, keepdim=True)
        normalized = (latent.float() - latent_mean) / latent_std.clamp_min(1e-6)
        matched = normalized * reference_std + reference_mean
        return torch.lerp(latent.float(), matched, self.adain_factor).to(latent.dtype)

    def process(
        self,
        states: Mapping[str, ModalityState],
        request: DiffusionRequest,
    ) -> Mapping[str, torch.Tensor]:
        """Unpatchify stage-1 states and return dense latents for stage 2."""
        fps = float(request.inputs.get("frame_rate", 24.0))
        stage_height, stage_width = self._stage_one_size(request)
        stage_one_pixels = VideoPixelShape(
            batch=request.batch_size,
            frames=request.num_frames,
            height=stage_height,
            width=stage_width,
            fps=fps,
        )
        channels = self.statistics.statistics.get_buffer("mean-of-means").numel() if self.split_checkpoint else 128
        video_shape = VideoLatentShape.from_pixel_shape(stage_one_pixels, latent_channels=channels)
        patchifier = VideoLatentPatchifier(1)
        latent = states["video"].latent
        if self.split_checkpoint or self.strip_reference_tokens:
            latent = latent[:, : patchifier.get_token_count(video_shape)]
        video = patchifier.unpatchify(latent, video_shape)
        raw = self.statistics.statistics.un_normalize(video)
        # Scheduler arithmetic can promote the first-stage latent to fp32,
        # while the released spatial upsampler is loaded in bf16.  Upstream
        # runs this boundary in the upsampler's weight dtype; make that
        # contract explicit so Conv3d never receives a mismatched input.
        with torch.autocast(
            device_type=raw.device.type,
            dtype=self.compute_dtype,
            enabled=not self.split_checkpoint and self.compute_dtype in {torch.float16, torch.bfloat16},
        ):
            upsampled = self.upsampler(raw.to(dtype=self.compute_dtype))
        result = {
            "video": self._adain(
                self.statistics.statistics.normalize(upsampled),
                video,
            )
        }
        if "audio" in states:
            audio_shape = AudioLatentShape.from_video_pixel_shape(stage_one_pixels)
            result["audio"] = AudioPatchifier(1).unpatchify(states["audio"].latent, audio_shape)
        return result


def build_ltx_spatial_latent_processor(
    context: ComponentBuildContext,
) -> LTXSpatialLatentProcessor:
    """Load the spatial upsampler + VAE statistics and return the processor."""
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    policy = replace(context.policy, quantization=QuantizationPolicy())
    split = bool(context.component_options.get("split_checkpoint", False))
    channels = 128
    if split:
        from ....loaders import NativeCheckpointResolver

        config = safetensors_json_metadata(
            NativeCheckpointResolver().materialize(context.require_checkpoint("statistics"))
        )
        vae = config["vae"]
        channels = int(vae.get("encoder", {}).get("out_channels", vae.get("latent_channels", 128)))

    module_map = {
        torch.nn.Linear: AutoWrappedLinear,
        torch.nn.Conv2d: AutoWrappedModule,
        torch.nn.Conv3d: AutoWrappedModule,
        torch.nn.GroupNorm: AutoWrappedModule,
    }
    upsampler = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=LTXLatentUpsamplerModule,
            config_resolver=lambda checkpoint: {"checkpoint_config": safetensors_json_metadata(checkpoint)},
            state_dict_converter=partial(convert_ltx_upsampler_state_dict, lazy=True)
            if split
            else convert_ltx_upsampler_state_dict,
            vram_module_map=module_map,
            layer_container="upsampler.res_blocks",
            lazy_state_dict=split,
            buffer_aware_offload=split,
        ),
        context.require_checkpoint("weights"),
        policy,
    )
    statistics = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=LTXVideoLatentStatistics,
            config={"latent_channels": channels},
            state_dict_converter=convert_ltx_video_statistics_state_dict,
        ),
        context.require_checkpoint("statistics"),
        replace(policy, offload=OffloadPolicy()),
    )
    if not isinstance(upsampler, LTXLatentUpsamplerModule):
        raise TypeError("LTX upsampler construction returned an unexpected module")
    if not isinstance(statistics, LTXVideoLatentStatistics):
        raise TypeError("LTX latent statistics construction returned an unexpected module")
    raw_scale = context.component_options.get("first_stage_scale")
    return LTXSpatialLatentProcessor(
        upsampler,
        statistics,
        compute_dtype=context.policy.dtype,
        first_stage_scale=None if raw_scale is None else float(raw_scale),
        adain_factor=float(context.component_options.get("adain_factor", 0.0)),
        split_checkpoint=split,
        strip_reference_tokens=bool(context.component_options.get("strip_reference_tokens", False)),
    )


__all__ = [
    "LTXSpatialLatentProcessor",
    "build_ltx_spatial_latent_processor",
    "convert_ltx_upsampler_state_dict",
    "convert_ltx_video_statistics_state_dict",
]
