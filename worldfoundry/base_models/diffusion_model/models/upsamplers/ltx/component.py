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

import torch

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest, ModalityState
from ....loaders import ModuleLoadSpec, NativeModuleLoader, safetensors_json_metadata
from ....optimizations import OffloadPolicy
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


def convert_ltx_upsampler_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Prefix official upsampler keys with ``upsampler.`` for the wrapper module."""
    return {f"upsampler.{key}": value for key, value in state_dict.items()}


def convert_ltx_video_statistics_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    prefix = "vae.per_channel_statistics."
    return {
        f"statistics.{key.removeprefix(prefix)}": value
        for key, value in state_dict.items()
        if key.startswith(prefix) and key.removeprefix(prefix) in {"mean-of-means", "std-of-means"}
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
    ) -> None:
        self.upsampler = upsampler
        self.statistics = statistics
        self.compute_dtype = compute_dtype
        if first_stage_scale is not None and not 0.0 < float(first_stage_scale) <= 1.0:
            raise ValueError("LTX first-stage scale must be in (0, 1]")
        self.first_stage_scale = None if first_stage_scale is None else float(first_stage_scale)
        self.adain_factor = float(adain_factor)

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
        video_shape = VideoLatentShape.from_pixel_shape(stage_one_pixels)
        video = VideoLatentPatchifier(1).unpatchify(states["video"].latent, video_shape)
        raw = self.statistics.statistics.un_normalize(video)
        # Scheduler arithmetic can promote the first-stage latent to fp32,
        # while the released spatial upsampler is loaded in bf16.  Upstream
        # runs this boundary in the upsampler's weight dtype; make that
        # contract explicit so Conv3d never receives a mismatched input.
        with torch.autocast(
            device_type=raw.device.type,
            dtype=self.compute_dtype,
            enabled=self.compute_dtype in {torch.float16, torch.bfloat16},
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
            state_dict_converter=convert_ltx_upsampler_state_dict,
            vram_module_map=module_map,
            layer_container="upsampler.res_blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    statistics = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=LTXVideoLatentStatistics,
            state_dict_converter=convert_ltx_video_statistics_state_dict,
        ),
        context.require_checkpoint("statistics"),
        replace(context.policy, offload=OffloadPolicy()),
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
    )


__all__ = [
    "LTXSpatialLatentProcessor",
    "build_ltx_spatial_latent_processor",
    "convert_ltx_upsampler_state_dict",
    "convert_ltx_video_statistics_state_dict",
]
