"""Wan :class:`~...contracts.LatentEncoder` + :class:`~...contracts.LatentDecoder` adapter.

:class:`WanVideoDecoder` wraps :class:`~.model.WanVideoVAE` /
:class:`WanVideoVAE38`.  ``encode`` maps BCTHW pixels in ``[-1, 1]`` to
latents; ``decode`` maps latents back, honoring
``DiffusionRequest.inputs``:

- ``vace_reference_count``: drop leading VACE reference tokens.
- ``frozen_context_latents``: drop the Echo-Memory context suffix.
- ``return_latent``: skip pixel decode (optionally concat
  ``first_frame_latents``).

Factories remap official / Diffusers checkpoints through
:func:`convert_wan21_vae_state_dict` and the Diffusers converters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import torch

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest
from ....loaders import CheckpointSpec, ModuleLoadSpec, NativeModuleLoader
from ....optimizations import (
    OffloadMode,
    OffloadPolicy,
    RuntimePolicy,
    parse_torch_dtype,
)
from .model import (
    CausalConv3d,
    RMS_norm,
    Upsample,
    WanVideoVAE,
    WanVideoVAE38,
    WanVideoVAEStateDictConverter,
)


class WanTAEPreviewDecoder:
    """Runner adapter for the optional TAEW2.2 fast preview decoder.

    TAEW2.2 consumes normalized Wan2.2 latents in ``BTCHW`` and emits RGB in
    ``[0, 1]``. The native runner uses ``BCTHW`` and ``[-1, 1]``; this adapter
    owns that boundary and deliberately does not pretend to provide an encoder.
    """

    def __init__(
        self,
        tae: torch.nn.Module,
        *,
        device: torch.device,
        dtype: torch.dtype,
        checkpoint_path: str | Path,
    ) -> None:
        self.vae = tae.to(device=device, dtype=dtype).eval()
        self.device = torch.device(device)
        self._dtype = dtype
        self.vae.decode_autocast_dtype = dtype
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self._decode_calls = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "WanTAEPreviewDecoder":
        from .variants.tae_22 import TAEW22StreamingDecoder

        return cls(
            TAEW22StreamingDecoder(checkpoint_path),
            device=device,
            dtype=dtype,
            checkpoint_path=checkpoint_path,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def spatial_compression_factor(self) -> int:
        return 16

    @property
    def temporal_compression_factor(self) -> int:
        return 4

    @property
    def pixel_chunk_duration(self) -> int:
        return 121

    @property
    def latent_chunk_duration(self) -> int:
        return 31

    @staticmethod
    def get_latent_num_frames(num_pixel_frames: int) -> int:
        return 1 + (int(num_pixel_frames) - 1) // 4

    @staticmethod
    def get_pixel_num_frames(num_latent_frames: int) -> int:
        return (int(num_latent_frames) - 1) * 4 + 1

    @property
    def latent_ch(self) -> int:
        return 48

    @property
    def spatial_resolution(self) -> int:
        return 1280

    @property
    def name(self) -> str:
        return "taew2.2-preview"

    def clear_cache(self) -> None:
        reset = getattr(self.vae, "reset", None)
        if callable(reset):
            reset()

    def reset_dtype(self) -> None:
        """Compatibility no-op; placement is fixed at construction."""

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        del images
        raise RuntimeError(
            "TAEW2.2 is a decode-only preview; image-conditioned Wan2.2 "
            "requires the official VAE encoder"
        )

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        request: DiffusionRequest | None = None,
    ) -> torch.Tensor:
        if latents.ndim != 5 or int(latents.shape[1]) != self.latent_ch:
            raise ValueError(
                "TAEW2.2 preview expects BCTHW latents with 48 channels, "
                f"got {tuple(latents.shape)}"
            )
        if request is not None and bool(request.inputs.get("return_latent", False)):
            return latents
        vace_reference_count = (
            int(request.inputs.get("vace_reference_count", 0))
            if request is not None
            else 0
        )
        if vace_reference_count:
            latents = latents[:, :, vace_reference_count:]
        frozen_context = request.inputs.get("frozen_context_latents") if request is not None else None
        if frozen_context is not None:
            if not isinstance(frozen_context, torch.Tensor):
                raise TypeError("frozen_context_latents must be a tensor")
            latents = latents[:, :, : -int(frozen_context.shape[-3])]
        self.clear_cache()
        self._decode_calls += 1
        preview = self.vae.decode(
            latents.to(device=self.device, dtype=self.dtype).transpose(1, 2)
        )
        if not isinstance(preview, torch.Tensor) or preview.ndim != 5:
            raise TypeError("TAEW2.2 decode must return one BTCHW tensor")
        return preview.transpose(1, 2).mul(2).sub(1).clamp_(-1, 1)

    def runtime_optimization_report(self) -> dict[str, object]:
        return {
            "requested": {
                "vae_preview_decoder_path": self.checkpoint_path,
                "vae_decode_autocast": str(self.dtype),
                "vae_spatial_tiling": False,
                "vae_temporal_chunk_size": False,
                "vae_parallel_degree": 1,
            },
            "effective": {
                "vae_decode": (
                    "taew2.2-preview" if self._decode_calls else "taew2.2-preview (pending)"
                ),
                "vae_preview_decoder": "taew2.2",
                "vae_decode_autocast": str(self.dtype),
                "vae_spatial_tiles": 0,
                "vae_parallel_degree": 1,
            },
            "fallbacks": [],
            "quality_tier": "algorithmically-approximate-preview",
            "runtime": {
                "decode_calls": self._decode_calls,
                "preview_decode_calls": self._decode_calls,
            },
        }


def convert_wan21_vae_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Map the official Wan2.1 VAE checkpoint into the native module."""

    return WanVideoVAEStateDictConverter().from_civitai(state_dict)


def _resolve_vae_decode_autocast(policy: RuntimePolicy) -> torch.dtype | None:
    """Resolve the optional output-only Wan VAE decode precision."""

    value = policy.options.get("vae_decode_autocast")
    if value is None or value is False or value == "":
        return None
    if value is True:
        value = policy.dtype if policy.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    if isinstance(value, str) and value.strip().lower() == "half":
        value = "fp16"
    dtype = value if isinstance(value, torch.dtype) else parse_torch_dtype(
        value,
        owner="Wan VAE decode autocast",
    )
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Wan VAE decode autocast supports only fp16 or bf16")
    return dtype


def _resolve_vae_weight_dtype(policy: RuntimePolicy) -> torch.dtype:
    """Resolve the opt-in resident Wan VAE parameter precision.

    Wan checkpoints historically stayed FP32 even when the surrounding
    pipeline used BF16.  Keep that compatibility default, while allowing the
    same BF16-resident codec used by LightX2V to be selected explicitly.
    """

    value = policy.options.get("vae_weight_dtype", policy.dtype)
    if value is None or value == "":
        value = torch.float32
    dtype = value if isinstance(value, torch.dtype) else parse_torch_dtype(
        value,
        owner="Wan VAE weight dtype",
    )
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Wan VAE weight dtype supports only fp16, bf16, or fp32")
    return dtype


def _resolve_vae_channels_last_3d(policy: RuntimePolicy) -> bool:
    """Resolve the explicit Conv3d memory-format optimization switch."""

    value = policy.options.get("vae_channels_last_3d", False)
    if not isinstance(value, bool):
        raise TypeError("vae_channels_last_3d must be a bool")
    return value


def _convert_conv3d_weights_channels_last_3d(
    module: torch.nn.Module,
) -> tuple[int, int]:
    """Convert and verify every materialized Conv3d weight in ``module``.

    Returns ``(effective, total)``.  Assigning the contiguous tensor to
    ``Parameter.data`` matches LightX2V's upstream optimization and preserves
    parameter identity for wrappers which already hold references to it.
    """

    total = 0
    effective = 0
    with torch.no_grad():
        for child in module.modules():
            if not isinstance(child, torch.nn.Conv3d):
                continue
            weight = child.weight
            if weight.device.type == "meta":
                raise RuntimeError(
                    "cannot apply VAE channels_last_3d to a meta Conv3d weight"
                )
            total += 1
            weight.data = weight.data.contiguous(
                memory_format=torch.channels_last_3d
            )
            if weight.is_contiguous(memory_format=torch.channels_last_3d):
                effective += 1
    return effective, total


def _wan_residual_suffix(value: str) -> str:
    replacements = {
        "norm1": "residual.0",
        "conv1": "residual.2",
        "norm2": "residual.3",
        "conv2": "residual.6",
        "conv_shortcut": "shortcut",
    }
    head, tail = value.split(".", 1)
    return f"{replacements[head]}.{tail}"


def convert_diffusers_wan22_vae_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Map the official Diffusers Wan2.2 VAE layout onto the shared native VAE38."""

    converted: dict[str, object] = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        if key.startswith("encoder.conv_in."):
            target = key.replace("encoder.conv_in", "model.encoder.conv1", 1)
        elif key.startswith("encoder.down_blocks."):
            block = parts[2]
            if parts[3] == "resnets":
                residual = parts[4]
                suffix = _wan_residual_suffix(".".join(parts[5:]))
                target = f"model.encoder.downsamples.{block}.downsamples.{residual}.{suffix}"
            else:
                suffix = ".".join(parts[4:])
                target = f"model.encoder.downsamples.{block}.downsamples.2.{suffix}"
        elif key.startswith("encoder.mid_block.resnets."):
            middle = "0" if parts[3] == "0" else "2"
            target = f"model.encoder.middle.{middle}.{_wan_residual_suffix('.'.join(parts[4:]))}"
        elif key.startswith("encoder.mid_block.attentions.0."):
            target = "model.encoder.middle.1." + ".".join(parts[4:])
        elif key.startswith("encoder.norm_out."):
            target = key.replace("encoder.norm_out", "model.encoder.head.0", 1)
        elif key.startswith("encoder.conv_out."):
            target = key.replace("encoder.conv_out", "model.encoder.head.2", 1)
        elif key.startswith("quant_conv."):
            target = key.replace("quant_conv", "model.conv1", 1)
        elif key.startswith("post_quant_conv."):
            target = key.replace("post_quant_conv", "model.conv2", 1)
        elif key.startswith("decoder.conv_in."):
            target = key.replace("decoder.conv_in", "model.decoder.conv1", 1)
        elif key.startswith("decoder.mid_block.resnets."):
            middle = "0" if parts[3] == "0" else "2"
            target = f"model.decoder.middle.{middle}.{_wan_residual_suffix('.'.join(parts[4:]))}"
        elif key.startswith("decoder.mid_block.attentions.0."):
            target = "model.decoder.middle.1." + ".".join(parts[4:])
        elif key.startswith("decoder.up_blocks."):
            block = parts[2]
            if parts[3] == "resnets":
                residual = parts[4]
                suffix = _wan_residual_suffix(".".join(parts[5:]))
                target = f"model.decoder.upsamples.{block}.upsamples.{residual}.{suffix}"
            else:
                suffix = ".".join(parts[4:])
                target = f"model.decoder.upsamples.{block}.upsamples.3.{suffix}"
        elif key.startswith("decoder.norm_out."):
            target = key.replace("decoder.norm_out", "model.decoder.head.0", 1)
        elif key.startswith("decoder.conv_out."):
            target = key.replace("decoder.conv_out", "model.decoder.head.2", 1)
        else:
            raise KeyError(f"unsupported Diffusers Wan VAE parameter: {key}")
        converted[target] = value
    return converted


def convert_diffusers_wan_vae_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Map the standard Diffusers 16-channel Wan VAE onto the native codec."""

    converted: dict[str, object] = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        if key.startswith("encoder.conv_in."):
            target = key.replace("encoder.conv_in", "model.encoder.conv1", 1)
        elif key.startswith("encoder.down_blocks."):
            index = parts[2]
            suffix = ".".join(parts[3:])
            if suffix.startswith(("norm1.", "conv1.", "norm2.", "conv2.", "conv_shortcut.")):
                suffix = _wan_residual_suffix(suffix)
            target = f"model.encoder.downsamples.{index}.{suffix}"
        elif key.startswith("encoder.mid_block.resnets."):
            middle = "0" if parts[3] == "0" else "2"
            target = f"model.encoder.middle.{middle}.{_wan_residual_suffix('.'.join(parts[4:]))}"
        elif key.startswith("encoder.mid_block.attentions.0."):
            target = "model.encoder.middle.1." + ".".join(parts[4:])
        elif key.startswith("encoder.norm_out."):
            target = key.replace("encoder.norm_out", "model.encoder.head.0", 1)
        elif key.startswith("encoder.conv_out."):
            target = key.replace("encoder.conv_out", "model.encoder.head.2", 1)
        elif key.startswith("quant_conv."):
            target = key.replace("quant_conv", "model.conv1", 1)
        elif key.startswith("post_quant_conv."):
            target = key.replace("post_quant_conv", "model.conv2", 1)
        elif key.startswith("decoder.conv_in."):
            target = key.replace("decoder.conv_in", "model.decoder.conv1", 1)
        elif key.startswith("decoder.mid_block.resnets."):
            middle = "0" if parts[3] == "0" else "2"
            target = f"model.decoder.middle.{middle}.{_wan_residual_suffix('.'.join(parts[4:]))}"
        elif key.startswith("decoder.mid_block.attentions.0."):
            target = "model.decoder.middle.1." + ".".join(parts[4:])
        elif key.startswith("decoder.up_blocks."):
            block = int(parts[2])
            if parts[3] == "resnets":
                index = block * 4 + int(parts[4])
                suffix = _wan_residual_suffix(".".join(parts[5:]))
            elif parts[3] == "upsamplers" and parts[4] == "0":
                index = block * 4 + 3
                suffix = ".".join(parts[5:])
            else:
                raise KeyError(f"unsupported Diffusers Wan VAE decoder parameter: {key}")
            target = f"model.decoder.upsamples.{index}.{suffix}"
        elif key.startswith("decoder.norm_out."):
            target = key.replace("decoder.norm_out", "model.decoder.head.0", 1)
        elif key.startswith("decoder.conv_out."):
            target = key.replace("decoder.conv_out", "model.decoder.head.2", 1)
        else:
            raise KeyError(f"unsupported Diffusers Wan VAE parameter: {key}")
        if target in converted:
            raise KeyError(f"Wan VAE conversion produced duplicate parameter: {target}")
        converted[target] = value
    return converted


class WanVideoDecoder:
    """Decode Wan latents to normalized ``[B, C, T, H, W]`` video."""

    def __init__(
        self,
        vae: WanVideoVAE,
        *,
        device: torch.device,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
        temporal_chunk_size: int = 0,
        parallel_degree: int = 1,
        chunk_duration: int = 81,
        offload_requested: str = "none",
        offload_effective: str = "resident",
        weight_dtype_requested: torch.dtype = torch.float32,
        channels_last_3d_requested: bool = False,
        channels_last_3d_effective: int = 0,
        conv3d_count: int = 0,
    ) -> None:
        self.vae = vae
        self.device = device
        self.parallel_degree = int(parallel_degree)
        if self.parallel_degree <= 0:
            raise ValueError("Wan VAE parallel degree must be positive")
        self.tiled = bool(tiled or self.parallel_degree > 1)
        self.tile_size = tuple(int(value) for value in tile_size)
        self.tile_stride = tuple(int(value) for value in tile_stride)
        if len(self.tile_size) != 2 or len(self.tile_stride) != 2:
            raise ValueError("Wan VAE tile_size and tile_stride must contain two values")
        if any(value <= 0 for value in (*self.tile_size, *self.tile_stride)):
            raise ValueError("Wan VAE tile sizes and strides must be positive")
        if any(
            stride > size
            for stride, size in zip(self.tile_stride, self.tile_size)
        ):
            raise ValueError("Wan VAE tile stride cannot exceed tile size")
        self.temporal_chunk_size = int(temporal_chunk_size)
        if self.temporal_chunk_size < 0:
            raise ValueError("Wan VAE temporal chunk size cannot be negative")
        if self.parallel_degree > 1:
            import torch.distributed as dist

            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "Wan VAE parallel decode requires an initialized torchrun process group"
                )
            active_world_size = int(dist.get_world_size())
            if active_world_size != self.parallel_degree:
                raise RuntimeError(
                    f"Wan VAE parallel degree {self.parallel_degree} requires "
                    f"WORLD_SIZE={self.parallel_degree}, got {active_world_size}"
                )
        self.chunk_duration = int(chunk_duration)
        if self.chunk_duration <= 0:
            raise ValueError("Wan codec chunk_duration must be positive")
        self.offload_requested = str(offload_requested)
        self.offload_effective = str(offload_effective)
        self.weight_dtype_requested = weight_dtype_requested
        self.channels_last_3d_requested = bool(channels_last_3d_requested)
        self.channels_last_3d_effective = int(channels_last_3d_effective)
        self.conv3d_count = int(conv3d_count)
        self._decode_calls = 0
        self._dense_decode_calls = 0
        self._spatial_tiled_decode_calls = 0
        self._temporal_chunked_decode_calls = 0
        self._parallel_tiled_decode_calls = 0
        self._last_spatial_tile_count = 0
        self._single_spatial_tile_calls = 0
        self._last_temporal_chunk_count = 0
        self._single_temporal_chunk_calls = 0
        self._fallbacks: list[str] = []
        self._decode_window = 0
        self._lifetime_decode_calls = 0
        self._lifetime_dense_decode_calls = 0
        self._lifetime_spatial_tiled_decode_calls = 0
        self._lifetime_temporal_chunked_decode_calls = 0
        self._lifetime_parallel_tiled_decode_calls = 0
        self._lifetime_single_spatial_tile_calls = 0
        self._lifetime_single_temporal_chunk_calls = 0

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    @property
    def spatial_compression_factor(self) -> int:
        return 8

    @property
    def temporal_compression_factor(self) -> int:
        return 4

    @property
    def pixel_chunk_duration(self) -> int:
        return self.chunk_duration

    @property
    def latent_chunk_duration(self) -> int:
        return self.get_latent_num_frames(self.chunk_duration)

    @property
    def latent_ch(self) -> int:
        return int(self.vae.z_dim)

    @property
    def spatial_resolution(self) -> int:
        return 512

    @property
    def name(self) -> str:
        return "wan2pt1_tokenizer"

    @staticmethod
    def get_latent_num_frames(num_pixel_frames: int) -> int:
        return 1 + (int(num_pixel_frames) - 1) // 4

    @staticmethod
    def get_pixel_num_frames(num_latent_frames: int) -> int:
        return (int(num_latent_frames) - 1) * 4 + 1

    def clear_cache(self) -> None:
        self.vae.model.clear_cache()

    def reset_dtype(self) -> None:
        """Compatibility no-op; dtype placement belongs to RuntimePolicy."""

    def _begin_decode_window(self) -> None:
        """Reset last-call telemetry without discarding lifetime diagnostics."""

        self._decode_window += 1
        self._decode_calls = 0
        self._dense_decode_calls = 0
        self._spatial_tiled_decode_calls = 0
        self._temporal_chunked_decode_calls = 0
        self._parallel_tiled_decode_calls = 0
        self._last_spatial_tile_count = 0
        self._single_spatial_tile_calls = 0
        self._last_temporal_chunk_count = 0
        self._single_temporal_chunk_calls = 0
        self._fallbacks.clear()

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode BCTHW pixels; optional spatial tiling via constructor flags."""
        if images.ndim != 5:
            raise ValueError(f"Wan encoder expects BCTHW pixels, got {tuple(images.shape)}")
        return self.vae.encode(
            images.to(device=self.device, dtype=self.dtype),
            self.device,
            tiled=self.tiled,
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
        )

    def decode(
        self,
        latents: torch.Tensor,
        request: DiffusionRequest | None = None,
    ) -> torch.Tensor:
        """Decode latents to BCTHW pixels, stripping VACE / Echo suffixes."""
        # One decoder invocation belongs to one diffusion request. Reset before
        # every early-return seam as well, otherwise a return-latent request can
        # inherit a previous request's multi-tile/chunk execution receipts.
        self._begin_decode_window()
        vace_reference_count = (
            int(request.inputs.get("vace_reference_count", 0))
            if request is not None
            else 0
        )
        if vace_reference_count < 0 or vace_reference_count >= int(latents.shape[2]):
            raise ValueError(
                "vace_reference_count must be non-negative and smaller than the "
                "latent sequence"
            )
        if vace_reference_count:
            # VACE prepends one latent-time token per reference image.  These
            # tokens condition the transformer but are not part of the output
            # video, matching the official ``decode_latent`` implementation.
            latents = latents[:, :, vace_reference_count:]
        frozen_context = request.inputs.get("frozen_context_latents") if request is not None else None
        if frozen_context is not None:
            if not isinstance(frozen_context, torch.Tensor):
                raise TypeError("frozen_context_latents must be a tensor")
            context_frames = int(frozen_context.shape[-3])
            if context_frames <= 0 or context_frames >= int(latents.shape[2]):
                raise ValueError("frozen context length must be smaller than the full latent sequence")
            latents = latents[:, :, :-context_frames]
        if request is not None and bool(request.inputs.get("return_latent", False)):
            prefix = request.inputs.get("first_frame_latents")
            if prefix is None:
                return latents
            if not isinstance(prefix, torch.Tensor):
                raise TypeError("first_frame_latents must be a tensor when returning Wan latents")
            prefix = prefix.to(device=latents.device, dtype=latents.dtype)
            if prefix.shape[:2] != latents.shape[:2] or prefix.shape[-2:] != latents.shape[-2:]:
                raise ValueError("first_frame_latents must match the generated latent batch/channel/spatial shape")
            return torch.cat([prefix, latents], dim=2)
        prepared = latents.to(device=self.device, dtype=self.dtype)
        self._decode_calls += 1
        self._lifetime_decode_calls += 1
        if self.tiled:
            tasks = self.vae._spatial_tile_tasks(
                int(prepared.shape[-2]),
                int(prepared.shape[-1]),
                self.tile_size,
                self.tile_stride,
            )
            self._last_spatial_tile_count = len(tasks)
            if len(tasks) <= 1:
                self._single_spatial_tile_calls += 1
                self._lifetime_single_spatial_tile_calls += 1
                fallback = (
                    "vae_spatial_tiling: requested shape produced one tile; "
                    "no spatial sharding benefit"
                )
                if fallback not in self._fallbacks:
                    self._fallbacks.append(fallback)
        if self.temporal_chunk_size:
            latent_frames = int(prepared.shape[2])
            self._last_temporal_chunk_count = (
                latent_frames + self.temporal_chunk_size - 1
            ) // self.temporal_chunk_size
            if self._last_temporal_chunk_count > 1:
                self._temporal_chunked_decode_calls += 1
                self._lifetime_temporal_chunked_decode_calls += 1
            else:
                self._single_temporal_chunk_calls += 1
                self._lifetime_single_temporal_chunk_calls += 1
                fallback = (
                    "vae_temporal_chunking: latent sequence fits in one chunk; "
                    "no multi-chunk streaming benefit"
                )
                if fallback not in self._fallbacks:
                    self._fallbacks.append(fallback)
        if self.parallel_degree > 1:
            self._parallel_tiled_decode_calls += 1
            self._lifetime_parallel_tiled_decode_calls += 1
            with self.vae._decode_autocast(prepared.dtype):
                videos = [
                    self.vae.parallel_tiled_decode(
                        hidden_state.unsqueeze(0),
                        self.device,
                        self.tile_size,
                        self.tile_stride,
                        temporal_chunk_size=self.temporal_chunk_size,
                    ).squeeze(0)
                    for hidden_state in prepared.to("cpu")
                ]
            return torch.stack(videos)

        if self.tiled:
            self._spatial_tiled_decode_calls += 1
            self._lifetime_spatial_tiled_decode_calls += 1
        elif not self.temporal_chunk_size:
            self._dense_decode_calls += 1
            self._lifetime_dense_decode_calls += 1
        return self.vae.decode(
            prepared,
            self.device,
            tiled=self.tiled,
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
            temporal_chunk_size=self.temporal_chunk_size,
        )

    def runtime_optimization_report(self) -> dict[str, object]:
        """Report requested and actually executed Wan VAE decode paths."""

        autocast_dtype = getattr(self.vae, "decode_autocast_dtype", None)
        autocast_elided = bool(
            autocast_dtype is not None
            and getattr(self.vae, "_decode_autocast_is_redundant", lambda _dtype: False)(
                self.dtype
            )
        )
        requested = {
            "vae_weight_dtype": str(self.weight_dtype_requested),
            "vae_channels_last_3d": self.channels_last_3d_requested,
            "vae_decode_autocast": str(autocast_dtype) if autocast_dtype else False,
            "vae_spatial_tiling": self.tiled,
            "vae_tile_size": self.tile_size,
            "vae_tile_stride": self.tile_stride,
            "vae_temporal_chunk_size": self.temporal_chunk_size or False,
            "vae_parallel_degree": self.parallel_degree,
            "vae_offload": self.offload_requested,
        }
        if self._parallel_tiled_decode_calls:
            decode_effective = "parallel-spatial-tiled"
        elif self._spatial_tiled_decode_calls:
            decode_effective = "spatial-tiled"
        elif self._temporal_chunked_decode_calls:
            decode_effective = "causal-temporal-streaming"
        elif self._dense_decode_calls:
            decode_effective = "dense"
        else:
            decode_effective = "pending"
        if self._single_spatial_tile_calls and "tiled" in decode_effective:
            decode_effective += " (single-tile)"
        if self._single_temporal_chunk_calls:
            decode_effective += " (single-temporal-chunk)"
        effective = {
            "vae_weight_dtype": str(self.dtype),
            "vae_channels_last_3d": (
                "enabled" if self.channels_last_3d_requested else "disabled"
            ),
            "vae_channels_last_3d_conv3d": self.channels_last_3d_effective,
            "vae_conv3d_count": self.conv3d_count,
            "vae_decode": decode_effective,
            "vae_decode_autocast": str(autocast_dtype) if autocast_dtype else "disabled",
            "vae_decode_autocast_context": (
                "elided-resident-dtype"
                if autocast_elided
                else ("enabled" if autocast_dtype else "disabled")
            ),
            "vae_spatial_tiles": self._last_spatial_tile_count,
            "vae_parallel_degree": (
                self.parallel_degree if self._parallel_tiled_decode_calls else 1
            ),
            "vae_offload": self.offload_effective,
        }
        approximate = bool(
            autocast_dtype is not None
            or self.tiled
            or self.dtype in (torch.float16, torch.bfloat16)
        )
        return {
            "requested": requested,
            "effective": effective,
            "fallbacks": list(self._fallbacks),
            "quality_tier": "numerically-approximate" if approximate else "exact",
            "runtime": {
                "window_id": self._decode_window,
                "vae_channels_last_3d_conv3d": self.channels_last_3d_effective,
                "vae_conv3d_count": self.conv3d_count,
                "vae_decode_autocast_context": (
                    "elided-resident-dtype"
                    if autocast_elided
                    else ("enabled" if autocast_dtype else "disabled")
                ),
                "decode_calls": self._decode_calls,
                "dense_decode_calls": self._dense_decode_calls,
                "spatial_tiled_decode_calls": self._spatial_tiled_decode_calls,
                "temporal_chunked_decode_calls": self._temporal_chunked_decode_calls,
                "parallel_tiled_decode_calls": self._parallel_tiled_decode_calls,
                "last_spatial_tile_count": self._last_spatial_tile_count,
                "single_spatial_tile_calls": self._single_spatial_tile_calls,
                "last_temporal_chunk_count": self._last_temporal_chunk_count,
                "single_temporal_chunk_calls": self._single_temporal_chunk_calls,
                "lifetime": {
                    "decode_calls": self._lifetime_decode_calls,
                    "dense_decode_calls": self._lifetime_dense_decode_calls,
                    "spatial_tiled_decode_calls": (
                        self._lifetime_spatial_tiled_decode_calls
                    ),
                    "temporal_chunked_decode_calls": (
                        self._lifetime_temporal_chunked_decode_calls
                    ),
                    "parallel_tiled_decode_calls": (
                        self._lifetime_parallel_tiled_decode_calls
                    ),
                    "single_spatial_tile_calls": (
                        self._lifetime_single_spatial_tile_calls
                    ),
                    "single_temporal_chunk_calls": (
                        self._lifetime_single_temporal_chunk_calls
                    ),
                },
            },
        }


def _load_wan_video_decoder(
    checkpoint: CheckpointSpec,
    policy: RuntimePolicy,
    *,
    module_class: type[WanVideoVAE],
    state_dict_converter=convert_wan21_vae_state_dict,
    tiled: bool = False,
    tile_size: tuple[int, int] = (34, 34),
    tile_stride: tuple[int, int] = (18, 16),
    temporal_chunk_size: int = 0,
    parallel_degree: int = 1,
    chunk_duration: int = 81,
) -> WanVideoDecoder:
    """Load one Wan VAE variant through the shared native module loader."""

    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    requested_offload = policy.offload.mode.value
    weight_dtype = _resolve_vae_weight_dtype(policy)
    channels_last_3d = _resolve_vae_channels_last_3d(policy)
    vae_policy = replace(policy, dtype=weight_dtype)
    vae_offload_effective = (
        "resident" if policy.offload.mode is OffloadMode.NONE else requested_offload
    )
    if policy.offload.mode is OffloadMode.BLOCK:
        # BLOCK is a two-layer transformer-window contract.  Wan's VAE has no
        # compatible block container; sending it through the generic wrapper
        # map used to deep-copy CausalConv3d/RMS_norm/Conv2d modules on every
        # encode/decode call.  Keep the comparatively small codec resident so
        # the denoiser remains the only asynchronous block-offload owner.
        # This is an explicit component policy, not an async-VAE claim.
        vae_policy = replace(vae_policy, offload=OffloadPolicy())
        vae_offload_effective = "resident-no-legacy-wrapper"

    vae = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=module_class,
            state_dict_converter=state_dict_converter,
            vram_module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
        ),
        checkpoint,
        vae_policy,
    )
    if not isinstance(vae, module_class):
        raise TypeError(f"expected {module_class.__name__}, got {type(vae).__name__}")
    vae.decode_autocast_dtype = _resolve_vae_decode_autocast(policy)
    channels_last_effective = 0
    conv3d_count = sum(
        isinstance(child, torch.nn.Conv3d) for child in vae.modules()
    )
    if channels_last_3d:
        channels_last_effective, conv3d_count = (
            _convert_conv3d_weights_channels_last_3d(vae)
        )
        if conv3d_count == 0:
            raise RuntimeError(
                "vae_channels_last_3d was requested but the Wan VAE has no Conv3d weights"
            )
        if channels_last_effective != conv3d_count:
            raise RuntimeError(
                "vae_channels_last_3d conversion was incomplete: "
                f"effective={channels_last_effective}, total={conv3d_count}"
            )
    if len(tile_size) != 2 or len(tile_stride) != 2:
        raise ValueError("Wan VAE tile_size and tile_stride must contain two values")
    return WanVideoDecoder(
        vae,
        device=policy.device,
        tiled=bool(tiled),
        tile_size=(int(tile_size[0]), int(tile_size[1])),
        tile_stride=(int(tile_stride[0]), int(tile_stride[1])),
        temporal_chunk_size=temporal_chunk_size,
        parallel_degree=parallel_degree,
        chunk_duration=chunk_duration,
        offload_requested=requested_offload,
        offload_effective=vae_offload_effective,
        weight_dtype_requested=weight_dtype,
        channels_last_3d_requested=channels_last_3d,
        channels_last_3d_effective=channels_last_effective,
        conv3d_count=conv3d_count,
    )


def _build_wan_video_decoder(
    context: ComponentBuildContext,
    *,
    module_class: type[WanVideoVAE],
    state_dict_converter=convert_wan21_vae_state_dict,
) -> WanVideoDecoder:
    if "dtype" in context.component_options:
        codec_dtype = context.component_options["dtype"]
    elif "vae_weight_dtype" in context.policy.options:
        codec_dtype = _resolve_vae_weight_dtype(context.policy)
    else:
        codec_dtype = torch.float32
    if not isinstance(codec_dtype, torch.dtype):
        raise TypeError(f"Wan codec dtype must be a torch.dtype, got {codec_dtype!r}")
    raw_parallel_degree = context.component_options.get(
        "parallel_degree",
        context.policy.options.get(
            "vae_parallel",
            context.policy.options.get("vae_parallel_degree", 1),
        ),
    )
    parallel_degree = (
        1 if raw_parallel_degree is None or raw_parallel_degree is False else int(raw_parallel_degree)
    )
    return _load_wan_video_decoder(
        context.require_checkpoint("weights"),
        replace(context.policy, dtype=codec_dtype),
        module_class=module_class,
        state_dict_converter=state_dict_converter,
        tiled=bool(
            context.component_options.get(
                "tiled",
                context.policy.options.get(
                    "vae_spatial_tiling",
                    context.policy.options.get("vae_tiled_decode", False),
                ),
            )
        ),
        tile_size=tuple(
            context.component_options.get(
                "tile_size",
                context.policy.options.get("vae_tile_size", (34, 34)),
            )
        ),
        tile_stride=tuple(
            context.component_options.get(
                "tile_stride",
                context.policy.options.get("vae_tile_stride", (18, 16)),
            )
        ),
        temporal_chunk_size=int(
            context.component_options.get(
                "temporal_chunk_size",
                context.policy.options.get("vae_temporal_chunk_size", 0),
            )
        ),
        parallel_degree=parallel_degree,
        chunk_duration=int(context.component_options.get("chunk_duration", 81)),
    )


def load_wan_video_codec(
    checkpoint_path: str | Path | CheckpointSpec,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    chunk_duration: int = 81,
) -> WanVideoDecoder:
    """Load the shared Wan2.1 codec for native and representation consumers."""

    if isinstance(checkpoint_path, CheckpointSpec):
        checkpoint = checkpoint_path
    else:
        source = str(checkpoint_path)
        if source.startswith("hf://"):
            from worldfoundry.core.io.assets.easy_io import resolve_checkpoint_path

            source = resolve_checkpoint_path(source)
        checkpoint = CheckpointSpec(source=source)
    return _load_wan_video_decoder(
        checkpoint,
        RuntimePolicy(device=device, dtype=dtype),
        module_class=WanVideoVAE,
        chunk_duration=chunk_duration,
    )


def build_wan_video_decoder(context: ComponentBuildContext) -> WanVideoDecoder:
    """Build the 16-channel Wan video decoder."""

    return _build_wan_video_decoder(context, module_class=WanVideoVAE)


def build_wan_video_vae38_decoder(
    context: ComponentBuildContext,
) -> WanVideoDecoder | WanTAEPreviewDecoder:
    """Build the 48-channel Wan2.2 video decoder."""

    preview_path = context.component_options.get(
        "preview_decoder_path",
        context.policy.options.get("vae_preview_decoder_path"),
    )
    if preview_path not in (None, False, ""):
        if not isinstance(preview_path, (str, Path)):
            raise TypeError("Wan2.2 preview decoder path must be a local path")
        preview_dtype = _resolve_vae_decode_autocast(context.policy) or context.policy.dtype
        if preview_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("TAEW2.2 preview supports fp16, bf16, or fp32")
        return WanTAEPreviewDecoder.from_checkpoint(
            preview_path,
            device=context.policy.device,
            dtype=preview_dtype,
        )
    return _build_wan_video_decoder(context, module_class=WanVideoVAE38)


def build_diffusers_wan_video_codec(context: ComponentBuildContext) -> WanVideoDecoder:
    """Build the shared 16-channel Wan codec from Diffusers-layout weights."""

    return _build_wan_video_decoder(
        context,
        module_class=WanVideoVAE,
        state_dict_converter=convert_diffusers_wan_vae_state_dict,
    )


__all__ = [
    "WanTAEPreviewDecoder",
    "WanVideoDecoder",
    "build_diffusers_wan_video_codec",
    "build_wan_video_decoder",
    "build_wan_video_vae38_decoder",
    "convert_wan21_vae_state_dict",
    "convert_diffusers_wan22_vae_state_dict",
    "convert_diffusers_wan_vae_state_dict",
    "load_wan_video_codec",
]
