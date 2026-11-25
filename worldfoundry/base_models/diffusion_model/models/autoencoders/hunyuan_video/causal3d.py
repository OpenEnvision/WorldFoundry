"""Monolithic facade for the original HunyuanVideo causal 3-D VAE.

Composes :mod:`.original_encoder` and :mod:`.original_decoder`. Public ``encode`` / ``decode`` match the official tiling API."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import distributed as dist
from torch import nn

from worldfoundry.core.model_loading import load_state_dict as load_core_state_dict
from worldfoundry.core.model_loading.model_configuration import ConfigNamespace
from worldfoundry.core.nn.latent.distributions import (
    AutoencoderKLOutput,
    DecoderOutput,
    DiagonalGaussianDistribution,
)

from .original_decoder import DecoderCausal3D
from .original_encoder import EncoderCausal3D


class HunyuanVideoCausal3DAutoencoder(nn.Module):
    """Framework-independent VAE compatible with original-release checkpoints.

    The public methods intentionally retain the small encode/decode/config API
    used by older in-tree Hunyuan-derived models.  Its implementation is plain
    PyTorch and shares the same encoder/decoder modules as the native recipe.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str, ...] = ("DownEncoderBlockCausal3D",),
        up_block_types: tuple[str, ...] = ("UpDecoderBlockCausal3D",),
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 16,
        norm_num_groups: int = 32,
        sample_size: int | tuple[int, int] = 256,
        sample_tsize: int = 64,
        scaling_factor: float = 0.476986,
        force_upcast: bool = True,
        spatial_compression_ratio: int = 8,
        time_compression_ratio: int = 4,
        mid_block_add_attention: bool = True,
        **unused: object,
    ) -> None:
        super().__init__()
        del down_block_types, up_block_types, act_fn, force_upcast, mid_block_add_attention, unused
        channels = tuple(int(value) for value in block_out_channels)
        self.encoder = EncoderCausal3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=list(channels),
            layers_per_block=layers_per_block,
            num_groups=norm_num_groups,
            time_compression_ratio=time_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
        )
        self.decoder = DecoderCausal3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=list(channels),
            layers_per_block=layers_per_block,
            num_groups=norm_num_groups,
            time_compression_ratio=time_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
        )
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)
        self.config = ConfigNamespace(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=channels,
            layers_per_block=layers_per_block,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            sample_tsize=sample_tsize,
            scaling_factor=float(scaling_factor),
            spatial_compression_ratio=int(spatial_compression_ratio),
            time_compression_ratio=int(time_compression_ratio),
        )
        self.use_slicing = False
        self.use_spatial_tiling = False
        self.use_temporal_tiling = False
        self.tile_sample_min_tsize = int(sample_tsize)
        self.tile_latent_min_tsize = max(int(sample_tsize) // int(time_compression_ratio), 1)
        spatial_size = sample_size[0] if isinstance(sample_size, (tuple, list)) else sample_size
        self.tile_sample_min_size = int(spatial_size)
        self.tile_latent_min_size = max(int(spatial_size) // int(spatial_compression_ratio), 1)
        self.tile_overlap_factor = 0.25
        self._tile_parallelism_enabled = False
        self._tile_parallel_group: dist.ProcessGroup | None = None

    @classmethod
    def load_config(cls, path: str | Path) -> dict[str, object]:
        config_path = Path(path).expanduser()
        if config_path.is_dir():
            config_path = config_path / "config.json"
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @classmethod
    def from_config(cls, config: Mapping[str, object], **overrides: object) -> HunyuanVideoCausal3DAutoencoder:
        values = {key: value for key, value in dict(config).items() if not key.startswith("_")}
        values.update(overrides)
        return cls(**values)

    @property
    def device(self) -> torch.device:
        return self.quant_conv.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.quant_conv.weight.dtype

    def enable_temporal_tiling(self, enabled: bool = True) -> None:
        self.use_temporal_tiling = bool(enabled)

    def disable_temporal_tiling(self) -> None:
        self.enable_temporal_tiling(False)

    def enable_spatial_tiling(self, enabled: bool = True) -> None:
        self.use_spatial_tiling = bool(enabled)

    def disable_spatial_tiling(self) -> None:
        self.enable_spatial_tiling(False)

    def enable_tiling(self, enabled: bool = True) -> None:
        self.enable_spatial_tiling(enabled)
        self.enable_temporal_tiling(enabled)

    def disable_tiling(self) -> None:
        self.enable_tiling(False)

    def enable_slicing(self) -> None:
        self.use_slicing = True

    def disable_slicing(self) -> None:
        self.use_slicing = False

    def enable_tile_parallelism(
        self,
        enabled: bool = True,
        *,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        """Enable or disable multi-GPU spatial tile sharding during tiled decode."""
        self._tile_parallelism_enabled = bool(enabled)
        if group is not None or not self._tile_parallelism_enabled:
            self._tile_parallel_group = group if self._tile_parallelism_enabled else None

    def disable_tile_parallelism(self) -> None:
        self.enable_tile_parallelism(False)

    @staticmethod
    def _blend(a: torch.Tensor, b: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
        extent = min(a.shape[dim], b.shape[dim], int(extent))
        if extent <= 0:
            return b
        axis = dim % b.ndim
        ratio_shape = [1] * b.ndim
        ratio_shape[axis] = extent
        ratios = (
            torch.arange(extent, device=b.device, dtype=torch.float32).div_(extent).view(*ratio_shape).to(dtype=b.dtype)
        )
        source = [slice(None)] * b.ndim
        target = [slice(None)] * b.ndim
        source[axis] = slice(a.shape[axis] - extent, a.shape[axis])
        target[axis] = slice(0, extent)
        b[tuple(target)] = a[tuple(source)] * (1.0 - ratios) + b[tuple(target)] * ratios
        return b

    def _encode_moments(self, value: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(value))

    def _decode_sample(self, value: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(value))

    def _spatial_tiled_encode(self, value: torch.Tensor, *, moments_only: bool = False):
        overlap = max(int(self.tile_sample_min_size * (1.0 - self.tile_overlap_factor)), 1)
        blend = int(self.tile_latent_min_size * self.tile_overlap_factor)
        limit = self.tile_latent_min_size - blend
        rows: list[list[torch.Tensor]] = []
        for height in range(0, value.shape[-2], overlap):
            row = []
            for width in range(0, value.shape[-1], overlap):
                tile = value[
                    ..., height : height + self.tile_sample_min_size, width : width + self.tile_sample_min_size
                ]
                row.append(self._encode_moments(tile))
            rows.append(row)
        merged_rows = []
        for row_index, row in enumerate(rows):
            merged = []
            for column_index, tile in enumerate(row):
                if row_index:
                    tile = self._blend(rows[row_index - 1][column_index], tile, blend, -2)
                if column_index:
                    tile = self._blend(row[column_index - 1], tile, blend, -1)
                merged.append(tile[..., :limit, :limit])
            merged_rows.append(torch.cat(merged, dim=-1))
        moments = torch.cat(merged_rows, dim=-2)
        if moments_only:
            return moments
        return AutoencoderKLOutput(DiagonalGaussianDistribution(moments))

    def _merge_spatial_rows(
        self,
        rows: list[list[torch.Tensor]],
        blend: int,
        limit: int,
    ) -> torch.Tensor:
        merged_rows = []
        for row_index, row in enumerate(rows):
            merged = []
            for column_index, tile in enumerate(row):
                if row_index:
                    tile = self._blend(rows[row_index - 1][column_index], tile, blend, -2)
                if column_index:
                    tile = self._blend(row[column_index - 1], tile, blend, -1)
                merged.append(tile[..., :limit, :limit])
            merged_rows.append(torch.cat(merged, dim=-1))
        return torch.cat(merged_rows, dim=-2)

    def parallel_spatial_tiled_decode(
        self,
        value: torch.Tensor,
        *,
        group: dist.ProcessGroup | None = None,
    ) -> DecoderOutput:
        """Shard spatial tiles across ranks, gather via contiguous collective, and blend on-device."""
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("parallel HunyuanVideo spatial tiled decode requires an initialized process group")
        comm_group = group if group is not None else self._tile_parallel_group
        world_size = int(dist.get_world_size(comm_group))
        rank = int(dist.get_rank(comm_group))
        if world_size <= 1:
            raise RuntimeError("parallel HunyuanVideo spatial tiled decode requires more than one rank")

        batch_size, _channels, latent_frames, height, width = value.shape
        overlap = max(int(self.tile_latent_min_size * (1.0 - self.tile_overlap_factor)), 1)
        blend = int(self.tile_sample_min_size * self.tile_overlap_factor)
        limit = self.tile_sample_min_size - blend

        row_starts = list(range(0, height, overlap))
        col_starts = list(range(0, width, overlap))
        num_rows = len(row_starts)
        num_cols = len(col_starts)
        total_tiles = num_rows * num_cols
        tiles_per_rank = math.ceil(total_tiles / world_size)

        spatial_ratio = int(self.config.spatial_compression_ratio)
        time_ratio = int(self.config.time_compression_ratio)
        out_channels = int(self.config.out_channels)
        out_frames = (latent_frames - 1) * time_ratio + 1
        max_tile_h = max(min(self.tile_latent_min_size, height - r) for r in row_starts) * spatial_ratio
        max_tile_w = max(min(self.tile_latent_min_size, width - c) for c in col_starts) * spatial_ratio

        if value.device.type in ("cuda", "cpu") and torch.is_autocast_enabled(value.device.type):
            out_dtype = torch.get_autocast_dtype(value.device.type)
        else:
            conv_out = getattr(self.decoder, "conv_out", None)
            inner_conv = getattr(conv_out, "conv", conv_out)
            weight = getattr(inner_conv, "weight", None)
            out_dtype = weight.dtype if isinstance(weight, torch.Tensor) else value.dtype

        local_tiles = torch.zeros(
            (tiles_per_rank, batch_size, out_channels, out_frames, max_tile_h, max_tile_w),
            device=value.device,
            dtype=out_dtype,
        )
        for slot, lin_idx in enumerate(range(rank, total_tiles, world_size)):
            ri = lin_idx // num_cols
            rj = lin_idx % num_cols
            r_start = row_starts[ri]
            c_start = col_starts[rj]
            tile = value[
                ...,
                r_start : r_start + self.tile_latent_min_size,
                c_start : c_start + self.tile_latent_min_size,
            ]
            dec = self._decode_sample(tile)
            if dec.dtype != local_tiles.dtype:
                local_tiles = local_tiles.to(dtype=dec.dtype)
            local_tiles[slot, :, :, :, : dec.shape[-2], : dec.shape[-1]].copy_(dec)

        gathered_tiles = torch.empty(
            (world_size, *local_tiles.shape),
            device=local_tiles.device,
            dtype=local_tiles.dtype,
        )
        if local_tiles.is_cuda and hasattr(dist, "all_gather_into_tensor"):
            dist.all_gather_into_tensor(gathered_tiles, local_tiles, group=comm_group)
        else:
            dist.all_gather(list(gathered_tiles.unbind(0)), local_tiles, group=comm_group)

        rows: list[list[torch.Tensor | None]] = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        for r in range(world_size):
            for k in range(tiles_per_rank):
                lin_idx = r + k * world_size
                if lin_idx >= total_tiles:
                    break
                ri = lin_idx // num_cols
                rj = lin_idx % num_cols
                tile_out_h = min(self.tile_latent_min_size, height - row_starts[ri]) * spatial_ratio
                tile_out_w = min(self.tile_latent_min_size, width - col_starts[rj]) * spatial_ratio
                rows[ri][rj] = gathered_tiles[r, k, :, :, :, :tile_out_h, :tile_out_w]

        populated_rows: list[list[torch.Tensor]] = []
        for row in rows:
            populated_row: list[torch.Tensor] = []
            for tile in row:
                if tile is None:
                    raise RuntimeError("parallel HunyuanVideo gather did not return every spatial tile")
                populated_row.append(tile)
            populated_rows.append(populated_row)
        return DecoderOutput(self._merge_spatial_rows(populated_rows, blend, limit))

    def _spatial_tiled_decode(self, value: torch.Tensor) -> DecoderOutput:
        if (
            self._tile_parallelism_enabled
            and dist.is_available()
            and dist.is_initialized()
            and int(dist.get_world_size(self._tile_parallel_group)) > 1
        ):
            return self.parallel_spatial_tiled_decode(value, group=self._tile_parallel_group)

        overlap = max(int(self.tile_latent_min_size * (1.0 - self.tile_overlap_factor)), 1)
        blend = int(self.tile_sample_min_size * self.tile_overlap_factor)
        limit = self.tile_sample_min_size - blend
        rows: list[list[torch.Tensor]] = []
        for height in range(0, value.shape[-2], overlap):
            row = []
            for width in range(0, value.shape[-1], overlap):
                tile = value[
                    ..., height : height + self.tile_latent_min_size, width : width + self.tile_latent_min_size
                ]
                row.append(self._decode_sample(tile))
            rows.append(row)
        return DecoderOutput(self._merge_spatial_rows(rows, blend, limit))

    def _temporal_tiled_encode(self, value: torch.Tensor) -> AutoencoderKLOutput:
        overlap = max(int(self.tile_sample_min_tsize * (1.0 - self.tile_overlap_factor)), 1)
        blend = int(self.tile_latent_min_tsize * self.tile_overlap_factor)
        limit = self.tile_latent_min_tsize - blend
        tiles = []
        for start in range(0, value.shape[2], overlap):
            tile = value[:, :, start : start + self.tile_sample_min_tsize + 1]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_sample_min_size or tile.shape[-2] > self.tile_sample_min_size
            ):
                encoded = self._spatial_tiled_encode(tile, moments_only=True)
            else:
                encoded = self._encode_moments(tile)
            tiles.append(encoded[:, :, 1:] if start else encoded)
        merged = []
        for index, tile in enumerate(tiles):
            if index:
                tile = self._blend(tiles[index - 1], tile, blend, 2)
                merged.append(tile[:, :, :limit])
            else:
                merged.append(tile[:, :, : limit + 1])
        return AutoencoderKLOutput(DiagonalGaussianDistribution(torch.cat(merged, dim=2)))

    def _temporal_tiled_decode(self, value: torch.Tensor) -> DecoderOutput:
        overlap = max(int(self.tile_latent_min_tsize * (1.0 - self.tile_overlap_factor)), 1)
        blend = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
        limit = self.tile_sample_min_tsize - blend
        tiles = []
        for start in range(0, value.shape[2], overlap):
            tile = value[:, :, start : start + self.tile_latent_min_tsize + 1]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_latent_min_size or tile.shape[-2] > self.tile_latent_min_size
            ):
                decoded = self._spatial_tiled_decode(tile).sample
            else:
                decoded = self._decode_sample(tile)
            tiles.append(decoded[:, :, 1:] if start else decoded)
        merged = []
        for index, tile in enumerate(tiles):
            if index:
                tile = self._blend(tiles[index - 1], tile, blend, 2)
                merged.append(tile[:, :, :limit])
            else:
                merged.append(tile[:, :, : limit + 1])
        return DecoderOutput(torch.cat(merged, dim=2))

    def encode(self, value: torch.Tensor, return_dict: bool = True):
        if value.ndim != 5:
            raise ValueError(f"HunyuanVideo VAE expects BCTHW input, got {tuple(value.shape)}")
        if self.use_temporal_tiling and value.shape[2] > self.tile_sample_min_tsize:
            output = self._temporal_tiled_encode(value)
        elif self.use_spatial_tiling and (
            value.shape[-1] > self.tile_sample_min_size or value.shape[-2] > self.tile_sample_min_size
        ):
            output = self._spatial_tiled_encode(value)
        else:
            if self.use_slicing and value.shape[0] > 1:
                moments = torch.cat([self._encode_moments(item) for item in value.split(1)])
            else:
                moments = self._encode_moments(value)
            output = AutoencoderKLOutput(DiagonalGaussianDistribution(moments))
        return output if return_dict else (output.latent_dist,)

    def decode(self, value: torch.Tensor, return_dict: bool = True, generator=None):
        del generator
        if value.ndim != 5:
            raise ValueError(f"HunyuanVideo VAE expects BCTHW latents, got {tuple(value.shape)}")
        if self.use_temporal_tiling and value.shape[2] > self.tile_latent_min_tsize:
            output = self._temporal_tiled_decode(value)
        elif self.use_spatial_tiling and (
            value.shape[-1] > self.tile_latent_min_size or value.shape[-2] > self.tile_latent_min_size
        ):
            output = self._spatial_tiled_decode(value)
        elif self.use_slicing and value.shape[0] > 1:
            output = DecoderOutput(torch.cat([self._decode_sample(item) for item in value.split(1)]))
        else:
            output = DecoderOutput(self._decode_sample(value))
        return output if return_dict else (output.sample,)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        return_posterior: bool = False,
        generator: torch.Generator | None = None,
    ):
        posterior = self.encode(sample).latent_dist
        latent = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        decoded = self.decode(latent).sample
        if not return_dict:
            return (decoded, posterior) if return_posterior else (decoded,)
        return DecoderOutput(decoded, posterior if return_posterior else None)


# Temporary source-compatible name for derived models while their type annotations migrate.
AutoencoderKLCausal3D = HunyuanVideoCausal3DAutoencoder


def load_hunyuan_video_causal3d(
    model_path: str | Path,
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    sample_size: int | tuple[int, int] | None = None,
) -> HunyuanVideoCausal3DAutoencoder:
    """Load an original-release VAE through WorldFoundry's checkpoint I/O."""

    root = Path(model_path).expanduser()
    config = HunyuanVideoCausal3DAutoencoder.load_config(root)
    overrides = {} if sample_size is None else {"sample_size": sample_size}
    vae = HunyuanVideoCausal3DAutoencoder.from_config(config, **overrides)
    state_dict = load_core_state_dict(root / "pytorch_model.pt", device="cpu")
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], Mapping):
        state_dict = state_dict["state_dict"]
    if any(name.startswith("vae.") for name in state_dict):
        state_dict = {name.removeprefix("vae."): value for name, value in state_dict.items() if name.startswith("vae.")}
    vae.load_state_dict(state_dict)
    if dtype is not None:
        vae.to(dtype=dtype)
    if device is not None:
        vae.to(device=device)
    return vae.requires_grad_(False).eval()


def load_hunyuan_video_vae(
    vae_type: str = "884-16c-hy",
    vae_precision: str | None = None,
    sample_size: int | tuple[int, int] | None = None,
    vae_path: str | Path | None = None,
    logger=None,
    device: str | torch.device | None = None,
):
    """Compatibility entry point for derived in-tree Hunyuan inference models."""

    if vae_path is None:
        raise ValueError("vae_path is required; checkpoint paths are resolved by the caller or recipe")
    precision_to_dtype = {
        None: None,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if vae_precision not in precision_to_dtype:
        raise ValueError(f"unsupported VAE precision: {vae_precision!r}")
    if logger is not None:
        logger.info(f"Loading HunyuanVideo VAE ({vae_type}) from: {vae_path}")
    vae = load_hunyuan_video_causal3d(
        vae_path,
        dtype=precision_to_dtype[vae_precision],
        device=device,
        sample_size=sample_size,
    )
    return (
        vae,
        str(vae_path),
        int(vae.config.spatial_compression_ratio),
        int(vae.config.time_compression_ratio),
    )


__all__ = [
    "AutoencoderKLCausal3D",
    "HunyuanVideoCausal3DAutoencoder",
    "load_hunyuan_video_causal3d",
    "load_hunyuan_video_vae",
]
