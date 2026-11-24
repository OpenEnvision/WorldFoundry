from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from worldfoundry.base_models.diffusion_model.models.representations.ltx.tiling import (
    AutoTiling,
    DimensionIntervals,
    DimensionSizeConfig,
    Tile,
    TileCountConfig,
    _assert_video_on_vae_grid,
    _validate_size_axis,
    compute_trapezoidal_mask_1d,
    default_mapping_operation,
    default_split_operation,
    scale_by_masks_1d,
    split_by_size,
    split_temporal_causal,
)
from worldfoundry.base_models.diffusion_model.models.representations.ltx.types import VideoPixelShape
from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.ltx_core.types import SpatioTemporalScaleFactors

# Operation to split a single dimension of the tensor into intervals based on the length along the dimension.
SplitOperation = Callable[[int], DimensionIntervals]
# Operation to map the intervals in input dimension to slices and masks along a corresponding output dimension.
MappingOperation = Callable[[DimensionIntervals], tuple[list[slice], list[torch.Tensor]]]


DEFAULT_SPLIT_OPERATION: SplitOperation = default_split_operation


DEFAULT_MAPPING_OPERATION: MappingOperation = default_mapping_operation


def identity_mapping_operation(intervals: DimensionIntervals) -> tuple[list[slice], list[torch.Tensor]]:
    """Map each DimensionInterval to an output region at the same position, with trapezoidal blend masks.
    For every interval the output start/end matches the input start/end and a
    1-D blending mask is built from the interval's left_ramp and right_ramp.
    """
    out_slices: list[slice] = []
    masks: list[torch.Tensor] = []
    for iv in intervals.intervals:
        out_slices.append(slice(iv.start, iv.end))
        masks.append(compute_trapezoidal_mask_1d(iv.end - iv.start, iv.left_ramp, iv.right_ramp))
    return out_slices, masks


def compute_summed_weights(
    tiles: Sequence[Tile],
    full_shape: Sequence[int],
) -> torch.Tensor:
    """Build the dense denominator for weighted blending over ``full_shape``.
    Uses separable per-axis mask broadcasts — never ``Tile.blend_mask``.
    Requires concrete ``out_coords`` (``stop`` not ``None``) on every axis.
    Always builds on CPU float32 so CUDA masks / a non-CPU default device cannot
    place a multi-GB ``[F,H,W]`` tensor on GPU.
    """
    weights = torch.zeros(*full_shape, dtype=torch.float32, device="cpu")
    for tile in tiles:
        masks = tuple(m.detach().float().cpu() for m in tile.masks_1d)
        region_shape = tuple(s.stop - s.start for s in tile.out_coords)
        region = torch.ones(region_shape, dtype=torch.float32, device="cpu")
        weights[tile.out_coords] += scale_by_masks_1d(region, masks)
    return weights.clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Video-grid tiling configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileSizeConfig:
    """Size-based tiling layout for a ``(F, H, W)`` video — mirror of ``TileCountConfig``.
    Each axis is a non-optional :class:`DimensionSizeConfig`; ``tile_size=0`` means
    untiled on that axis (:meth:`DimensionSizeConfig.is_tiled`). Sizes and overlaps
    are in pixel / frame units. Conversion to a split grid is an explicit
    ``scale_factors`` argument to :meth:`to_splitters` (not stored on the config).
    Legality vs a VAE grid is checked by :meth:`validate` (same factors decode will
    pass to :meth:`to_splitters`), not at construction.
    Attributes:
        frames: Temporal tile size/overlap in frames.
        height: Spatial height tile size/overlap in pixels.
        width: Spatial width tile size/overlap in pixels.
    """

    frames: DimensionSizeConfig = field(default_factory=DimensionSizeConfig)
    height: DimensionSizeConfig = field(default_factory=DimensionSizeConfig)
    width: DimensionSizeConfig = field(default_factory=DimensionSizeConfig)

    def validate(self, scale_factors: SpatioTemporalScaleFactors, video_shape: VideoPixelShape) -> None:
        """Raise if this size layout is illegal for ``video_shape`` under ``scale_factors``.
        Checks tile/overlap divisibility and minimums against the VAE grid, and that
        the video extents are compatible with that grid. ``video_shape.frames <= 0``
        skips the temporal axis (duration not yet known); height/width always checked.
        """
        check_temporal = _assert_video_on_vae_grid(scale_factors, video_shape)
        _validate_size_axis(self.height, scale_factors.height, "height")
        _validate_size_axis(self.width, scale_factors.width, "width")
        if check_temporal:
            _validate_size_axis(self.frames, scale_factors.time, "frames")

    @classmethod
    def default(cls) -> TileSizeConfig:
        return cls(
            frames=DimensionSizeConfig(tile_size=80, overlap=24),
            height=DimensionSizeConfig(tile_size=768, overlap=64),
            width=DimensionSizeConfig(tile_size=768, overlap=64),
        )

    @classmethod
    def from_long_side(
        cls,
        *,
        long_side: DimensionSizeConfig,
        height: int,
        width: int,
        scale_factors: SpatioTemporalScaleFactors,
        frames: DimensionSizeConfig | None = None,
    ) -> TileSizeConfig:
        """Aspect-coupled construction — old single-spatial long-side behavior, explicit.
        Matches main-era ``latent_tile_splitters``: scale the long-side tile in
        *latent* units with ``round(size_lat * axis_lat / long_lat)``, then
        multiply back by the VAE factor. Pixel-space ``round`` + ceil-snap would
        bias the short axis up by almost one latent (e.g. 680 → 704 vs 672).
        Both axes share ``long_side.overlap``.
        """
        if height < 1 or width < 1:
            raise ValueError(f"height/width must be >= 1, got {height}x{width}")
        if not long_side.is_tiled():
            raise ValueError("long_side must be tiled (tile_size > 0)")
        if scale_factors.height < 1 or scale_factors.width < 1:
            raise ValueError(f"scale_factors height/width must be >= 1, got {scale_factors}")
        span = max(height, width)

        def axis_size(axis_len: int, factor: int) -> int:
            # Latent-grid round (same as main decode enable_on_axis), not pixel ceil.
            axis_lat = axis_len // factor
            long_lat = span // factor
            size_lat = long_side.tile_size // factor
            overlap_lat = long_side.overlap // factor
            lower_threshold = max(2, overlap_lat + 1)
            tile_lat = max(lower_threshold, round(size_lat * axis_lat / long_lat))
            tile_px = tile_lat * factor
            min_legal = max(2 * factor, long_side.overlap + factor)
            return max(tile_px, min_legal)

        return cls(
            frames=DimensionSizeConfig() if frames is None else frames,
            height=DimensionSizeConfig(tile_size=axis_size(height, scale_factors.height), overlap=long_side.overlap),
            width=DimensionSizeConfig(tile_size=axis_size(width, scale_factors.width), overlap=long_side.overlap),
        )

    def to_splitters(
        self,
        scale_factors: SpatioTemporalScaleFactors,
        min_tile_size: tuple[int, int, int] | None = None,
        *,
        causal_temporal: bool = True,
    ) -> tuple[SplitOperation, SplitOperation, SplitOperation]:
        """Build ``(T, H, W)`` grid split ops from pixel/frame sizes via ``scale_factors``.
        When ``causal_temporal`` is True, frames use :func:`split_temporal_causal`.
        """
        min_t = min_h = min_w = None
        if min_tile_size is not None:
            min_t, min_h, min_w = min_tile_size

        def enable_size_axis(
            factor: int,
            axis_min: int | None,
            cfg: DimensionSizeConfig,
            axis_name: str,
            *,
            temporal: bool,
        ) -> SplitOperation:
            if not cfg.is_tiled():
                return DEFAULT_SPLIT_OPERATION
            _validate_size_axis(cfg, factor, axis_name)
            size = cfg.tile_size // factor
            overlap = cfg.overlap // factor
            lower_threshold = max(2, overlap + 1)
            tile = max(lower_threshold, size)
            if temporal and causal_temporal:
                return split_temporal_causal(tile, overlap, min_tile_size=axis_min)
            return split_by_size(tile, overlap, min_tile_size=axis_min)

        return (
            enable_size_axis(scale_factors.time, min_t, self.frames, "frames", temporal=True),
            enable_size_axis(scale_factors.height, min_h, self.height, "height", temporal=False),
            enable_size_axis(scale_factors.width, min_w, self.width, "width", temporal=False),
        )

    def video_chunks_number(self, num_frames: int) -> int:
        """Number of temporal decode chunks for ``num_frames`` under this layout."""
        if not self.frames.is_tiled():
            return 1
        frame_stride = self.frames.tile_size - self.frames.overlap
        return (num_frames - 1 + frame_stride - 1) // frame_stride


# Either size-based (single-GPU VAE) or count-based (MGPU / explicit counts).
TilingConfig = TileSizeConfig | TileCountConfig


AUTO_TILING = AutoTiling()

# Pipeline ``tiling_config`` argument: explicit layout, auto-recommend, or untiled.
PipelineTiling = TilingConfig | AutoTiling | None


def balanced_tile_split(num_tiles: int) -> tuple[int, int]:
    """Factor ``num_tiles`` into ``(small, large)`` as square as possible.
    ``small`` is the largest divisor not exceeding the square root, so
    ``small * large == num_tiles`` and ``small <= large``. E.g. 2 -> (1, 2),
    4 -> (2, 2), 8 -> (2, 4), 16 -> (4, 4). The caller decides which tiled
    dimension gets which factor.
    """
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    small = next(d for d in range(math.isqrt(num_tiles), 0, -1) if num_tiles % d == 0)
    return small, num_tiles // small
