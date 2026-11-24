import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class SpatialTilingConfig:
    """Configuration for dividing each frame into spatial tiles with optional overlap.
    Args:
        tile_size_in_pixels (int): Size of each tile in pixels. Must be at least 64 and divisible by 32.
        tile_overlap_in_pixels (int, optional): Overlap between tiles in pixels. Must be divisible by 32. Defaults to 0.
    """

    tile_size_in_pixels: int
    tile_overlap_in_pixels: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_pixels < 64:
            raise ValueError(f"tile_size_in_pixels must be at least 64, got {self.tile_size_in_pixels}")
        if self.tile_size_in_pixels % 32 != 0:
            raise ValueError(f"tile_size_in_pixels must be divisible by 32, got {self.tile_size_in_pixels}")
        if self.tile_overlap_in_pixels % 32 != 0:
            raise ValueError(f"tile_overlap_in_pixels must be divisible by 32, got {self.tile_overlap_in_pixels}")
        if self.tile_overlap_in_pixels >= self.tile_size_in_pixels:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_pixels} and {self.tile_size_in_pixels}"
            )


@dataclass(frozen=True)
class TemporalTilingConfig:
    """Configuration for dividing a video into temporal tiles (chunks of frames) with optional overlap.
    Args:
        tile_size_in_frames (int): Number of frames in each tile. Must be at least 16 and divisible by 8.
        tile_overlap_in_frames (int, optional): Number of overlapping frames between consecutive tiles.
            Must be divisible by 8. Defaults to 0.
    """

    tile_size_in_frames: int
    tile_overlap_in_frames: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_frames < 16:
            raise ValueError(f"tile_size_in_frames must be at least 16, got {self.tile_size_in_frames}")
        if self.tile_size_in_frames % 8 != 0:
            raise ValueError(f"tile_size_in_frames must be divisible by 8, got {self.tile_size_in_frames}")
        if self.tile_overlap_in_frames % 8 != 0:
            raise ValueError(f"tile_overlap_in_frames must be divisible by 8, got {self.tile_overlap_in_frames}")
        if self.tile_overlap_in_frames >= self.tile_size_in_frames:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_frames} and {self.tile_size_in_frames}"
            )


@dataclass(frozen=True)
class TilingConfig:
    """Configuration for splitting video into tiles with optional overlap.
    Attributes:
        spatial_config: Configuration for splitting spatial dimensions into tiles.
        temporal_config: Configuration for splitting temporal dimension into tiles.
    """

    spatial_config: SpatialTilingConfig | None = None
    temporal_config: TemporalTilingConfig | None = None

    @classmethod
    def default(cls) -> "TilingConfig":
        return cls(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64),
            temporal_config=TemporalTilingConfig(tile_size_in_frames=64, tile_overlap_in_frames=24),
        )


@dataclass(frozen=True)
class DimensionIntervals:
    """Defines how a single dimension is split into overlapping intervals (tiles).
    Each list has length N where N is the number of intervals. The i-th element
    of each list describes the i-th interval.
    Attributes:
        starts: Start index of each interval (inclusive).
        ends: End index of each interval (exclusive).
        left_ramps: Length of the left blend ramp for each interval.
            Used to create masks that fade in from 0 to 1.
        right_ramps: Length of the right blend ramp for each interval.
            Used to create masks that fade out from 1 to 0.
    """

    starts: list[int]
    ends: list[int]
    left_ramps: list[int]
    right_ramps: list[int]


@dataclass(frozen=True)
class TensorTilingSpec:
    """Specifies how a tensor of a given shape is split into intervals (tiles) along each dimension.
    Attributes:
        original_shape: Shape of the tensor being tiled.
        dimension_intervals: Per-dimension intervals (starts, ends, ramps) for each axis.
    """

    original_shape: torch.Size
    dimension_intervals: tuple[DimensionIntervals, ...]


# Operation to split a single dimension of the tensor into intervals based on the length along the dimension.
SplitOperation = Callable[[int], DimensionIntervals]
# Operation to map the intervals in input dimension to slices and masks along a corresponding output dimension.
MappingOperation = Callable[[DimensionIntervals], tuple[list[slice], list[torch.Tensor | None]]]


def default_split_operation(length: int) -> DimensionIntervals:
    return DimensionIntervals(starts=[0], ends=[length], left_ramps=[0], right_ramps=[0])


DEFAULT_SPLIT_OPERATION: SplitOperation = default_split_operation


def default_mapping_operation(
    _intervals: DimensionIntervals,
) -> tuple[list[slice], list[torch.Tensor | None]]:
    return [slice(0, None)], [None]


DEFAULT_MAPPING_OPERATION: MappingOperation = default_mapping_operation


class Tile(NamedTuple):
    """
    Represents a single tile.
    Attributes:
        in_coords:
            Tuple of slices specifying where to cut the tile from the INPUT tensor.
        out_coords:
            Tuple of slices specifying where this tile's OUTPUT should be placed in the reconstructed OUTPUT tensor.
        masks_1d:
            Per-dimension masks in OUTPUT units.
            These are used to create all-dimensional blending mask.
    Methods:
        blend_mask:
            Create a single N-D mask from the per-dimension masks.
    """

    in_coords: tuple[slice, ...]
    out_coords: tuple[slice, ...]
    masks_1d: tuple[tuple[torch.Tensor, ...]]

    @property
    def blend_mask(self) -> torch.Tensor:
        num_dims = len(self.out_coords)
        per_dimension_masks: list[torch.Tensor] = []

        for dim_idx in range(num_dims):
            mask_1d = self.masks_1d[dim_idx]
            view_shape = [1] * num_dims
            if mask_1d is None:
                # Broadcast mask along this dimension (length 1).
                one = torch.ones(1)

                view_shape[dim_idx] = 1
                per_dimension_masks.append(one.view(*view_shape))
                continue

            # Reshape (L,) -> (1, ..., L, ..., 1) so masks across dimensions broadcast-multiply.
            view_shape[dim_idx] = mask_1d.shape[0]
            per_dimension_masks.append(mask_1d.view(*view_shape))

        # Multiply per-dimension masks to form the full N-D mask (separable blending window).
        combined_mask = per_dimension_masks[0]
        for mask in per_dimension_masks[1:]:
            combined_mask = combined_mask * mask

        return combined_mask


def create_tiles_from_intervals_and_mappers(
    intervals: TensorTilingSpec,
    mappers: list[MappingOperation],
) -> list[Tile]:
    full_dim_input_slices = []
    full_dim_output_slices = []
    full_dim_masks_1d = []
    for axis_index in range(len(intervals.original_shape)):
        dimension_intervals = intervals.dimension_intervals[axis_index]
        starts = dimension_intervals.starts
        ends = dimension_intervals.ends
        input_slices = [slice(s, e) for s, e in zip(starts, ends, strict=True)]
        output_slices, masks_1d = mappers[axis_index](dimension_intervals)
        full_dim_input_slices.append(input_slices)
        full_dim_output_slices.append(output_slices)
        full_dim_masks_1d.append(masks_1d)

    tiles = []
    tile_in_coords = list(itertools.product(*full_dim_input_slices))
    tile_out_coords = list(itertools.product(*full_dim_output_slices))
    tile_mask_1ds = list(itertools.product(*full_dim_masks_1d))
    for in_coord, out_coord, mask_1d in zip(tile_in_coords, tile_out_coords, tile_mask_1ds, strict=True):
        tiles.append(
            Tile(
                in_coords=in_coord,
                out_coords=out_coord,
                masks_1d=mask_1d,
            )
        )
    return tiles


def create_tiles(
    tensor_shape: torch.Size,
    splitters: list[SplitOperation],
    mappers: list[MappingOperation],
) -> list[Tile]:
    if len(splitters) != len(tensor_shape):
        raise ValueError(
            f"Number of splitters must be equal to number of dimensions in tensor shape, "
            f"got {len(splitters)} and {len(tensor_shape)}"
        )
    if len(mappers) != len(tensor_shape):
        raise ValueError(
            f"Number of mappers must be equal to number of dimensions in tensor shape, "
            f"got {len(mappers)} and {len(tensor_shape)}"
        )
    intervals = [splitter(length) for splitter, length in zip(splitters, tensor_shape, strict=True)]
    tiling_spec = TensorTilingSpec(original_shape=tensor_shape, dimension_intervals=tuple(intervals))
    return create_tiles_from_intervals_and_mappers(tiling_spec, mappers)
