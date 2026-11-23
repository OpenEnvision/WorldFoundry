# SPDX-FileCopyrightText: 2026 Physical Intelligence Team, Shanghai AI Laboratory
# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Sparse 3D depth refinement network for Pi4.

Adapted from ``yyfz/Pi4`` revision ``32dfbea9ce6268ea17446208a804710df9475c32``
(``pi4/models/refiner.py``). Upstream code and weights are licensed under CC BY-NC 4.0.

When running on CUDA with the optional ``flex_gemm`` package installed, the sparse
3D operators delegate directly to ``flex_gemm.nn``. Otherwise, they execute a
checkpoint-identical pure-PyTorch sparse 3D fallback preserving exact parameter names
and layouts (``weight: [C_out, 3, 3, 3, C_in]``, ``bias: [C_out]``) on both CPU and
CUDA.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

_FLEX_GEMM_PROBED = False
_FLEX_GEMM_NN: Any | None = None

_OFFSETS_3X3X3: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
)


def _get_flex_gemm_nn() -> Any | None:
    """Lazily probe ``flex_gemm.nn`` without requiring it at import time."""
    global _FLEX_GEMM_PROBED, _FLEX_GEMM_NN
    if "flex_gemm.nn" in sys.modules:
        return sys.modules["flex_gemm.nn"]
    if not _FLEX_GEMM_PROBED:
        _FLEX_GEMM_PROBED = True
        try:
            _FLEX_GEMM_NN = importlib.import_module("flex_gemm.nn")
        except ImportError:
            _FLEX_GEMM_NN = None
    return _FLEX_GEMM_NN


@dataclass(frozen=True)
class _SubmanifoldNeighborCache:
    """Cached 3x3x3 neighbor index table for pure-PyTorch submanifold convolution."""

    neighbor_indices: torch.Tensor  # [27, N], -1 for inactive neighbors


@dataclass(frozen=True)
class _PoolUpsampleCache:
    """Cached fine-to-coarse voxel correspondence with ``.T`` transpose support."""

    fine_to_coarse: torch.Tensor  # [N_fine] indices into [0, N_coarse)
    transposed: bool = False

    @property
    def T(self) -> _PoolUpsampleCache:
        """Return transposed view matching ``flex_gemm``'s ``down_caches[level].T`` contract."""
        return _PoolUpsampleCache(fine_to_coarse=self.fine_to_coarse, transposed=not self.transposed)


def _spatial_bounds(coordinates: torch.Tensor, shape: torch.Size | tuple[int, ...] | None) -> tuple[int, int, int]:
    """Compute exclusive positive spatial extents ``(sx, sy, sz)`` for injective 64-bit hashing."""
    max_x = int(coordinates[:, 1].max().item()) + 1
    max_y = int(coordinates[:, 2].max().item()) + 1
    max_z = int(coordinates[:, 3].max().item()) + 1
    if shape is not None and len(shape) >= 4:
        max_x = max(max_x, int(shape[1]))
        max_y = max(max_y, int(shape[2]))
        max_z = max(max_z, int(shape[3]))
    return max(max_x, 1), max(max_y, 1), max(max_z, 1)


def _encode_coords_64(coords: torch.Tensor, sx: int, sy: int, sz: int) -> torch.Tensor:
    """Encode ``[N, 4]`` integer coordinates ``(b, x, y, z)`` into unique int64 keys."""
    c = coords.to(torch.int64)
    stride_y = sz
    stride_x = sy * sz
    stride_b = sx * sy * sz
    return c[:, 0] * stride_b + c[:, 1] * stride_x + c[:, 2] * stride_y + c[:, 3]


def _build_submanifold_neighbor_cache(
    coordinates: torch.Tensor,
    shape: torch.Size | tuple[int, ...] | None,
) -> _SubmanifoldNeighborCache:
    """Build a ``[27, N]`` neighbor lookup table via vectorized 64-bit coordinate hashing."""
    num_voxels = int(coordinates.shape[0])
    device = coordinates.device
    if num_voxels == 0:
        return _SubmanifoldNeighborCache(neighbor_indices=torch.empty((27, 0), dtype=torch.int64, device=device))

    sx, sy, sz = _spatial_bounds(coordinates, shape)
    c = coordinates.to(torch.int64)
    stride_y = sz
    stride_x = sy * sz
    stride_b = sx * sy * sz

    keys = c[:, 0] * stride_b + c[:, 1] * stride_x + c[:, 2] * stride_y + c[:, 3]
    sorted_keys, perm = torch.sort(keys)

    neighbor_indices = torch.full((27, num_voxels), -1, dtype=torch.int64, device=device)
    identity = torch.arange(num_voxels, dtype=torch.int64, device=device)
    base_b = c[:, 0] * stride_b
    cx, cy, cz = c[:, 1], c[:, 2], c[:, 3]

    for kernel_idx, (dx, dy, dz) in enumerate(_OFFSETS_3X3X3):
        if dx == 0 and dy == 0 and dz == 0:
            neighbor_indices[kernel_idx] = identity
            continue
        nx = cx + dx
        ny = cy + dy
        nz = cz + dz
        valid = (nx >= 0) & (nx < sx) & (ny >= 0) & (ny < sy) & (nz >= 0) & (nz < sz)
        nb_keys = base_b + nx.clamp(0, sx - 1) * stride_x + ny.clamp(0, sy - 1) * stride_y + nz.clamp(0, sz - 1)
        pos = torch.searchsorted(sorted_keys, nb_keys).clamp(max=num_voxels - 1)
        found = valid & (sorted_keys[pos] == nb_keys)
        neighbor_indices[kernel_idx] = torch.where(found, perm[pos], -1)

    return _SubmanifoldNeighborCache(neighbor_indices=neighbor_indices)


class SubmanifoldConv3d(nn.Module):
    """Submanifold 3D sparse convolution with channels-last kernel layout ``[C_out, K, K, K, C_in]``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        bias: bool = True,
        algorithm: str = "implicit_gemm",
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_tuple = (kernel_size, kernel_size, kernel_size)
        else:
            kernel_tuple = (int(kernel_size[0]), int(kernel_size[1]), int(kernel_size[2]))
        if kernel_tuple != (3, 3, 3):
            raise ValueError(f"Pi4 SubmanifoldConv3d only supports 3x3x3 kernels, got {kernel_tuple}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = kernel_tuple
        self.algorithm = algorithm
        self.weight = nn.Parameter(torch.empty(out_channels, *kernel_tuple, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        # Store optional flex_gemm delegate outside ``_modules`` so state_dict keys never duplicate.
        self._flex_cache: dict[str, Any] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with Kaiming uniform scaling matching sparse conv defaults."""
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        bound = 1.0 / math.sqrt(max(fan_in, 1))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def _get_flex_delegate(self, flex_nn: Any) -> Any:
        delegate = self._flex_cache.get("delegate")
        if delegate is None or self._flex_cache.get("flex_nn") is not flex_nn:
            with torch.device("meta"):
                delegate = flex_nn.SubmanifoldConv3d(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size[0],
                    bias=self.bias is not None,
                    algorithm=self.algorithm,
                )
            self._flex_cache["delegate"] = delegate
            self._flex_cache["flex_nn"] = flex_nn
        delegate._parameters["weight"] = self.weight
        if self.bias is not None:
            delegate._parameters["bias"] = self.bias
        return delegate

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        shape: torch.Size | tuple[int, ...],
        neighbor_cache: Any = None,
    ) -> tuple[torch.Tensor, Any]:
        flex_nn = _get_flex_gemm_nn() if features.device.type == "cuda" else None
        if flex_nn is not None:
            delegate = self._get_flex_delegate(flex_nn)
            return delegate(features, coordinates, shape, neighbor_cache=neighbor_cache)

        if not isinstance(neighbor_cache, _SubmanifoldNeighborCache):
            neighbor_cache = _build_submanifold_neighbor_cache(coordinates, shape)

        num_voxels = int(features.shape[0])
        if num_voxels == 0:
            return features.new_empty((0, self.out_channels)), neighbor_cache

        weight = self.weight.to(dtype=features.dtype).reshape(self.out_channels, 27, self.in_channels)
        # Center offset (0, 0, 0) is index 13; every active voxel maps to itself.
        out = features @ weight[:, 13, :].transpose(0, 1)
        if self.bias is not None:
            out = out + self.bias.to(dtype=features.dtype)

        neighbor_indices = neighbor_cache.neighbor_indices
        for kernel_idx in range(27):
            if kernel_idx == 13:
                continue
            idx = neighbor_indices[kernel_idx]
            valid_out = (idx >= 0).nonzero(as_tuple=False).squeeze(-1)
            if valid_out.numel() == 0:
                continue
            valid_in = idx[valid_out]
            contrib = features[valid_in] @ weight[:, kernel_idx, :].transpose(0, 1)
            out.index_add_(0, valid_out, contrib)

        return out, neighbor_cache


class SparsePool3d(nn.Module):
    """Stride-2 3D sparse pooling with mean/sum reduction and cached upsample mapping."""

    def __init__(self, kernel_size: int = 2, stride: int = 2, reduce: str = "mean") -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.reduce = reduce
        self._flex_cache: dict[str, Any] = {}

    def _get_flex_delegate(self, flex_nn: Any) -> Any:
        delegate = self._flex_cache.get("delegate")
        if delegate is None or self._flex_cache.get("flex_nn") is not flex_nn:
            delegate = flex_nn.SparsePool3d(kernel_size=self.kernel_size, stride=self.stride, reduce=self.reduce)
            self._flex_cache["delegate"] = delegate
            self._flex_cache["flex_nn"] = flex_nn
        return delegate

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        shape: torch.Size | tuple[int, ...],
        output_shape: torch.Size | tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Size, Any]:
        flex_nn = _get_flex_gemm_nn() if features.device.type == "cuda" else None
        if flex_nn is not None:
            delegate = self._get_flex_delegate(flex_nn)
            return delegate(features, coordinates, shape, output_shape=output_shape)

        resolved_shape = (
            torch.Size(output_shape)
            if output_shape is not None
            else torch.Size(
                (shape[0], *((int(d) + self.stride - 1) // self.stride for d in shape[1:4]), features.shape[-1])
            )
        )
        num_voxels = int(coordinates.shape[0])
        if num_voxels == 0:
            empty_cache = _PoolUpsampleCache(
                fine_to_coarse=torch.empty(0, dtype=torch.int64, device=coordinates.device)
            )
            return features.new_empty((0, features.shape[-1])), coordinates.clone(), resolved_shape, empty_cache

        coarse_per_fine = coordinates.to(torch.int64).clone()
        coarse_per_fine[:, 1:] = torch.div(coarse_per_fine[:, 1:], self.stride, rounding_mode="floor")
        unique_coarse, inverse = torch.unique(coarse_per_fine, dim=0, return_inverse=True)
        num_coarse = int(unique_coarse.shape[0])

        if self.reduce in ("mean", "sum"):
            pooled = torch.zeros((num_coarse, features.shape[-1]), dtype=features.dtype, device=features.device)
            pooled.index_add_(0, inverse, features)
            if self.reduce == "mean":
                counts = torch.zeros(num_coarse, dtype=features.dtype, device=features.device)
                counts.index_add_(0, inverse, torch.ones(num_voxels, dtype=features.dtype, device=features.device))
                pooled = pooled / counts.clamp_min(1.0).unsqueeze(-1)
        elif self.reduce == "max":
            pooled = torch.full(
                (num_coarse, features.shape[-1]), float("-inf"), dtype=features.dtype, device=features.device
            )
            pooled.scatter_reduce_(0, inverse.unsqueeze(-1).expand_as(features), features, reduce="amax")
        else:
            raise ValueError(f"Unsupported SparsePool3d reduce mode: {self.reduce!r}")

        cache = _PoolUpsampleCache(fine_to_coarse=inverse)
        return pooled, unique_coarse.to(dtype=coordinates.dtype), resolved_shape, cache


class SparseUpsample3d(nn.Module):
    """Nearest-neighbor 3D sparse upsampling onto target fine coordinates."""

    def __init__(self, scale_factor: int = 2, mode: str = "nearest") -> None:
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.mode = mode
        self._flex_cache: dict[str, Any] = {}

    def _get_flex_delegate(self, flex_nn: Any) -> Any:
        delegate = self._flex_cache.get("delegate")
        if delegate is None or self._flex_cache.get("flex_nn") is not flex_nn:
            delegate = flex_nn.SparseUpsample3d(scale_factor=self.scale_factor, mode=self.mode)
            self._flex_cache["delegate"] = delegate
            self._flex_cache["flex_nn"] = flex_nn
        return delegate

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        shape: torch.Size | tuple[int, ...],
        output_coords: torch.Tensor,
        output_shape: torch.Size | tuple[int, ...],
        neighbor_cache: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Size, Any]:
        flex_nn = _get_flex_gemm_nn() if features.device.type == "cuda" else None
        if flex_nn is not None:
            delegate = self._get_flex_delegate(flex_nn)
            return delegate(
                features,
                coordinates,
                shape,
                output_coords=output_coords,
                output_shape=output_shape,
                neighbor_cache=neighbor_cache,
            )

        if self.mode != "nearest":
            raise ValueError(f"Unsupported SparseUpsample3d mode: {self.mode!r}")

        target_shape = torch.Size((*output_shape[:4], features.shape[-1]))
        num_target = int(output_coords.shape[0])
        if num_target == 0:
            return features.new_empty((0, features.shape[-1])), output_coords, target_shape, neighbor_cache

        if isinstance(neighbor_cache, _PoolUpsampleCache) and int(neighbor_cache.fine_to_coarse.shape[0]) == num_target:
            return features[neighbor_cache.fine_to_coarse], output_coords, target_shape, neighbor_cache

        # Fallback coordinate lookup when called standalone without a pool cache.
        query_coarse = output_coords.to(torch.int64).clone()
        query_coarse[:, 1:] = torch.div(query_coarse[:, 1:], self.scale_factor, rounding_mode="floor")
        coarse_int64 = coordinates.to(torch.int64)
        sx = max(int(coarse_int64[:, 1].max().item()), int(query_coarse[:, 1].max().item())) + 1
        sy = max(int(coarse_int64[:, 2].max().item()), int(query_coarse[:, 2].max().item())) + 1
        sz = max(int(coarse_int64[:, 3].max().item()), int(query_coarse[:, 3].max().item())) + 1
        coarse_keys = _encode_coords_64(coarse_int64, sx, sy, sz)
        sorted_keys, perm = torch.sort(coarse_keys)
        query_keys = _encode_coords_64(query_coarse, sx, sy, sz)
        pos = torch.searchsorted(sorted_keys, query_keys).clamp(max=max(int(coarse_keys.shape[0]) - 1, 0))
        matched = sorted_keys[pos] == query_keys
        mapped = perm[pos]
        out_features = torch.zeros((num_target, features.shape[-1]), dtype=features.dtype, device=features.device)
        if matched.any():
            out_features[matched] = features[mapped[matched]]
        return out_features, output_coords, target_shape, _PoolUpsampleCache(fine_to_coarse=mapped)


class SparseResidual(nn.Module):
    """Residual block of two 3x3x3 submanifold sparse convolutions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        algorithm = "masked_implicit_gemm" if channels >= 128 else "implicit_gemm"
        self.conv1 = SubmanifoldConv3d(channels, channels, 3, algorithm=algorithm)
        self.conv2 = SubmanifoldConv3d(channels, channels, 3, algorithm=algorithm)

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        shape: torch.Size,
        cache: Any = None,
    ) -> tuple[torch.Tensor, Any]:
        residual = F.silu(self.norm1(features).type_as(features))
        residual, cache = self.conv1(residual, coordinates, shape, neighbor_cache=cache)
        residual, cache = self.conv2(F.silu(residual), coordinates, shape, neighbor_cache=cache)
        return features + residual, cache


class SparseDown(nn.Module):
    """Stride-2 sparse pooling followed by channel projection."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.pool = SparsePool3d(kernel_size=2, stride=2, reduce="mean")
        self.linear = nn.Linear(input_channels, output_channels)

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        shape: torch.Size,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Size, Any]:
        output_shape = torch.Size((shape[0], *((dimension + 1) // 2 for dimension in shape[1:4]), shape[4]))
        features, coordinates, shape, cache = self.pool(features, coordinates, shape, output_shape=output_shape)
        features = self.linear(features)
        return (
            features,
            coordinates,
            torch.Size((*shape[:-1], features.shape[-1])),
            cache,
        )


class SparseUp(nn.Module):
    """Channel projection followed by nearest-neighbor sparse upsampling."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.upsample = SparseUpsample3d(scale_factor=2, mode="nearest")
        self.linear = nn.Linear(input_channels, output_channels)

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        shape: torch.Size,
        target_coordinates: torch.Tensor,
        target_shape: torch.Size,
        cache: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
        features = self.linear(features)
        shape = torch.Size((*shape[:-1], features.shape[-1]))
        features, coordinates, shape, _ = self.upsample(
            features,
            coordinates,
            shape,
            output_coords=target_coordinates,
            output_shape=target_shape,
            neighbor_cache=cache,
        )
        return features, coordinates, shape


class SparseUNet(nn.Module):
    """5-level sparse 3D U-Net conditioned on dense patch-level geometry features."""

    def __init__(self) -> None:
        super().__init__()
        channels = (32, 64, 128, 256, 512)
        self.input_proj = nn.Linear(3, 32)
        self.down_stages = nn.ModuleList(nn.ModuleList((SparseResidual(dimension),)) for dimension in channels)
        self.downsample_blocks = nn.ModuleList(SparseDown(first, second) for first, second in pairwise(channels))
        self.fuse_proj = nn.Sequential(nn.Linear(1024, 512), nn.SiLU(), nn.Linear(512, 512))
        self.bottleneck_stage = nn.ModuleList((SparseResidual(512),))
        self.upsample_blocks = nn.ModuleList(
            SparseUp(first, second) for first, second in zip(channels[:0:-1], channels[-2::-1], strict=True)
        )
        self.up_stages = nn.ModuleList(nn.ModuleList((SparseResidual(dimension),)) for dimension in channels[-2::-1])
        self.out_proj = nn.Linear(32, 1)

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        shape: torch.Size,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        features = self.input_proj(features)
        shape = torch.Size((*shape[:4], features.shape[-1]))
        caches: list[Any] = [None] * 5
        down_caches: list[Any] = []
        skips: list[tuple[torch.Tensor, torch.Tensor, torch.Size]] = []
        for index, stage in enumerate(self.down_stages):
            if not isinstance(stage, nn.ModuleList):
                raise TypeError(f"Expected nn.ModuleList stage, got {type(stage).__name__}")
            features, caches[index] = stage[0](features, coordinates, shape, caches[index])
            if index < 4:
                skips.append((features, coordinates, shape))
                features, coordinates, shape, cache = self.downsample_blocks[index](features, coordinates, shape)
                down_caches.append(cache)
        sampled = condition[
            coordinates[:, 0].long(),
            :,
            coordinates[:, 1].long(),
            coordinates[:, 2].long(),
        ]
        features = self.fuse_proj(torch.cat((features, sampled), dim=-1))
        features, caches[4] = self.bottleneck_stage[0](features, coordinates, shape, caches[4])
        for index, (upsample, stage) in enumerate(zip(self.upsample_blocks, self.up_stages, strict=True)):
            level = 3 - index
            skip_features, skip_coordinates, skip_shape = skips[level]
            features, coordinates, shape = upsample(
                features,
                coordinates,
                shape,
                skip_coordinates,
                skip_shape,
                down_caches[level].T,
            )
            if not isinstance(stage, nn.ModuleList):
                raise TypeError(f"Expected nn.ModuleList stage, got {type(stage).__name__}")
            features, caches[level] = stage[0](features + skip_features, coordinates, shape, caches[level])
        return self.out_proj(features)


class DepthRefiner(nn.Module):
    """3-iteration sparse 3D depth residual refiner converting rays + log-depth to local XYZ."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_projector = nn.Sequential(nn.Conv2d(1026, 1024, 1), nn.GELU(), nn.Conv2d(1024, 512, 1))
        self.network = SparseUNet()

    def forward(self, rays: torch.Tensor, log_depth: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        condition = self.feature_projector(feature.float())
        batch_size, height, width = log_depth.shape
        batch, rows, columns = torch.meshgrid(
            torch.arange(batch_size, device=rays.device),
            torch.arange(height, device=rays.device),
            torch.arange(width, device=rays.device),
            indexing="ij",
        )
        for _ in range(3):
            bins = torch.round(log_depth.float() * 256).long()
            bins = bins - bins.amin(dim=(1, 2), keepdim=True)
            coordinates = torch.stack((batch, rows, columns, bins), dim=-1).reshape(-1, 4).int()
            features = torch.cat((rays, log_depth[..., None]), dim=-1).reshape(-1, 3).float()
            shape = torch.Size((batch_size, height, width, int(bins.max().item()) + 1, 3))
            residual = self.network(features, coordinates, shape, condition)
            log_depth = log_depth + residual.float().reshape_as(log_depth)
        depth = log_depth.exp()[..., None]
        return torch.cat((rays * depth, depth), dim=-1)
