# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Array and tensor camera transforms with explicit pixel and depth conventions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor


def pixel_coordinate_grid(
    W: int,
    H: int,
    device: torch.device | str | None = None,
    origin: tuple[float, float] = (0, 0),
    unsqueeze: int | None = None,
    cat_dim: int | None = -1,
    homogeneous: bool = False,
    **arange_kw: Any,
) -> np.ndarray | Tensor | Sequence[np.ndarray | Tensor]:
    """Build an XY pixel grid, using NumPy unless a Torch device is supplied.

    ``cat_dim=None`` returns the unstacked meshgrid. Homogeneous coordinates and
    ``unsqueeze`` require a Torch device. The homogeneous plane uses Torch's
    default floating dtype. Unsqueezing returns only the XY planes.
    """
    backend: Any = np if device is None else torch
    placement = {} if device is None else {"device": device}
    x = backend.arange(origin[0], origin[0] + W, **placement, **arange_kw)
    y = backend.arange(origin[1], origin[1] + H, **placement, **arange_kw)
    coordinates = backend.meshgrid(x, y, indexing="xy")
    if homogeneous:
        coordinates = coordinates + (backend.ones((H, W), **placement),)  # noqa: RUF005 - Preserve sequence-type behavior.
    if unsqueeze is not None:
        coordinates = tuple(axis.unsqueeze(unsqueeze) for axis in coordinates[:2])
    return coordinates if cat_dim is None else backend.stack(coordinates, cat_dim)


def transform_points(
    Trf: np.ndarray | Tensor,
    pts: Any,
    ncol: int | None = None,
    norm: bool | float = False,
) -> np.ndarray | Tensor:
    """Apply batched linear, affine or homogeneous matrices to row-vector points.

    Leading matrix batch dimensions match the leading point dimensions. A
    matrix with one extra input column supplies translation. ``norm`` divides
    by the last output coordinate before optionally scaling to that plane;
    ``ncol`` selects the returned coordinates, defaulting to the input count.
    Torch inputs adopt the matrix dtype, without changing their device.
    """
    assert Trf.ndim >= 2
    points = torch.as_tensor(pts, dtype=Trf.dtype) if isinstance(Trf, Tensor) else np.asarray(pts)
    dimensions = points.shape[-1]
    columns = ncol or dimensions
    result_shape = (*points.shape[:-1], columns)
    matrix = Trf

    if isinstance(matrix, Tensor) and matrix.ndim == 3 and points.ndim == 4:
        if matrix.shape[-1] not in (dimensions, dimensions + 1):
            raise ValueError(f"bad shape, not ending with 3 or 4, for pts.shape={points.shape}")
        if matrix.shape[-1] == dimensions:
            result = torch.einsum("bij,bhwj->bhwi", matrix, points)
        else:
            result = torch.einsum("bij,bhwj->bhwi", matrix[:, :dimensions, :dimensions], points)
            result = result + matrix[:, None, None, :dimensions, dimensions]
    else:
        if matrix.ndim > 2:
            batch_axes = matrix.ndim - 2
            assert matrix.shape[:batch_axes] == points.shape[:batch_axes], "batch size does not match"
            matrix = matrix.reshape(-1, *matrix.shape[-2:])
            if points.ndim > 3:
                points = points.reshape(matrix.shape[0], -1, dimensions)
            elif points.ndim == 2:
                points = points[:, None, :]

        if matrix.shape[-1] in (dimensions, dimensions + 1):
            rows = matrix.swapaxes(-1, -2)
            if matrix.shape[-1] == dimensions + 1:
                result = points @ rows[..., :-1, :] + rows[..., -1:, :]
            else:
                result = points @ rows
        else:
            result = matrix @ points.T
            if result.ndim >= 2:
                result = result.swapaxes(-1, -2)

    if norm:
        result = result / result[..., -1:]
        if norm != 1:
            result *= norm
    return result[..., :columns].reshape(result_shape)


def invert_matrix(mat: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """Use native NumPy or Torch inversion without explicit casts or transfers."""
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    if isinstance(mat, Tensor):
        return torch.linalg.inv(mat)
    raise ValueError(f"bad matrix type = {type(mat)}")


def unproject_depth_with_focal(depth: Tensor, pseudo_focal: Tensor, pp: Tensor | None = None, **_: Any) -> Tensor:
    """Unproject BHW or BHWN depths using BHW, B1HW or B2HW focal lengths.

    Pixel centers default to ``((W - 1) / 2, (H - 1) / 2)``; ``pp`` supplies
    per-image XY centers. Results are BHW3 or BHW3N on the depth device, in
    Torch's default floating dtype. Single-plane depths multiply before focal
    division; multi-plane depths scale already-normalized rays.
    """
    if depth.ndim == 4:
        batch, height, width, _planes = depth.shape
    else:
        batch, height, width = depth.shape
    if pseudo_focal.ndim not in (3, 4):
        raise NotImplementedError("Error, unknown input focal shape format.")
    focal_x = pseudo_focal if pseudo_focal.ndim == 3 else pseudo_focal[:, 0]
    focal_y = pseudo_focal[:, 1] if pseudo_focal.ndim == 4 and pseudo_focal.shape[1] == 2 else focal_x
    assert focal_x.shape == depth.shape[:3]
    assert focal_y.shape == depth.shape[:3]

    xy = cast(Tensor, pixel_coordinate_grid(width, height, device=depth.device, cat_dim=0))
    x, y = xy[:, None]
    if pp is None:
        x, y = x - (width - 1) / 2, y - (height - 1) / 2
    else:
        x = x.expand(batch, -1, -1) - pp[:, 0, None, None]
        y = y.expand(batch, -1, -1) - pp[:, 1, None, None]

    if depth.ndim == 3:
        coordinates = (depth * x / focal_x, depth * y / focal_y, depth)
        result = torch.stack(coordinates, dim=-1)
    else:
        coordinates = (depth * (x / focal_x)[..., None], depth * (y / focal_y)[..., None], depth)
        result = torch.stack(coordinates, dim=-2)
    return result.to(dtype=torch.get_default_dtype())
