"""Shared panorama camera, reprojection, and depth-merging utilities."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from scipy.ndimage import convolve
from scipy.sparse import csr_array, vstack
from scipy.sparse.linalg import lsmr

from ....general_3d.eastern_journalist.utils3d.numpy.maps import uv_map
from ....general_3d.eastern_journalist.utils3d.numpy.mesh import create_icosahedron_mesh
from ....general_3d.eastern_journalist.utils3d.numpy.transforms import (
    extrinsics_look_at,
    intrinsics_from_fov,
    project_cv,
    unproject_cv,
    uv_to_pixel,
)


def get_panorama_cameras(fov_deg: float = 90.0) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized intrinsics and icosahedron camera rotations."""
    vertices, _ = create_icosahedron_mesh()
    fov = np.deg2rad(fov_deg)
    intrinsics = intrinsics_from_fov(fov_x=fov, fov_y=fov).astype(np.float32)
    extrinsics = extrinsics_look_at([0, 0, 0], vertices, [0, 0, 1]).astype(np.float32)
    return extrinsics, np.repeat(intrinsics[None], len(vertices), axis=0)


def get_cubemap_cameras(fov_deg: float = 90.0) -> tuple[np.ndarray, np.ndarray]:
    """Return six cubemap cameras with normalized OpenCV intrinsics."""
    targets = np.asarray(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=np.float32,
    )
    ups = np.asarray(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, -1, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )
    extrinsics = extrinsics_look_at(np.zeros_like(targets), targets, ups).astype(np.float32)
    fov = np.deg2rad(fov_deg)
    intrinsics = intrinsics_from_fov(fov_x=fov, fov_y=fov).astype(np.float32)
    return extrinsics, np.repeat(intrinsics[None], len(targets), axis=0)


def spherical_uv_to_directions(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)
    return directions


def directions_to_spherical_uv(directions: np.ndarray):
    directions = directions / np.clip(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-8, None)
    u = 1 - np.arctan2(directions[..., 1], directions[..., 0]) / (2 * np.pi) % 1.0
    v = np.arccos(np.clip(directions[..., 2], -1, 1)) / np.pi
    return np.stack([u, v], axis=-1)


def split_panorama_image(image: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray, resolution: int):
    height, width = image.shape[:2]
    uv = uv_map((resolution, resolution))
    splitted_images = []
    for i in range(len(extrinsics)):
        spherical_uv = directions_to_spherical_uv(
            unproject_cv(uv, np.ones_like(uv[..., 0]), extrinsics=extrinsics[i], intrinsics=intrinsics[i])
        )
        pixels = uv_to_pixel(spherical_uv, (height, width)).astype(np.float32)

        splitted_image = cv2.remap(
            image,
            pixels[..., 0],
            pixels[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        splitted_images.append(splitted_image)
    return splitted_images


def zdepth_to_distance(depth_map: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Convert perspective z-depth to distance from the camera origin."""
    rays = unproject_cv(uv_map(depth_map.shape), intrinsics=intrinsics)
    return depth_map * np.linalg.norm(rays, axis=-1)


def merge_cubemap_blended_to_panorama(
    width: int,
    height: int,
    distance_maps: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    extrinsics: Sequence[np.ndarray],
    intrinsics: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Merge overlapping cubemap faces using optical-axis cosine weights."""
    directions = spherical_uv_to_directions(uv_map(height, width))
    weighted_depth = np.zeros((height, width), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)

    for distance, mask, extrinsic, intrinsic in zip(distance_maps, pred_masks, extrinsics, intrinsics, strict=False):
        projected_uv, projected_depth = project_cv(directions, extrinsics=extrinsic, intrinsics=intrinsic)
        valid = (
            (projected_depth > 0)
            & (projected_uv[..., 0] >= 0)
            & (projected_uv[..., 0] <= 1)
            & (projected_uv[..., 1] >= 0)
            & (projected_uv[..., 1] <= 1)
        )
        pixels = uv_to_pixel(np.clip(projected_uv, 0, 1), distance.shape).astype(np.float32)
        sampled_distance = cv2.remap(
            distance.astype(np.float32),
            pixels[..., 0],
            pixels[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        sampled_mask = cv2.remap(
            mask.astype(np.uint8),
            pixels[..., 0],
            pixels[..., 1],
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REPLICATE,
        ).astype(bool)

        ray_x = (projected_uv[..., 0] - intrinsic[0, 2]) / intrinsic[0, 0]
        ray_y = (projected_uv[..., 1] - intrinsic[1, 2]) / intrinsic[1, 1]
        weight = 1.0 / (ray_x * ray_x + ray_y * ray_y + 1.0) ** 2
        weight = np.where(valid & sampled_mask, weight, 0.0)
        weighted_depth += sampled_distance.astype(np.float64) * weight
        weight_sum += weight

    panorama_mask = weight_sum > 0
    panorama_depth = np.zeros((height, width), dtype=np.float32)
    panorama_depth[panorama_mask] = (weighted_depth[panorama_mask] / weight_sum[panorama_mask]).astype(np.float32)
    return panorama_depth, panorama_mask


def poisson_equation(width: int, height: int, wrap_x: bool = False, wrap_y: bool = False) -> csr_array:
    grid_index = np.arange(height * width).reshape(height, width)
    grid_index = np.pad(grid_index, ((0, 0), (1, 1)), mode="wrap" if wrap_x else "edge")
    grid_index = np.pad(grid_index, ((1, 1), (0, 0)), mode="wrap" if wrap_y else "edge")

    data = np.array([[-4, 1, 1, 1, 1]], dtype=np.float32).repeat(height * width, axis=0).reshape(-1)
    indices = np.stack(
        [
            grid_index[1:-1, 1:-1],
            grid_index[:-2, 1:-1],  # up
            grid_index[2:, 1:-1],  # down
            grid_index[1:-1, :-2],  # left
            grid_index[1:-1, 2:],  # right
        ],
        axis=-1,
    ).reshape(-1)
    indptr = np.arange(0, height * width * 5 + 1, 5)
    A = csr_array((data, indices, indptr), shape=(height * width, height * width))

    return A


def grad_equation(width: int, height: int, wrap_x: bool = False, wrap_y: bool = False) -> csr_array:
    grid_index = np.arange(width * height).reshape(height, width)
    if wrap_x:
        grid_index = np.pad(grid_index, ((0, 0), (0, 1)), mode="wrap")
    if wrap_y:
        grid_index = np.pad(grid_index, ((0, 1), (0, 0)), mode="wrap")

    data = np.concatenate(
        [
            np.concatenate(
                [
                    np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(-1, 1),  # x[i,j]
                    -np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(
                        -1, 1
                    ),  # x[i,j-1]
                ],
                axis=1,
            ).reshape(-1),
            np.concatenate(
                [
                    np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(-1, 1),  # x[i,j]
                    -np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(
                        -1, 1
                    ),  # x[i-1,j]
                ],
                axis=1,
            ).reshape(-1),
        ]
    )
    indices = np.concatenate(
        [
            np.concatenate(
                [
                    grid_index[:, :-1].reshape(-1, 1),
                    grid_index[:, 1:].reshape(-1, 1),
                ],
                axis=1,
            ).reshape(-1),
            np.concatenate(
                [
                    grid_index[:-1, :].reshape(-1, 1),
                    grid_index[1:, :].reshape(-1, 1),
                ],
                axis=1,
            ).reshape(-1),
        ]
    )
    indptr = np.arange(
        0,
        grid_index.shape[0] * (grid_index.shape[1] - 1) * 2 + (grid_index.shape[0] - 1) * grid_index.shape[1] * 2 + 1,
        2,
    )
    A = csr_array(
        (data, indices, indptr),
        shape=(
            grid_index.shape[0] * (grid_index.shape[1] - 1) + (grid_index.shape[0] - 1) * grid_index.shape[1],
            height * width,
        ),
    )

    return A


def merge_panorama_depth(
    width: int,
    height: int,
    distance_maps: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    extrinsics: Sequence[np.ndarray],
    intrinsics: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if max(width, height) > 256:
        panorama_depth_init, _ = merge_panorama_depth(
            width // 2, height // 2, distance_maps, pred_masks, extrinsics, intrinsics
        )
        panorama_depth_init = cv2.resize(panorama_depth_init, (width, height), cv2.INTER_LINEAR)
    else:
        panorama_depth_init = None

    uv = uv_map(height, width)
    spherical_directions = spherical_uv_to_directions(uv)

    # Warp each view to the panorama
    panorama_log_distance_grad_maps, panorama_grad_masks = [], []
    panorama_log_distance_laplacian_maps, panorama_laplacian_masks = [], []
    panorama_pred_masks = []
    for i in range(len(distance_maps)):
        projected_uv, projected_depth = project_cv(
            spherical_directions, extrinsics=extrinsics[i], intrinsics=intrinsics[i]
        )
        projection_valid_mask = (
            (projected_depth > 0) & (projected_uv > 0).all(axis=-1) & (projected_uv < 1).all(axis=-1)
        )

        projected_pixels = uv_to_pixel(np.clip(projected_uv, 0, 1), distance_maps[i].shape).astype(np.float32)

        log_splitted_distance = np.log(np.clip(distance_maps[i], 1e-6, None))
        panorama_log_distance_map = np.where(
            projection_valid_mask,
            cv2.remap(
                log_splitted_distance,
                projected_pixels[..., 0],
                projected_pixels[..., 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ),
            0,
        )
        panorama_pred_mask = projection_valid_mask & (
            cv2.remap(
                pred_masks[i].astype(np.uint8),
                projected_pixels[..., 0],
                projected_pixels[..., 1],
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REPLICATE,
            )
            > 0
        )

        # calculate gradient map
        padded = np.pad(panorama_log_distance_map, ((0, 0), (0, 1)), mode="wrap")
        grad_x, grad_y = padded[:, :-1] - padded[:, 1:], padded[:-1, :] - padded[1:, :]

        padded = np.pad(panorama_pred_mask, ((0, 0), (0, 1)), mode="wrap")
        mask_x, mask_y = padded[:, :-1] & padded[:, 1:], padded[:-1, :] & padded[1:, :]

        panorama_log_distance_grad_maps.append((grad_x, grad_y))
        panorama_grad_masks.append((mask_x, mask_y))

        # calculate laplacian map
        padded = np.pad(panorama_log_distance_map, ((1, 1), (0, 0)), mode="edge")
        padded = np.pad(padded, ((0, 0), (1, 1)), mode="wrap")
        laplacian = convolve(padded, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32))[1:-1, 1:-1]

        padded = np.pad(panorama_pred_mask, ((1, 1), (0, 0)), mode="edge")
        padded = np.pad(padded, ((0, 0), (1, 1)), mode="wrap")
        mask = (
            convolve(padded.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))[1:-1, 1:-1]
            == 5
        )

        panorama_log_distance_laplacian_maps.append(laplacian)
        panorama_laplacian_masks.append(mask)

        panorama_pred_masks.append(panorama_pred_mask)

    panorama_log_distance_grad_x = np.stack([grad_map[0] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_log_distance_grad_y = np.stack([grad_map[1] for grad_map in panorama_log_distance_grad_maps], axis=0)
    panorama_grad_mask_x = np.stack([mask_map[0] for mask_map in panorama_grad_masks], axis=0)
    panorama_grad_mask_y = np.stack([mask_map[1] for mask_map in panorama_grad_masks], axis=0)

    panorama_log_distance_grad_x = np.sum(panorama_log_distance_grad_x * panorama_grad_mask_x, axis=0) / np.sum(
        panorama_grad_mask_x, axis=0
    ).clip(1e-3)
    panorama_log_distance_grad_y = np.sum(panorama_log_distance_grad_y * panorama_grad_mask_y, axis=0) / np.sum(
        panorama_grad_mask_y, axis=0
    ).clip(1e-3)

    panorama_laplacian_maps = np.stack(panorama_log_distance_laplacian_maps, axis=0)
    panorama_laplacian_masks = np.stack(panorama_laplacian_masks, axis=0)
    panorama_laplacian_map = np.sum(panorama_laplacian_maps * panorama_laplacian_masks, axis=0) / np.sum(
        panorama_laplacian_masks, axis=0
    ).clip(1e-3)

    grad_x_mask = np.any(panorama_grad_mask_x, axis=0).reshape(-1)
    grad_y_mask = np.any(panorama_grad_mask_y, axis=0).reshape(-1)
    grad_mask = np.concatenate([grad_x_mask, grad_y_mask])
    laplacian_mask = np.any(panorama_laplacian_masks, axis=0).reshape(-1)

    # Solve overdetermined system
    A = vstack(
        [
            grad_equation(width, height, wrap_x=True, wrap_y=False)[grad_mask],
            poisson_equation(width, height, wrap_x=True, wrap_y=False)[laplacian_mask],
        ]
    )
    b = np.concatenate(
        [
            panorama_log_distance_grad_x.reshape(-1)[grad_x_mask],
            panorama_log_distance_grad_y.reshape(-1)[grad_y_mask],
            panorama_laplacian_map.reshape(-1)[laplacian_mask],
        ]
    )
    x, *_ = lsmr(
        A,
        b,
        atol=1e-5,
        btol=1e-5,
        x0=np.log(panorama_depth_init).reshape(-1) if panorama_depth_init is not None else None,
        show=False,
    )

    panorama_depth = np.exp(x).reshape(height, width).astype(np.float32)
    panorama_mask = np.any(panorama_pred_masks, axis=0)

    return panorama_depth, panorama_mask
