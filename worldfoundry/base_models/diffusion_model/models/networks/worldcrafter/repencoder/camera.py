# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


NUM_SOURCE_VIEWS = 9
NUM_TARGET_VIEWS = 4
RAY_HEIGHT = 384
RAY_WIDTH = 640
RAY_HW = (RAY_HEIGHT, RAY_WIDTH)
CAMERA_SCALE_MULTIPLIER = 1.35
DEFAULT_NEAR_ZERO_BASELINE_M = 1e-6


@dataclass(frozen=True)
class RepEncoderCameraConditioning:
    source_camera_tokens: torch.Tensor
    target_rays: torch.Tensor
    source_c2w_normalized: torch.Tensor
    target_c2w_normalized: torch.Tensor
    scale_tokens: torch.Tensor
    scene_scale_m: torch.Tensor


def _validate_metric_c2w(
    source_c2w_metric: torch.Tensor,
    target_c2w_metric: torch.Tensor,
) -> None:
    if not isinstance(source_c2w_metric, torch.Tensor) or not isinstance(
        target_c2w_metric, torch.Tensor
    ):
        raise TypeError("source and target c2w values must be torch tensors")
    if source_c2w_metric.ndim != 4 or source_c2w_metric.shape[1:] != (
        NUM_SOURCE_VIEWS,
        4,
        4,
    ):
        raise ValueError(
            "source_c2w_metric must be [B,9,4,4], got "
            f"{tuple(source_c2w_metric.shape)}"
        )
    if target_c2w_metric.ndim != 4 or target_c2w_metric.shape[1:] != (
        NUM_TARGET_VIEWS,
        4,
        4,
    ):
        raise ValueError(
            "target_c2w_metric must be [B,4,4,4], got "
            f"{tuple(target_c2w_metric.shape)}"
        )
    if source_c2w_metric.shape[0] != target_c2w_metric.shape[0]:
        raise ValueError("source and target camera batches differ")
    if source_c2w_metric.device != target_c2w_metric.device:
        raise ValueError("source and target cameras must share one device")
    if not torch.is_floating_point(source_c2w_metric) or not torch.is_floating_point(
        target_c2w_metric
    ):
        raise TypeError("camera matrices must use a floating-point dtype")
    if not torch.isfinite(source_c2w_metric).all() or not torch.isfinite(
        target_c2w_metric
    ).all():
        raise ValueError("camera matrices contain NaN or Inf")


def normalize_source_target_cameras(
    source_c2w_metric: torch.Tensor,
    target_c2w_metric: torch.Tensor,
    *,
    near_zero_baseline_m: float = DEFAULT_NEAR_ZERO_BASELINE_M,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    _validate_metric_c2w(source_c2w_metric, target_c2w_metric)
    epsilon = float(near_zero_baseline_m)
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("near_zero_baseline_m must be finite and non-negative")

    with torch.autocast(device_type=source_c2w_metric.device.type, enabled=False):
        reference_inverse = torch.linalg.inv(source_c2w_metric[:, 0])
        source = reference_inverse[:, None] @ source_c2w_metric
        target = reference_inverse[:, None] @ target_c2w_metric
        max_source = torch.linalg.vector_norm(source[..., :3, 3], dim=-1).amax(dim=1)
        is_zero = max_source <= epsilon
        scene_scale = torch.where(
            is_zero,
            torch.ones_like(max_source),
            CAMERA_SCALE_MULTIPLIER * max_source,
        )
        zero = torch.zeros_like(max_source)
        one = torch.ones_like(max_source)
        scale_tokens = torch.stack(
            [
                torch.where(is_zero, zero, max_source / scene_scale),
                torch.where(is_zero, one, zero),
            ],
            dim=-1,
        )
        source = source.clone()
        target = target.clone()
        source[..., :3, 3] /= scene_scale[:, None, None]
        target[..., :3, 3] /= scene_scale[:, None, None]

    identity = torch.eye(4, dtype=source.dtype, device=source.device)
    if not torch.allclose(
        source[:, 0], identity.expand(source.shape[0], -1, -1), atol=2e-4, rtol=0
    ):
        raise RuntimeError("source[0] is not identity after RepEncoder normalization")
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise FloatingPointError("normalized RepEncoder cameras contain NaN or Inf")
    return (
        source.contiguous(),
        target.contiguous(),
        scale_tokens.contiguous(),
        scene_scale.contiguous(),
    )


def _sqrt_positive_part(value: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(value)
    positive = value > 0
    if torch.is_grad_enabled():
        result[positive] = torch.sqrt(value[positive])
        return result
    return torch.where(positive, torch.sqrt(value), result)


def _mat_to_quat_scalar_last(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"invalid rotation matrix shape {tuple(matrix.shape)}")
    batch_shape = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_shape, 9), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack(
                [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01],
                dim=-1,
            ),
            torch.stack(
                [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20],
                dim=-1,
            ),
            torch.stack(
                [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21],
                dim=-1,
            ),
            torch.stack(
                [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2],
                dim=-1,
            ),
        ],
        dim=-2,
    )
    floor = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(floor))
    selected = candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(*batch_shape, 4)
    selected = selected[..., [1, 2, 3, 0]]
    return torch.where(selected[..., 3:4] < 0, -selected, selected)


def _nominal_intrinsics(
    batch: int,
    views: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    matrices = torch.zeros(batch, views, 3, 3, device=device, dtype=torch.float32)
    matrices[..., 0, 0] = float(RAY_WIDTH)
    matrices[..., 1, 1] = float(RAY_WIDTH)
    matrices[..., 0, 2] = RAY_WIDTH / 2.0
    matrices[..., 1, 2] = RAY_HEIGHT / 2.0
    matrices[..., 2, 2] = 1.0
    return matrices


def _compute_plucker_rays(
    target_c2w: torch.Tensor,
    target_intrinsics: torch.Tensor,
) -> torch.Tensor:
    batch, views = target_c2w.shape[:2]
    device = target_c2w.device
    pixel_x = torch.linspace(
        0.5, RAY_WIDTH - 0.5, RAY_WIDTH, device=device, dtype=torch.float32
    )
    pixel_y = torch.linspace(
        0.5, RAY_HEIGHT - 0.5, RAY_HEIGHT, device=device, dtype=torch.float32
    )
    grid_y, grid_x = torch.meshgrid(pixel_y, pixel_x, indexing="ij")
    uv = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1)
    inverse_intrinsics = torch.linalg.inv(target_intrinsics).float()
    directions_local = torch.einsum("bvij,hwj->bvhwi", inverse_intrinsics, uv)
    directions_local = directions_local / torch.linalg.vector_norm(
        directions_local, dim=-1, keepdim=True
    )
    directions_global = torch.einsum(
        "bvij,bvhwj->bvhwi", target_c2w[..., :3, :3].float(), directions_local
    )
    ray_origin = target_c2w[..., :3, 3].float()[:, :, None, None, :].expand(
        batch, views, RAY_HEIGHT, RAY_WIDTH, 3
    )
    moment = torch.cross(ray_origin, directions_global, dim=-1)
    return (
        torch.cat([moment, directions_global], dim=-1)
        .permute(0, 1, 4, 2, 3)
        .contiguous()
    )


def prepare_repencoder_camera_conditioning(
    source_c2w_metric: torch.Tensor,
    target_c2w_metric: torch.Tensor,
    *,
    near_zero_baseline_m: float = DEFAULT_NEAR_ZERO_BASELINE_M,
    device: torch.device | str | None = None,
) -> RepEncoderCameraConditioning:
    target_device = (
        source_c2w_metric.device if device is None else torch.device(device)
    )
    with torch.autocast(device_type=target_device.type, enabled=False):
        source_metric = source_c2w_metric.to(device=target_device, dtype=torch.float32)
        target_metric = target_c2w_metric.to(device=target_device, dtype=torch.float32)
        source, target, scale_tokens, scene_scale = normalize_source_target_cameras(
            source_metric,
            target_metric,
            near_zero_baseline_m=near_zero_baseline_m,
        )
        batch = source.shape[0]
        intrinsics = _nominal_intrinsics(
            batch,
            NUM_SOURCE_VIEWS + NUM_TARGET_VIEWS,
            device=target_device,
        )
        quaternion = _mat_to_quat_scalar_last(source[..., :3, :3])
        fy = intrinsics[:, :NUM_SOURCE_VIEWS, 1, 1]
        fx = intrinsics[:, :NUM_SOURCE_VIEWS, 0, 0]
        fov_h = 2.0 * torch.atan((RAY_HEIGHT / 2.0) / fy)
        fov_w = 2.0 * torch.atan((RAY_WIDTH / 2.0) / fx)
        pose_encoding = torch.cat(
            [source[..., :3, 3], quaternion, fov_h[..., None], fov_w[..., None]],
            dim=-1,
        ).float()
        camera_tokens = torch.cat(
            [
                pose_encoding,
                scale_tokens[:, None, :].expand(batch, NUM_SOURCE_VIEWS, 2),
            ],
            dim=-1,
        ).contiguous()
        target_rays = _compute_plucker_rays(
            target,
            intrinsics[:, NUM_SOURCE_VIEWS:],
        )
    if camera_tokens.shape != (batch, NUM_SOURCE_VIEWS, 11):
        raise RuntimeError(f"camera token shape is invalid: {camera_tokens.shape}")
    if target_rays.shape != (batch, NUM_TARGET_VIEWS, 6, RAY_HEIGHT, RAY_WIDTH):
        raise RuntimeError(f"target ray shape is invalid: {target_rays.shape}")
    if not torch.isfinite(camera_tokens).all() or not torch.isfinite(target_rays).all():
        raise FloatingPointError("RepEncoder camera conditioning contains NaN or Inf")
    return RepEncoderCameraConditioning(
        source_camera_tokens=camera_tokens,
        target_rays=target_rays,
        source_c2w_normalized=source,
        target_c2w_normalized=target,
        scale_tokens=scale_tokens,
        scene_scale_m=scene_scale,
    )


__all__ = [
    "CAMERA_SCALE_MULTIPLIER",
    "DEFAULT_NEAR_ZERO_BASELINE_M",
    "RepEncoderCameraConditioning",
    "NUM_SOURCE_VIEWS",
    "NUM_TARGET_VIEWS",
    "RAY_HW",
    "normalize_source_target_cameras",
    "prepare_repencoder_camera_conditioning",
]
