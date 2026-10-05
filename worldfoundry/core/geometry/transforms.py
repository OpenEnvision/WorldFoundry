"""Geometry helpers shared by camera-conditioned visual generation runtimes.

Camera-control recipes need the same SE(3) / pinhole primitives whether
the model is Wan, SANA-WM, or GEN3C. This module is the NumPy/Torch
home for those ops so family packages do not each copy Euler and
quaternion conversions:

- Depth unprojection (:func:`depth_to_world_points`).
- ZYX Euler ↔ rotation matrix, OpenCV ``(z, y, x)`` order, and
  WXYZ / XYZW quaternions (XYZW path adapted from PyTorch3D).
- :func:`ray_condition` — per-pixel Plücker or origin+direction
  features from ``[fx, fy, cx, cy]`` and c2w (half-pixel centers).
- :func:`render_point_cloud_frames_torch` — debug splat of world
  points back into camera frames.

OpenCV convention: +X right, +Y down, +Z forward.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────
# Pinhole unprojection + Euler / quaternion conversions (OpenCV axes)
# ──────────────────────────────────────────────────────────────────────────


def depth_to_world_points(depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Unproject a depth image with an OpenCV camera-to-world transform."""

    depth = np.asarray(depth)
    intrinsics = np.asarray(intrinsics)
    pose = np.asarray(pose)
    if depth.ndim != 2:
        raise ValueError(f"expected a 2D depth image, got shape {depth.shape}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"expected 3x3 camera intrinsics, got shape {intrinsics.shape}")
    if pose.shape not in {(3, 4), (4, 4)}:
        raise ValueError(f"expected a 3x4 or 4x4 camera pose, got shape {pose.shape}")

    height, width = depth.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    # OpenCV pinhole: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy, then R @ p + t.
    camera_points = np.stack(
        (
            (x - intrinsics[0, 2]) * depth / intrinsics[0, 0],
            (y - intrinsics[1, 2]) * depth / intrinsics[1, 1],
            depth,
        ),
        axis=-1,
    )
    return camera_points @ pose[:3, :3].T + pose[:3, 3]


def euler_angles_to_rotation_matrix_zyx(euler_angles: np.ndarray) -> np.ndarray:
    """Convert ``[x, y, z]`` Euler angles to ``Rz @ Ry @ Rx``."""
    x, y, z = euler_angles
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def rotation_matrix_to_euler_angles_zyx(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to ``[x, y, z]`` Euler angles."""
    rotation = np.asarray(rotation)
    identity = np.identity(3, dtype=rotation.dtype)
    if rotation.shape != (3, 3) or np.linalg.norm(identity - rotation.T @ rotation) >= 1e-6:
        raise ValueError("expected a valid 3x3 rotation matrix")
    # ZYX Euler: sy ~ 0 is gimbal lock; recover x from row 1 and freeze z.
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    if sy >= 1e-6:
        x = math.atan2(rotation[2, 1], rotation[2, 2])
        y = math.atan2(-rotation[2, 0], sy)
        z = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        x = math.atan2(-rotation[1, 2], rotation[1, 1])
        y = math.atan2(-rotation[2, 0], sy)
        z = 0.0
    return (np.array([x, y, z]) + np.pi) % (2 * np.pi) - np.pi


def rotation_matrix_to_euler_angles_opencv(rotation: np.ndarray) -> tuple[float, float, float]:
    """Return the ZYX angles in OpenCV's ``(z, y, x)`` order."""
    x, y, z = rotation_matrix_to_euler_angles_zyx(rotation)
    return float(z), float(y), float(x)


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized ``[w, x, y, z]`` quaternion."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("expected a 3x3 rotation matrix")
    trace = np.trace(rotation)
    if trace > 0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / scale
        x = (rotation[2, 1] - rotation[1, 2]) * scale
        y = (rotation[0, 2] - rotation[2, 0]) * scale
        z = (rotation[1, 0] - rotation[0, 1]) * scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.array([w, x, y, z])
    return quaternion / np.linalg.norm(quaternion)


# The XYZW quaternion conversion routines below are adapted from PyTorch3D's
# rotation_conversions.py (BSD-3-Clause), Copyright Meta Platforms, Inc.
def quaternion_xyzw_to_rotation_matrix(quaternions: Tensor) -> Tensor:
    """Convert ``[..., x, y, z, w]`` quaternions to rotation matrices."""

    x, y, z, w = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    matrix = torch.stack(
        (
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ),
        dim=-1,
    )
    return matrix.reshape(quaternions.shape[:-1] + (3, 3))


def rotation_matrix_to_quaternion_xyzw(matrix: Tensor) -> Tensor:
    """Convert ``[..., 3, 3]`` rotation matrices to canonical XYZW quaternions.

    The implementation chooses the best-conditioned of four equivalent
    quaternion candidates and keeps gradients well-defined away from zero.
    """

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_shape = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_shape + (9,)),
        dim=-1,
    )
    q_abs = _sqrt_positive_part(
        torch.stack(
            (
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ),
            dim=-1,
        )
    )
    quaternion_by_wxyz = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    floor = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    candidates = quaternion_by_wxyz / (2.0 * q_abs[..., None].max(floor))
    quaternion_wxyz = candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4).bool(), :].reshape(
        batch_shape + (4,)
    )
    return standardize_quaternion_xyzw(quaternion_wxyz[..., (1, 2, 3, 0)])


def standardize_quaternion_xyzw(quaternions: Tensor) -> Tensor:
    """Choose the equivalent XYZW quaternion whose real part is non-negative."""

    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def _sqrt_positive_part(value: Tensor) -> Tensor:
    """sqrt of the positive part only; the grad-enabled path avoids ``sqrt(0)`` singularities."""
    result = torch.zeros_like(value)
    positive = value > 0
    if torch.is_grad_enabled():
        result[positive] = torch.sqrt(value[positive])
        return result
    return torch.where(positive, torch.sqrt(value), result)


# ──────────────────────────────────────────────────────────────────────────
# Plücker / origin+direction rays and debug splat
# ──────────────────────────────────────────────────────────────────────────


def torch_meshgrid_ij(*args):
    """Return an ``ij``-indexed meshgrid across supported torch versions."""

    try:
        return torch.meshgrid(*args, indexing="ij")
    except TypeError:
        return torch.meshgrid(*args)


def ray_condition(
    K,
    c2w,
    H: int,
    W: int,
    device,
    flip_flag=None,
    *,
    use_ray_o: bool = False,
):
    """Build per-pixel Plucker ray features from intrinsics and camera-to-world poses.

    Args:
        K: Camera intrinsics in ``[fx, fy, cx, cy]`` layout with shape ``[B,V,4]``.
            When ``None``, rays use a constant forward camera-space direction.
        c2w: Camera-to-world matrices with shape ``[B,V,4,4]``.
        H: Output grid height.
        W: Output grid width.
        device: Device for generated coordinate grids.
        flip_flag: Optional boolean mask selecting horizontally flipped views.
        use_ray_o: If true, concatenate ``[ray_origin, ray_direction]`` instead of
            the Plucker ``[ray_origin x ray_direction, ray_direction]`` form.
    """

    batch, views = c2w.shape[:2]
    j, i = torch_meshgrid_ij(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape(1, 1, H * W).expand(batch, views, H * W) + 0.5
    j = j.reshape(1, 1, H * W).expand(batch, views, H * W) + 0.5

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = torch_meshgrid_ij(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype),
        )
        i_flip = i_flip.reshape(1, 1, H * W).expand(batch, 1, H * W) + 0.5
        j_flip = j_flip.reshape(1, 1, H * W).expand(batch, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    if K is None:
        directions = torch.zeros(batch, views, H * W, 3, device=device, dtype=c2w.dtype)
        directions[..., 2] = 1.0
    else:
        fx, fy, cx, cy = K.chunk(4, dim=-1)
        zs = torch.ones_like(i)
        xs = (i - cx) / fx * zs
        ys = (j - cy) / fy * zs
        zs = zs.expand_as(ys)
        directions = torch.stack((xs, ys, zs), dim=-1)
        directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3]
    rays_o = rays_o[:, :, None].expand_as(rays_d)
    if use_ray_o:
        plucker = torch.cat([rays_o, rays_d], dim=-1)
    else:
        rays_dxo = torch.cross(rays_o, rays_d, dim=-1)
        plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    return plucker.reshape(batch, views, H, W, 6)


def render_point_cloud_frames_torch(
    intrinsics: Tensor,
    world_to_camera: Tensor,
    points: Tensor | np.ndarray,
    colors: Tensor | np.ndarray,
    *,
    height: int,
    width: int,
    device: torch.device | str | None = None,
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius_ndc: float = 0.008,
    max_splat_entries: int = 4_000_000,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render OpenCV-camera point clouds with a chunked torch z-buffer.

    The function is a dependency-free fallback for inference runtimes that
    ordinarily use PyTorch3D point rasterization. Inputs use pixel-space
    intrinsics and OpenCV world-to-camera matrices. Returned tensors are RGB
    ``[F,3,H,W]``, hole masks ``[F,1,H,W]`` (one means no point), and depth
    ``[F,1,H,W]`` (minus one means no point).

    ``max_splat_entries`` bounds temporary memory independently of point-cloud
    size, which keeps the same implementation usable on consumer GPUs and on
    larger A100/H100-class devices.
    """

    height, width = int(height), int(width)
    if height < 1 or width < 1:
        raise ValueError(f"height and width must be positive, got {(height, width)}")
    if intrinsics.ndim != 3 or tuple(intrinsics.shape[-2:]) != (3, 3):
        raise ValueError(f"intrinsics must have shape [F,3,3], got {tuple(intrinsics.shape)}")
    if world_to_camera.ndim != 3 or tuple(world_to_camera.shape[-2:]) != (4, 4):
        raise ValueError(
            f"world_to_camera must have shape [F,4,4], got {tuple(world_to_camera.shape)}"
        )
    if intrinsics.shape[0] != world_to_camera.shape[0]:
        raise ValueError("intrinsics and world_to_camera must have the same frame count")

    resolved_device = torch.device(device) if device is not None else world_to_camera.device
    intrinsics = intrinsics.to(device=resolved_device, dtype=torch.float32)
    world_to_camera = world_to_camera.to(device=resolved_device, dtype=torch.float32)
    point_tensor = torch.as_tensor(points, device=resolved_device, dtype=torch.float32)
    color_tensor = torch.as_tensor(colors, device=resolved_device, dtype=torch.float32)
    if point_tensor.ndim != 2 or point_tensor.shape[1] != 3:
        raise ValueError(f"points must have shape [N,3], got {tuple(point_tensor.shape)}")
    if color_tensor.ndim != 2 or color_tensor.shape != point_tensor.shape:
        raise ValueError(
            f"colors must match points shape [N,3], got {tuple(color_tensor.shape)}"
        )

    radius_pixels = max(0, int(round(float(radius_ndc) * min(height, width) * 0.5)))
    if radius_pixels == 0:
        offsets = torch.zeros((1, 2), device=resolved_device, dtype=torch.long)
    else:
        axis = torch.arange(-radius_pixels, radius_pixels + 1, device=resolved_device)
        offset_y, offset_x = torch.meshgrid(axis, axis, indexing="ij")
        disk = offset_x.square() + offset_y.square() <= radius_pixels * radius_pixels
        offsets = torch.stack((offset_x[disk], offset_y[disk]), dim=-1).to(torch.long)
    offset_x, offset_y = offsets[:, 0], offsets[:, 1]
    offsets_per_point = int(offsets.shape[0])
    chunk_points = max(1, int(max_splat_entries) // offsets_per_point)
    flat_pixels = height * width
    background = torch.as_tensor(background_color, device=resolved_device, dtype=torch.float32)
    if background.numel() != 3:
        raise ValueError("background_color must contain three values")

    rendered_frames: list[Tensor] = []
    hole_masks: list[Tensor] = []
    depth_frames: list[Tensor] = []
    for frame_index in range(int(world_to_camera.shape[0])):
        matrix = world_to_camera[frame_index]
        camera_points = point_tensor @ matrix[:3, :3].T + matrix[:3, 3]
        z = camera_points[:, 2]
        finite = torch.isfinite(camera_points).all(dim=-1) & (z > 1.0e-4)
        camera_points, z = camera_points[finite], z[finite]
        frame_colors = color_tensor[finite]

        if camera_points.numel() == 0:
            rendered_frames.append(background[:, None, None].expand(3, height, width).clone())
            hole_masks.append(torch.ones((1, height, width), device=resolved_device))
            depth_frames.append(torch.full((1, height, width), -1.0, device=resolved_device))
            continue

        intrinsic = intrinsics[frame_index]
        u = intrinsic[0, 0] * camera_points[:, 0] / z + intrinsic[0, 2]
        v = intrinsic[1, 1] * camera_points[:, 1] / z + intrinsic[1, 2]
        finite_uv = torch.isfinite(u) & torch.isfinite(v)
        u, v, z, frame_colors = u[finite_uv], v[finite_uv], z[finite_uv], frame_colors[finite_uv]
        center_u, center_v = torch.round(u).to(torch.long), torch.round(v).to(torch.long)

        depth_buffer = torch.full((flat_pixels,), torch.inf, device=resolved_device)
        for start in range(0, int(z.shape[0]), chunk_points):
            stop = min(start + chunk_points, int(z.shape[0]))
            pixel_u = center_u[start:stop, None] + offset_x[None]
            pixel_v = center_v[start:stop, None] + offset_y[None]
            chunk_z = z[start:stop, None].expand(-1, offsets_per_point)
            inside = (pixel_u >= 0) & (pixel_u < width) & (pixel_v >= 0) & (pixel_v < height)
            if inside.any():
                indices = (pixel_v[inside] * width + pixel_u[inside]).to(torch.long)
                depth_buffer.scatter_reduce_(0, indices, chunk_z[inside], reduce="amin", include_self=True)

        color_accumulator = torch.zeros((flat_pixels, 3), device=resolved_device)
        weight_accumulator = torch.zeros((flat_pixels,), device=resolved_device)
        radius_for_weight = max(float(radius_pixels) + 0.5, 0.5)
        for start in range(0, int(z.shape[0]), chunk_points):
            stop = min(start + chunk_points, int(z.shape[0]))
            pixel_u = center_u[start:stop, None] + offset_x[None]
            pixel_v = center_v[start:stop, None] + offset_y[None]
            chunk_z = z[start:stop, None].expand(-1, offsets_per_point)
            inside = (pixel_u >= 0) & (pixel_u < width) & (pixel_v >= 0) & (pixel_v < height)
            if not inside.any():
                continue
            indices = (pixel_v[inside] * width + pixel_u[inside]).to(torch.long)
            candidate_depth = chunk_z[inside]
            surface_tolerance = torch.maximum(
                depth_buffer[indices].abs() * 1.0e-3,
                torch.full_like(candidate_depth, 1.0e-4),
            )
            closest = candidate_depth <= depth_buffer[indices] + surface_tolerance
            if not closest.any():
                continue
            expanded_colors = frame_colors[start:stop, None, :].expand(-1, offsets_per_point, -1)
            distance_squared = (
                (u[start:stop, None] - pixel_u.to(u.dtype)).square()
                + (v[start:stop, None] - pixel_v.to(v.dtype)).square()
            )
            weights = (1.0 - distance_squared / (radius_for_weight * radius_for_weight)).clamp_min(1.0e-6)
            selected_indices = indices[closest]
            selected_weights = weights[inside][closest].to(torch.float32)
            selected_colors = expanded_colors[inside][closest].to(torch.float32)
            color_accumulator.index_add_(0, selected_indices, selected_colors * selected_weights[:, None])
            weight_accumulator.index_add_(0, selected_indices, selected_weights)

        visible = weight_accumulator > 0
        frame = background[None].expand(flat_pixels, 3).clone()
        frame[visible] = color_accumulator[visible] / weight_accumulator[visible, None]
        depth = depth_buffer.clone()
        depth[~torch.isfinite(depth)] = -1.0
        rendered_frames.append(frame.view(height, width, 3).permute(2, 0, 1).contiguous())
        hole_masks.append((~visible).to(torch.float32).view(1, height, width))
        depth_frames.append(depth.view(1, height, width))

    return (
        torch.stack(rendered_frames, dim=0),
        torch.stack(hole_masks, dim=0),
        torch.stack(depth_frames, dim=0),
    )


__all__ = [
    "euler_angles_to_rotation_matrix_zyx",
    "quaternion_xyzw_to_rotation_matrix",
    "ray_condition",
    "render_point_cloud_frames_torch",
    "rotation_matrix_to_euler_angles_opencv",
    "rotation_matrix_to_euler_angles_zyx",
    "rotation_matrix_to_quaternion_wxyz",
    "rotation_matrix_to_quaternion_xyzw",
    "standardize_quaternion_xyzw",
    "torch_meshgrid_ij",
]
