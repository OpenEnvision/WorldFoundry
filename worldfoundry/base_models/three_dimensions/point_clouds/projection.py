"""Reusable point-cloud projection utilities for inference.

Project 3D points onto a 2D image plane to produce:
- depth: depth map
- mask: valid-pixel mask
- rgb: optional projected colors

Resolve occlusion with a z-buffer that keeps the nearest point.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


class ZBufferResult:
    """Cached z-buffer result for recoloring with different point colors."""

    __slots__ = ("H", "W", "depth_img", "first_idx", "mask_img", "order", "unique_flat", "valid")

    def __init__(
        self,
        H: int,
        W: int,
        valid: np.ndarray,
        order: np.ndarray,
        unique_flat: np.ndarray,
        first_idx: np.ndarray,
        depth_img: np.ndarray,
        mask_img: np.ndarray,
    ):
        self.H = H
        self.W = W
        self.valid = valid
        self.order = order
        self.unique_flat = unique_flat
        self.first_idx = first_idx
        self.depth_img = depth_img
        self.mask_img = mask_img


def _compute_zbuffer(
    points_world: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    image_size: tuple[int, int],
    device: str = "cuda",
) -> ZBufferResult:
    """Compute z-buffer projection (CPU numpy)."""
    H, W = image_size

    # Transform to camera space
    w2c = np.linalg.inv(c2w.astype(np.float64)).astype(np.float32)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    points_cam = (points_world.astype(np.float32) @ R.T) + t

    # Project
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    with np.errstate(divide="ignore", invalid="ignore"):
        z_safe = np.where(z > 0, z, 1.0)
        u_f = x / z_safe * fx + cx
        v_f = y / z_safe * fy + cy
        u_f = np.where(z > 0, u_f, np.nan)
        v_f = np.where(z > 0, v_f, np.nan)

    # Validity check
    int32_max = np.iinfo(np.int32).max
    uv_valid = np.isfinite(u_f) & np.isfinite(v_f) & (z > 0) & (np.abs(u_f) < int32_max) & (np.abs(v_f) < int32_max)
    u_i = np.zeros(len(u_f), dtype=np.int32)
    v_i = np.zeros(len(v_f), dtype=np.int32)
    if uv_valid.any():
        u_i[uv_valid] = np.rint(u_f[uv_valid]).astype(np.int32)
        v_i[uv_valid] = np.rint(v_f[uv_valid]).astype(np.int32)
    valid = uv_valid & (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)

    # Z-buffer scatter
    depth_img = np.zeros((H, W), dtype=np.float32)
    mask_img = np.zeros((H, W), dtype=bool)
    order = np.empty(0, dtype=np.intp)
    unique_flat = np.empty(0, dtype=np.intp)
    first_idx = np.empty(0, dtype=np.intp)

    if valid.any():
        flat = v_i[valid] * W + u_i[valid]
        depth_valid = z[valid]
        order = np.argsort(depth_valid)
        flat_sorted = flat[order]
        depth_sorted = depth_valid[order]
        unique_flat, first_idx = np.unique(flat_sorted, return_index=True)

        depth_min = np.full(H * W, np.inf, dtype=np.float32)
        depth_min[unique_flat] = depth_sorted[first_idx]
        mask_img = np.isfinite(depth_min).reshape(H, W)
        depth_img = depth_min.reshape(H, W)
        depth_img[~mask_img] = 0.0

    return ZBufferResult(H, W, valid, order, unique_flat, first_idx, depth_img, mask_img)


def _assemble_channels(
    zbuf: ZBufferResult,
    channels: Iterable[str],
    rgb_img: np.ndarray | None,
) -> np.ndarray:
    """Assemble output array from z-buffer result and optional RGB."""
    H, W = zbuf.H, zbuf.W
    parts: list[np.ndarray] = []
    for ch in channels:
        if ch == "depth":
            parts.append(zbuf.depth_img[..., None])
        elif ch == "mask":
            parts.append(zbuf.mask_img.astype(np.float32)[..., None])
        elif ch == "rgb":
            if rgb_img is None:
                raise ValueError("rgb_img required for 'rgb' channel")
            parts.append(rgb_img.astype(np.float32))
        else:
            raise ValueError(f"Unsupported projection channel: {ch}")
    if not parts:
        return np.zeros((H, W, 0), dtype=np.float32)
    return np.concatenate(parts, axis=-1).astype(np.float32)


def _colorize_zbuffer(
    zbuf: ZBufferResult,
    colors: np.ndarray,
) -> np.ndarray:
    """Apply colors to a pre-computed z-buffer result."""
    H, W = zbuf.H, zbuf.W
    rgb_img = np.zeros((H, W, 3), dtype=np.float32)
    if zbuf.valid.any() and zbuf.unique_flat.size > 0:
        colors_valid = colors[zbuf.valid]
        colors_sorted = colors_valid[zbuf.order]
        rgb_flat = np.zeros((H * W, 3), dtype=np.float32)
        rgb_flat[zbuf.unique_flat] = colors_sorted[zbuf.first_idx].astype(np.float32)
        rgb_img = rgb_flat.reshape(H, W, 3)
    return rgb_img


def render_projection(
    points_world: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    image_size: tuple[int, int],
    channels: Iterable[str],
    colors: np.ndarray | None = None,
    fill_holes_kernel: int = 0,
    return_zbuffer: bool = False,
    device: str = "cuda",
) -> np.ndarray | tuple[np.ndarray, ZBufferResult]:
    """
    Render a point cloud with NumPy on the CPU.

    Project world-space points into the specified camera view, supporting:
    - depth: depth channel
    - mask: valid-region mask
    - rgb: color channels (requires colors)

    Args:
        return_zbuffer: If True, also return the ZBufferResult for recoloring.
        device: Accepted but unused by the CPU projection.
    """
    channels = list(channels)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError("points_world must be Nx3")

    zbuf = _compute_zbuffer(points_world, K, c2w, image_size, device=device)

    rgb_img = None
    if "rgb" in channels:
        if colors is None:
            raise ValueError("colors must be provided when requesting rgb projection")
        rgb_img = _colorize_zbuffer(zbuf, colors)

    result = _assemble_channels(zbuf, channels, rgb_img)
    if return_zbuffer:
        return result, zbuf
    return result


def recolor_projection(
    zbuf: ZBufferResult,
    channels: Iterable[str],
    colors: np.ndarray,
) -> np.ndarray:
    """
    Recolor a cached z-buffer projection with new point colors.

    Reuse the projected coordinates and depth ordering in zbuf, replacing only colors.
    Avoid the matrix products and sorting required by render_projection.
    """
    channels = list(channels)
    rgb_img = _colorize_zbuffer(zbuf, colors) if "rgb" in channels else None
    return _assemble_channels(zbuf, channels, rgb_img)


# ---------------------------------------------------------------------------
# GPU-batched projection: share one point cloud across all camera views.
# ---------------------------------------------------------------------------


def render_projection_batch_gpu(
    points_world: np.ndarray,
    Ks: np.ndarray,
    c2ws: np.ndarray,
    image_size: tuple[int, int],
    channels: Iterable[str],
    colors: np.ndarray | None = None,
    device: str = "cuda",
    return_zbuffers: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[ZBufferResult]]:
    """
    Project one point cloud into multiple camera views in a batched tensor pass.

    Compute transforms and projection on device, then scatter into z-buffers on CPU.
    Share the point upload and batch the transforms instead of projecting each view separately.

    Args:
        points_world: [N, 3] world-space points.
        Ks: [F, 3, 3] per-frame intrinsics.
        c2ws: [F, 4, 4] per-frame camera-to-world matrices.
        image_size: (H, W)
        channels: Requested output channels.
        colors: [N, 3] optional point colors.
        device: Tensor computation device.
        return_zbuffers: Whether to return ZBufferResult objects for recoloring.

    Returns:
        [F, H, W, C] projection, optionally paired with a list of ZBufferResult objects.
    """
    channels = list(channels)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError("points_world must be Nx3")

    F = Ks.shape[0]
    H, W = image_size
    N = points_world.shape[0]

    with torch.no_grad():
        # Upload points to GPU once (shared across all frames), explicit float32
        pts = torch.as_tensor(points_world, dtype=torch.float32, device=device)  # [N, 3]
        pts_homo = torch.cat([pts, torch.ones(N, 1, device=device, dtype=torch.float32)], dim=1)  # [N, 4]

        # Upload camera matrices and invert on GPU (avoids numpy float64 issues)
        c2ws_t = torch.as_tensor(c2ws, dtype=torch.float32, device=device)  # [F, 4, 4]
        w2cs = torch.linalg.inv(c2ws_t)  # [F, 4, 4] float32
        Ks_t = torch.as_tensor(Ks, dtype=torch.float32, device=device)  # [F, 3, 3]
        del c2ws_t

        # Batched transform: [F, 4, 4] @ [4, N] -> [F, 4, N] -> [F, N, 3]
        pts_cam = torch.bmm(w2cs, pts_homo.T.unsqueeze(0).expand(F, -1, -1))  # [F, 4, N]
        pts_cam = pts_cam[:, :3, :].permute(0, 2, 1)  # [F, N, 3]

        # Extract xyz
        x = pts_cam[:, :, 0]  # [F, N]
        y = pts_cam[:, :, 1]  # [F, N]
        z = pts_cam[:, :, 2]  # [F, N]

        # Batched projection: u = x/z * fx + cx, v = y/z * fy + cy
        fx = Ks_t[:, 0, 0].unsqueeze(1)  # [F, 1]
        fy = Ks_t[:, 1, 1].unsqueeze(1)
        cx = Ks_t[:, 0, 2].unsqueeze(1)
        cy = Ks_t[:, 1, 2].unsqueeze(1)

        safe_z = z.clamp(min=1e-8)
        u_f = x / safe_z * fx + cx  # [F, N]
        v_f = y / safe_z * fy + cy  # [F, N]

        # Round to pixel coordinates
        u_i = u_f.round().to(torch.int64)  # [F, N]
        v_i = v_f.round().to(torch.int64)  # [F, N]

        # Validity mask: z > 0, in bounds
        valid = (z > 0) & (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)  # [F, N]

        # Flat pixel indices
        flat = v_i * W + u_i  # [F, N]
        flat[~valid] = 0  # safe index for invalid points

        # Move to CPU for z-buffer scatter
        z_cpu = z.float().cpu().numpy()  # [F, N] ensure float32
        flat_cpu = flat.cpu().numpy()  # [F, N]
        valid_cpu = valid.cpu().numpy()  # [F, N]

        # Free GPU memory
        del pts, pts_homo, w2cs, Ks_t, pts_cam, x, y, z, safe_z
        del u_f, v_f, u_i, v_i, valid, flat
    torch.cuda.empty_cache()

    # Process z-buffer per frame on CPU (scatter is hard to batch efficiently)
    need_rgb = "rgb" in channels
    colors_f32 = colors.astype(np.float32) if colors is not None else None

    results = []
    zbuffers = []

    for f in range(F):
        v_mask = valid_cpu[f]  # [N]

        depth_img = np.zeros((H, W), dtype=np.float32)
        mask_img = np.zeros((H, W), dtype=bool)
        order = np.empty(0, dtype=np.intp)
        unique_flat = np.empty(0, dtype=np.intp)
        first_idx = np.empty(0, dtype=np.intp)

        if v_mask.any():
            fl = flat_cpu[f, v_mask]
            dv = z_cpu[f, v_mask]
            order = np.argsort(dv)
            fl_sorted = fl[order]
            d_sorted = dv[order]
            unique_flat, first_idx = np.unique(fl_sorted, return_index=True)

            depth_min = np.full(H * W, np.inf, dtype=np.float32)
            depth_min[unique_flat] = d_sorted[first_idx]
            mask_img = np.isfinite(depth_min).reshape(H, W)
            depth_img = depth_min.reshape(H, W)
            depth_img[~mask_img] = 0.0

        zbuf = ZBufferResult(H, W, v_mask, order, unique_flat, first_idx, depth_img, mask_img)

        rgb_img = None
        if need_rgb:
            if colors_f32 is None:
                raise ValueError("colors required for 'rgb' channel")
            rgb_img = _colorize_zbuffer(zbuf, colors_f32)

        result = _assemble_channels(zbuf, channels, rgb_img)
        results.append(result)
        if return_zbuffers:
            zbuffers.append(zbuf)

    output = np.stack(results, axis=0).astype(np.float32)
    if return_zbuffers:
        return output, zbuffers
    return output
