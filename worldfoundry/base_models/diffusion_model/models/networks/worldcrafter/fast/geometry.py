# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat


def _to_tensor_1d(value, device, dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype).reshape(-1)
    return torch.tensor([value], dtype=dtype, device=device)


def pixel_coordinates(
    height: int,
    width: int,
    pixel_center: bool = False,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sampling coordinates for a ``height x width`` grid.

    With ``pixel_center=False`` the samples are the integers ``0 .. W-1``, which
    is what ``equilib.create_grid`` produces. Combined with ``cx = W / 2`` the
    covered extent is ``[-0.5, 0.5 - 1/W]`` in normalized units: asymmetric by
    one pixel, and the asymmetry is ``2/W`` relative to the half width. That is
    invisible at high resolution but reaches 20% of the half width on a 10-column
    grid, so a pyramid built this way samples a progressively offset frustum.

    With ``pixel_center=True`` the samples are ``0.5 .. W-0.5``, giving the
    symmetric extent ``[-0.5 + 0.5/W, 0.5 - 0.5/W]``. Each coarse sample then
    lands exactly on the centroid of the corresponding block of finer samples,
    which makes the geometry consistent across pyramid levels.
    """
    offset = 0.5 if pixel_center else 0.0
    xs = torch.arange(width, device=device, dtype=dtype) + offset
    ys = torch.arange(height, device=device, dtype=dtype) + offset
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x, grid_y


def compute_fx_from_fov_xi(
    x_fov: torch.Tensor | float,
    xi: torch.Tensor | float,
    width: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Focal length in pixels from horizontal FOV (degrees) and the UCM mirror parameter.

    ``fx`` is proportional to ``width``, which is what keeps the covered field of
    view constant when the same camera is sampled on a coarser grid.
    """
    x_fov = _to_tensor_1d(x_fov, device, dtype)
    xi = _to_tensor_1d(xi, device, dtype)

    batch = max(x_fov.shape[0], xi.shape[0])
    x_fov = x_fov.view(-1).expand(batch)
    xi = xi.view(-1).expand(batch)

    theta = torch.deg2rad(0.5 * x_fov)
    eps = torch.finfo(dtype).eps
    denom = torch.sin(theta).clamp_min(eps)
    return (width * 0.5) * (torch.cos(theta) + xi) / denom


def compute_fov_from_fx_xi(
    fx: torch.Tensor | float,
    xi: torch.Tensor | float,
    width: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Inverse of :func:`compute_fx_from_fov_xi`, returning degrees."""
    fx = _to_tensor_1d(fx, device, dtype)
    xi = _to_tensor_1d(xi, device, dtype)
    batch = max(fx.shape[0], xi.shape[0])
    fx = fx.expand(batch)
    xi = xi.expand(batch)

    a = 2.0 * fx / width
    phi = torch.atan(1.0 / a)
    denom = torch.sqrt(a * a + 1.0)
    ratio = (xi / denom).clamp(-1.0, 1.0)
    theta = torch.asin(ratio) + phi
    return torch.rad2deg(2.0 * theta)


def ucm_unproject_grid(
    height: int,
    width: int,
    fx: torch.Tensor,
    fy: torch.Tensor,
    cx: float | torch.Tensor,
    cy: float | torch.Tensor,
    xi: torch.Tensor,
    pixel_center: bool = False,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unproject a pixel grid to unit-sphere ray directions under the UCM model.

    Returns ``[B, H, W, 3]``.
    """
    fx = _to_tensor_1d(fx, device, dtype)
    fy = _to_tensor_1d(fy, device, dtype)
    cx = _to_tensor_1d(cx, device, dtype)
    cy = _to_tensor_1d(cy, device, dtype)
    xi = _to_tensor_1d(xi, device, dtype)
    batch = max(fx.shape[0], fy.shape[0], cx.shape[0], cy.shape[0], xi.shape[0])

    grid_x, grid_y = pixel_coordinates(
        height, width, pixel_center, device=device, dtype=dtype
    )
    grid_x = grid_x.unsqueeze(0).expand(batch, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(batch, -1, -1)

    fx = fx.expand(batch)[:, None, None]
    fy = fy.expand(batch)[:, None, None]
    cx = cx.expand(batch)[:, None, None]
    cy = cy.expand(batch)[:, None, None]
    xi = xi.expand(batch)[:, None, None]

    x = (grid_x - cx) / fx
    y = (grid_y - cy) / fy

    r2 = x * x + y * y
    alpha = xi + torch.sqrt(1 + (1 - xi * xi) * r2)
    gamma = alpha / (1 + r2)

    return torch.stack([gamma * x, gamma * y, gamma - xi], dim=-1)


def ucm_unproject_grid_fov(
    x_fov: float | torch.Tensor,
    xi: float | torch.Tensor,
    height: int,
    width: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    pixel_center: bool = False,
) -> torch.Tensor:
    """Ray directions for a ``height x width`` grid covering the given horizontal FOV.

    Intrinsics are always derived from the grid that is passed in, so requesting a
    coarser grid resamples the same frustum rather than cropping it. Reusing a
    high-resolution ``fx`` with a low-resolution grid would instead shrink the
    covered FOV, which is the failure mode this function exists to avoid.

    Returns ``[H, W, 3]`` for scalar parameters, ``[B, H, W, 3]`` otherwise.
    """
    is_batched = any(
        torch.is_tensor(p) and p.reshape(-1).numel() > 1 for p in (x_fov, xi)
    )

    fx = compute_fx_from_fov_xi(x_fov, xi, width, device, dtype)
    d_cam = ucm_unproject_grid(
        height=height,
        width=width,
        fx=fx,
        fy=fx,
        cx=width / 2,
        cy=height / 2,
        xi=xi,
        pixel_center=pixel_center,
        device=device,
        dtype=dtype,
    )
    return d_cam if is_batched else d_cam[0]


def project_ucm_points(X, Y, Z, fx, fy, cx, cy, xi):
    """Project camera-frame points onto the UCM image plane."""

    def broadcast_param(param):
        if not torch.is_tensor(param):
            return torch.tensor(param, device=X.device, dtype=X.dtype)
        param = param.to(device=X.device, dtype=X.dtype)
        if param.ndim == 0:
            return param
        flat = param.reshape(-1)
        if flat.numel() == 1:
            return flat.view(1)
        if X.ndim >= 1 and flat.numel() == X.shape[0]:
            return flat.view(flat.shape[0], *([1] * (X.ndim - 1)))
        return param

    fx = broadcast_param(fx)
    fy = broadcast_param(fy)
    cx = broadcast_param(cx)
    cy = broadcast_param(cy)
    xi = broadcast_param(xi)

    r = torch.sqrt(X * X + Y * Y + Z * Z)
    alpha = Z + xi * r
    du = fx * (X / alpha) + cx
    dv = fy * (Y / alpha) + cy
    return du, dv


def project_ucm_points_fov(X, Y, Z, x_fov, xi, height, width):
    fx = compute_fx_from_fov_xi(x_fov, xi, width, X.device, X.dtype)
    return project_ucm_points(X, Y, Z, fx, fx, width / 2, height / 2, xi)


def d_cam_to_angles(d_cam: torch.Tensor) -> torch.Tensor:
    """Direction vectors to ``[azimuth, elevation]`` in radians."""
    d_unit = F.normalize(d_cam, dim=-1)
    x, y, z = d_unit[..., 0], d_unit[..., 1], d_unit[..., 2]
    azimuth = torch.atan2(x, z)
    elevation = -torch.asin(y.clamp(-1.0, 1.0))
    return torch.stack([azimuth, elevation], dim=-1)


def world_to_ray_mats(
    d_cam: torch.Tensor,  # [B, H, W, 3]
    c2w: torch.Tensor,  # [B, T, 4, 4]
) -> torch.Tensor:
    """Per-ray world-to-ray-local transforms, ``[B, T, H, W, 4, 4]``.

    The ray-local frame is z along the ray, x = cam_y x z, y = z x x.
    """
    if d_cam.ndim == 3:
        d_cam = d_cam.unsqueeze(0)
    if c2w.ndim == 3:
        c2w = c2w.unsqueeze(0)
    if d_cam.ndim != 4 or d_cam.shape[-1] != 3:
        raise ValueError(
            f"d_cam must have shape [H,W,3] or [B,H,W,3], got {tuple(d_cam.shape)}"
        )
    if c2w.ndim != 4 or c2w.shape[-2:] != (4, 4):
        raise ValueError(
            f"c2w must have shape [T,4,4] or [B,T,4,4], got {tuple(c2w.shape)}"
        )
    if d_cam.shape[0] == 1 and c2w.shape[0] != 1:
        d_cam = d_cam.expand(c2w.shape[0], -1, -1, -1)
    elif c2w.shape[0] == 1 and d_cam.shape[0] != 1:
        c2w = c2w.expand(d_cam.shape[0], -1, -1, -1)
    elif d_cam.shape[0] != c2w.shape[0]:
        raise ValueError(
            f"d_cam and c2w batch mismatch: {d_cam.shape[0]} vs {c2w.shape[0]}"
        )

    B, H, W, _ = d_cam.shape
    T = c2w.shape[1]
    device = d_cam.device
    dtype = d_cam.dtype

    d_cam = repeat(d_cam, "b h w c -> b t h w c", t=T)
    R_cam = c2w[..., :3, :3]
    t_cam = c2w[..., :3, 3]

    d_world = einsum(R_cam, d_cam, "b t i j, b t h w j -> b t h w i")

    cam_y = R_cam[..., :, 1]
    cam_y = repeat(cam_y, "b t c -> b t h w c", h=H, w=W)

    z_ray = F.normalize(d_world, dim=-1, eps=1e-6)
    x_ray = F.normalize(torch.cross(cam_y, z_ray, dim=-1), dim=-1, eps=1e-6)
    y_ray = F.normalize(torch.cross(z_ray, x_ray, dim=-1), dim=-1, eps=1e-6)

    R_l2w = torch.stack([x_ray, y_ray, z_ray], dim=-1)
    R_w2l = rearrange(R_l2w, "b t h w i j -> b t h w j i")

    t_world = repeat(t_cam, "b t c -> b t h w c", h=H, w=W)
    t_w2l = -einsum(R_w2l, t_world, "b t h w i j, b t h w j -> b t h w i")

    raymats = torch.zeros(B, T, H, W, 4, 4, device=device, dtype=dtype)
    raymats[..., :3, :3] = R_w2l
    raymats[..., :3, 3] = t_w2l
    raymats[..., 3, 3] = 1.0

    mask = torch.isnan(d_world).any(-1)
    raymats[mask] = torch.eye(4, device=device, dtype=dtype)

    return raymats


def compute_up_lat_map(
    R: torch.Tensor,  # [B, T, 3, 3]
    x_fov: torch.Tensor,
    xi: torch.Tensor,
    height: int,
    width: int,
    device: torch.device = torch.device("cpu"),
    delta: float = 0.1,
    pixel_center: bool = False,
):
    """World-up direction and latitude maps used by the ``absmap`` conditioning.

    Returns ``(up_map [B,T,H,W,2], lat_map [B,T,H,W,1])``.
    """
    B, T, _, _ = R.shape
    dtype = R.dtype
    R = R.float()

    d_cam = ucm_unproject_grid_fov(
        x_fov=x_fov,
        xi=xi,
        height=height,
        width=width,
        device=device,
        dtype=torch.float32,
        pixel_center=pixel_center,
    )
    if d_cam.ndim == 3:
        d_cam = d_cam.unsqueeze(0)
    mask = d_cam.isnan().any(dim=-1, keepdim=True)

    d_cam_exp = repeat(d_cam, "B H W C -> B T H W C", T=T)
    d_world = torch.einsum("btij,bthwj->bthwi", R, d_cam_exp)
    d_world = d_world / torch.clamp_min(d_world.norm(dim=-1, keepdim=True), 1e-8)

    Xw, Yw, Zw = d_world[..., 0], d_world[..., 1], d_world[..., 2]
    lat_map = torch.atan2(-Yw, torch.sqrt(Xw**2 + Zw**2)).unsqueeze(-1)

    v = d_world
    up_world = torch.tensor([0, -1, 0], device=device, dtype=torch.float32)
    k = torch.cross(
        v, up_world.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(v), dim=-1
    )
    k = k / torch.clamp_min(k.norm(dim=-1, keepdim=True), 1e-8)

    delta = torch.tensor(delta, device=device, dtype=torch.float32)
    cos_eps = torch.cos(delta)
    sin_eps = torch.sin(delta)
    v_rot = (
        v * cos_eps
        + torch.cross(k, v, dim=-1) * sin_eps
        + k * (k * (v * 1).sum(dim=-1, keepdim=True)) * (1 - cos_eps)
    )

    dirs_cam = torch.einsum("btij,bthwj->bthwi", R.transpose(-1, -2), v_rot)
    Xs, Ys, Zs = dirs_cam[..., 0], dirs_cam[..., 1], dirs_cam[..., 2]

    du, dv = project_ucm_points_fov(
        Xs,
        Ys,
        Zs,
        x_fov=x_fov.float() if torch.is_tensor(x_fov) else x_fov,
        xi=xi.float() if torch.is_tensor(xi) else xi,
        height=height,
        width=width,
    )

    grid_x, grid_y = pixel_coordinates(
        height, width, pixel_center, device=device, dtype=torch.float32
    )
    grid_x = grid_x.view(1, 1, height, width)
    grid_y = grid_y.view(1, 1, height, width)

    up_map = torch.stack((du - grid_x, dv - grid_y), dim=-1)
    up_map = up_map / torch.clamp_min(up_map.norm(dim=-1, keepdim=True), 1e-8)

    up_map = up_map.to(dtype=dtype)
    lat_map = lat_map.to(dtype=dtype)

    mask_exp = mask.unsqueeze(1).expand(B, T, height, width, 1)
    return up_map.masked_fill(mask_exp, 0.0), lat_map.masked_fill(mask_exp, 0.0)
