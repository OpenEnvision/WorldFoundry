from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# from frustum.frustrum_gpu import (
#     ensure_pixel_intrinsics,
#     # fix_extrinsic_translation_axis_and_scale,
#     invert_pose,
# )


def ensure_pixel_intrinsics(K: np.ndarray, normalized=False, height=None, width=None) -> np.ndarray:
    """Convert between pixel-space and normalized intrinsic conventions.

    Pixel convention (used internally by ``make_hw_grid`` / reproject):
        Each pixel center sits at an integer coordinate, so the image of
        size W spans x in ``[-0.5, W - 0.5]`` and the geometric image
        center is at ``(W - 1) / 2``.

    Normalized convention (assumed for input/output when ``normalized``):
        ``cx_norm = (cx_pix + 0.5) / W``, with the inverse
        ``cx_pix = cx_norm * W - 0.5``. This means a centered principal
        point corresponds to ``cx_norm == 0.5`` exactly. The half-pixel
        offset matches what most depth / mvs / SfM code uses (it aligns
        the normalized origin with the pixel-corner grid the integer
        indices in ``make_hw_grid`` actually sample).
    """
    K = K.copy().astype(float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if fx > 2 and fy > 2 and cx > 2 and cy > 2:
        # Already in pixel space: undo to normalized so the rest of the
        # function can re-derive the pixel-space K with the right offset.
        # We assume the principal point in the input is (W-1)/2 / (H-1)/2
        # for centered cameras, so dimensions can be inferred via 2*cx + 1.
        W = max(2.0 * cx + 1.0, 1.0)
        H = max(2.0 * cy + 1.0, 1.0)
        fx /= W
        fy /= H
        cx = (cx + 0.5) / W
        cy = (cy + 0.5) / H
    K = np.array([fx, fy, cx, cy], dtype=float)

    if K.shape != (3, 3):
        Kbase = np.eye(3, dtype=float)
        Kbase[0, 0] = K[0]
        Kbase[1, 1] = K[1]
        Kbase[0, 2] = K[2]
        Kbase[1, 2] = K[3]
        K = Kbase
    if normalized:
        return K
    else:
        K[0, 0] *= width
        K[1, 1] *= height
        # Half-pixel offset: keeps the normalized "center of image" (cx=0.5)
        # mapping to (W-1)/2 in pixel space, matching make_hw_grid's
        # integer-pixel-center convention.
        K[0, 2] = K[0, 2] * width - 0.5
        K[1, 2] = K[1, 2] * height - 0.5
        return K


def invert_pose(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def sparse_grid_sample(
    input: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "sparse_nearest",
    padding_mode: str = "zeros",
    align_corners: bool | None = False,
) -> torch.Tensor:
    """
    A sparse-fill replacement for torch.nn.functional.grid_sample with the same
    input/output signature.

    Main behavior (mode="nearest", padding_mode="zeros"):
    - Convert each output location's normalized grid coord to the nearest integer
      source pixel.
    - Only one output location is allowed to consume each source pixel
      (simplest conflict rule: keep the first target location in flattened order).
    - Output locations that do not get a unique source assignment remain zero.

    Notes:
    - This is intentionally NOT the same semantics as F.grid_sample(..., mode="nearest").
      It enforces sparse filling / hole preservation.
    - For unsupported settings, this function falls back to F.grid_sample.
    - Currently focuses on 4D input: [N, C, H_in, W_in] and 4D grid: [N, H_out, W_out, 2].
    """
    if input.ndim != 4:
        raise ValueError(f"Expected 4D input [N,C,H,W], got shape={tuple(input.shape)}")
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError(f"Expected 4D grid [N,H_out,W_out,2], got shape={tuple(grid.shape)}")
    if input.shape[0] != grid.shape[0]:
        raise ValueError(f"Batch size mismatch: input N={input.shape[0]}, grid N={grid.shape[0]}")

    # Keep direct-replace usability: unsupported cases fall back to standard grid_sample.
    if mode != "nearest" or padding_mode != "zeros":
        return F.grid_sample(
            input=input,
            grid=grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

    n, c, h_in, w_in = input.shape
    _, h_out, w_out, _ = grid.shape
    device = input.device
    dtype = input.dtype

    def _unnormalize(coord: torch.Tensor, size: int, align: bool | None) -> torch.Tensor:
        if align:
            # [-1, 1] -> [0, size - 1]
            return ((coord + 1.0) * (size - 1)) * 0.5
        # [-1, 1] -> [-0.5, size - 0.5]
        return ((coord + 1.0) * size - 1.0) * 0.5

    def _round_nearest(x: torch.Tensor) -> torch.Tensor:
        # Avoid torch.round banker rounding on .5
        return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5)).to(torch.long)

    x = _unnormalize(grid[..., 0], w_in, align_corners)
    y = _unnormalize(grid[..., 1], h_in, align_corners)

    x_nn = _round_nearest(x)
    y_nn = _round_nearest(y)

    valid = (x_nn >= 0) & (x_nn < w_in) & (y_nn >= 0) & (y_nn < h_in)

    out = torch.zeros((n, c, h_out, w_out), device=device, dtype=dtype)
    out_flat = out.view(n, c, h_out * w_out)
    inp_flat = input.view(n, c, h_in * w_in)

    # Target flattened index [0, H_out * W_out)
    tgt_lin_all = torch.arange(h_out * w_out, device=device, dtype=torch.long).view(1, -1)
    tgt_lin_all = tgt_lin_all.expand(n, -1)

    x_flat = x_nn.view(n, -1)
    y_flat = y_nn.view(n, -1)
    valid_flat = valid.view(n, -1)

    num_tgt = h_out * w_out

    for b in range(n):
        vb = valid_flat[b]
        if not torch.any(vb):
            continue

        tgt_lin = tgt_lin_all[b, vb]  # [M]
        src_x = x_flat[b, vb]  # [M]
        src_y = y_flat[b, vb]  # [M]
        src_lin = src_y * w_in + src_x  # [M]

        # Simplest conflict rule:
        # keep the first target position (in flattened target order) for each source pixel.
        # We do this by sorting with key = src_lin * num_tgt + tgt_lin,
        # then keeping the first occurrence for each src_lin.
        key = src_lin * num_tgt + tgt_lin
        order = torch.argsort(key)
        src_lin_sorted = src_lin[order]

        keep = torch.ones_like(src_lin_sorted, dtype=torch.bool)
        if src_lin_sorted.numel() > 1:
            keep[1:] = src_lin_sorted[1:] != src_lin_sorted[:-1]

        chosen = order[keep]
        chosen_tgt = tgt_lin[chosen]  # [K]
        chosen_src = src_lin[chosen]  # [K]

        out_flat[b, :, chosen_tgt] = inp_flat[b, :, chosen_src]

    return out


def fix_extrinsic_translation_axis_and_scale(T, trans_axis_order="xyz", flip_up_sign=False):
    Tout = T.copy().astype(float)
    t_given = Tout[:3, 3].copy()
    # t_xyz = _reorder_vec_to_xyz(t_given, order=trans_axis_order)
    if flip_up_sign:
        t_given[2] = -t_given[2]
    Tout[:3, 3] = t_given
    return Tout


# 权重
DIVERSITY_WEIGHT = 0
STABILITY__WEIGHT = 0
FIRST_FRAME_PENALTY = 0
VOTE_WEIGHT = 0  # 组内投票一致性权重

try:
    from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.api import (
        DepthAnything3,
    )

    HAS_DA3 = True
except ImportError:
    HAS_DA3 = False
DepthEstimator = Callable[[Sequence[np.ndarray]], np.ndarray]

# def resample_cthw_t1(
#     image, coords, mode="nearest", padding_mode="zeros", align_corners=True
# ):
#     deprecated
#     """
#     image:  (C, 1, H, W)
#     coords: (H_out, W_out, 2)  float pixel coords (x, y)
#     return: (C, 1, H_out, W_out)
#     """
#     if image.ndim != 4 or image.shape[1] != 1:
#         raise ValueError(f"Expected image shape (C,1,H,W), got {tuple(image.shape)}")

#     C, _, H, W = image.shape
#     H_out, W_out, _ = coords.shape
#     if isinstance(coords, np.ndarray):
#         coords = torch.from_numpy(coords)
#     coords = coords.to(device=image.device, dtype=image.dtype)

#     # pixel -> normalized [-1, 1]
#     x = coords[..., 0]
#     y = coords[..., 1]
#     if align_corners:
#         x = 2.0 * x / (W - 1) - 1.0
#         y = 2.0 * y / (H - 1) - 1.0
#     else:
#         x = 2.0 * (x + 0.5) / W - 1.0
#         y = 2.0 * (y + 0.5) / H - 1.0

#     grid = torch.stack((x, y), dim=-1).unsqueeze(0)  # (1, H_out, W_out, 2)

#     # (C,1,H,W) -> (1,C,H,W)
#     img = image[:, 0].unsqueeze(0)

#     out = sparse_grid_sample(
#         img,
#         grid,
#         mode=mode,
#         padding_mode=padding_mode,
#         align_corners=align_corners,
#     )  # (1, C, H_out, W_out)
#     # dtype_ = img.dtype
#     # out = F.grid_sample(
#     #     img,
#     #     grid,
#     #     mode=mode,
#     #     padding_mode=padding_mode,
#     #     align_corners=align_corners,
#     # )  # (1, C, H_out, W_out)
#     # # back -> (C,1,H_out,W_out)
#     # out = out.to(dtype=dtype_)
#     out = out.squeeze(0).unsqueeze(1)
#     return out


# =========================
# Caching structures
# =========================
def make_hw_grid(H, W):
    h = np.arange(H)
    w = np.arange(W)
    hh, ww = np.meshgrid(h, w, indexing="ij")  # 保证第一维是H
    grid = np.stack([hh, ww], axis=-1)  # (H, W, 2)
    return grid


def reproject_pixels_nd_to_nd(
    pix_hw: np.ndarray = None,  # (...,2) with (h, w) == (v, u)
    depth_z: np.ndarray = None,  # (...)   Zc in old camera (meters)
    K_old: np.ndarray = None,  # (...,3,3)
    T_old_w2c: np.ndarray = None,  # (...,4,4) world-to-camera (X_c = R X_w + t)
    K_new: np.ndarray = None,  # (...,3,3) or (3,3) or (1,3,3)
    T_new_w2c: np.ndarray = None,  # (...,4,4) or (4,4) or (1,4,4) world-to-camera
    eps: float = 0.0,
    *,
    T_old_c2w=None,  # DEPRECATED alias; actually expected w2c
    T_new_c2w=None,  # DEPRECATED alias; actually expected w2c
):
    """Reproject pixels from one camera to another (no distortion, no visibility check).

    Convention (IMPORTANT):
      - ``pix_hw[..., 0] = h (v, row)``, ``pix_hw[..., 1] = w (u, col)``
      - ``depth_z`` is Z in the OLD camera coord system (must be metric / consistent
        with the world units implied by the extrinsics' translation vectors).
      - ``T_old_w2c`` / ``T_new_w2c`` are **world-to-camera** rigid transforms:
        ``X_c = R @ X_w + t``. The legacy parameter names ``T_old_c2w`` /
        ``T_new_c2w`` are misleading -- the function math has always required
        w2c -- but are retained as keyword aliases for backward compatibility.
        New code should pass ``T_old_w2c`` / ``T_new_w2c``.

    Returns:
        pix_new_hw (..., 2) float -- projected (h, w) in the NEW camera; invalid
            points are filled with -10000 sentinel for downstream filtering.
        zc_new (...) -- Z in the new camera coord (NOT NaN-masked; check
            ``isfinite`` and ``> 0`` before trusting).
        Xw (..., 3) -- intermediate world point.
        Xc_new (..., 3) -- intermediate new-camera point.
    """
    # Resolve legacy c2w-named kwargs
    if T_old_w2c is None and T_old_c2w is not None:
        warnings.warn(
            "reproject_pixels_nd_to_nd: 'T_old_c2w' is a legacy misnomer (the "
            "function has always required w2c). Use 'T_old_w2c' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        T_old_w2c = T_old_c2w
    if T_new_w2c is None and T_new_c2w is not None:
        warnings.warn(
            "reproject_pixels_nd_to_nd: 'T_new_c2w' is a legacy misnomer (the "
            "function has always required w2c). Use 'T_new_w2c' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        T_new_w2c = T_new_c2w
    if pix_hw is None or depth_z is None or K_old is None or K_new is None:
        raise ValueError("reproject_pixels_nd_to_nd: pix_hw, depth_z, K_old, K_new are required.")
    if T_old_w2c is None or T_new_w2c is None:
        raise ValueError(
            "reproject_pixels_nd_to_nd: must pass T_old_w2c and T_new_w2c (or legacy T_old_c2w / T_new_c2w aliases)."
        )

    pix_hw = np.asarray(pix_hw)
    depth_z = np.asarray(depth_z)
    K_old = np.asarray(K_old)
    T_old_w2c = np.asarray(T_old_w2c)
    K_new = np.asarray(K_new)
    T_new_w2c = np.asarray(T_new_w2c)

    assert pix_hw.shape[-1] == 2, f"pix_hw must be (...,2), got {pix_hw.shape}"
    lead = pix_hw.shape[:-1]
    assert depth_z.shape == lead, f"depth_z must be {lead}, got {depth_z.shape}"
    # Remember whether K_old is a SINGLE intrinsic shared across every grid
    # point (the common case: one camera per call). Inverting the broadcast
    # array below would re-invert the SAME 3x3 once per pixel (e.g. 3520x for
    # a 44x80 latent grid), which dominated the reproject cost. We instead
    # invert the unique 3x3 once and broadcast the result -- bit-identical
    # (same LAPACK inv on the same matrix) but ~N_pixels cheaper.
    _K_old_single = K_old.shape == (3, 3)
    if not K_old.shape == lead + (
        3,
        3,
    ):
        K_old = np.broadcast_to(K_old, lead + (3, 3))
    if not T_old_w2c.shape == lead + (
        4,
        4,
    ):
        T_old_w2c = np.broadcast_to(T_old_w2c, lead + (4, 4))

    # Normalize new cam shapes:
    # - if (3,3)/(4,4), broadcast to lead
    # - if (1,3,3)/(1,4,4), squeeze then broadcast
    if K_new.shape == (1, 3, 3):
        K_new = K_new[0]
    if T_new_w2c.shape == (1, 4, 4):
        T_new_w2c = T_new_w2c[0]

    if K_new.shape == (3, 3):
        K_new = np.broadcast_to(K_new, lead + (3, 3))
    if T_new_w2c.shape == (4, 4):
        T_new_w2c = np.broadcast_to(T_new_w2c, lead + (4, 4))

    assert K_new.shape == lead + (
        3,
        3,
    ), f"K_new must be {lead + (3, 3)} or (3,3), got {K_new.shape}"
    assert T_new_w2c.shape == lead + (
        4,
        4,
    ), f"T_new_w2c must be {lead + (4, 4)} or (4,4), got {T_new_w2c.shape}"

    # (h,w)->(v,u)
    v = pix_hw[..., 0]
    u = pix_hw[..., 1]

    ones = np.ones(lead, dtype=np.result_type(pix_hw.dtype, np.float32))
    pix_homo = np.stack([u, v, ones], axis=-1)  # (...,3)

    # back-project in old camera
    if _K_old_single:
        # Invert the unique 3x3 once, then broadcast: identical values to
        # ``inv(broadcast(K))`` but avoids the per-pixel re-inversion.
        invK_old = np.broadcast_to(np.linalg.inv(K_old[(0,) * len(lead)]), lead + (3, 3))
    else:
        invK_old = np.linalg.inv(K_old)  # (...,3,3)
    rays = np.einsum("...ij,...j->...i", invK_old, pix_homo)  # (...,3)
    Xc_old = rays * depth_z[..., None]  # (...,3)

    # old camera -> world (T_old is w2c, so Xc = R Xw + t  =>  Xw = R^T (Xc - t))
    R_old = T_old_w2c[..., :3, :3]  # (...,3,3)
    t_old = T_old_w2c[..., :3, 3]  # (...,3)
    R_old_T = np.swapaxes(R_old, -1, -2)
    Xw = np.einsum("...ij,...j->...i", R_old_T, (Xc_old - t_old))  # (...,3)

    # world -> new camera (per-point, T_new is w2c): Xc_new = R_new Xw + t_new
    R_new = T_new_w2c[..., :3, :3]  # (...,3,3)
    t_new = T_new_w2c[..., :3, 3]  # (...,3)
    Xc_new = np.einsum("...ij,...j->...i", R_new, Xw) + t_new  # (...,3)

    zc_new = Xc_new[..., 2]  # (...)

    # 在 pinhole 相机模型中，通常 z>0 才表示点位于相机前方，才可视为有效投影
    valid_mask = np.isfinite(zc_new) & (zc_new > 1e-2)

    # normalize + project with per-point K_new
    if eps == 0.0:
        denom = np.where(valid_mask, zc_new, 1.0)
    else:
        denom = zc_new + np.where(zc_new >= 0, eps, -eps)
        # 避免无效点参与除法，后续会统一写成 NaN
        denom = np.where(valid_mask, denom, 1.0)

    x = np.full_like(zc_new, np.nan, dtype=np.result_type(Xc_new.dtype, np.float32))
    y = np.full_like(zc_new, np.nan, dtype=np.result_type(Xc_new.dtype, np.float32))
    np.divide(Xc_new[..., 0], denom, out=x, where=valid_mask)
    np.divide(Xc_new[..., 1], denom, out=y, where=valid_mask)

    xy1 = np.stack([x, y, ones], axis=-1)  # (...,3)
    uv1 = np.einsum("...ij,...j->...i", K_new, xy1)  # (...,3)

    u_new = uv1[..., 0]
    v_new = uv1[..., 1]

    pix_new_hw = np.stack([v_new, u_new], axis=-1)  # (...,2) float
    # 将无效点显式标记为 -1，便于下游直接按 mask 或 NaN 过滤
    pix_new_hw = np.where(valid_mask[..., None], pix_new_hw, -10000)

    return pix_new_hw, zc_new, Xw, Xc_new


@dataclass
class SourceCache:
    """Container for per-frame cached geometry with global indexing support."""

    # Global index of the first frame stored in this cache (e.g., 1)
    first_index: int

    # Per-frame arrays (length = T)
    c2w: np.ndarray  # (T, 4, 4)
    w2c: np.ndarray  # (T, 4, 4)
    camera_centers: np.ndarray  # (T, 3)
    depths: np.ndarray  # (T, H, W)
    is_keyframe: np.ndarray  # (T,) bool，硬性关键帧标记
    soft_key_score: np.ndarray  # (T,) float32，软性关键分数，范围 [0,1]

    # ---- helpers ----
    @property
    def num_frames(self) -> int:
        return int(self.depths.shape[0])

    def local_to_global(self, local_idx: int) -> int:
        return int(self.first_index + local_idx)

    def global_to_local(self, global_idx: int) -> Optional[int]:
        li = int(global_idx - self.first_index)
        return li if 0 <= li < self.num_frames else None

    def all_global_indices(self) -> List[int]:
        start = int(self.first_index)
        end = int(self.first_index + self.num_frames)
        return list(range(start, end))


# =========================
# Frustum Handler
# =========================


class FrustumHandler:
    """
    High level frustum overlap helper with APPEND-able cache and global indexing.

    Typical usage:
        handler = FrustumHandler(intrinsic, image_size=(H, W))
        handler.register_source_sequence(device, frames_block1, extrinsics_block1, start_index=1)
        handler.register_source_sequence(device, frames_block2, extrinsics_block2)  # auto-append at 1+len(block1)
        groups = handler.query_hits(device, query_extrinsics, ...)
    """

    def __init__(
        self,
        intrinsic: np.ndarray,
        *,
        image_size: Optional[Tuple[int, int]] = None,
        grid_size: Tuple[int, int] = (64, 64),
        point_stride: Optional[int] = None,
        depth_inf_thresh: Optional[float] = None,
        is_c2w: bool = True,
        trans_axis_order: str = "xyz",
        flip_up_sign: bool = False,
        use_gpu: bool = True,  # kept for compatibility (not toggling logic here)
        depth_estimator: Optional[DepthEstimator] = None,
        depth_model_ckpt: Optional[str] = None,
        depth_model_config: Optional[Dict] = None,
        depth_infer_fps: int = 30,
        depth_input_size: Optional[int] = None,
        depth_fp32: bool = False,
        keyframe_rot_thresh_deg: float = 45,
        keyframe_trans_thresh: float = 1,
        keyframe_max_interval: int = 100,
        init_extrinsic=None,
        latent_stride: int = 16,
        fixed_intrinsics: bool = False,
    ):
        # Intrinsics / image size
        self._intrinsic = ensure_pixel_intrinsics(
            np.asarray(intrinsic, dtype=float),
            normalized=True,
        )
        self._image_size = (int(image_size[0]), int(image_size[1])) if image_size is not None else None

        # Grid / sampling
        self._grid_size = tuple(int(v) for v in grid_size)
        self._point_stride = point_stride
        self._depth_inf_thresh = depth_inf_thresh

        # Pose conversion
        self._is_c2w = bool(is_c2w)
        self._trans_axis_order = trans_axis_order
        self._flip_up_sign = bool(flip_up_sign)

        # GPU flag (for kernels)
        self._use_gpu = bool(use_gpu)

        # Depth estimation
        self._depth_estimator = depth_estimator
        self._depth_model_ckpt = depth_model_ckpt
        self._depth_model_config = depth_model_config or {}
        self._depth_infer_fps = int(depth_infer_fps)
        self._depth_input_size = depth_input_size
        self._depth_fp32 = bool(depth_fp32)

        # 关键帧参数：硬阈值 + 时间强制阈值
        self._keyframe_rot_thresh_deg = float(keyframe_rot_thresh_deg)
        self._keyframe_trans_thresh = float(keyframe_trans_thresh)
        self._keyframe_max_interval = max(1, int(keyframe_max_interval))

        # VAE 空间下采样比 (latent ↔ pixel). WAN VAE 默认 16, 不同 VAE 需调
        # 此参数影响 reproject 时 pix_hw 的尺度换算与回换算 (proj/stride).
        if int(latent_stride) <= 0:
            raise ValueError(f"latent_stride must be positive, got {latent_stride}")
        self._latent_stride = int(latent_stride)

        # Cache & matrices
        self._cache: Optional[SourceCache] = None
        self.depths = []
        self._Kpix: Optional[np.ndarray] = None
        self._Kpixs = []
        self._Kpix_inv: Optional[np.ndarray] = None
        self._fixed_intrinsics = bool(fixed_intrinsics)
        self._stride: Optional[int] = None
        self.extrinsics = []
        self.frames = []
        if init_extrinsic is not None:
            self.last_extrinsic = init_extrinsic
        else:
            self.last_extrinsic = np.eye(4)[None, ...]  # 用于对齐新加入的相机位姿

    @staticmethod
    def _compute_view_change_for_points(points_world, source_w2c, query_w2c):
        source_w2c = np.asarray(source_w2c, dtype=np.float64)
        query_w2c = np.asarray(query_w2c, dtype=np.float64)
        points_world = np.asarray(points_world, dtype=np.float64)
        Rs = source_w2c[:3, :3]
        ts = source_w2c[:3, 3]
        Rt = query_w2c[:3, :3]
        tt = query_w2c[:3, 3]
        c_src = -Rs.T @ ts
        c_tgt = -Rt.T @ tt
        a = c_tgt[None, :] - points_world
        b = c_src[None, :] - points_world
        ratio = np.linalg.norm(a, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-8)
        u_b = (Rt @ b.T).T
        u_a = (Rt @ a.T).T
        u_b = u_b / np.maximum(np.linalg.norm(u_b, axis=-1, keepdims=True), 1e-8)
        u_a = u_a / np.maximum(np.linalg.norm(u_a, axis=-1, keepdims=True), 1e-8)
        phi = np.arctan2(u_b[:, 0], -u_b[:, 2]) - np.arctan2(u_a[:, 0], -u_a[:, 2])
        phi = np.arctan2(np.sin(phi), np.cos(phi))
        theta = np.arcsin(np.clip(-u_b[:, 1], -1.0, 1.0)) - np.arcsin(np.clip(-u_a[:, 1], -1.0, 1.0))
        return np.stack([ratio, phi, theta], axis=-1).astype(np.float32)

    @contextlib.contextmanager
    def _timing_scope(self, name: str, metadata: Optional[Dict] = None):
        del name, metadata
        yield self

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get_state(self) -> Dict:
        """Get a serializable snapshot of the handler's state (for checkpointing, etc.)."""
        state = {
            "_Kpix": self._Kpix,
            "image_size": self._image_size,
            "grid_size": self._grid_size,
            "depths": self.depths,
            "extrinsics": self.extrinsics,
            "frames": self.frames,
            "is_c2w": self._is_c2w,
        }
        return state

    def load_state(self, state: Dict, clip_num) -> None:
        """Load a state snapshot previously obtained from get_state()."""
        self._Kpix = state.get("_Kpix")
        self._image_size = state.get("image_size")
        self._grid_size = state.get("grid_size")
        self.depths = state.get("depths", [])[clip_num:]  # 只保留最近的 depth_infer_fps 帧
        self.extrinsics = state.get("extrinsics", [])[clip_num:]  # 只保留最近的 depth_infer_fps 帧
        self.frames = state.get("frames", [])[clip_num:]
        self._is_c2w = state.get("is_c2w", True)
        self.last_extrinsic = self.extrinsics[-1] if self.extrinsics else np.eye(4)

    def reestimate_depth(self, frames: Sequence[np.ndarray]) -> None:
        with torch.no_grad():
            self._depth_estimator = self._depth_estimator.cuda()
            prediction = self._depth_estimator.inference(
                frames,
                use_ray_pose=True,
                process_res=700,
            )
            self._depth_estimator = self._depth_estimator.cpu()
        depths = prediction.depth  # [N, H, W] float32 array
        base_extrinsics = np.eye(4)[None, ...].repeat(depths.shape[0], 0)
        base_extrinsics[:, :3] = prediction.extrinsics  # [N, 4, 4] array
        self.depths = list(depths)
        self.extrinsics = list(base_extrinsics)
        self.frames = list(frames)
        self._Kpixs = list(prediction.intrinsics)  # [N, 3, 3] array
        for i, kpix in enumerate(self._Kpixs):
            self._Kpixs[i] = ensure_pixel_intrinsics(
                kpix,
                normalized=False,
                height=self._image_size[0],
                width=self._image_size[1],
            )
        self._Kpix = self._Kpixs[-1]

    def align_w2c_trajectory(self, A_w2c: np.ndarray, B_w2c: np.ndarray = None, eps: float = 1e-8) -> np.ndarray:
        """
        Align a w2c trajectory A (T,4,4) to a reference w2c B (1,4,4) (or (4,4)),
        enforcing A_aligned[0] == B, and keeping the rest continuous.

        Returns:
            A_aligned: (T,4,4)
        """
        assert A_w2c.ndim == 3 and A_w2c.shape[1:] == (4, 4)
        if B_w2c is None:
            B_w2c = self.last_extrinsic.copy()
        if B_w2c.ndim == 3:
            assert B_w2c.shape[0] == 1 and B_w2c.shape[1:] == (4, 4)
            B = B_w2c[0]
        else:
            assert B_w2c.shape == (4, 4)
            B = B_w2c

        A0 = A_w2c[0]

        # Invert A0 stably (they should be rigid transforms; this is fine in practice)
        A0_inv = np.linalg.inv(A0)

        # Right-multiply a fixed transform
        # A'[t] = A[t] @ A0^{-1} @ B
        M = A0_inv @ B
        A_aligned = A_w2c @ M  # broadcast matmul (T,4,4)@(4,4)

        # (Optional) sanity: force exact last row if you expect perfect rigid matrices
        # A_aligned[:, 3, :] = np.array([0, 0, 0, 1], dtype=A_aligned.dtype)

        return A_aligned.copy()

    def flush_cache(self) -> None:
        """Clear all cached frames and per-frame geometry."""
        self._cache = None
        # keep intrinsics & config; Kpix is recomputed on next register if needed
        # (leave _Kpix/_Kpix_inv/_stride as-is; they'll be overwritten on next build)

    def register_source_sequence(
        self,
        device,
        frames: Sequence[np.ndarray],
        *,
        extrinsics=None,
        intrinsics=None,
        depths: Optional[np.ndarray] = None,
        start_index: Optional[int] = None,
        overwrite_stride: Optional[int] = None,
        force_using_input_extrinics: bool = False,
        cat_to_start=False,
        cache_frames: bool = False,
    ) -> None:
        """
        Append source frames into cache (incremental). On first call, build a fresh cache.

        Args:
            device: torch device spec ("cuda:0", torch.device, etc.).
            frames: sequence of RGB frames (H,W,[3]) as numpy arrays.
            extrinsics: array with shape (T,4,4); interpreted as c2w if is_c2w=True.
                Required when ``force_using_input_extrinics=True``; otherwise
                ignored on the DA3 branch (poses are recovered from DA3) and
                used directly on the VDA branch.
            intrinsics: optional per-frame intrinsics with shape (T,3,3).
                Only consumed when ``force_using_input_extrinics=True``;
                in that case overrides DA3's predicted intrinsics so each
                frame's K matches what generated the RGB (e.g. the
                trajectory used to drive the model). ``None`` falls back
                to the legacy behaviour (DA3 intrinsics for the DA3 path,
                handler-level ``self._Kpix`` for the VDA path).
            depths: optional per-frame depth maps (T,Hd,Wd). If None, will infer.
            start_index: global index of the FIRST frame in `frames`. If omitted:
                         - first call: default 1
                         - subsequent calls: must be contiguous (auto = last+1)
            force_using_input_extrinics: when True, skip DA3 trajectory
                alignment and write the caller-supplied ``extrinsics``
                (and ``intrinsics`` if provided) verbatim into the handler
                state. Use this in eval to replay the exact poses that
                drove the previous generation step instead of letting DA3
                re-infer a drifted trajectory from rendered RGB.
            cache_frames: when True, retain RGB frames in ``self.frames`` for
                later v40 photo refinement. Defaults off to preserve memory.
        """
        frames = list(frames)
        if not frames:
            raise ValueError("Frames sequence must not be empty.")

        if force_using_input_extrinics and extrinsics is None:
            raise ValueError(
                "register_source_sequence: force_using_input_extrinics=True requires explicit `extrinsics`; got None."
            )
        if extrinsics is not None:
            extrinsics = np.asarray(extrinsics, dtype=float)
            if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
                raise ValueError(f"extrinsics must have shape (T,4,4), got {extrinsics.shape}.")
            if extrinsics.shape[0] != len(frames):
                raise ValueError(f"extrinsics length {extrinsics.shape[0]} does not match frames length {len(frames)}.")
        if intrinsics is not None:
            if not force_using_input_extrinics:
                # Silently ignoring the arg would hide caller misconfiguration
                # since the legacy paths consume DA3-predicted intrinsics
                # regardless. Fail loud instead.
                raise ValueError(
                    "register_source_sequence: `intrinsics` is only honoured when `force_using_input_extrinics=True`."
                )
            intrinsics = np.asarray(intrinsics, dtype=float)
            if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
                raise ValueError(f"intrinsics must have shape (T,3,3), got {intrinsics.shape}.")
            if intrinsics.shape[0] != len(frames):
                raise ValueError(f"intrinsics length {intrinsics.shape[0]} does not match frames length {len(frames)}.")
        H_img, W_img = frames[0].shape[:2]
        if self._image_size is None:
            self._image_size = (int(H_img), int(W_img))
        else:
            if tuple(self._image_size) != (H_img, W_img):
                raise ValueError(
                    f"Registered image size {H_img, W_img} does not match handler image size {self._image_size}."
                )

        # Build intrinsics (once)
        if self._Kpix is None:
            self._Kpix = ensure_pixel_intrinsics(self._intrinsic, height=self._image_size[0], width=self._image_size[1])
            self._Kpix_inv = np.linalg.inv(self._Kpix)

        # ---- Determine the global starting index for this batch ----
        if self._cache is None:
            # First batch: default to 1 if not provided
            batch_first_index = 1 if start_index is None else int(start_index)
        else:
            expected_next = self._cache.first_index + self._cache.num_frames
            batch_first_index = int(expected_next) if start_index is None else int(start_index)
            if batch_first_index != expected_next:
                raise ValueError(
                    f"Non-contiguous append. Expected start_index={expected_next}, got {batch_first_index}."
                )

        # ---- Prepare depths (for NEW frames only) ----
        if depths is None:
            # 检测模型类型: DepthAnything3 或 VideoDepthAnything

            self._depth_estimator = self._depth_estimator.to(device)

            # Route through this branch for DA3 itself OR any DA3-compatible
            # estimator that exposes the same ``.inference()`` contract and sets
            # ``da3_compatible = True`` (e.g. the VGGT-Omega metric estimator).
            # This keeps ``frustum/`` free of any mosaic-side imports.
            if (HAS_DA3 and isinstance(self._depth_estimator, DepthAnything3)) or getattr(
                self._depth_estimator, "da3_compatible", False
            ):
                # 使用 Depth Anything V3 / DA3 兼容估计器推理
                model_name = getattr(self._depth_estimator, "model_name", "unknown")
                print(f"注册帧深度/位姿推理(estimator={model_name})")

                # 将 numpy 数组 frames 转换为 PIL Images (DA3 需要)
                from PIL import Image

                pil_frames = []
                for frame in frames:
                    if isinstance(frame, np.ndarray):
                        # 确保是 uint8 格式
                        if frame.dtype != np.uint8:
                            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                        pil_frames.append(Image.fromarray(frame))
                    else:
                        pil_frames.append(frame)

                # DA3 推理
                with torch.no_grad():
                    prediction = self._depth_estimator.inference(
                        pil_frames,
                        use_ray_pose=True,
                        process_res=700,
                    )
                depths = prediction.depth  # [N, H, W] float32 array
                if not force_using_input_extrinics:
                    extrinsics = prediction.extrinsics  # [N, 4, 4] array
                    if not self._fixed_intrinsics:
                        intrinsics = prediction.intrinsics  # [N, 3, 3] array

                    base_extrinsics = np.eye(4)[None, ...].repeat(extrinsics.shape[0], 0)
                    base_extrinsics[:, :3, :] = extrinsics
                    extrinsics = base_extrinsics
                    # is_first_section = self.last_extrinsic is None
                    extrinsics = self.align_w2c_trajectory(
                        extrinsics,
                        self.last_extrinsic,
                    )
                    self.last_extrinsic = extrinsics[-1:].copy()
                    raw_extrinsics = extrinsics.copy()
                    self.extrinsics.extend(list(raw_extrinsics)[:])
                    if self._fixed_intrinsics:
                        for _ in range(len(extrinsics)):
                            self._Kpixs.append(self._Kpix.copy())
                    else:
                        for intr in intrinsics[:]:
                            self._Kpixs.append(
                                ensure_pixel_intrinsics(
                                    intr,
                                    height=self._image_size[0],
                                    width=self._image_size[1],
                                )
                            )
                    # assert len(self._Kpixs) == len(self.extrinsics)
                else:
                    if cat_to_start:
                        self.extrinsics = list(extrinsics) + self.extrinsics
                        if self._fixed_intrinsics:
                            prepend_intrinsics = [self._Kpix.copy() for _ in range(len(extrinsics))]
                        elif intrinsics is not None:
                            prepend_intrinsics = [
                                ensure_pixel_intrinsics(
                                    intr,
                                    height=self._image_size[0],
                                    width=self._image_size[1],
                                )
                                for intr in intrinsics
                            ]
                        else:
                            prepend_intrinsics = [
                                ensure_pixel_intrinsics(
                                    intr,
                                    height=self._image_size[0],
                                    width=self._image_size[1],
                                )
                                for intr in prediction.intrinsics
                            ]
                        self._Kpixs = prepend_intrinsics + self._Kpixs
                    else:
                        self.extrinsics.extend(list(extrinsics))
                        self.last_extrinsic = extrinsics[-1:].copy()
                        # Honour caller-supplied per-frame intrinsics when
                        # provided; otherwise fall back to DA3's predicted
                        # intrinsics. Both go through ensure_pixel_intrinsics
                        # to normalise the convention before being stored.
                        if self._fixed_intrinsics:
                            input_intrinsics_iter = [self._Kpix.copy() for _ in range(len(extrinsics))]
                        elif intrinsics is not None:
                            input_intrinsics_iter = intrinsics
                        else:
                            input_intrinsics_iter = prediction.intrinsics
                        for intr in input_intrinsics_iter[:]:
                            normalized_intr = ensure_pixel_intrinsics(
                                intr,
                                height=self._image_size[0],
                                width=self._image_size[1],
                            )
                            self._Kpixs.append(normalized_intr)
                            self._Kpix = normalized_intr.copy()
            else:
                # 使用原有的 VideoDepthAnything 推理
                print("使用 VideoDepthAnything 推理深度图")
                depths, _ = self._depth_estimator.infer_video_depth(
                    frames,
                    target_fps=self._depth_infer_fps,
                    input_size=W_img,
                    device=device,
                    fp32=self._depth_fp32,
                )
                # VDA cannot predict camera poses; the caller-supplied
                # ``extrinsics`` are the sole source regardless of the
                # ``force_using_input_extrinics`` flag.
                self.extrinsics.extend(list(extrinsics))
                if self._fixed_intrinsics:
                    for _ in range(len(extrinsics)):
                        self._Kpixs.append(self._Kpix.copy())
                elif intrinsics is not None:
                    # Mirror the DA3 force-branch behaviour: when the caller
                    # explicitly supplies per-frame intrinsics, register them
                    # in lockstep with the new frames so downstream IoU /
                    # reproject can pick the correct K per memory frame.
                    for intr in intrinsics:
                        normalized_intr = ensure_pixel_intrinsics(
                            intr,
                            height=self._image_size[0],
                            width=self._image_size[1],
                        )
                        self._Kpixs.append(normalized_intr)
                        self._Kpix = normalized_intr.copy()
        self._depth_estimator = self._depth_estimator.to("cpu")
        depths = np.asarray(depths, dtype=np.float32)
        if depths.shape[0] != len(frames):
            raise ValueError("Depth estimator must return one depth per frame.")
        if depths.ndim != 3:
            raise ValueError("Depth array must have shape (T,H,W).")

        # ---- Determine stride (first build) ----
        Hc, Wc = self._grid_size
        if Hc <= 0 or Wc <= 0:
            raise ValueError("grid_size must contain positive integers.")

        if self._stride is None:
            self._stride = (
                self._point_stride
                if self._point_stride is not None
                else max(depths.shape[1] // Hc, depths.shape[2] // Wc, 1) // 2
            )
        if overwrite_stride is not None:
            self._stride = int(max(1, overwrite_stride))

        if cat_to_start:
            self.depths = list(depths) + self.depths
        else:
            self.depths.extend(list(depths))
        if cache_frames:
            cached = [np.asarray(frame).copy() for frame in frames]
            if cat_to_start:
                self.frames = cached + self.frames
            else:
                self.frames.extend(cached)
        return depths

    def _latent_frame_to_frame_span(
        self,
        latent_frame_idx: int,
        total_frames: int,
        frames_per_latent: int = 4,
    ) -> Tuple[int, int]:
        """Map a latent-frame index to the covered raw-frame half-open interval [start, end)."""
        latent_frame_idx = int(latent_frame_idx)
        total_frames = int(total_frames)
        frames_per_latent = max(1, int(frames_per_latent))
        if total_frames <= 0:
            return 0, 0
        if frames_per_latent == 1:
            start = int(np.clip(latent_frame_idx, 0, total_frames - 1))
            end = min(start + 1, total_frames)
            return start, end

        if latent_frame_idx <= 0:
            return 0, min(1, total_frames)
        start = 1 + frames_per_latent * (latent_frame_idx - 1)
        end = min(start + frames_per_latent, total_frames)
        if start >= total_frames:
            start = total_frames - 1
            end = total_frames
        return start, max(start + 1, end)

    def _frame_id_to_latent_id(
        self,
        frame_id: int,
        total_latents: int,
        *,
        frames_per_latent: int = 1,
    ) -> int:
        """将原始 frame 索引映射为 latent 索引，支持 1:1 与 4:1 压缩模式。"""
        frame_id = int(frame_id)
        total_latents = int(total_latents)
        if total_latents <= 0:
            return 0

        frames_per_latent = max(1, int(frames_per_latent))
        if frames_per_latent == 1:
            latent_id = frame_id
        else:
            # 与 _latent_frame_to_frame_span 的分组规则保持一致：
            # latent0 覆盖 frame0，随后每个 latent 覆盖 frames_per_latent 帧。
            latent_id = 0 if frame_id <= 0 else (1 + (frame_id - 1) // frames_per_latent)
        return int(np.clip(latent_id, 0, total_latents - 1))

    @staticmethod
    def _coerce_source_valid_masks(
        source_valid_masks: Optional[torch.Tensor | np.ndarray],
        *,
        total_latents: int,
        H_lat: int,
        W_lat: int,
    ) -> Optional[torch.Tensor]:
        """Validate optional source-latent validity masks.

        ``True`` means the source latent pixel may participate in frustum
        splatting. ``False`` excludes the source pixel before projection/fuse,
        which is different from writing a zero latent into the mosaic canvas.
        """
        if source_valid_masks is None:
            return None
        if isinstance(source_valid_masks, torch.Tensor):
            masks = source_valid_masks.detach().to(device="cpu", dtype=torch.bool)
        else:
            masks = torch.as_tensor(np.asarray(source_valid_masks), dtype=torch.bool)
        expected = (int(total_latents), int(H_lat), int(W_lat))
        if tuple(masks.shape) != expected:
            raise ValueError(f"source_valid_masks must have shape {expected}, got {tuple(masks.shape)}.")
        return masks.contiguous()

    @staticmethod
    def _coerce_memory_K(
        *,
        K: Optional[np.ndarray],
        memory_K: Optional[np.ndarray],
        M: int,
    ) -> Optional[np.ndarray]:
        """Resolve the public ``select_candidates`` / ``fuse_candidates``
        ``K`` + ``memory_K`` kwargs into the single internal ``K_override``
        array.

        * ``K=(3,3)``                 → legacy single-K override (returned as-is).
        * ``memory_K=(3,3)``          → broadcast: same as legacy.
        * ``memory_K=(M,3,3)``        → per-memory-frame override; ``K``
                                        must NOT also be provided.
        * Both None                   → returns None (handler falls back to
                                        ``self._Kpixs`` / ``self._Kpix``).
        """
        if memory_K is None:
            if K is None:
                return None
            arr = np.asarray(K, dtype=np.float32)
            if arr.shape != (3, 3):
                raise ValueError(f"K must have shape (3,3), got {arr.shape}")
            return arr
        if K is not None:
            raise ValueError("Pass either ``K`` (legacy single (3,3) override) or ``memory_K`` (per-frame), not both.")
        arr = np.asarray(memory_K, dtype=np.float32)
        if arr.shape == (3, 3):
            return arr
        if arr.ndim == 3 and arr.shape[1:] == (3, 3):
            if arr.shape[0] != M:
                raise ValueError(f"memory_K shape {arr.shape} must match w2c length M={M}.")
            return arr
        raise ValueError(f"memory_K must have shape (3,3) or (M,3,3), got {arr.shape}.")

    def _get_frame_intrinsic(
        self,
        frame_id: int,
        Ks=None,
        K_override: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Resolve the per-frame intrinsic. Two K_override shapes supported:
        #   * (3, 3)   -- legacy: every frame shares one K.
        #   * (M, 3, 3) -- training-time: per-memory-frame K, indexed by
        #                  frame_id. Out-of-range frame_ids are clamped to
        #                  the last frame for backward compatibility, but a
        #                  warning is emitted because it almost always means
        #                  an upstream wiring bug (e.g. a frame_id from
        #                  another sequence leaked into the candidate list).
        if K_override is not None:
            K = np.asarray(K_override, dtype=np.float32)
            if K.ndim == 2 and K.shape == (3, 3):
                return K
            if K.ndim == 3 and K.shape[1:] == (3, 3):
                fid_in = int(frame_id)
                M = K.shape[0]
                if fid_in < 0 or fid_in >= M:
                    warnings.warn(
                        f"_get_frame_intrinsic: frame_id={fid_in} is out of "
                        f"range for K_override of shape (M={M}, 3, 3). "
                        f"Clamping to [0, M-1]; check the caller for an "
                        f"index/mapping bug.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                fid = int(np.clip(fid_in, 0, M - 1))
                return K[fid]
            raise ValueError(f"K_override must have shape (3,3) or (M,3,3), got {K.shape}")
        if Ks is not None and len(Ks) > int(frame_id):
            K = np.asarray(Ks[int(frame_id)], dtype=np.float32)
            if K.shape == (3, 3):
                return K
        return np.asarray(self._Kpix, dtype=np.float32)

    def _build_group_representative_geometry(
        self,
        latent_frame_idx: int,
        *,
        depths: np.ndarray,
        w2c: np.ndarray,
        geometry_rep_mode: str = "single_frame",
        frames_per_latent: int = 4,
    ) -> Dict[str, np.ndarray]:
        """Resolve a latent frame into representative geometry.

        Returns a dict with:
            - latent_frame_idx
            - frame_ids: raw frame ids covered by this latent frame
            - frame_id: representative raw frame id
            - depth: representative depth map
            - w2c: representative w2c
        """
        total_frames = int(w2c.shape[0])
        start, end = self._latent_frame_to_frame_span(
            latent_frame_idx,
            total_frames,
            frames_per_latent=frames_per_latent,
        )
        frame_ids = np.arange(start, end, dtype=np.int64)
        if frame_ids.size == 0:
            frame_ids = np.array([0], dtype=np.int64)

        mode = str(geometry_rep_mode).lower().strip()
        if mode == "second_frame":
            rep_frame_id = int(np.clip(start + 1 if frame_ids.size >= 2 else start, 0, total_frames - 1))
            rep_depth = np.asarray(depths[rep_frame_id], dtype=np.float32)
            rep_w2c = np.asarray(w2c[rep_frame_id], dtype=np.float32)
        elif mode == "medoid_depth_mean":
            cand_w2c = np.asarray(w2c[frame_ids], dtype=np.float64)
            centers = np.stack([-m[:3, :3].T @ m[:3, 3] for m in cand_w2c], axis=0)
            dirs = cand_w2c[:, 2, :3]
            dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
            cos_mat = np.clip(dirs @ dirs.T, -1.0, 1.0)
            angle_mat = np.degrees(np.arccos(cos_mat))
            dist_mat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
            score = angle_mat + dist_mat
            medoid_local = int(np.argmin(score.sum(axis=1)))
            rep_frame_id = int(frame_ids[medoid_local])
            rep_w2c = np.asarray(w2c[rep_frame_id], dtype=np.float32)
            depth_stack = np.asarray(depths[frame_ids], dtype=np.float32)
            valid = np.isfinite(depth_stack) & (depth_stack > 1e-6)
            depth_sum = np.where(valid, depth_stack, 0.0).sum(axis=0)
            depth_cnt = valid.sum(axis=0)
            rep_depth = np.where(depth_cnt > 0, depth_sum / np.maximum(depth_cnt, 1), 0.0).astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported geometry_rep_mode={geometry_rep_mode}. Use 'second_frame' or 'medoid_depth_mean'."
            )

        return {
            "latent_frame_idx": int(latent_frame_idx),
            "frame_ids": frame_ids,
            "frame_id": int(rep_frame_id),
            "depth": rep_depth,
            "w2c": rep_w2c,
        }

    def _downsample_depth_to_latent(
        self,
        depth_map: np.ndarray,
        W_lat: int,
        H_lat: int,
        reduce_mode: str = "nearest",
        valid_eps: float = 1e-3,
    ) -> np.ndarray:
        """Downsample a per-pixel depth map to (H_lat, W_lat).

        Modes:
            - ``nearest``      : cv2 INTER_NEAREST. Fast, no edge averaging.
                                 RECOMMENDED for forward-warp fuse paths --
                                 averaging depth across object boundaries
                                 produces "flying" depth values that splat
                                 to wildly wrong query pixels.
            - ``block_median`` : per-block median over valid (>valid_eps,
                                 isfinite) values; -1 sentinel if empty.
                                 Used by IoU rerank where a stable median
                                 is more robust than a single sample.
            - ``block_min``    : per-block minimum (foreground-preferring),
                                 also robust against edge averaging.
            - ``cv2_area``     : true ``cv2.INTER_AREA`` downsampling. NOTE:
                                 historically this branch was wrongly using
                                 ``INTER_LINEAR``; that bug is fixed here.
                                 Avoid for depth on edges -- prefer
                                 ``nearest`` / ``block_min``.
        """
        depth_map = np.asarray(depth_map, dtype=np.float32)
        if reduce_mode == "nearest":
            return cv2.resize(
                depth_map,
                (W_lat, H_lat),
                interpolation=cv2.INTER_NEAREST,
            )
        if reduce_mode == "cv2_area":
            return cv2.resize(
                depth_map,
                (W_lat, H_lat),
                interpolation=cv2.INTER_AREA,
            )

        h_src, w_src = depth_map.shape[:2]
        h_edges = np.linspace(0, h_src, H_lat + 1, dtype=int)
        w_edges = np.linspace(0, w_src, W_lat + 1, dtype=int)
        out = np.full((H_lat, W_lat), -1.0, dtype=np.float32)
        if reduce_mode == "block_min":
            reducer = np.min
        elif reduce_mode == "block_median":
            reducer = np.median
        else:
            raise ValueError(
                f"Unsupported reduce_mode={reduce_mode}. Use 'nearest', 'block_median', 'block_min' or 'cv2_area'."
            )
        for ih in range(H_lat):
            hs, he = h_edges[ih], h_edges[ih + 1]
            for iw in range(W_lat):
                ws, we = w_edges[iw], w_edges[iw + 1]
                block = depth_map[hs:he, ws:we]
                valid_vals = block[np.isfinite(block) & (block > valid_eps)]
                if valid_vals.size > 0:
                    out[ih, iw] = np.float32(reducer(valid_vals))
        return out

    @staticmethod
    def _depth_downsample_pool_key(
        *,
        depths: np.ndarray,
        candidate_latent_frame_idx: int,
        geometry_rep_mode: str,
        frames_per_latent: int,
        H_lat: int,
        W_lat: int,
        depth_reduce_mode: str,
    ) -> Tuple:
        """Key scoped to a specific depth array and representative geometry."""
        return (
            id(depths),
            int(candidate_latent_frame_idx),
            str(geometry_rep_mode),
            int(frames_per_latent),
            int(H_lat),
            int(W_lat),
            str(depth_reduce_mode),
        )

    @staticmethod
    def _clipped_proj_bbox(
        projected_hw: np.ndarray,
        canvas_hw: Tuple[int, int],
        *,
        outlier_clip_factor: float = 1.0,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Compute the canvas-clipped (h0, h1, w0, w1) bounding box of a
        projected point cloud, using exactly the same sentinel filter and
        outlier-clip semantics as ``_compute_canvas_iou_score``'s bbox
        branch. Returns ``None`` if no finite point survives the filter.

        Reused for both IoU rerank scoring (legacy path) and inter-candidate
        bbox-NMS exclusion (new path) so the two definitions can never drift.
        """
        H, W = int(canvas_hw[0]), int(canvas_hw[1])
        coords = np.asarray(projected_hw, dtype=np.float32).reshape(-1, 2)
        sentinel_threshold = -float(max(H, W)) * 5.0
        finite = (
            np.isfinite(coords).all(axis=1) & (coords[:, 0] > sentinel_threshold) & (coords[:, 1] > sentinel_threshold)
        )
        coords = coords[finite]
        if coords.shape[0] == 0:
            return None
        if outlier_clip_factor > 0:
            max_dim = float(max(H, W))
            lo_h = -outlier_clip_factor * max_dim
            lo_w = -outlier_clip_factor * max_dim
            hi_h = (1.0 + outlier_clip_factor) * float(H)
            hi_w = (1.0 + outlier_clip_factor) * float(W)
            coords = np.stack(
                [
                    np.clip(coords[:, 0], lo_h, hi_h),
                    np.clip(coords[:, 1], lo_w, hi_w),
                ],
                axis=1,
            )
        h0 = float(coords[:, 0].min())
        h1 = float(coords[:, 0].max() + 1.0)
        w0 = float(coords[:, 1].min())
        w1 = float(coords[:, 1].max() + 1.0)
        return (h0, h1, w0, w1)

    @staticmethod
    def _bbox_iou_on_canvas(
        a: Optional[Tuple[float, float, float, float]],
        b: Optional[Tuple[float, float, float, float]],
    ) -> float:
        """Axis-aligned bbox IoU between two candidate projection bboxes.

        Inputs are ``(h0, h1, w0, w1)`` tuples (as produced by
        ``_clipped_proj_bbox``) or ``None`` for an invalid projection.
        ``None`` on either side yields ``0.0`` so an invalid projection
        cannot suppress its neighbours via NMS.
        """
        if a is None or b is None:
            return 0.0
        ah0, ah1, aw0, aw1 = a
        bh0, bh1, bw0, bw1 = b
        a_area = max(ah1 - ah0, 0.0) * max(aw1 - aw0, 0.0)
        b_area = max(bh1 - bh0, 0.0) * max(bw1 - bw0, 0.0)
        inter_h = max(min(ah1, bh1) - max(ah0, bh0), 0.0)
        inter_w = max(min(aw1, bw1) - max(aw0, bw0), 0.0)
        inter = inter_h * inter_w
        union = a_area + b_area - inter
        if union <= 1e-6:
            return 0.0
        return float(inter / union)

    @staticmethod
    def _pick_candidates_with_nms(
        scored: List[Dict],
        k: int,
        mode: Optional[str],
        thresholds: Dict[str, float],
    ) -> List[Dict]:
        """Greedy NMS-style candidate picker.

        Inputs:
            scored: list of dicts pre-sorted by the legacy ranking key
                ``(-iou_score, angle_deg, dist, coarse_rank, gid)``.
                Each dict must contain at least ``frame_id``, ``proj_bbox``,
                ``rep_translation`` (the latter two may be missing for
                non-projection_iou paths and are only consulted for the
                corresponding NMS mode).
            k: target number of candidates to return.
            mode: one of ``None`` / ``"projection_iou"`` / ``"temporal"`` /
                ``"pose"``. ``None`` short-circuits to ``scored[:k]``.
            thresholds: dict with keys
                ``projection_iou_threshold`` (float, in [0,1]),
                ``min_temporal_gap`` (int, >=0),
                ``pose_distance_threshold`` (float, >=0).

        Behaviour:
            1. Seed with the top-scoring candidate.
            2. Repeatedly take the next candidate that is NOT proximate to
               any already-picked one (per ``mode``).
            3. Fallback: when no non-proximate candidate is left in the
               pool, fill the remaining slots with the highest-scoring
               unpicked candidates regardless of proximity, so the picker
               always returns up to ``min(k, len(scored))`` items.

        ``rep_translation`` is consumed as a 3-element 1D array. ``None``
        (or a bbox of ``None`` for the projection_iou mode) is treated as
        "definitely not proximate" so an invalid projection never blocks
        downstream selection.
        """
        if k <= 0 or not scored:
            return []
        if mode is None:
            return list(scored[:k])

        proj_thr = float(thresholds.get("projection_iou_threshold", 0.7))
        temp_gap = int(thresholds.get("min_temporal_gap", 0))
        pose_thr = float(thresholds.get("pose_distance_threshold", 0.0))

        def is_proximate(cand: Dict, picked_one: Dict) -> bool:
            if mode == "projection_iou":
                iou = FrustumHandler._bbox_iou_on_canvas(cand.get("proj_bbox"), picked_one.get("proj_bbox"))
                return iou >= proj_thr
            if mode == "temporal":
                if temp_gap <= 0:
                    return False
                return abs(int(cand["frame_id"]) - int(picked_one["frame_id"])) < temp_gap
            if mode == "pose":
                if pose_thr <= 0.0:
                    return False
                t_a = cand.get("rep_translation")
                t_b = picked_one.get("rep_translation")
                if t_a is None or t_b is None:
                    return False
                return float(np.linalg.norm(np.asarray(t_a) - np.asarray(t_b))) < pose_thr
            raise ValueError(
                f"Unsupported candidate_nms_mode={mode!r}. Use 'projection_iou', 'temporal', 'pose' or None."
            )

        picked: List[Dict] = []
        used = [False] * len(scored)
        picked.append(scored[0])
        used[0] = True

        while len(picked) < k:
            next_idx = None
            for i, cand in enumerate(scored):
                if used[i]:
                    continue
                if any(is_proximate(cand, p) for p in picked):
                    continue
                next_idx = i
                break
            if next_idx is None:
                break
            picked.append(scored[next_idx])
            used[next_idx] = True

        if len(picked) < k:
            # NMS-constrained pool exhausted: fall back to highest-scoring
            # unpicked candidates so we still return up to ``k`` items.
            for i, cand in enumerate(scored):
                if used[i]:
                    continue
                picked.append(cand)
                used[i] = True
                if len(picked) >= k:
                    break

        return picked

    def _select_candidates_coverage_vectorized(
        self,
        *,
        query_extrinsics: np.ndarray,
        depths: np.ndarray,
        w2c: np.ndarray,
        memory_K: np.ndarray,
        query_K: np.ndarray,
        H_lat: int,
        W_lat: int,
        candidates_per_query_group: int,
        query_reference_frame: int = 2,
        coverage_grid_downsample: int = 4,
        coverage_pool_stride: int = 2,
        depth_downsample_pool: Optional[Dict] = None,
    ) -> List[List[int]]:
        """Fast UNION-coverage candidate selection (the ``coverage`` picker).

        A fully VECTORISED, FULL-POOL greedy-coverage selector that avoids both
        slow parts of the per-candidate ``projection_iou`` path:

          * No per-(query_group, candidate) Python loop / reproject call -- ALL
            pool frames are forward-warped into ALL query reference cameras in a
            handful of batched numpy ops.
          * No angle pre-filter / coarse_k cap -- the greedy picks from the whole
            memory pool (optionally strided), so it can grab a frame that covers
            the query periphery from a very different viewpoint (exactly what
            pose_nearest / individual-IoU pickers miss).

        Geometry is identical to ``reproject_pixels_nd_to_nd`` (same lift +
        project math, ``nearest`` depth), so the coverage masks reflect what the
        full-resolution fuse will actually fill. The masks are rasterised on a
        coarse ``(H_lat//d, W_lat//d)`` grid (``d = coverage_grid_downsample``)
        -- selection-only; the fuse still runs at full latent resolution.

        Returns ``candidate_frame_ids: List[List[int]]`` (one inner list of
        ``candidates_per_query_group`` frame ids per query group).
        """
        stride = int(self._latent_stride)
        d = max(1, int(coverage_grid_downsample))
        ps = max(1, int(coverage_pool_stride))
        cpqg = max(1, int(candidates_per_query_group))
        H_lat, W_lat = int(H_lat), int(W_lat)
        Hd, Wd = H_lat // d, W_lat // d
        query_extrinsics = np.asarray(query_extrinsics, dtype=np.float64)
        if query_extrinsics.ndim == 2:
            query_extrinsics = query_extrinsics[None]
        Q = int(query_extrinsics.shape[0])
        n_groups = Q // 4 if Q % 4 == 0 else Q
        G = int(w2c.shape[0])
        if G == 0 or n_groups == 0 or Hd == 0 or Wd == 0:
            return [[-1] * cpqg for _ in range(max(n_groups, 0))]

        pool = np.arange(0, G, ps)
        M = len(pool)
        w2c = np.asarray(w2c, dtype=np.float64)
        memory_K = np.asarray(memory_K, dtype=np.float64)
        if memory_K.shape == (3, 3):
            memory_K = np.broadcast_to(memory_K, (G, 3, 3))
        query_K = np.asarray(query_K, dtype=np.float64)
        if query_K.shape == (3, 3):
            query_K = np.broadcast_to(query_K, (Q, 3, 3))

        # ---- per-pool-frame coarse depth (cached per (frame, d)) ----
        dgrid = np.empty((M, Hd, Wd), dtype=np.float64)
        for k, fid in enumerate(pool):
            fid = int(fid)
            cache_key = ("covdepth", fid, Hd, Wd) if depth_downsample_pool is not None else None
            dl = depth_downsample_pool.get(cache_key) if cache_key is not None else None
            if dl is None:
                dl = self._downsample_depth_to_latent(
                    np.asarray(depths[fid], dtype=np.float32),
                    W_lat=Wd,
                    H_lat=Hd,
                    reduce_mode="nearest",
                )
                if cache_key is not None:
                    depth_downsample_pool[cache_key] = dl
            dgrid[k] = dl

        # ---- lift each pool frame's coarse grid to world (vectorised) ----
        gi, gj = np.meshgrid(np.arange(Hd), np.arange(Wd), indexing="ij")
        u = (gj.reshape(-1) * stride * d).astype(np.float64)
        v = (gi.reshape(-1) * stride * d).astype(np.float64)
        pix = np.stack([u, v, np.ones_like(u)], axis=0)  # (3, P)
        P = pix.shape[1]
        dflat = dgrid.reshape(M, P)
        Rm = w2c[pool, :3, :3]
        tm = w2c[pool, :3, 3]
        invKm = np.linalg.inv(memory_K[pool])
        rays = np.einsum("mij,jp->mip", invKm, pix)  # (M,3,P)
        Xc = rays * dflat[:, None, :] - tm[:, :, None]  # (M,3,P)
        Xw = np.einsum("mji,mjp->mip", Rm, Xc)  # R^T (Xc - t): (M,3,P)

        # ---- project to every query group reference camera (vectorised) ----
        qoff = self._query_reference_offset(query_reference_frame)
        qidx = [g * 4 + qoff if Q % 4 == 0 else g for g in range(n_groups)]
        Rq = query_extrinsics[qidx, :3, :3]
        tq = query_extrinsics[qidx, :3, 3]
        Kq = query_K[qidx]
        Xcq = np.einsum("gij,mjp->gmip", Rq, Xw) + tq[:, None, :, None]
        proj = np.einsum("gij,gmjp->gmip", Kq, Xcq)  # (Qg,M,3,P)
        z = proj[:, :, 2, :]
        # Query-side z threshold MUST match the fuse path: reproject_pixels_nd_to_nd
        # (used by fuse) marks points with zc_new <= 1e-2 as invalid (-10000
        # sentinel), so they never rasterise. Using a looser 1e-3 here would let
        # the selector count cells that fuse will actually drop -> overestimated
        # coverage and worse picks. (dflat is the *memory* depth; 1e-3 is just a
        # degenerate-depth floor, fuse has no separate memory-depth threshold.)
        valid = np.isfinite(z) & (z > 1e-2) & np.isfinite(dflat)[None] & (dflat[None] > 1e-3)
        denom = np.where(valid, z, 1.0)
        # Match the full-resolution fuse path. fuse computes the latent cell as
        # ``np.rint(proj_hw / stride)`` before z-buffer scatter (see
        # ``_splat_one``), so the coverage mask must mark the SAME cell. We
        # compute that full-res rint cell here, then floor-divide by the coarse
        # factor ``d`` -> the coarse mask cell is exactly the coarse cell that
        # CONTAINS the full-res cell fusion will fill. (Flooring the coarse-grid
        # float coord directly, as before, drifts by up to one coarse cell for
        # sub-cell projections >.5, which can make two candidates look
        # complementary when fusion actually fills overlapping cells.)
        fi = np.rint(proj[:, :, 1, :] / denom / stride).astype(np.int64)
        fj = np.rint(proj[:, :, 0, :] / denom / stride).astype(np.int64)
        ci = fi // d
        cj = fj // d
        inb = valid & (ci >= 0) & (ci < Hd) & (cj >= 0) & (cj < Wd)
        masks = np.zeros((n_groups, M, Hd, Wd), dtype=bool)
        gg, mm, pp = np.where(inb)
        masks[gg, mm, ci[gg, mm, pp], cj[gg, mm, pp]] = True

        # ---- per-group greedy union-coverage pick over the full pool ----
        out: List[List[int]] = []
        flat = masks.reshape(n_groups, M, Hd * Wd)
        area = flat.sum(axis=2)  # (Qg, M)
        for g in range(n_groups):
            mg = flat[g]
            union = np.zeros(Hd * Wd, dtype=bool)
            avail = np.ones(M, dtype=bool)
            picked: List[int] = []
            for _ in range(cpqg):
                gain = (mg & ~union).sum(axis=1)
                gain[~avail] = -1
                bi = int(gain.argmax())
                if gain[bi] <= 0:
                    break
                union |= mg[bi]
                picked.append(int(pool[bi]))
                avail[bi] = False
            if len(picked) < cpqg:
                # No remaining frame adds coverage: fill with largest-area
                # unpicked frames so the per-group budget stays honoured.
                ar = area[g].copy()
                ar[~avail] = -1
                for bi in np.argsort(-ar):
                    if avail[bi] and ar[bi] > 0:
                        picked.append(int(pool[bi]))
                        avail[bi] = False
                    if len(picked) >= cpqg:
                        break
            if not picked:
                picked = [-1]
            while len(picked) < cpqg:
                picked.append(picked[-1])
            out.append(picked)
        return out

    @staticmethod
    def _pick_candidates_by_coverage_greedy(
        scored: List[Dict],
        k: int,
    ) -> List[Dict]:
        """Greedy UNION-coverage candidate picker.

        Unlike ``_pick_candidates_with_nms`` (which seeds with the single
        highest individual-IoU candidate and then merely SUPPRESSES proximate
        ones), this picker directly maximises the union of per-candidate
        query-canvas coverage masks: at each step it adds the candidate that
        contributes the most NEW (currently-uncovered) latent cells. This is
        the production analogue of the offline "oracle" greedy-union selection
        and is the fix for the nearest-pose / individual-IoU pickers, whose
        objective (pose proximity / per-frame IoU) is orthogonal to maximising
        canvas coverage and so picks mutually-overlapping (redundant) frames.

        Each dict in ``scored`` must carry a boolean ``coverage_mask``
        (H_lat, W_lat); candidates without one (e.g. fully-invalid
        projections) contribute zero coverage. ``scored`` is assumed already
        sorted by the legacy key so ties / zero-gain fills are deterministic.

        When the union stops growing before ``k`` picks (no candidate adds any
        new cell), the remaining slots are filled with the highest-scoring
        unpicked candidates so the caller's budget stays honoured (matching
        ``_pick_candidates_with_nms`` fallback semantics).
        """
        if k <= 0 or not scored:
            return []

        masks = []
        for c in scored:
            m = c.get("coverage_mask")
            masks.append(None if m is None else np.asarray(m, dtype=bool))

        # Establish the canvas shape from the first available mask; if none of
        # the candidates projected at all, degrade to plain top-k by score.
        union = None
        for m in masks:
            if m is not None:
                union = np.zeros_like(m)
                break
        if union is None:
            return list(scored[:k])

        picked: List[Dict] = []
        used = [False] * len(scored)
        while len(picked) < k:
            best_i, best_gain = None, 0
            for i, m in enumerate(masks):
                if used[i] or m is None:
                    continue
                gain = int(np.count_nonzero(m & ~union))
                # Strict ``>`` keeps the first (highest-score) candidate on
                # ties, since ``scored`` is pre-sorted.
                if gain > best_gain:
                    best_gain, best_i = gain, i
            if best_i is None:
                break  # no candidate adds new coverage
            union |= masks[best_i]
            picked.append(scored[best_i])
            used[best_i] = True

        if len(picked) < k:
            for i, cand in enumerate(scored):
                if used[i]:
                    continue
                picked.append(cand)
                used[i] = True
                if len(picked) >= k:
                    break

        return picked

    def _compute_canvas_iou_score(
        self,
        projected_hw: np.ndarray,
        canvas_hw: Tuple[int, int],
        *,
        iou_mode: str = "bbox",
        iou_mix_weight: float = 0.5,
        outlier_clip_factor: float = 1.0,
    ) -> float:
        """Score how well a candidate projection matches the full target canvas.

        Args:
            projected_hw: (...,2) target-canvas coordinates in (h,w). May be
                in pixel space or latent space (caller's choice); both work
                because the outlier filter is canvas-relative.
            canvas_hw: (H, W) of the target latent canvas.
            iou_mode: ``"bbox"`` (axis-aligned bounding-box IoU) or
                ``"mask"`` (per-pixel coverage / IoU mix).
            iou_mix_weight: ``mask`` mode only. Final score is
                ``(1-w)*iou + w*coverage``, where
                ``coverage = inter_count / canvas_area``. The pure IoU formula
                punishes wide off-canvas projections via the inflated union,
                but does not directly reward FILLING the canvas; mixing in
                coverage gives candidates that cover more of the canvas a
                higher score. Set to 0 to fall back to pure mask IoU. Default
                0.5 (50/50 mix).
            outlier_clip_factor: clip coords to
                ``[-f * max(H, W), (1+f) * max(H, W)]`` after the finite
                filter, to prevent a single outlier projection (a stale
                sentinel surviving the latent-space scale, or a numerical
                NaN-equivalent) from blowing up the candidate bbox by
                orders of magnitude. Set 0 to disable. Default 1.0
                (allow up to one canvas-width slop on each side).
        """
        H, W = int(canvas_hw[0]), int(canvas_hw[1])
        coords = np.asarray(projected_hw, dtype=np.float32).reshape(-1, 2)

        # The reproject helper marks invalid points with ``-10000`` in PIXEL
        # space. Callers commonly pre-divide by ``self._latent_stride`` (e.g.
        # the IoU rerank), turning the sentinel into ``-625`` which the
        # legacy ``> -9999`` filter was too lax to catch. Use a
        # canvas-relative threshold so the sentinel is rejected regardless
        # of whether the caller passes pixel-space or latent-space coords.
        sentinel_threshold = -float(max(H, W)) * 5.0
        finite = (
            np.isfinite(coords).all(axis=1) & (coords[:, 0] > sentinel_threshold) & (coords[:, 1] > sentinel_threshold)
        )
        coords = coords[finite]
        if coords.shape[0] == 0:
            return 0.0

        if outlier_clip_factor > 0:
            max_dim = float(max(H, W))
            lo_h = -outlier_clip_factor * max_dim
            lo_w = -outlier_clip_factor * max_dim
            hi_h = (1.0 + outlier_clip_factor) * float(H)
            hi_w = (1.0 + outlier_clip_factor) * float(W)
            coords = np.stack(
                [
                    np.clip(coords[:, 0], lo_h, hi_h),
                    np.clip(coords[:, 1], lo_w, hi_w),
                ],
                axis=1,
            )

        mode = str(iou_mode).lower().strip()
        canvas_area = float(max(H * W, 1))

        if mode == "bbox":
            h0 = float(coords[:, 0].min())
            h1 = float(coords[:, 0].max() + 1.0)
            w0 = float(coords[:, 1].min())
            w1 = float(coords[:, 1].max() + 1.0)
            cand_area = max(h1 - h0, 0.0) * max(w1 - w0, 0.0)
            inter_h = max(min(h1, float(H)) - max(h0, 0.0), 0.0)
            inter_w = max(min(w1, float(W)) - max(w0, 0.0), 0.0)
            inter_area = inter_h * inter_w
            union_area = canvas_area + cand_area - inter_area
            return float(inter_area / max(union_area, 1e-6))

        if mode == "mask":
            pix = np.rint(coords).astype(np.int64)
            uniq = np.unique(pix, axis=0)
            inside = (uniq[:, 0] >= 0) & (uniq[:, 0] < H) & (uniq[:, 1] >= 0) & (uniq[:, 1] < W)
            inter_count = int(inside.sum())
            outside_count = int((~inside).sum())
            union_count = int(canvas_area) + outside_count
            iou = inter_count / max(union_count, 1)
            mix_w = float(np.clip(iou_mix_weight, 0.0, 1.0))
            if mix_w <= 0.0:
                return float(iou)
            coverage = inter_count / canvas_area
            return float((1.0 - mix_w) * iou + mix_w * coverage)

        raise ValueError(f"Unsupported iou_mode={iou_mode}. Use 'bbox' or 'mask'.")

    def _project_memory_to_query(
        self,
        *,
        memory_depth: np.ndarray,
        memory_K: np.ndarray,
        memory_w2c: np.ndarray,
        query_K: np.ndarray,
        query_w2c: np.ndarray,
        H_lat: int,
        W_lat: int,
        depth_reduce_mode: str = "nearest",
        depth_downsample_pool: Optional[Dict] = None,
        depth_downsample_key: Optional[Tuple] = None,
        grid_downsample: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward warp the (downsampled) memory latent grid into the query camera.

        old=memory and new=query: depth, K, pose are all consistent on the
        memory side. This is the geometrically correct direction for both
        IoU rerank scoring and the new forward-warp + scatter fuse.

        ``grid_downsample`` (d>=1) projects on a COARSER ``(H_lat//d, W_lat//d)``
        grid (depth downsampled to match, grid step = ``stride*d``). The cost
        of the reproject scales with the point count, so d=2/4 cuts it 4x/16x.
        Used by the ``coverage`` greedy picker, whose frame-SELECTION coverage
        estimate tolerates a coarse grid; the actual fuse still runs at full
        latent resolution, so the produced mosaic is unaffected. d=1 is
        bit-identical to the original full-resolution path.

        Returns:
            proj_hw: (H_g, W_g, 2) float pixel coords in the query camera
                     (PIXEL space; divide by ``stride*d`` for coarse-latent
                     coords, or ``stride`` for full-latent). Invalid points
                     are -10000 sentinel. (H_g, W_g) == (H_lat//d, W_lat//d).
            zc_new : (H_g, W_g) float, z in the query camera.
        """
        stride = int(self._latent_stride)
        d = max(1, int(grid_downsample))
        H_g, W_g = int(H_lat) // d, int(W_lat) // d
        eff_stride = stride * d
        # Keep the per-frame depth cache disjoint across downsample factors.
        pool_key = None
        if depth_downsample_key is not None:
            pool_key = tuple(depth_downsample_key) + (("gd", d) if d != 1 else ())
        if depth_downsample_pool is not None and pool_key is not None:
            depth_lat = depth_downsample_pool.get(pool_key)
            if depth_lat is None:
                depth_lat = self._downsample_depth_to_latent(
                    np.asarray(memory_depth, dtype=np.float32),
                    W_lat=W_g,
                    H_lat=H_g,
                    reduce_mode=depth_reduce_mode,
                )
                depth_downsample_pool[pool_key] = depth_lat
        else:
            depth_lat = self._downsample_depth_to_latent(
                np.asarray(memory_depth, dtype=np.float32),
                W_lat=W_g,
                H_lat=H_g,
                reduce_mode=depth_reduce_mode,
            )
        proj_hw, zc_new, _, _ = reproject_pixels_nd_to_nd(
            make_hw_grid(H_g, W_g) * eff_stride,
            depth_z=depth_lat,
            K_old=np.asarray(memory_K, dtype=np.float32),
            T_old_w2c=np.asarray(memory_w2c, dtype=np.float32),
            K_new=np.asarray(query_K, dtype=np.float32),
            T_new_w2c=np.asarray(query_w2c, dtype=np.float32),
        )
        return proj_hw, np.asarray(zc_new, dtype=np.float32)

    @staticmethod
    def _query_reference_offset(query_reference_frame: int) -> int:
        query_reference_frame = int(query_reference_frame)
        if query_reference_frame < 1 or query_reference_frame > 4:
            raise ValueError(
                "query_reference_frame must be in [1, 4] (1-based within each "
                f"4-frame query group), got {query_reference_frame}."
            )
        return query_reference_frame - 1

    @staticmethod
    def _resolve_query_K_for_group(
        *,
        query_K_one: Optional[np.ndarray],
        K_override: Optional[np.ndarray],
        fallback_K: np.ndarray,
    ) -> np.ndarray:
        """Resolve the (3,3) query-side intrinsic for a single query group.

        Priority:
            1. ``query_K_one`` (shape (3,3))                 -- explicit per-group K.
            2. ``K_override`` (shape (3,3))                  -- legacy single K
                shared by both sides; only valid when no explicit query_K.
            3. ``fallback_K``                                -- handler default.

        ``K_override`` of shape (M,3,3) without a matching ``query_K_one`` is
        rejected: it indicates per-memory-frame K was wired in but the query
        side intrinsic was forgotten, which silently wrong-projects.
        """
        if query_K_one is not None:
            arr = np.asarray(query_K_one, dtype=np.float32)
            if arr.shape != (3, 3):
                raise ValueError(f"query_K_one must have shape (3,3), got {arr.shape}")
            return arr
        if K_override is not None:
            arr = np.asarray(K_override, dtype=np.float32)
            if arr.shape == (3, 3):
                return arr
            raise ValueError(
                "K_override has shape (M,3,3) but query_K_one was not "
                "provided; pass both query_K and memory_K together when "
                "using per-frame intrinsics."
            )
        return np.asarray(fallback_K, dtype=np.float32)

    def _score_group_candidate_by_projection_iou(
        self,
        candidate_latent_frame_idx: int,
        *,
        query_w2c: np.ndarray,
        H_lat: int,
        W_lat: int,
        depths: np.ndarray,
        w2c: np.ndarray,
        Ks=None,
        K_override: Optional[np.ndarray] = None,
        query_K_one: Optional[np.ndarray] = None,
        geometry_rep_mode: str = "second_frame",
        iou_mode: str = "bbox",
        frames_per_latent: int = 4,
        depth_downsample_pool: Optional[Dict] = None,
        return_coverage_mask: bool = False,
        depth_reduce_mode: str = "block_median",
        coverage_grid_downsample: int = 1,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        rep = self._build_group_representative_geometry(
            candidate_latent_frame_idx,
            depths=depths,
            w2c=w2c,
            geometry_rep_mode=geometry_rep_mode,
            frames_per_latent=frames_per_latent,
        )
        H_lat, W_lat = int(H_lat), int(W_lat)
        # K_old (memory side) is per-rep; K_new (query side) is the resolved
        # group-level query K. IoU rerank defaults to ``block_median`` for
        # depth (a robust median is preferable when only ranking); the
        # ``coverage`` greedy picker passes ``nearest`` instead -- it is far
        # cheaper (cv2 vs a per-cell Python reducer) AND matches the actual
        # fuse path (which splats with ``nearest``), so the coverage mask
        # reflects what fuse will really fill.
        memory_K = self._get_frame_intrinsic(int(rep["frame_id"]), Ks=Ks, K_override=K_override)
        query_K = self._resolve_query_K_for_group(
            query_K_one=query_K_one,
            K_override=K_override,
            fallback_K=self._Kpix,
        )
        stride = int(self._latent_stride)
        depth_key = self._depth_downsample_pool_key(
            depths=depths,
            candidate_latent_frame_idx=candidate_latent_frame_idx,
            geometry_rep_mode=geometry_rep_mode,
            frames_per_latent=frames_per_latent,
            H_lat=H_lat,
            W_lat=W_lat,
            depth_reduce_mode=depth_reduce_mode,
        )
        if return_coverage_mask:
            # COVERAGE greedy path: project on a (optionally coarser) grid and
            # rasterise a binary coverage mask. The mask is the only thing the
            # union-greedy picker consumes, so we skip the bbox IoU score and
            # use the cheap covered-cell fraction as the deterministic tiebreak.
            d = max(1, int(coverage_grid_downsample))
            H_g, W_g = int(H_lat) // d, int(W_lat) // d
            proj_hw, _ = self._project_memory_to_query(
                memory_depth=rep["depth"],
                memory_K=memory_K,
                memory_w2c=rep["w2c"],
                query_K=query_K,
                query_w2c=query_w2c,
                H_lat=H_lat,
                W_lat=W_lat,
                depth_reduce_mode=depth_reduce_mode,
                depth_downsample_pool=depth_downsample_pool,
                depth_downsample_key=depth_key,
                grid_downsample=d,
            )
            # proj_hw is in PIXEL space on the (H_g, W_g) coarse grid; the
            # coarse-latent cell size is ``stride*d`` pixels.
            coords = (proj_hw / float(stride * d)).reshape(-1, 2)
            finite = np.isfinite(coords).all(axis=1)
            pix = np.rint(coords[finite]).astype(np.int64)
            mask = np.zeros((H_g, W_g), dtype=bool)
            if pix.shape[0] > 0:
                inside = (pix[:, 0] >= 0) & (pix[:, 0] < H_g) & (pix[:, 1] >= 0) & (pix[:, 1] < W_g)
                pin = pix[inside]
                if pin.shape[0] > 0:
                    mask[pin[:, 0], pin[:, 1]] = True
            rep["coverage_mask"] = mask
            rep["proj_bbox"] = None
            score = float(mask.mean())
            rep["score"] = score
            return score, rep
        # PROJECTION_IOU path (rank by per-candidate bbox/mask IoU): full
        # latent-resolution projection + canvas-clipped bbox for optional NMS.
        proj_hw, _ = self._project_memory_to_query(
            memory_depth=rep["depth"],
            memory_K=memory_K,
            memory_w2c=rep["w2c"],
            query_K=query_K,
            query_w2c=query_w2c,
            H_lat=H_lat,
            W_lat=W_lat,
            depth_reduce_mode=depth_reduce_mode,
            depth_downsample_pool=depth_downsample_pool,
            depth_downsample_key=depth_key,
        )
        proj_hw_lat = proj_hw / float(stride)
        score = self._compute_canvas_iou_score(proj_hw_lat, (H_lat, W_lat), iou_mode=iou_mode)
        rep["score"] = float(score)
        # Stash the canvas-clipped projection bbox for inter-candidate NMS.
        # Computed via the shared helper so the bbox here matches exactly
        # the geometry used by ``_compute_canvas_iou_score`` for the bbox
        # IoU score above. ``None`` when no finite projection point survives.
        rep["proj_bbox"] = self._clipped_proj_bbox(proj_hw_lat, (H_lat, W_lat))
        return float(score), rep

    def _select_group_ids_by_projection_iou(
        self,
        *,
        query_extrinsics: np.ndarray,
        H_lat: int,
        W_lat: int,
        depths: np.ndarray,
        w2c: np.ndarray,
        Ks=None,
        K_override: Optional[np.ndarray] = None,
        query_K: Optional[np.ndarray] = None,
        candidates_per_query_group: int = 1,
        coarse_candidate_groups: Optional[int] = None,
        angle_threshold: Optional[float] = None,
        distance_threshold: Optional[float] = None,
        temporal_threshold: Optional[int] = None,
        iou_mode: str = "bbox",
        geometry_rep_mode: str = "second_frame",
        frames_per_latent: int = 4,
        candidate_nms_mode: Optional[str] = None,
        candidate_nms_projection_iou_threshold: float = 0.7,
        candidate_nms_min_temporal_gap: int = 0,
        candidate_nms_pose_distance_threshold: float = 0.0,
        candidate_nms_pool_multiplier: float = 2.0,
        query_reference_frame: int = 2,
        depth_downsample_pool: Optional[Dict] = None,
        coverage_grid_downsample: int = 1,
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[List[int]]]:
        Q = int(query_extrinsics.shape[0])
        if Q % 4 != 0:
            raise ValueError(f"query_extrinsics 数量必须是4的倍数, 当前 Q={Q}")
        query_reference_offset = self._query_reference_offset(query_reference_frame)

        if candidate_nms_mode is not None and candidate_nms_mode not in {
            "projection_iou",
            "temporal",
            "pose",
            "coverage",
        }:
            raise ValueError(
                f"Unsupported candidate_nms_mode={candidate_nms_mode!r}. Use "
                "'projection_iou', 'temporal', 'pose', 'coverage' or None."
            )

        total_frames = int(w2c.shape[0])
        frames_per_latent = max(1, int(frames_per_latent))
        if frames_per_latent == 1:
            num_candidate_groups = max(total_frames, 1)
        else:
            num_candidate_groups = (
                1 + (((total_frames - 1) + frames_per_latent - 1) // frames_per_latent) if total_frames > 1 else 1
            )
        candidate_gids = list(range(num_candidate_groups))
        rep_infos = [
            self._build_group_representative_geometry(
                gid,
                depths=depths,
                w2c=w2c,
                geometry_rep_mode=geometry_rep_mode,
                frames_per_latent=frames_per_latent,
            )
            for gid in candidate_gids
        ]
        rep_w2c = np.stack([r["w2c"] for r in rep_infos], axis=0)
        coarse_k = int(
            max(
                candidates_per_query_group,
                (coarse_candidate_groups if coarse_candidate_groups is not None else candidates_per_query_group * 4),
            )
        )
        # When candidate NMS is active the inter-candidate suppression can
        # drain the coarse pool; enlarge it preemptively so the picker has
        # enough non-proximate options to honour the request. Clamp to the
        # total number of candidate groups available.
        if candidate_nms_mode is not None:
            coarse_k = int(round(coarse_k * float(candidate_nms_pool_multiplier)))
            coarse_k = max(coarse_k, candidates_per_query_group)
            coarse_k = min(coarse_k, len(candidate_gids))

        selected_group_ids: List[List[int]] = []
        selected_frame_ids: List[List[int]] = []
        keyframe_group_ids: List[List[int]] = []
        keyframe_frame_ids: List[List[int]] = []

        num_query_groups = Q // 4
        # Per-query-frame K: when ``query_K`` is supplied, derive a single
        # (3,3) for THIS query group from query_K[query_frame_idx]; falls
        # back to None so legacy K_override (3,3) keeps applying to both
        # sides as before.
        if query_K is not None:
            query_K_arr = np.asarray(query_K, dtype=np.float32)
            if query_K_arr.shape == (3, 3):
                pass  # broadcast: same K for all query frames
            elif query_K_arr.ndim == 3 and query_K_arr.shape[1:] == (3, 3):
                if query_K_arr.shape[0] != Q:
                    raise ValueError(f"query_K shape {query_K_arr.shape} must match query_extrinsics length Q={Q}.")
            else:
                raise ValueError(f"query_K must have shape (3,3) or (Q,3,3), got {query_K_arr.shape}.")
        else:
            query_K_arr = None

        for qg in range(num_query_groups):
            query_frame_idx = qg * 4 + query_reference_offset
            query_w2c = np.asarray(query_extrinsics[query_frame_idx], dtype=np.float32)
            if query_K_arr is None:
                query_K_one = None
            elif query_K_arr.shape == (3, 3):
                query_K_one = query_K_arr
            else:
                query_K_one = query_K_arr[query_frame_idx]
            ranked = self._rank_candidates_by_angle_then_translation(
                rep_w2c,
                query_w2c[None, ...],
                angle_topk=min(max(coarse_k, 1), len(candidate_gids)),
                angle_threshold=angle_threshold,
                query_frame_idx=query_frame_idx,
                distance_threshold=distance_threshold,
                temporal_threshold=temporal_threshold,
            )
            if not ranked:
                selected_group_ids.append([-1] * candidates_per_query_group)
                selected_frame_ids.append([-1] * candidates_per_query_group)
                keyframe_group_ids.append([-1] * candidates_per_query_group)
                keyframe_frame_ids.append([-1] * candidates_per_query_group)
                continue

            coarse = ranked[: min(len(ranked), coarse_k)]
            scored = []
            for coarse_rank, (local_idx, angle_deg, dist) in enumerate(coarse):
                gid = candidate_gids[int(local_idx)]
                iou_score, rep = self._score_group_candidate_by_projection_iou(
                    gid,
                    query_w2c=query_w2c,
                    H_lat=H_lat,
                    W_lat=W_lat,
                    depths=depths,
                    w2c=w2c,
                    Ks=Ks,
                    K_override=K_override,
                    query_K_one=query_K_one,
                    geometry_rep_mode=geometry_rep_mode,
                    iou_mode=iou_mode,
                    frames_per_latent=frames_per_latent,
                    depth_downsample_pool=depth_downsample_pool,
                    return_coverage_mask=(candidate_nms_mode == "coverage"),
                    depth_reduce_mode=("nearest" if candidate_nms_mode == "coverage" else "block_median"),
                    coverage_grid_downsample=coverage_grid_downsample,
                )
                # Derive camera-centre translation (C = -R^T t) for the
                # ``pose`` NMS mode. Built unconditionally so the entry is
                # always present and cheap; the picker only consults it
                # when mode == "pose".
                rep_w2c_one = np.asarray(rep["w2c"], dtype=np.float32)
                R_rep = rep_w2c_one[:3, :3]
                t_rep = rep_w2c_one[:3, 3]
                cam_center = -R_rep.T @ t_rep
                scored.append(
                    {
                        "gid": int(gid),
                        "frame_id": int(rep["frame_id"]),
                        "iou_score": float(iou_score),
                        "angle_deg": float(angle_deg),
                        "dist": float(dist),
                        "coarse_rank": int(coarse_rank),
                        "proj_bbox": rep.get("proj_bbox"),
                        "rep_translation": np.asarray(cam_center, dtype=np.float32),
                        "coverage_mask": rep.get("coverage_mask"),
                    }
                )

            scored.sort(
                key=lambda x: (
                    -x["iou_score"],
                    x["angle_deg"],
                    x["dist"],
                    x["coarse_rank"],
                    x["gid"],
                )
            )
            if candidate_nms_mode is None:
                picked = scored[:candidates_per_query_group]
            elif candidate_nms_mode == "coverage":
                picked = self._pick_candidates_by_coverage_greedy(scored, candidates_per_query_group)
            else:
                picked = self._pick_candidates_with_nms(
                    scored,
                    candidates_per_query_group,
                    candidate_nms_mode,
                    {
                        "projection_iou_threshold": candidate_nms_projection_iou_threshold,
                        "min_temporal_gap": candidate_nms_min_temporal_gap,
                        "pose_distance_threshold": candidate_nms_pose_distance_threshold,
                    },
                )
            picked_group_ids = [int(x["gid"]) for x in picked]
            picked_frame_ids = [int(x["frame_id"]) for x in picked]
            while len(picked_group_ids) < candidates_per_query_group:
                picked_group_ids.append(-1)
                picked_frame_ids.append(-1)
            selected_group_ids.append(picked_group_ids)
            selected_frame_ids.append(picked_frame_ids)
            keyframe_group_ids.append([-1] * candidates_per_query_group)
            keyframe_frame_ids.append([-1] * candidates_per_query_group)

        return (
            selected_group_ids,
            selected_frame_ids,
            keyframe_group_ids,
            keyframe_frame_ids,
        )

    def query_hits_mode_old(
        self,
        device,
        query_extrinsics: np.ndarray,
        latents,
        w2c=None,
        depths=None,
        K=None,
        candidates_per_query_group: int = 1,
        keyframe_candidate_min_count: Optional[int] = 0,
        keyframe_candidate_max_count: Optional[int] = 0,
        keyframe_candidate_ratio: float = 0,
        distance_threshold: Optional[float] = None,
        temporal_threshold: Optional[int] = None,
        angle_threshold: Optional[float] = None,
        diversity_weight: float = DIVERSITY_WEIGHT,
        vote_weight: float = VOTE_WEIGHT,
        stability_weight: float = STABILITY__WEIGHT,
        first_frame_penalty: float = FIRST_FRAME_PENALTY,
        fuse_mode: str = "masked_mean",
        fill_ratio_threshold: float = 0.95,
        fill_candidate_limit: Optional[int] = None,
        return_candidate_frame_ids: bool = False,
        return_revgrid: bool = False,
        zbuffer_depth_preference: str = "near",
        interpolation_mode: str = "bilinear",
        geometry_rep_mode: str = "second_frame",
    ) -> Tuple[
        Dict[int, Dict[str, np.ndarray]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Run frustum overlap queries for novel camera parameters.
        latents: CTHW
        Notes:
            - `candidate_ids` are **GLOBAL** frame indices (e.g., 1..N). If None, use all cached frames.
            - Output `grouped_output[q]["n"]` stores **GLOBAL** frame indices.
        """
        keyframe_mask = None
        if w2c is None and depths is None:
            cache = self._require_cache()
            w2c = cache.w2c.copy()  # (T, 4, 4) 已缓存的所有源帧 w2c
            depths = cache.depths.copy()
            keyframe_mask = cache.is_keyframe.copy()
            # w2c_backup = w2c.copy()

        query_extrinsics = np.asarray(query_extrinsics, dtype=float)
        query_extrinsics_backup = query_extrinsics.copy()
        if query_extrinsics.ndim == 2:
            query_extrinsics = query_extrinsics[None, ...]  # (1,4,4)
        if query_extrinsics.ndim != 3 or query_extrinsics.shape[1:] != (4, 4):
            raise ValueError("query_extrinsics must have shape (Q,4,4).")

        # 将候选分组投票逻辑抽成类方法，避免 query_hits_mode_old 过长
        (
            _selected_group_ids,
            candidate_frame_ids,
            _keyframe_group_ids,
            keyframe_candidate_frame_ids,
        ) = self._select_group_ids_by_vote(
            w2c,
            query_extrinsics,
            candidates_per_query_group=candidates_per_query_group,
            keyframe_mask=keyframe_mask,
            keyframe_candidate_min_count=keyframe_candidate_min_count,
            keyframe_candidate_max_count=keyframe_candidate_max_count,
            keyframe_candidate_ratio=keyframe_candidate_ratio,
            angle_threshold=angle_threshold,
            distance_threshold=distance_threshold,
            temporal_threshold=temporal_threshold,
            diversity_weight=diversity_weight,
            vote_weight=vote_weight,
            stability_weight=stability_weight,
            first_frame_penalty=first_frame_penalty,
        )
        iqx = 1
        queried_latents = []
        queried_revgrids: List[np.ndarray] = []
        for frame_candidates, keyframe_frame_candidates in zip(candidate_frame_ids, keyframe_candidate_frame_ids):
            keyframe_frame_candidates = []
            fused_out = self._fuse_group_latent_with_keyframes(
                frame_candidates=frame_candidates,
                keyframe_frame_candidates=keyframe_frame_candidates,
                latents=latents,
                depths=depths,
                Ks=self._Kpixs,
                w2c=w2c,
                query_w2c=query_extrinsics_backup[iqx],
                fuse_mode=fuse_mode,
                interpolation_mode=interpolation_mode,
                fill_ratio_threshold=fill_ratio_threshold,
                fill_candidate_limit=fill_candidate_limit,
                return_revgrid=return_revgrid,
                zbuffer_depth_preference=zbuffer_depth_preference,
                geometry_rep_mode=geometry_rep_mode,
            )
            if return_revgrid:
                fused_latent, fused_revgrid = fused_out
                queried_revgrids.append(fused_revgrid)
            else:
                fused_latent = fused_out
            queried_latents.append(fused_latent)
            iqx += 4
        output = torch.cat(queried_latents, dim=1)
        revgrid_output = np.stack(queried_revgrids, axis=0) if return_revgrid else None
        # 按需返回每个 query-group 的候选帧 id 列表（长度为 candidates_per_query_group）
        if return_candidate_frame_ids and return_revgrid:
            return (
                output,
                candidate_frame_ids,
                keyframe_candidate_frame_ids,
                revgrid_output,
            )
        if return_candidate_frame_ids:
            return (output, candidate_frame_ids, keyframe_candidate_frame_ids)
        if return_revgrid:
            return (output, revgrid_output)
        return output

    def replace_ext_with_end_extr(self, frame_id, extrinsics: np.ndarray) -> np.ndarray:
        assert extrinsics.shape == (4, 4)
        self.extrinsics[frame_id] = extrinsics.copy()

    def select_candidates(
        self,
        *,
        query_extrinsics: np.ndarray,
        depths: np.ndarray,
        w2c: np.ndarray,
        K: Optional[np.ndarray] = None,
        memory_K: Optional[np.ndarray] = None,
        query_K: Optional[np.ndarray] = None,
        H_lat: int,
        W_lat: int,
        total_latents: Optional[int] = None,
        candidates_per_query_group: int = 1,
        distance_threshold: Optional[float] = None,
        temporal_threshold: Optional[int] = None,
        angle_threshold: Optional[float] = None,
        coarse_candidate_groups: Optional[int] = None,
        iou_mode: str = "bbox",
        geometry_rep_mode: str = "second_frame",
        latent_merge_4frames: bool = False,
        selection_mode: str = "projection_iou",
        query_frame_indices: Optional[Sequence[int]] = None,
        query_reference_frame: int = 2,
        candidate_nms_mode: Optional[str] = None,
        candidate_nms_projection_iou_threshold: float = 0.7,
        candidate_nms_min_temporal_gap: int = 0,
        candidate_nms_pose_distance_threshold: float = 0.0,
        candidate_nms_pool_multiplier: float = 2.0,
        depth_downsample_pool: Optional[Dict] = None,
        coverage_grid_downsample: int = 4,
        coverage_pool_stride: int = 2,
    ) -> List[List[int]]:
        """Phase 1 of the split mosaic-query pipeline: pick candidate frame ids
        per query group **without** touching any latent buffer.

        Returns ``candidate_frame_ids: List[List[int]]`` — one inner list per
        query group, each element a frame index in ``[0, w2c.shape[0])``
        (or ``-1`` padding when no candidate survives the filters). When
        ``latent_merge_4frames=False`` (the video-flow default), the frame
        index is also the memory-latent buffer index 1:1.

        ``total_latents`` is accepted for caller documentation parity but is
        not consumed at selection time (candidate enumeration is driven by
        ``w2c.shape[0]``); the value is asserted to match
        ``w2c.shape[0]`` when ``latent_merge_4frames=False`` to catch
        upstream off-by-one bugs early.

        Args:
            candidate_nms_mode: opt-in inter-candidate NMS. One of
                ``None`` (default; legacy top-N by score), ``"projection_iou"``
                (only valid in ``selection_mode='projection_iou'``),
                ``"temporal"`` (frame-id gap), ``"pose"`` (camera-centre
                translation L2 distance). Suppresses neighbouring candidates
                so a small budget covers a wider portion of the canvas /
                camera trajectory instead of clustering near the top score.
            candidate_nms_projection_iou_threshold: ``projection_iou`` mode;
                two candidates with projection-bbox IoU >= this value count
                as proximate. Defaults to 0.7.
            candidate_nms_min_temporal_gap: ``temporal`` mode; absolute
                frame-id gap below this counts as proximate. ``0`` disables.
            candidate_nms_pose_distance_threshold: ``pose`` mode; camera
                centre L2 distance below this counts as proximate. ``0.0``
                disables.
            candidate_nms_pool_multiplier: scale factor applied to
                ``coarse_k`` when NMS is active, so the coarse pool stays
                large enough to fill the budget after suppression.
        """
        query_extrinsics = np.asarray(query_extrinsics, dtype=float)
        if query_extrinsics.ndim == 2:
            query_extrinsics = query_extrinsics[None, ...]
        if query_extrinsics.ndim != 3 or query_extrinsics.shape[1:] != (4, 4):
            raise ValueError("query_extrinsics must have shape (Q,4,4).")

        selection_mode = str(selection_mode)
        query_reference_offset = self._query_reference_offset(query_reference_frame)

        if candidate_nms_mode is not None and candidate_nms_mode not in {
            "projection_iou",
            "temporal",
            "pose",
            "coverage",
        }:
            raise ValueError(
                f"Unsupported candidate_nms_mode={candidate_nms_mode!r}. Use "
                "'projection_iou', 'temporal', 'pose', 'coverage' or None."
            )

        pose_selection_modes = {"pose_nearest", "pose_pool_temporal_earliest"}
        if selection_mode in pose_selection_modes:
            if candidate_nms_mode in ("projection_iou", "coverage"):
                # ``pose_nearest`` skips the per-candidate projection step
                # entirely, so we have no proj_bbox / coverage_mask to pick
                # on. Both projection-based pickers require the projection_iou
                # selection path; reject loudly rather than silently degrading.
                raise ValueError(
                    f"{candidate_nms_mode!r} candidate_nms_mode requires "
                    "selection_mode='projection_iou'; switch the selection "
                    "mode or pick 'temporal'/'pose' instead."
                )
            max_candidates = max(1, int(candidates_per_query_group))
            num_query_frames = int(query_extrinsics.shape[0])
            num_query_groups = num_query_frames // 4 if num_query_frames % 4 == 0 else num_query_frames
            if query_frame_indices is not None:
                query_frame_indices = [int(idx) for idx in query_frame_indices]
                if len(query_frame_indices) not in {num_query_frames, num_query_groups}:
                    raise ValueError(
                        "query_frame_indices length must match query frames or query groups; "
                        f"got {len(query_frame_indices)}, frames={num_query_frames}, "
                        f"groups={num_query_groups}."
                    )
            candidate_frame_ids = []
            for qg in range(num_query_groups):
                query_array_idx = qg * 4 + query_reference_offset if num_query_frames % 4 == 0 else qg
                if query_frame_indices is None:
                    query_frame_idx = query_array_idx
                elif len(query_frame_indices) == num_query_frames:
                    query_frame_idx = query_frame_indices[query_array_idx]
                else:
                    query_frame_idx = query_frame_indices[qg]
                # Enlarge the coarse pool when NMS is active so we still
                # have non-proximate candidates after suppression.
                if candidate_nms_mode is None:
                    pool_topk = max(max_candidates, min(10, int(w2c.shape[0])))
                else:
                    pool_topk = int(
                        round(max(max_candidates, min(10, int(w2c.shape[0]))) * float(candidate_nms_pool_multiplier))
                    )
                    pool_topk = min(max(pool_topk, max_candidates), int(w2c.shape[0]))
                ranked = self._rank_candidates_by_angle_then_translation(
                    w2c,
                    query_extrinsics[query_array_idx : query_array_idx + 1],
                    angle_topk=pool_topk,
                    angle_threshold=angle_threshold,
                    query_frame_idx=query_frame_idx,
                    distance_threshold=distance_threshold,
                    temporal_threshold=temporal_threshold,
                )
                if selection_mode == "pose_pool_temporal_earliest":
                    temporal_pool_k = int(round(max_candidates * float(candidate_nms_pool_multiplier)))
                    temporal_pool_k = min(max(temporal_pool_k, max_candidates), len(ranked))
                    ranked_for_pick = sorted(ranked[:temporal_pool_k], key=lambda item: int(item[0]))
                else:
                    ranked_for_pick = ranked
                if candidate_nms_mode is None:
                    ids = [int(idx) for idx, _angle, _dist in ranked_for_pick[:max_candidates]]
                else:
                    scored_pose = []
                    for idx, _angle, _dist in ranked_for_pick:
                        w2c_one = np.asarray(w2c[int(idx)], dtype=np.float32)
                        R_one = w2c_one[:3, :3]
                        t_one = w2c_one[:3, 3]
                        cam_center = -R_one.T @ t_one
                        scored_pose.append(
                            {
                                "gid": int(idx),
                                "frame_id": int(idx),
                                "iou_score": 0.0,
                                "angle_deg": 0.0,
                                "dist": 0.0,
                                "coarse_rank": int(len(scored_pose)),
                                "proj_bbox": None,
                                "rep_translation": np.asarray(cam_center, dtype=np.float32),
                            }
                        )
                    picked_pose = self._pick_candidates_with_nms(
                        scored_pose,
                        max_candidates,
                        candidate_nms_mode,
                        {
                            "projection_iou_threshold": candidate_nms_projection_iou_threshold,
                            "min_temporal_gap": candidate_nms_min_temporal_gap,
                            "pose_distance_threshold": candidate_nms_pose_distance_threshold,
                        },
                    )
                    ids = [int(x["frame_id"]) for x in picked_pose]
                if not ids:
                    ids = [-1] * max_candidates
                while len(ids) < max_candidates:
                    ids.append(-1)
                candidate_frame_ids.append(ids)
            return candidate_frame_ids

        if selection_mode != "projection_iou":
            raise ValueError(
                "selection_mode must be 'projection_iou', 'pose_nearest', or "
                "'pose_pool_temporal_earliest', "
                f"got {selection_mode!r}."
            )

        if candidate_nms_mode == "coverage":
            # This branch returns early, so it must re-assert the same
            # buffer-sanity guards the main projection_iou path enforces below
            # (the early return would otherwise silently skip them). The
            # vectorised coverage picker assumes exactly one latent per memory
            # frame, so latent_merge_4frames is unsupported, and total_latents
            # (when given) must equal the memory length.
            if bool(latent_merge_4frames):
                raise ValueError(
                    "candidate_nms_mode='coverage' does not support "
                    "latent_merge_4frames=True (the vectorised coverage picker "
                    "assumes one latent per memory frame)."
                )
            if total_latents is not None and int(total_latents) != int(w2c.shape[0]):
                raise ValueError(
                    f"select_candidates: total_latents={total_latents} does not "
                    f"match w2c.shape[0]={w2c.shape[0]} (coverage mode is "
                    f"one-latent-per-frame)."
                )
            # Fast vectorised full-pool greedy-coverage picker (bypasses the
            # per-candidate angle-rank + projection_iou scoring loop entirely).
            #
            # Resolve memory/query intrinsics with the SAME fallback chain the
            # regular projection_iou path uses (``_get_frame_intrinsic`` /
            # ``_resolve_query_K_for_group``): training passes memory_K +
            # query_K explicitly, but inference callers (e.g.
            # ``query_hits_mode_new``) register the source sequence and rely on
            # ``self._Kpixs`` / ``self._Kpix`` instead of passing K. Hard-
            # requiring query_K here crashed those callers.
            M = int(w2c.shape[0])
            cov_memory_K = self._coerce_memory_K(K=K, memory_K=memory_K, M=M)
            cov_query_K = query_K
            if cov_memory_K is None:
                # No explicit memory K: fall back to the registered intrinsics.
                if self._Kpixs is not None and len(self._Kpixs) >= M:
                    cov_memory_K = np.asarray(self._Kpixs, dtype=np.float64)
                elif self._Kpix is not None:
                    cov_memory_K = np.asarray(self._Kpix, dtype=np.float64)
                else:
                    raise ValueError(
                        "candidate_nms_mode='coverage': no memory intrinsics. "
                        "Pass memory_K/K, or register a source sequence first."
                    )
            if cov_query_K is None:
                cov_arr = np.asarray(cov_memory_K)
                if cov_arr.shape == (3, 3) and memory_K is None:
                    # Legacy single shared K (or single registered _Kpix):
                    # query reuses the same (3,3), matching the regular path.
                    cov_query_K = cov_arr
                elif self._Kpix is not None:
                    # Per-frame memory K but no query K supplied: inference uses
                    # the single registered query intrinsic (regular path does
                    # the same via _resolve_query_K_for_group's fallback_K).
                    cov_query_K = np.asarray(self._Kpix, dtype=np.float64)
                else:
                    raise ValueError(
                        "candidate_nms_mode='coverage': per-frame memory_K was "
                        "given but query_K was not, and no registered query "
                        "intrinsic is available. Pass query_K together with "
                        "memory_K."
                    )
            return self._select_candidates_coverage_vectorized(
                query_extrinsics=query_extrinsics,
                depths=depths,
                w2c=w2c,
                memory_K=cov_memory_K,
                query_K=cov_query_K,
                H_lat=int(H_lat),
                W_lat=int(W_lat),
                candidates_per_query_group=candidates_per_query_group,
                query_reference_frame=query_reference_frame,
                coverage_grid_downsample=coverage_grid_downsample,
                coverage_pool_stride=coverage_pool_stride,
                depth_downsample_pool=depth_downsample_pool,
            )

        # ``K_override`` is the memory-side intrinsic. Two valid shapes:
        #   * (3,3)        — legacy: same K shared by query + memory (inference).
        #   * (M,3,3)      — per-memory-frame K (training); REQUIRES query_K.
        # ``memory_K`` is just an alias for the (M,3,3) case to keep the
        # call-site self-documenting; collapses into ``K_override`` here.
        K_override = self._coerce_memory_K(K=K, memory_K=memory_K, M=int(w2c.shape[0]))

        frames_per_latent = 4 if bool(latent_merge_4frames) else 1

        if total_latents is not None and frames_per_latent == 1 and int(total_latents) != int(w2c.shape[0]):
            # Fail loud: callers that pass a mismatched total_latents almost
            # always have an off-by-one in their memory-buffer setup. Better
            # to surface that here than produce silently-bad candidate ids.
            raise ValueError(
                f"select_candidates: total_latents={total_latents} does not "
                f"match w2c.shape[0]={w2c.shape[0]} under latent_merge_4frames=False."
            )

        (
            _selected_group_ids,
            candidate_frame_ids,
            _keyframe_group_ids,
            _keyframe_frame_ids,
        ) = self._select_group_ids_by_projection_iou(
            query_extrinsics=query_extrinsics,
            H_lat=int(H_lat),
            W_lat=int(W_lat),
            depths=depths,
            w2c=w2c,
            Ks=self._Kpixs,
            K_override=K_override,
            query_K=query_K,
            candidates_per_query_group=candidates_per_query_group,
            coarse_candidate_groups=coarse_candidate_groups,
            angle_threshold=angle_threshold,
            distance_threshold=distance_threshold,
            temporal_threshold=temporal_threshold,
            iou_mode=iou_mode,
            geometry_rep_mode=geometry_rep_mode,
            frames_per_latent=frames_per_latent,
            candidate_nms_mode=candidate_nms_mode,
            candidate_nms_projection_iou_threshold=candidate_nms_projection_iou_threshold,
            candidate_nms_min_temporal_gap=candidate_nms_min_temporal_gap,
            candidate_nms_pose_distance_threshold=candidate_nms_pose_distance_threshold,
            candidate_nms_pool_multiplier=candidate_nms_pool_multiplier,
            query_reference_frame=query_reference_frame,
            depth_downsample_pool=depth_downsample_pool,
            coverage_grid_downsample=coverage_grid_downsample,
        )
        return candidate_frame_ids

    def fuse_candidates(
        self,
        *,
        query_extrinsics: np.ndarray,
        candidate_frame_ids: Sequence[Sequence[int]],
        latents: torch.Tensor,
        w2c: np.ndarray,
        depths: np.ndarray,
        K: Optional[np.ndarray] = None,
        memory_K: Optional[np.ndarray] = None,
        query_K: Optional[np.ndarray] = None,
        fuse_mode: str = "masked_mean",
        fill_ratio_threshold: float = 0.95,
        fill_candidate_limit: Optional[int] = None,
        return_revgrid: bool = False,
        zbuffer_depth_preference: str = "near",
        interpolation_mode: str = "bilinear",
        geometry_rep_mode: str = "second_frame",
        latent_merge_4frames: bool = False,
        query_reference_frame: int = 2,
        special_query_frame=None,
        # Accept and ignore so the dataset can pass H_lat/W_lat alongside
        # the rest of ``fused_query_inputs`` without a separate split.
        H_lat: Optional[int] = None,
        W_lat: Optional[int] = None,
        memory_total_frames: Optional[int] = None,
        return_view_change: bool = False,
        fuse_block_size: int = 1,
        return_source_frame_ids: bool = False,
        source_valid_masks: Optional[torch.Tensor | np.ndarray] = None,
    ) -> Tuple:
        """Phase 2 of the split pipeline: given ``candidate_frame_ids`` from
        ``select_candidates`` and a populated memory ``latents`` buffer, run
        per-query-group reproject+fuse and stitch the result.

        Returns ``queried_latent`` plus optional extras in flag order:
        ``revgrid``, ``view_change``, ``source_frame_ids``.

        Removed (vs. the legacy ``query_hits_mode_new`` signature):
            * ``keyframe_candidate_frame_ids`` — IoU mode never produces real
              keyframe ids (was always ``-1`` placeholder, then forced to
              ``[]`` before fuse). Internal call uses ``[]`` directly.
        """
        with self._timing_scope(
            "materialize.fuse_candidates.normalize_inputs",
            metadata={
                "query_count": len(query_extrinsics),
                "candidate_groups": len(candidate_frame_ids),
                "fuse_mode": str(fuse_mode),
                "return_revgrid": bool(return_revgrid),
                "return_view_change": bool(return_view_change),
                "return_source_frame_ids": bool(return_source_frame_ids),
                "fuse_block_size": int(fuse_block_size),
            },
        ):
            query_extrinsics = np.asarray(query_extrinsics, dtype=float)
            query_extrinsics_backup = query_extrinsics.copy()
            if query_extrinsics.ndim == 2:
                query_extrinsics = query_extrinsics[None, ...]
            if query_extrinsics.ndim != 3 or query_extrinsics.shape[1:] != (4, 4):
                raise ValueError("query_extrinsics must have shape (Q,4,4).")

            K_override = self._coerce_memory_K(K=K, memory_K=memory_K, M=int(w2c.shape[0]))

            # Per-query-frame K resolution (mirrors _select_group_ids_by_projection_iou).
            Q = int(query_extrinsics.shape[0])
            if query_K is not None:
                query_K_arr = np.asarray(query_K, dtype=np.float32)
                if query_K_arr.shape == (3, 3):
                    pass
                elif query_K_arr.ndim == 3 and query_K_arr.shape[1:] == (3, 3):
                    if query_K_arr.shape[0] != Q:
                        raise ValueError(f"query_K shape {query_K_arr.shape} must match query_extrinsics length Q={Q}.")
                else:
                    raise ValueError(f"query_K must have shape (3,3) or (Q,3,3), got {query_K_arr.shape}.")
            else:
                query_K_arr = None

            frames_per_latent = 4 if bool(latent_merge_4frames) else 1
            source_valid_masks_arg = self._coerce_source_valid_masks(
                source_valid_masks,
                total_latents=int(latents.shape[1]),
                H_lat=int(latents.shape[-2]),
                W_lat=int(latents.shape[-1]),
            )

        query_reference_offset = self._query_reference_offset(query_reference_frame)
        if int(query_extrinsics.shape[0]) % 4 == 0:
            iqx = query_reference_offset
            query_step = 4
        else:
            iqx = 0
            query_step = 1
        queried_latents: List[torch.Tensor] = []
        queried_revgrids: List[np.ndarray] = []
        queried_view_changes: List[np.ndarray] = []
        queried_source_fids: List[np.ndarray] = []
        for group_idx, frame_candidates in enumerate(candidate_frame_ids):
            if special_query_frame is not None:
                frame_candidates = list(special_query_frame) + list(frame_candidates)
            if query_K_arr is None:
                query_K_one = None
            elif query_K_arr.shape == (3, 3):
                query_K_one = query_K_arr
            else:
                query_K_one = query_K_arr[iqx]
            with self._timing_scope(
                "materialize.fuse_candidates.group",
                metadata={
                    "group_idx": int(group_idx),
                    "query_frame_index": int(iqx),
                    "query_reference_frame": int(query_reference_frame),
                    "candidate_count": len(frame_candidates),
                    "fuse_mode": str(fuse_mode),
                    "fuse_block_size": int(fuse_block_size),
                    "return_view_change": bool(return_view_change),
                    "return_source_frame_ids": bool(return_source_frame_ids),
                },
            ):
                fused_out = self._fuse_group_latent_with_keyframes(
                    frame_candidates=frame_candidates,
                    keyframe_frame_candidates=[],
                    latents=latents,
                    depths=depths,
                    Ks=self._Kpixs,
                    K_override=K_override,
                    query_K_one=query_K_one,
                    w2c=w2c,
                    query_w2c=query_extrinsics_backup[iqx],
                    fuse_mode=fuse_mode,
                    interpolation_mode=interpolation_mode,
                    fill_ratio_threshold=fill_ratio_threshold,
                    fill_candidate_limit=fill_candidate_limit,
                    return_revgrid=return_revgrid,
                    zbuffer_depth_preference=zbuffer_depth_preference,
                    geometry_rep_mode=geometry_rep_mode,
                    frames_per_latent=frames_per_latent,
                    return_view_change=return_view_change,
                    fuse_block_size=fuse_block_size,
                    return_source_frame_ids=return_source_frame_ids,
                    source_valid_masks=source_valid_masks_arg,
                )
            if return_revgrid or return_view_change or return_source_frame_ids:
                fused_latent = fused_out[0]
                extra_idx = 1
                if return_revgrid:
                    queried_revgrids.append(fused_out[extra_idx])
                    extra_idx += 1
                if return_view_change:
                    queried_view_changes.append(fused_out[extra_idx])
                    extra_idx += 1
                if return_source_frame_ids:
                    queried_source_fids.append(fused_out[extra_idx])
            else:
                fused_latent = fused_out
            queried_latents.append(fused_latent)
            iqx += query_step

        with self._timing_scope(
            "materialize.fuse_candidates.concat_output",
            metadata={
                "groups": len(queried_latents),
                "return_revgrid": bool(return_revgrid),
                "return_view_change": bool(return_view_change),
                "return_source_frame_ids": bool(return_source_frame_ids),
            },
        ):
            output = torch.cat(queried_latents, dim=1)
            revgrid_output = np.stack(queried_revgrids, axis=0) if return_revgrid else None
            view_change_output = np.stack(queried_view_changes, axis=0) if return_view_change else None
            source_fid_output = np.stack(queried_source_fids, axis=0) if return_source_frame_ids else None
        extras: List = []
        if return_revgrid:
            extras.append(revgrid_output)
        if return_view_change:
            extras.append(view_change_output)
        if return_source_frame_ids:
            extras.append(source_fid_output)
        if extras:
            return (output, *extras)
        return output

    def query_hits_mode_new(
        self,
        device,
        query_extrinsics: np.ndarray,
        latents,
        w2c=None,
        depths=None,
        K=None,
        memory_K: Optional[np.ndarray] = None,
        query_K: Optional[np.ndarray] = None,
        candidates_per_query_group: int = 1,
        distance_threshold: Optional[float] = None,
        temporal_threshold: Optional[int] = None,
        angle_threshold: Optional[float] = None,
        fuse_mode: str = "masked_mean",
        fill_ratio_threshold: float = 0.95,
        fill_candidate_limit: Optional[int] = None,
        return_candidate_frame_ids: bool = False,
        return_revgrid: bool = False,
        zbuffer_depth_preference: str = "near",
        interpolation_mode: str = "bilinear",
        geometry_rep_mode: str = "second_frame",
        coarse_candidate_groups: Optional[int] = None,
        iou_mode: str = "bbox",
        selection_mode: str = "projection_iou",
        special_query_frame=None,
        latent_merge_4frames: bool = False,
        query_reference_frame: int = 2,
        candidate_nms_mode: Optional[str] = None,
        candidate_nms_projection_iou_threshold: float = 0.7,
        candidate_nms_min_temporal_gap: int = 0,
        candidate_nms_pose_distance_threshold: float = 0.0,
        candidate_nms_pool_multiplier: float = 2.0,
        coverage_grid_downsample: int = 4,
        coverage_pool_stride: int = 2,
        return_view_change: bool = False,
        source_valid_masks: Optional[torch.Tensor | np.ndarray] = None,
    ) -> Tuple[
        Dict[int, Dict[str, np.ndarray]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Mode-2: coarse recall by extrinsics, then rerank latent-frame groups by projection-vs-canvas IoU.

        Thin shim over ``select_candidates`` + ``fuse_candidates`` retained
        for backward compatibility (older inference code calls this entry
        point with the legacy signature).

        Args:
            latent_merge_4frames: True 时按 4 帧压缩 latent 处理；False 时按 1 帧 1 latent 处理。
            candidate_nms_mode: opt-in inter-candidate NMS. See
                ``select_candidates`` for full semantics. Default ``None``
                preserves the legacy "top-N by IoU score" behaviour.
            candidate_nms_projection_iou_threshold: projection-bbox IoU
                threshold for ``candidate_nms_mode='projection_iou'``.
            candidate_nms_min_temporal_gap: frame-id gap threshold for
                ``candidate_nms_mode='temporal'``.
            candidate_nms_pose_distance_threshold: camera-centre L2
                distance threshold for ``candidate_nms_mode='pose'``.
            candidate_nms_pool_multiplier: coarse-pool enlargement factor
                when NMS is active.
        """
        if w2c is None and depths is None:
            depths = np.array(self.depths)
            w2c = np.array(self.extrinsics)

        H_lat = int(latents.shape[-2])
        W_lat = int(latents.shape[-1])

        # Scope a depth-downsample cache to this single inference query so
        # the IoU rerank inside ``select_candidates`` doesn't re-downsample
        # the same ``rep["depth"]`` for every (query_group, candidate)
        # pair. The pool is owned here and discarded before ``fuse_candidates``
        # runs so memory stays bounded (one (H_lat, W_lat) float32 per
        # candidate group at peak).
        depth_downsample_pool: Dict = {}
        try:
            candidate_frame_ids = self.select_candidates(
                query_extrinsics=query_extrinsics,
                depths=depths,
                w2c=w2c,
                K=K,
                memory_K=memory_K,
                query_K=query_K,
                H_lat=H_lat,
                W_lat=W_lat,
                candidates_per_query_group=candidates_per_query_group,
                distance_threshold=distance_threshold,
                temporal_threshold=temporal_threshold,
                angle_threshold=angle_threshold,
                coarse_candidate_groups=coarse_candidate_groups,
                iou_mode=iou_mode,
                geometry_rep_mode=geometry_rep_mode,
                selection_mode=selection_mode,
                latent_merge_4frames=latent_merge_4frames,
                query_reference_frame=query_reference_frame,
                candidate_nms_mode=candidate_nms_mode,
                candidate_nms_projection_iou_threshold=candidate_nms_projection_iou_threshold,
                candidate_nms_min_temporal_gap=candidate_nms_min_temporal_gap,
                candidate_nms_pose_distance_threshold=candidate_nms_pose_distance_threshold,
                candidate_nms_pool_multiplier=candidate_nms_pool_multiplier,
                coverage_grid_downsample=coverage_grid_downsample,
                coverage_pool_stride=coverage_pool_stride,
                depth_downsample_pool=depth_downsample_pool,
            )
        finally:
            depth_downsample_pool.clear()

        fused_out = self.fuse_candidates(
            query_extrinsics=query_extrinsics,
            candidate_frame_ids=candidate_frame_ids,
            latents=latents,
            w2c=w2c,
            depths=depths,
            K=K,
            memory_K=memory_K,
            query_K=query_K,
            fuse_mode=fuse_mode,
            fill_ratio_threshold=fill_ratio_threshold,
            fill_candidate_limit=fill_candidate_limit,
            return_revgrid=return_revgrid,
            zbuffer_depth_preference=zbuffer_depth_preference,
            interpolation_mode=interpolation_mode,
            geometry_rep_mode=geometry_rep_mode,
            latent_merge_4frames=latent_merge_4frames,
            query_reference_frame=query_reference_frame,
            special_query_frame=special_query_frame,
            return_view_change=return_view_change,
            source_valid_masks=source_valid_masks,
        )
        view_change_output = None
        if return_revgrid and return_view_change:
            output, revgrid_output, view_change_output = fused_out
        elif return_revgrid:
            output, revgrid_output = fused_out
        elif return_view_change:
            output, view_change_output = fused_out
            revgrid_output = None
        else:
            output = fused_out
            revgrid_output = None

        # Legacy entry point used to also surface the keyframe candidate
        # list (always all -1 in IoU mode); we synthesise an equivalent
        # placeholder so the caller signatures don't have to change.
        keyframe_candidate_frame_ids = [[-1] * candidates_per_query_group for _ in candidate_frame_ids]

        if return_candidate_frame_ids and return_revgrid and return_view_change:
            return (
                output,
                candidate_frame_ids,
                keyframe_candidate_frame_ids,
                revgrid_output,
                view_change_output,
            )
        if return_candidate_frame_ids and return_revgrid:
            return (
                output,
                candidate_frame_ids,
                keyframe_candidate_frame_ids,
                revgrid_output,
            )
        if return_candidate_frame_ids and return_view_change:
            return (
                output,
                candidate_frame_ids,
                keyframe_candidate_frame_ids,
                view_change_output,
            )
        if return_candidate_frame_ids:
            return (output, candidate_frame_ids, keyframe_candidate_frame_ids)
        if return_revgrid and return_view_change:
            return (output, revgrid_output, view_change_output)
        if return_revgrid:
            return (output, revgrid_output)
        if return_view_change:
            return (output, view_change_output)
        return output

    def _fuse_group_latent_with_keyframes(
        self,
        *,
        frame_candidates: Sequence[int],
        keyframe_frame_candidates: Sequence[int],
        latents: torch.Tensor,
        depths: np.ndarray,
        Ks: np.ndarray,
        w2c: np.ndarray,
        query_w2c: np.ndarray,
        fuse_mode: str,
        interpolation_mode: str,
        fill_ratio_threshold: float,
        fill_candidate_limit: Optional[int],
        K_override: Optional[np.ndarray] = None,
        query_K_one: Optional[np.ndarray] = None,
        return_revgrid: bool = False,
        zbuffer_depth_preference: str = "near",
        geometry_rep_mode: str = "second_frame",
        frames_per_latent: int = 1,
        return_view_change: bool = False,
        fuse_block_size: int = 1,
        return_source_frame_ids: bool = False,
        source_valid_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward-warp + z-buffer scatter fusion of memory latents into the query view.

        For each candidate memory frame:
            1. Lift its (downsampled) latent-pixel grid to 3D using
               memory depth + memory K + memory pose (forward warp).
            2. Project that 3D point into the query camera using query K + query pose.
            3. Round to integer query latent pixels, resolve intra-candidate
               occlusion with a NEAREST z-buffer (multiple memory pixels
               mapping to the same query pixel pick the nearer surface),
               and scatter the corresponding memory latent value into the
               output buffer.

        Cross-candidate fusion is then driven by ``fuse_mode``:
            - ``masked_mean``       : pixel-wise mean across candidates that
                                      hit a given query pixel. Uses the
                                      geometric valid mask, NOT the legacy
                                      ``(latent==0).all()`` heuristic.
            - ``fill_stop``         : in candidate order, write only where
                                      not yet filled. Stops once filled ratio
                                      or candidate cap is hit.
            - ``fill_stop_zbuffer`` : maintain a global z-buffer across
                                      candidates; ``zbuffer_depth_preference``
                                      ('near' / 'far') decides who wins at
                                      each query pixel.

        Keyframe candidates are processed first, then normal candidates.

        NOTE: ``interpolation_mode`` previously selected the grid_sample
        gather mode. Under forward-warp + scatter the parameter is currently
        a no-op; only nearest single-pixel splat is implemented. A future
        patch may add 2x2 bilinear splat -- callers should keep passing
        "nearest" for forward compatibility.
        """
        del interpolation_mode  # see docstring

        H_lat = int(latents.shape[-2])
        W_lat = int(latents.shape[-1])
        C = int(latents.shape[0])
        N_pix = H_lat * W_lat
        device = latents.device
        dtype = latents.dtype
        stride = int(self._latent_stride)

        if int(fuse_block_size) not in (1, 2):
            raise ValueError(f"fuse_block_size must be 1 or 2, got {fuse_block_size}.")
        if int(fuse_block_size) == 2:
            if H_lat % 2 != 0 or W_lat % 2 != 0:
                raise ValueError(f"fuse_block_size=2 requires even (H_lat, W_lat); got ({H_lat}, {W_lat}).")

        z_pref = str(zbuffer_depth_preference).lower().strip()
        if z_pref not in {"near", "far"}:
            raise ValueError(f"Unsupported zbuffer_depth_preference={zbuffer_depth_preference}. Use 'near' or 'far'.")

        neutral_view_change = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        def _pack_fuse_result(fused_latent, revgrid=None, view_change=None, source_fid=None):
            extras: List = []
            if return_revgrid:
                extras.append(revgrid)
            if return_view_change:
                extras.append(view_change)
            if return_source_frame_ids:
                extras.append(source_fid)
            if extras:
                return (fused_latent, *extras)
            return fused_latent

        # Resolve query-side K once for this entire query group.
        with self._timing_scope(
            "materialize.fuse_candidates.group.resolve_query_K",
            metadata={"fuse_mode": str(fuse_mode)},
        ):
            query_K = self._resolve_query_K_for_group(
                query_K_one=query_K_one,
                K_override=K_override,
                fallback_K=self._Kpix,
            )

        # NOTE: revgrid semantics changed.
        # OLD (deprecated): for each canvas pixel, store the *integer* (x, y)
        # of the SOURCE memory pixel that won the splat. This discarded the
        # fractional projection and, when fed back as RoPE indices, only
        # told the model "this token came from source row/col r/c" -- the
        # sub-latent geometry was lost.
        #
        # NEW (option A): for each canvas pixel, store the *fractional*
        # canvas-space projection coords (x_canvas_f, y_canvas_f) =
        # (j_q_f, i_q_f) of the source pixel that won the splat. These are
        # in latent-grid units (range [0, W_lat] / [0, H_lat]) and can be
        # multiplied by a power-of-two scale and `.long()`-quantised against
        # the fractional-step RoPE table (`precompute_freqs_cis_with_step`)
        # to recover sub-latent positional encoding.

        def _resolve_candidate_records(
            frame_ids: Sequence[int],
        ) -> List[Dict[str, np.ndarray]]:
            records: List[Dict[str, np.ndarray]] = []
            seen_local = set()
            total_frames = int(w2c.shape[0])
            for frame_id in frame_ids:
                frame_id = int(frame_id)
                if frame_id < 0 or frame_id in seen_local:
                    continue
                fid = int(np.clip(frame_id, 0, total_frames - 1))
                records.append(
                    {
                        "frame_id": fid,
                        "depth": np.asarray(depths[fid], dtype=np.float32),
                        "w2c": np.asarray(w2c[fid], dtype=np.float32),
                        "K": self._get_frame_intrinsic(fid, Ks=Ks, K_override=K_override),
                    }
                )
                seen_local.add(frame_id)
            return records

        with self._timing_scope(
            "materialize.fuse_candidates.group.resolve_candidate_records",
            metadata={
                "normal_candidates": len(frame_candidates),
                "keyframe_candidates": len(keyframe_frame_candidates),
            },
        ):
            keyframe_records = _resolve_candidate_records(keyframe_frame_candidates)
            normal_records: List[Dict[str, np.ndarray]] = []
            seen_frame_ids = {int(r["frame_id"]) for r in keyframe_records}
            for rep in _resolve_candidate_records(frame_candidates):
                fid_int = int(rep["frame_id"])
                if fid_int in seen_frame_ids:
                    continue
                seen_frame_ids.add(fid_int)
                normal_records.append(rep)

            if len(keyframe_records) == 0 and len(normal_records) == 0:
                normal_records = [
                    {
                        "frame_id": 0,
                        "depth": np.asarray(depths[0], dtype=np.float32),
                        "w2c": np.asarray(w2c[0], dtype=np.float32),
                        "K": self._get_frame_intrinsic(0, Ks=Ks, K_override=K_override),
                    }
                ]

        def _splat_one(
            rec: Dict[str, np.ndarray],
        ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]:
            """Forward-warp + intra-candidate z-buffer scatter for one memory frame."""
            fid = int(rec["frame_id"])
            with self._timing_scope(
                "materialize.fuse_candidates.group.splat.project_memory_to_query",
                metadata={"frame_id": fid},
            ):
                proj_hw, zc_new = self._project_memory_to_query(
                    memory_depth=np.asarray(rec["depth"], dtype=np.float32),
                    memory_K=np.asarray(rec["K"], dtype=np.float32),
                    memory_w2c=np.asarray(rec["w2c"], dtype=np.float32),
                    query_K=query_K,
                    query_w2c=np.asarray(query_w2c, dtype=np.float32),
                    H_lat=H_lat,
                    W_lat=W_lat,
                    depth_reduce_mode="nearest",
                )

            with self._timing_scope(
                "materialize.fuse_candidates.group.splat.prepare_canvas_indices",
                metadata={"frame_id": fid},
            ):
                stride_f = float(stride)
                i_q_f = proj_hw[..., 0] / stride_f
                j_q_f = proj_hw[..., 1] / stride_f
                i_q = np.rint(i_q_f).astype(np.int64)
                j_q = np.rint(j_q_f).astype(np.int64)
                in_canvas_np = (
                    np.isfinite(i_q_f)
                    & np.isfinite(j_q_f)
                    & np.isfinite(zc_new)
                    & (i_q >= 0)
                    & (i_q < H_lat)
                    & (j_q >= 0)
                    & (j_q < W_lat)
                    & (zc_new > 0.0)
                )

                gl = self._frame_id_to_latent_id(
                    fid,
                    total_latents=int(latents.shape[1]),
                    frames_per_latent=frames_per_latent,
                )
                if source_valid_masks is not None:
                    src_valid = source_valid_masks[int(gl)].detach().cpu().numpy()
                    in_canvas_np &= src_valid

                out_latent = torch.zeros(C, 1, H_lat, W_lat, device=device, dtype=dtype)
                valid_t_flat = torch.zeros(N_pix, device=device, dtype=torch.bool)
                z_buf = torch.full((N_pix,), float("inf"), device=device, dtype=torch.float32)
                revgrid_np = np.full((N_pix, 2), -1.0, dtype=np.float32)
                view_change_np = np.empty((N_pix, 3), dtype=np.float32)
                view_change_np[:] = neutral_view_change
                source_fid_np = np.full((N_pix,), -1, dtype=np.int64)

            if not in_canvas_np.any():
                return (
                    out_latent,
                    valid_t_flat.view(H_lat, W_lat),
                    z_buf.view(H_lat, W_lat),
                    revgrid_np.reshape(H_lat, W_lat, 2),
                    view_change_np.reshape(H_lat, W_lat, 3),
                    source_fid_np.reshape(H_lat, W_lat),
                )

            flat_idx_np = (i_q * W_lat + j_q).astype(np.int64)
            flat_idx_v_np = flat_idx_np[in_canvas_np]
            z_v_np = zc_new[in_canvas_np].astype(np.float32)

            flat_idx_v = torch.from_numpy(flat_idx_v_np).to(device=device)
            z_v = torch.from_numpy(z_v_np).to(device=device)

            with self._timing_scope(
                "materialize.fuse_candidates.group.splat.scatter_zbuffer",
                metadata={"frame_id": fid, "valid_pixels": int(flat_idx_v_np.shape[0])},
            ):
                z_buf.scatter_reduce_(0, flat_idx_v, z_v, reduce="amin", include_self=True)
                winner_v = z_v == z_buf[flat_idx_v]

            if winner_v.any():
                with self._timing_scope(
                    "materialize.fuse_candidates.group.splat.scatter_latent_revgrid",
                    metadata={
                        "frame_id": fid,
                        "winner_pixels": int(winner_v.sum().item()),
                    },
                ):
                    target_flat = flat_idx_v[winner_v]
                    in_canvas_t = torch.from_numpy(in_canvas_np).to(device=device)
                    src_flat_all = torch.arange(N_pix, device=device, dtype=torch.long).view(H_lat, W_lat)
                    src_flat_v = src_flat_all[in_canvas_t]
                    src_flat = src_flat_v[winner_v]

                    mem_lat = latents[:, gl : gl + 1]  # (C, 1, H_lat, W_lat)
                    mem_flat = mem_lat.reshape(C, N_pix)
                    src_data = mem_flat[:, src_flat]

                    out_latent_flat = out_latent.reshape(C, N_pix)
                    out_latent_flat[:, target_flat] = src_data
                    valid_t_flat[target_flat] = True

                    target_np = target_flat.detach().cpu().numpy()
                    winner_v_np = winner_v.detach().cpu().numpy()
                    i_q_f_v_np = i_q_f[in_canvas_np]
                    j_q_f_v_np = j_q_f[in_canvas_np]
                    i_q_f_w_np = i_q_f_v_np[winner_v_np].astype(np.float32)
                    j_q_f_w_np = j_q_f_v_np[winner_v_np].astype(np.float32)
                    canvas_xy_w_np = np.stack([j_q_f_w_np, i_q_f_w_np], axis=-1).astype(np.float32)
                    revgrid_np[target_np] = canvas_xy_w_np
                    source_fid_np[target_np] = fid

                    if return_view_change:
                        depth_lat = self._downsample_depth_to_latent(
                            np.asarray(rec["depth"], dtype=np.float32),
                            W_lat=W_lat,
                            H_lat=H_lat,
                            reduce_mode="nearest",
                        )
                        src_h = (src_flat.detach().cpu().numpy() // W_lat).astype(np.int64)
                        src_w = (src_flat.detach().cpu().numpy() % W_lat).astype(np.int64)
                        src_h_pix = (src_h.astype(np.float64) + 0.5) * stride_f
                        src_w_pix = (src_w.astype(np.float64) + 0.5) * stride_f
                        _pix_new_hw, _zc, points_world, _xc = reproject_pixels_nd_to_nd(
                            pix_hw=np.stack([src_h_pix, src_w_pix], axis=-1),
                            depth_z=depth_lat[src_h, src_w].astype(np.float64),
                            K_old=np.asarray(rec["K"], dtype=np.float64),
                            T_old_w2c=np.asarray(rec["w2c"], dtype=np.float64),
                            K_new=np.asarray(query_K, dtype=np.float64),
                            T_new_w2c=np.asarray(query_w2c, dtype=np.float64),
                            eps=0.0,
                        )
                        vc = self._compute_view_change_for_points(
                            points_world,
                            np.asarray(rec["w2c"], dtype=np.float64),
                            np.asarray(query_w2c, dtype=np.float64),
                        )
                        valid_vc = np.isfinite(vc).all(axis=-1) & (vc[:, 0] > 0)
                        vc = np.where(valid_vc[:, None], vc, neutral_view_change)
                        view_change_np[target_np] = vc

            return (
                out_latent,
                valid_t_flat.view(H_lat, W_lat),
                z_buf.view(H_lat, W_lat),
                revgrid_np.reshape(H_lat, W_lat, 2),
                view_change_np.reshape(H_lat, W_lat, 3),
                source_fid_np.reshape(H_lat, W_lat),
            )

        def _splat_one_block(
            rec: Dict[str, np.ndarray],
        ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]:
            """2x2-block variant aligned with the 1x2x2 patch embedding conv."""
            fid = int(rec["frame_id"])
            H_pat = H_lat // 2
            W_pat = W_lat // 2
            N_pat = H_pat * W_pat

            with self._timing_scope(
                "materialize.fuse_candidates.group.splat_block.anchor_depth",
                metadata={"frame_id": fid},
            ):
                depth_lat = self._downsample_depth_to_latent(
                    np.asarray(rec["depth"], dtype=np.float32),
                    W_lat=W_lat,
                    H_lat=H_lat,
                    reduce_mode="nearest",
                )
                anchor_depth = (
                    0.25 * (depth_lat[:-1, :-1] + depth_lat[:-1, 1:] + depth_lat[1:, :-1] + depth_lat[1:, 1:])
                ).astype(np.float32)

            with self._timing_scope(
                "materialize.fuse_candidates.group.splat_block.project_anchor",
                metadata={"frame_id": fid},
            ):
                h_a = np.arange(H_lat - 1, dtype=np.float32) + 0.5
                w_a = np.arange(W_lat - 1, dtype=np.float32) + 0.5
                hh, ww = np.meshgrid(h_a, w_a, indexing="ij")
                anchor_center_hw = np.stack([hh, ww], axis=-1)
                proj_hw, zc_new, _, _ = reproject_pixels_nd_to_nd(
                    anchor_center_hw * float(stride),
                    depth_z=anchor_depth,
                    K_old=np.asarray(rec["K"], dtype=np.float32),
                    T_old_w2c=np.asarray(rec["w2c"], dtype=np.float32),
                    K_new=query_K,
                    T_new_w2c=np.asarray(query_w2c, dtype=np.float32),
                )
                zc_new = np.asarray(zc_new, dtype=np.float32)

            stride_f = float(stride)
            i_q_f = proj_hw[..., 0] / stride_f
            j_q_f = proj_hw[..., 1] / stride_f
            I_q = np.floor(i_q_f / 2.0).astype(np.int64)
            J_q = np.floor(j_q_f / 2.0).astype(np.int64)
            in_canvas_np = (
                np.isfinite(i_q_f)
                & np.isfinite(j_q_f)
                & np.isfinite(zc_new)
                & (I_q >= 0)
                & (I_q < H_pat)
                & (J_q >= 0)
                & (J_q < W_pat)
                & (zc_new > 0.0)
            )
            gl = self._frame_id_to_latent_id(
                fid,
                total_latents=int(latents.shape[1]),
                frames_per_latent=frames_per_latent,
            )
            if source_valid_masks is not None:
                src_valid = source_valid_masks[int(gl)].detach().cpu().numpy()
                # A WAN patch token consumes a 2x2 latent block. If any cell in
                # that source block is masked out, drop the whole block before
                # projection so dynamic-object pixels cannot leak through the
                # block-fuse path.
                src_valid_anchor = src_valid[:-1, :-1] & src_valid[:-1, 1:] & src_valid[1:, :-1] & src_valid[1:, 1:]
                in_canvas_np &= src_valid_anchor

            out_latent = torch.zeros(C, 1, H_lat, W_lat, device=device, dtype=dtype)
            valid_P_flat = torch.zeros(N_pat, device=device, dtype=torch.bool)
            z_buf_P_flat = torch.full((N_pat,), float("inf"), device=device, dtype=torch.float32)
            revgrid_np = np.full((H_lat, W_lat, 2), -1.0, dtype=np.float32)
            view_change_np = np.empty((H_lat, W_lat, 3), dtype=np.float32)
            view_change_np[:] = neutral_view_change
            source_fid_np = np.full((H_lat, W_lat), -1, dtype=np.int64)

            def _broadcast_p_to_l(t: torch.Tensor) -> torch.Tensor:
                return t.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

            if not in_canvas_np.any():
                return (
                    out_latent,
                    _broadcast_p_to_l(valid_P_flat.view(H_pat, W_pat)),
                    _broadcast_p_to_l(z_buf_P_flat.view(H_pat, W_pat)),
                    revgrid_np,
                    view_change_np,
                    source_fid_np,
                )

            with self._timing_scope(
                "materialize.fuse_candidates.group.splat_block.scatter_zbuffer",
                metadata={"frame_id": fid, "valid_anchors": int(in_canvas_np.sum())},
            ):
                P_flat_np = (I_q * W_pat + J_q).astype(np.int64)
                P_flat_v_np = P_flat_np[in_canvas_np]
                z_v_np = zc_new[in_canvas_np]
                P_flat_v = torch.from_numpy(P_flat_v_np).to(device=device)
                z_v = torch.from_numpy(z_v_np).to(device=device)
                z_buf_P_flat.scatter_reduce_(0, P_flat_v, z_v, reduce="amin", include_self=True)
                winner_v = z_v == z_buf_P_flat[P_flat_v]

            if winner_v.any():
                with self._timing_scope(
                    "materialize.fuse_candidates.group.splat_block.copy_2x2_blocks",
                    metadata={
                        "frame_id": fid,
                        "winner_patches": int(winner_v.sum().item()),
                    },
                ):
                    a_h_grid = np.arange(H_lat - 1, dtype=np.int64)[:, None].repeat(W_lat - 1, axis=1)
                    a_w_grid = np.arange(W_lat - 1, dtype=np.int64)[None, :].repeat(H_lat - 1, axis=0)
                    winner_v_np = winner_v.detach().cpu().numpy()
                    anchor_h_w = a_h_grid[in_canvas_np][winner_v_np]
                    anchor_w_w = a_w_grid[in_canvas_np][winner_v_np]
                    i_q_f_w = i_q_f[in_canvas_np][winner_v_np].astype(np.float32)
                    j_q_f_w = j_q_f[in_canvas_np][winner_v_np].astype(np.float32)
                    target_P_flat_np = P_flat_v_np[winner_v_np]

                    di = np.array([0, 0, 1, 1], dtype=np.int64)
                    dj = np.array([0, 1, 0, 1], dtype=np.int64)
                    I_w = target_P_flat_np // W_pat
                    J_w = target_P_flat_np % W_pat
                    tgt_h = 2 * I_w[:, None] + di[None, :]
                    tgt_w = 2 * J_w[:, None] + dj[None, :]
                    src_h = anchor_h_w[:, None] + di[None, :]
                    src_w = anchor_w_w[:, None] + dj[None, :]
                    tgt_flat = (tgt_h * W_lat + tgt_w).reshape(-1)
                    src_flat = (src_h * W_lat + src_w).reshape(-1)

                    mem_flat = latents[:, gl].reshape(C, -1)
                    src_idx_t = torch.from_numpy(src_flat).to(device=device, dtype=torch.long)
                    tgt_idx_t = torch.from_numpy(tgt_flat).to(device=device, dtype=torch.long)
                    out_flat = out_latent.reshape(C, -1)
                    out_flat[:, tgt_idx_t] = mem_flat[:, src_idx_t]

                    valid_P_flat[torch.from_numpy(target_P_flat_np).to(device=device)] = True
                    source_fid_np_flat = source_fid_np.reshape(-1)
                    source_fid_np_flat[tgt_flat] = fid

                    dx = np.array([-0.5, 0.5, -0.5, 0.5], dtype=np.float32)
                    dy = np.array([-0.5, -0.5, 0.5, 0.5], dtype=np.float32)
                    rev_x = (j_q_f_w[:, None] + dx[None, :]).reshape(-1)
                    rev_y = (i_q_f_w[:, None] + dy[None, :]).reshape(-1)
                    revgrid_flat = revgrid_np.reshape(-1, 2)
                    revgrid_flat[tgt_flat, 0] = rev_x
                    revgrid_flat[tgt_flat, 1] = rev_y

                    if return_view_change:
                        src_h_flat = src_h.reshape(-1).astype(np.int64)
                        src_w_flat = src_w.reshape(-1).astype(np.int64)
                        src_h_pix = (src_h_flat.astype(np.float64) + 0.5) * stride_f
                        src_w_pix = (src_w_flat.astype(np.float64) + 0.5) * stride_f
                        _pix_new_hw, _zc, points_world, _xc = reproject_pixels_nd_to_nd(
                            pix_hw=np.stack([src_h_pix, src_w_pix], axis=-1),
                            depth_z=depth_lat[src_h_flat, src_w_flat].astype(np.float64),
                            K_old=np.asarray(rec["K"], dtype=np.float64),
                            T_old_w2c=np.asarray(rec["w2c"], dtype=np.float64),
                            K_new=np.asarray(query_K, dtype=np.float64),
                            T_new_w2c=np.asarray(query_w2c, dtype=np.float64),
                            eps=0.0,
                        )
                        vc = self._compute_view_change_for_points(
                            points_world,
                            np.asarray(rec["w2c"], dtype=np.float64),
                            np.asarray(query_w2c, dtype=np.float64),
                        )
                        valid_vc = np.isfinite(vc).all(axis=-1) & (vc[:, 0] > 0)
                        vc = np.where(valid_vc[:, None], vc, neutral_view_change)
                        view_change_np.reshape(-1, 3)[tgt_flat] = vc

            return (
                out_latent,
                _broadcast_p_to_l(valid_P_flat.view(H_pat, W_pat)),
                _broadcast_p_to_l(z_buf_P_flat.view(H_pat, W_pat)),
                revgrid_np,
                view_change_np,
                source_fid_np,
            )

        all_records = list(keyframe_records) + list(normal_records)
        splat_list: List[torch.Tensor] = []
        valid_list: List[torch.Tensor] = []
        zmap_list: List[torch.Tensor] = []
        revgrid_list: List[np.ndarray] = []
        view_change_list: List[np.ndarray] = []
        source_fid_list: List[np.ndarray] = []
        if int(fuse_block_size) == 2:
            splat_fn = _splat_one_block
            splat_scope = "materialize.fuse_candidates.group.splat_block_candidate"
        else:
            splat_fn = _splat_one
            splat_scope = "materialize.fuse_candidates.group.splat_candidate"
        for candidate_idx, rec in enumerate(all_records):
            with self._timing_scope(
                splat_scope,
                metadata={
                    "candidate_idx": int(candidate_idx),
                    "frame_id": int(rec["frame_id"]),
                    "fuse_block_size": int(fuse_block_size),
                },
            ):
                s, v, z, rg, vc, sf = splat_fn(rec)
            splat_list.append(s)
            valid_list.append(v)
            zmap_list.append(z)
            revgrid_list.append(rg)
            view_change_list.append(vc)
            source_fid_list.append(sf)

        revgrid_out = np.full((H_lat, W_lat, 2), -1.0, dtype=np.float32)
        view_change_out = np.empty((H_lat, W_lat, 3), dtype=np.float32)
        view_change_out[:] = neutral_view_change
        source_fid_out = np.full((H_lat, W_lat), -1, dtype=np.int64)

        def _maybe_pack_returns(fused_latent):
            return _pack_fuse_result(
                fused_latent,
                revgrid_out,
                view_change_out,
                source_fid_out[::2, ::2].copy(),
            )

        if fuse_mode == "masked_mean":
            with self._timing_scope(
                "materialize.fuse_candidates.group.combine.masked_mean",
                metadata={"candidate_count": len(splat_list)},
            ):
                invalid_masks = [~v for v in valid_list]
                fused_latent = self.masked_pixelwise_mean(splat_list, invalid_masks=invalid_masks)
            if return_revgrid or return_view_change or return_source_frame_ids:
                with self._timing_scope(
                    "materialize.fuse_candidates.group.revgrid.masked_mean",
                    metadata={"candidate_count": len(revgrid_list)},
                ):
                    for rg, vc, sf in zip(revgrid_list, view_change_list, source_fid_list):
                        empty = revgrid_out[..., 0] < 0
                        cand_valid = rg[..., 0] >= 0
                        write = empty & cand_valid
                        if np.any(write):
                            revgrid_out[write] = rg[write]
                            view_change_out[write] = vc[write]
                            source_fid_out[write] = sf[write]
            return _maybe_pack_returns(fused_latent)

        if fuse_mode == "fill_stop":
            with self._timing_scope(
                "materialize.fuse_candidates.group.combine.fill_stop",
                metadata={"candidate_count": len(splat_list)},
            ):
                fused_latent = torch.zeros_like(splat_list[0])
                filled = torch.zeros(H_lat, W_lat, device=device, dtype=torch.bool)
                max_candidates = (
                    len(splat_list)
                    if fill_candidate_limit is None
                    else max(1, min(int(fill_candidate_limit), len(splat_list)))
                )
                ratio_threshold = float(np.clip(fill_ratio_threshold, 0.0, 1.0))

                used = 0
                for splat, valid_mask, _zmap, rg, vc, sf in zip(
                    splat_list,
                    valid_list,
                    zmap_list,
                    revgrid_list,
                    view_change_list,
                    source_fid_list,
                ):
                    write = valid_mask & ~filled
                    if write.any():
                        fused_latent = torch.where(write.unsqueeze(0).unsqueeze(0), splat, fused_latent)
                        filled = filled | write
                        if return_revgrid or return_view_change or return_source_frame_ids:
                            write_np = write.detach().cpu().numpy().astype(bool)
                            rg_valid = rg[..., 0] >= 0
                            rev_write = write_np & rg_valid
                            if np.any(rev_write):
                                revgrid_out[rev_write] = rg[rev_write]
                                view_change_out[rev_write] = vc[rev_write]
                                source_fid_out[rev_write] = sf[rev_write]
                    used += 1
                    filled_ratio = float(filled.float().mean().item())
                    if used >= max_candidates or filled_ratio >= ratio_threshold:
                        break
            return _maybe_pack_returns(fused_latent)

        if fuse_mode == "fill_stop_zbuffer":
            with self._timing_scope(
                "materialize.fuse_candidates.group.combine.fill_stop_zbuffer",
                metadata={"candidate_count": len(splat_list), "z_pref": z_pref},
            ):
                fused_latent = torch.zeros_like(splat_list[0])
                filled = torch.zeros(H_lat, W_lat, device=device, dtype=torch.bool)
                init_z = float("inf") if z_pref == "near" else float("-inf")
                best_z = torch.full(
                    (H_lat, W_lat),
                    init_z,
                    device=device,
                    dtype=torch.float32,
                )
                max_candidates = (
                    len(splat_list)
                    if fill_candidate_limit is None
                    else max(1, min(int(fill_candidate_limit), len(splat_list)))
                )
                ratio_threshold = float(np.clip(fill_ratio_threshold, 0.0, 1.0))

                used = 0
                for splat, valid_mask, z_map, rg, vc, sf in zip(
                    splat_list,
                    valid_list,
                    zmap_list,
                    revgrid_list,
                    view_change_list,
                    source_fid_list,
                ):
                    if z_pref == "near":
                        z_for_cmp = torch.where(valid_mask, z_map, torch.full_like(z_map, float("inf")))
                        win = valid_mask & (z_for_cmp < best_z)
                    else:
                        z_for_cmp = torch.where(valid_mask, z_map, torch.full_like(z_map, float("-inf")))
                        win = valid_mask & (z_for_cmp > best_z)
                    if win.any():
                        fused_latent = torch.where(win.unsqueeze(0).unsqueeze(0), splat, fused_latent)
                        filled = filled | win
                        best_z = torch.where(win, z_for_cmp, best_z)
                        if return_revgrid or return_view_change or return_source_frame_ids:
                            win_np = win.detach().cpu().numpy().astype(bool)
                            rg_valid = rg[..., 0] >= 0
                            rev_write = win_np & rg_valid
                            if np.any(rev_write):
                                revgrid_out[rev_write] = rg[rev_write]
                                view_change_out[rev_write] = vc[rev_write]
                                source_fid_out[rev_write] = sf[rev_write]
                    used += 1
                    filled_ratio = float(filled.float().mean().item())
                    if used >= max_candidates or filled_ratio >= ratio_threshold:
                        break
            return _maybe_pack_returns(fused_latent)

        raise ValueError(f"Unsupported fuse_mode={fuse_mode}. Use 'masked_mean', 'fill_stop', or 'fill_stop_zbuffer'.")

    def masked_pixelwise_mean(self, xs, invalid_masks: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
        """
        Pixel-wise mean over multiple images with per-input invalid masks (True=invalid).

        Args:
            *xs: multiple tensors, each shaped (C, B, H, W) (or any same shape, but assumes channel is dim=0 for auto-mask).
            invalid_masks: optional sequence of masks, each True=invalid.
                - If provided, each mask can be (B,H,W) or broadcastable to (C,B,H,W).
                - If None, auto-generate as (x==0).all(dim=0) -> (B,H,W).

        Returns:
            out: (C,B,H,W) tensor where:
                - if a pixel has k valid inputs, output = sum(valid values)/k
                - if k==0, output = 0
        """
        if len(xs) == 0:
            raise ValueError("masked_pixelwise_mean requires at least one input tensor.")
        ref = xs[0]
        for i, x in enumerate(xs):
            if x.shape != ref.shape:
                raise ValueError(f"All inputs must have the same shape. xs[0]={ref.shape}, xs[{i}]={x.shape}")

        # build invalid masks (True=invalid)
        if invalid_masks is None:
            inv = [(x == 0).all(dim=0) for x in xs]  # (B,H,W)
        else:
            if len(invalid_masks) != len(xs):
                raise ValueError(f"invalid_masks length must match inputs: {len(invalid_masks)} vs {len(xs)}")
            inv = list(invalid_masks)

        # valid mask: True=valid
        # stack for vectorized reduction over N inputs
        x_stack = torch.stack(xs, dim=0)  # (N,C,B,H,W)

        # make valid masks broadcast to (N,1,B,H,W)
        valid_list = []
        for m in inv:
            # m: True=invalid -> valid = ~m
            v = ~m
            # allow (B,H,W) or (1,B,H,W) or (C,B,H,W)
            while v.dim() < ref.dim():
                v = v.unsqueeze(0)
            # now v is (something, B, H, W) or (C,B,H,W); we only need a single channel mask
            if v.shape[0] != 1 and v.shape[0] != ref.shape[0]:
                raise ValueError(f"Mask first dim must be 1 or C. Got mask shape {m.shape} -> broadcasted {v.shape}")
            if v.shape[0] == ref.shape[0]:
                v = v[:1]  # take one channel; assumes invalid/valid consistent across channels
            valid_list.append(v)

        valid_stack = torch.stack(valid_list, dim=0).to(dtype=x_stack.dtype)  # (N,1,B,H,W) float

        # weighted sum and count
        summed = (x_stack * valid_stack).sum(dim=0)  # (C,B,H,W)
        denom = valid_stack.sum(dim=0)  # (1,B,H,W)

        # mean where denom>0 else 0
        out = torch.where(denom > 0, summed / denom.clamp(min=1), torch.zeros_like(summed))
        return out

    def masked_average(self, a, b, maska, maskb):
        """
        a, b: (C,B,H,W) tensor
        maska, maskb: bool tensor, same shape as a/b
        """
        both = maska & maskb
        only_a = maska & ~maskb
        only_b = maskb & ~maska

        out = torch.zeros_like(a)
        out = torch.where(both, (a + b) * 0.5, out)
        out = torch.where(only_a, a, out)
        out = torch.where(only_b, b, out)

        return out

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _require_cache(self) -> SourceCache:
        if self._cache is None:
            raise RuntimeError("Source sequence has not been registered yet.")
        return self._cache

    def _rotation_geodesic_deg(self, R_prev: np.ndarray, R_curr: np.ndarray) -> float:
        """计算相邻两帧旋转的测地角（单位：度）。"""
        R_rel = R_prev.T @ R_curr
        cos_a = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_a)))

    def _compute_keyframe_labels_for_new_batch(self, c2w_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """为新注册批次计算硬关键帧标记与软关键分数。

        硬关键帧触发条件：
            1) 旋转变化超过阈值；或
            2) 平移变化超过阈值；或
            3) 距上次关键帧间隔达到上限（时间强制关键帧）。

        软关键分数范围 [0,1]，并在关键帧处设为 1.0。
        """
        c2w_new = np.asarray(c2w_new, dtype=np.float64)
        if c2w_new.ndim != 3 or c2w_new.shape[1:] != (4, 4):
            raise ValueError("c2w_new must have shape (T,4,4).")

        rot_thresh = max(float(self._keyframe_rot_thresh_deg), 1e-6)
        trans_thresh = max(float(self._keyframe_trans_thresh), 1e-9)
        max_interval = max(1, int(self._keyframe_max_interval))

        Tn = c2w_new.shape[0]
        hard_flags = np.zeros((Tn,), dtype=bool)
        soft_scores = np.zeros((Tn,), dtype=np.float32)

        # 从历史缓存衔接：保证跨批次关键帧逻辑连续
        if self._cache is None or self._cache.num_frames == 0:
            prev_pose = None
            frames_since_last_kf = max_interval
        else:
            prev_pose = self._cache.c2w[-1]
            key_indices = np.flatnonzero(self._cache.is_keyframe)
            if key_indices.size > 0:
                last_kf = int(key_indices[-1])
            else:
                last_kf = int(self._cache.num_frames - 1)
            frames_since_last_kf = int(self._cache.num_frames - 1 - last_kf)

        for idx in range(Tn):
            cur_pose = c2w_new[idx]

            if prev_pose is None:
                # 第一帧强制关键帧，作为后续分段锚点
                is_kf = True
                soft = 1.0
            else:
                R_prev = prev_pose[:3, :3]
                R_cur = cur_pose[:3, :3]
                rot_deg = self._rotation_geodesic_deg(R_prev, R_cur)

                c_prev = prev_pose[:3, 3]
                c_cur = cur_pose[:3, 3]
                trans_dist = float(np.linalg.norm(c_cur - c_prev))

                rot_ratio = rot_deg / rot_thresh
                trans_ratio = trans_dist / trans_thresh
                time_ratio = float(frames_since_last_kf + 1) / float(max_interval)

                # 软分数：运动与时间进度联合增长，关键帧处会被抬到 1.0
                soft = float(np.clip(max(rot_ratio, trans_ratio, time_ratio), 0.0, 1.0))

                is_kf = (
                    (rot_deg >= rot_thresh)
                    or (trans_dist >= trans_thresh)
                    or ((frames_since_last_kf + 1) >= max_interval)
                )
                if is_kf:
                    soft = 1.0

            hard_flags[idx] = is_kf
            soft_scores[idx] = np.float32(soft)

            if is_kf:
                frames_since_last_kf = 0
            else:
                frames_since_last_kf += 1
            prev_pose = cur_pose

        return hard_flags, soft_scores

    def _compute_group_motion_scores(self, w2c: np.ndarray) -> np.ndarray:
        """计算每帧所属4帧组的运动量分数.

        T = 1 + 4t 结构:
          - 帧 0: 单独帧（不属于任何组，赋中位数运动量）
          - 帧 1~4: 第0组
          - 帧 5~8: 第1组
          - ...

        运动量 = 组内首尾帧的旋转测地距离(度) + 平移距离

        Args:
            w2c: (T, 4, 4) 所有帧的 w2c 外参

        Returns:
            scores: (T,) 每帧的运动量分数，值越大表示所属组运动越剧烈
        """
        T = w2c.shape[0]
        scores = np.zeros(T, dtype=np.float64)

        def _cam_center(m: np.ndarray) -> np.ndarray:
            """从 w2c 矩阵提取世界坐标系下的相机中心"""
            R = m[:3, :3]
            t = m[:3, 3]
            return -R.T @ t

        def _rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
            """两个旋转矩阵之间的测地距离（度）"""
            R_rel = R1 @ R2.T
            cos_a = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
            return float(np.degrees(np.arccos(cos_a)))

        # 计算每个4帧组的运动量
        num_groups = (T - 1) // 4 if T > 1 else 0
        group_motions: List[float] = []

        for g in range(num_groups):
            start_idx = 1 + g * 4  # 组内第一帧
            end_idx = start_idx + 3  # 组内最后一帧
            if end_idx >= T:
                end_idx = T - 1

            # 旋转变化量
            R_start = w2c[start_idx, :3, :3]
            R_end = w2c[end_idx, :3, :3]
            rot_change = _rotation_angle_deg(R_start, R_end)

            # 平移变化量
            c_start = _cam_center(w2c[start_idx])
            c_end = _cam_center(w2c[end_idx])
            trans_change = float(np.linalg.norm(c_end - c_start))

            motion = rot_change + trans_change
            group_motions.append(motion)

            # 将运动量赋给组内所有帧
            for fi in range(start_idx, min(start_idx + 4, T)):
                scores[fi] = motion

        # 帧0特殊处理：赋中位数运动量（中性，既不特别偏好也不特别惩罚）
        if group_motions:
            scores[0] = float(np.median(group_motions))
        else:
            scores[0] = 0.0

        return scores

    def _select_candidates(
        self,
        base_local_indices: List[int],
        query_center: np.ndarray,
        max_candidates: Optional[int],
        distance_threshold: Optional[float],
    ) -> List[int]:
        """Select LOCAL candidate indices by distance to query center."""
        if not base_local_indices:
            return []
        cache = self._require_cache()
        indices = np.asarray(base_local_indices, dtype=int)
        centers = cache.camera_centers[indices]
        distances = np.linalg.norm(centers - query_center[None, :], axis=1)

        mask = np.ones_like(distances, dtype=bool)
        if distance_threshold is not None:
            mask &= distances <= float(distance_threshold)

        indices = indices[mask]
        distances = distances[mask]
        if indices.size == 0:
            return []

        if max_candidates is not None and max_candidates > 0 and indices.size > max_candidates:
            order = np.argsort(distances)
            indices = indices[order[: int(max_candidates)]]
        return indices.tolist()

    def _select_group_ids_by_vote(
        self,
        w2c: np.ndarray,
        query_extrinsics: np.ndarray,
        candidates_per_query_group: int = 1,
        keyframe_mask: Optional[np.ndarray] = None,
        keyframe_candidate_min_count: Optional[int] = 0,
        keyframe_candidate_max_count: Optional[int] = 0,
        keyframe_candidate_ratio: float = 0,
        angle_threshold: float = 30.0,
        distance_threshold: Optional[float] = None,
        temporal_threshold: Optional[int] = None,
        diversity_weight: float = DIVERSITY_WEIGHT,
        vote_weight: float = VOTE_WEIGHT,
        stability_weight: float = STABILITY__WEIGHT,
        first_frame_penalty: float = FIRST_FRAME_PENALTY,
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[List[int]]]:
        """按每 4 个 query 为一组做候选分组投票。

        Returns:
            selected_group_ids_per_group: 每个 query-group 的候选组 id 列表
            selected_frame_ids_per_group: 每个 query-group 的候选帧 id 列表
            keyframe_group_ids_per_group: 每个 query-group 中来自关键帧的候选组 id 列表
            keyframe_frame_ids_per_group: 每个 query-group 中来自关键帧的候选帧 id 列表
        """
        Q = query_extrinsics.shape[0]
        N = w2c.shape[0]
        candidates_per_query_group = max(1, int(candidates_per_query_group))
        keyframe_candidate_ratio = float(np.clip(keyframe_candidate_ratio, 0.0, 1.0))

        if keyframe_mask is None:
            # 未提供关键帧信息时，不强制关键帧比例
            keyframe_mask = np.zeros((N,), dtype=bool)
            keyframe_candidate_ratio = 0.0
        else:
            keyframe_mask = np.asarray(keyframe_mask, dtype=bool)
            if keyframe_mask.shape[0] != N:
                raise ValueError(f"keyframe_mask length must match frame count {N}, got {keyframe_mask.shape[0]}.")

        # 帧索引映射到候选组：帧0→组0，帧1~4→组1，帧5~8→组2...
        def _frame_to_group(idx: int) -> int:
            if idx == 0:
                return 0
            return (idx - 1) // 4 + 1

        # 预计算稳定性分（按组运动量归一化）
        group_motions = self._compute_group_motion_scores(w2c.copy())
        gm_min, gm_max = group_motions.min(), group_motions.max()
        gm_range = gm_max - gm_min + 1e-9
        group_motions_norm = (group_motions - gm_min) / gm_range

        # 先为每个 query 预计算候选排序（角度筛选 + 位移排序）
        angle_topk_for_ranking = max(10, N // 4)
        all_candidates: List[List[Tuple[int, float, float]]] = []
        for qi in range(Q):
            ranked = self._rank_candidates_by_angle_then_translation(
                w2c,
                query_extrinsics[qi : qi + 1],
                angle_topk=angle_topk_for_ranking,
                angle_threshold=angle_threshold,
                query_frame_idx=qi,
                distance_threshold=distance_threshold,
                temporal_threshold=temporal_threshold,
            )
            all_candidates.append(ranked)

        # 每 4 个 query 组成一个 query-group
        assert Q % 4 == 0, f"query_extrinsics 数量必须是4的倍数, 当前 Q={Q}"
        num_query_groups = Q // 4

        selected_group_ids: List[List[int]] = []
        selected_frame_ids: List[List[int]] = []
        keyframe_group_ids: List[List[int]] = []
        keyframe_frame_ids: List[List[int]] = []
        for qg in range(num_query_groups):
            q_start = qg * 4
            q_end = q_start + 4

            from collections import defaultdict

            group_votes: Dict[int, float] = defaultdict(float)
            group_hit_count: Dict[int, int] = defaultdict(int)
            # gid -> (frame_idx, angle, dist, best_sub_score)
            group_best_frame: Dict[int, Tuple[int, float, float, float]] = {}

            # 2a) 子查询投票
            for qi in range(q_start, q_end):
                candidates = all_candidates[qi]
                if not candidates:
                    continue

                dists_arr = np.array([c[2] for c in candidates])
                dist_min, dist_max = dists_arr.min(), dists_arr.max()
                dist_range = dist_max - dist_min + 1e-9

                for idx, angle, dist in candidates:
                    match_score = 1.0 - (dist - dist_min) / dist_range
                    stability_score = 1.0 - group_motions_norm[idx]
                    fp = first_frame_penalty if idx == 0 else 0.0
                    sub_score = match_score + stability_weight * stability_score + fp

                    gid = _frame_to_group(idx)
                    group_votes[gid] += sub_score
                    group_hit_count[gid] += 1

                    if gid not in group_best_frame or sub_score > group_best_frame[gid][3]:
                        group_best_frame[gid] = (idx, angle, dist, sub_score)

            if not group_votes:
                selected_group_ids.append([-1] * candidates_per_query_group)
                selected_frame_ids.append([-1] * candidates_per_query_group)
                keyframe_group_ids.append([-1] * candidates_per_query_group)
                keyframe_frame_ids.append([-1] * candidates_per_query_group)
                continue

            # 2b) 组级综合打分并贪心选组（支持每组输出多个候选）
            all_gids = list(group_votes.keys())

            vote_scores = np.array([group_votes[g] for g in all_gids])
            vote_min, vote_max = vote_scores.min(), vote_scores.max()
            vote_range = vote_max - vote_min + 1e-9

            # 新规则：
            # 1) 4个query先各自贡献候选，组内允许重复命中同一个gid；
            # 2) 最终按票数(众数)从高到低排序；
            # 3) 同票时按综合分(final_score)打破平局。
            gid_metrics: List[Tuple[int, int, float]] = []
            for gid in all_gids:
                vote_score_norm = (group_votes[gid] - vote_min) / vote_range
                hit_ratio = group_hit_count[gid] / 4.0

                # 组间完全独立：不再使用跨组历史选择约束。
                # diversity_weight 保留参数兼容，这里给常数1.0，不引入跨组抑制。
                diversity_score = 1.0

                repr_idx = group_best_frame[gid][0]
                stability_score = 1.0 - group_motions_norm[repr_idx]
                fp = first_frame_penalty if gid == 0 else 0.0

                final_score = (
                    vote_score_norm
                    + vote_weight * hit_ratio
                    + diversity_weight * diversity_score
                    + stability_weight * stability_score
                    + fp
                )
                gid_metrics.append((gid, int(group_hit_count[gid]), float(final_score)))

            # 先按票数降序，再按综合分降序
            gid_metrics.sort(key=lambda x: (-x[1], -x[2]))

            # 关键帧数量双阈值：最少数量 + 最大数量
            # - 若未显式设置 min_count，则退回 ratio 行为，保证兼容旧调用
            required_keyframes = (
                int(np.ceil(candidates_per_query_group * keyframe_candidate_ratio))
                if keyframe_candidate_min_count is None
                else int(keyframe_candidate_min_count)
            )
            max_keyframes = (
                candidates_per_query_group
                if keyframe_candidate_max_count is None
                else int(keyframe_candidate_max_count)
            )

            required_keyframes = max(0, min(required_keyframes, candidates_per_query_group))
            max_keyframes = max(0, min(max_keyframes, candidates_per_query_group))
            # 若配置冲突（max < min），以满足最少数量为优先
            if max_keyframes < required_keyframes:
                max_keyframes = required_keyframes

            keyframe_metrics = [m for m in gid_metrics if keyframe_mask[int(group_best_frame[m[0]][0])]]

            chosen_gids: List[int] = []
            chosen_kf_count = 0

            # 1) 先满足关键帧硬配额
            for m in keyframe_metrics:
                if len(chosen_gids) >= required_keyframes:
                    break
                chosen_gids.append(int(m[0]))
                chosen_kf_count += 1

            # 2) 再按总排序补齐到 candidates_per_query_group，同时不超过关键帧上限
            for m in gid_metrics:
                gid = int(m[0])
                if gid in chosen_gids:
                    continue
                if len(chosen_gids) >= candidates_per_query_group:
                    break

                frame_id = int(group_best_frame[gid][0])
                is_kf_frame = bool(keyframe_mask[frame_id])
                if is_kf_frame and chosen_kf_count >= max_keyframes:
                    continue

                chosen_gids.append(gid)
                if is_kf_frame:
                    chosen_kf_count += 1

            picked_group_ids: List[int] = chosen_gids
            picked_frame_ids: List[int] = [int(group_best_frame[gid][0]) for gid in picked_group_ids]

            # 记录“来自关键帧”的子集（用于下游单独分析/可视化）
            picked_kf_group_ids: List[int] = []
            picked_kf_frame_ids: List[int] = []
            for gid in picked_group_ids:
                frame_id = int(group_best_frame[gid][0])
                if keyframe_mask[frame_id]:
                    picked_kf_group_ids.append(gid)
                    picked_kf_frame_ids.append(frame_id)

            # 不足 candidates_per_query_group 时用 -1 补齐，保持输出形状稳定
            while len(picked_group_ids) < candidates_per_query_group:
                picked_group_ids.append(-1)
                picked_frame_ids.append(-1)

            # 关键帧列表也补齐为固定长度，便于下游统一处理
            while len(picked_kf_group_ids) < candidates_per_query_group:
                picked_kf_group_ids.append(-1)
                picked_kf_frame_ids.append(-1)

            selected_group_ids.append(picked_group_ids)
            selected_frame_ids.append(picked_frame_ids)
            keyframe_group_ids.append(picked_kf_group_ids)
            keyframe_frame_ids.append(picked_kf_frame_ids)

        return (
            selected_group_ids,
            selected_frame_ids,
            keyframe_group_ids,
            keyframe_frame_ids,
        )

    def find_best_match_by_angle_then_translation(
        self,
        w2c_candidates: np.ndarray,
        w2c_query: np.ndarray,
        *,
        angle_topk: int = 5,
        angle_threshold: Optional[float] = None,
    ) -> Tuple[int, float, float]:
        """根据朝向角度和位移距离，从候选外参中选出最佳匹配.

        策略:
            1. 计算每个候选相机与查询相机的光轴夹角，取夹角最小的一批候选；
            2. 在这批候选中，选取平移距离最近的那一个作为最终结果。

        Args:
            w2c_candidates: 候选 w2c 外参 (N, 4, 4).
            w2c_query: 查询 w2c 外参 (1, 4, 4) 或 (4, 4).
            angle_topk: 按夹角取前 k 个候选（默认 5）.
            angle_threshold: 若指定，则改用角度阈值筛选（单位：度），
                               此时 angle_topk 仅作为最少保留数量的下限。

        Returns:
            (best_idx, best_angle_deg, best_dist):
                best_idx      – 在 w2c_candidates 中的索引
                best_angle_deg – 该候选与查询的光轴夹角（度）
                best_dist      – 该候选与查询的平移距离
        """
        w2c_candidates = np.asarray(w2c_candidates, dtype=np.float64)
        w2c_query = np.asarray(w2c_query, dtype=np.float64)
        if w2c_query.ndim == 3:
            w2c_query = w2c_query[0]  # (4,4)
        assert w2c_candidates.ndim == 3 and w2c_candidates.shape[1:] == (4, 4)
        assert w2c_query.shape == (4, 4)

        N = w2c_candidates.shape[0]
        if N == 0:
            raise ValueError("候选外参数量不能为 0")

        # ---- 提取光轴方向（相机坐标系 z 轴在世界坐标系中的方向）----
        # w2c 的旋转部分 R = w2c[:3,:3]，其转置即 c2w 旋转
        # 相机 z 轴在世界系中 = R^T @ [0,0,1] = R 的第三列（转置后第三行）
        # 即 c2w[:3, 2] = w2c[:3, 2]（因为 R^T 的第三列 = R 的第三行）
        # 更准确：c2w = inv(w2c)，对于刚体变换 c2w[:3,:3] = R^T
        # 光轴 = c2w[:3, 2] = R^T[:, 2] = R[2, :3]（w2c 旋转矩阵的第三行）
        dirs_cand = w2c_candidates[:, 2, :3]  # (N, 3) — 每个候选相机的光轴方向
        dir_query = w2c_query[2, :3]  # (3,)   — 查询相机的光轴方向

        # 归一化
        dirs_cand = dirs_cand / (np.linalg.norm(dirs_cand, axis=1, keepdims=True) + 1e-12)
        dir_query = dir_query / (np.linalg.norm(dir_query) + 1e-12)

        # 计算夹角（弧度 → 度）
        cos_angles = np.clip(dirs_cand @ dir_query, -1.0, 1.0)  # (N,)
        angles_rad = np.arccos(cos_angles)
        angles_deg = np.degrees(angles_rad)  # (N,)

        # ---- 第一步：按夹角筛选候选 ----
        if angle_threshold is not None:
            # 角度阈值模式：选所有角度 <= 阈值的，至少保留 angle_topk 个
            mask = angles_deg <= float(angle_threshold)
            if mask.sum() < min(angle_topk, N):
                # 不够时退化为 topk
                topk_indices = np.argsort(angles_deg)[: min(angle_topk, N)]
            else:
                topk_indices = np.where(mask)[0]
        else:
            # 纯 topk 模式
            k = min(angle_topk, N)
            topk_indices = np.argsort(angles_deg)[:k]

        # ---- 第二步：在候选中按平移距离选最近 ----
        # w2c 的平移部分 t = w2c[:3, 3]；相机中心 = -R^T @ t
        def _cam_center(w2c_mat: np.ndarray) -> np.ndarray:
            R = w2c_mat[:3, :3]
            t = w2c_mat[:3, 3]
            return -R.T @ t

        center_query = _cam_center(w2c_query)  # (3,)
        centers_cand = np.stack([_cam_center(w2c_candidates[i]) for i in topk_indices], axis=0)  # (K, 3)

        dists = np.linalg.norm(centers_cand - center_query[None, :], axis=1)  # (K,)
        best_local = int(np.argmin(dists))
        best_idx = int(topk_indices[best_local])

        return best_idx, float(angles_deg[best_idx]), float(dists[best_local])

    def _rank_candidates_by_angle_then_translation(
        self,
        w2c_candidates: np.ndarray,
        w2c_query: np.ndarray,
        *,
        angle_topk: int = 10,
        angle_threshold: Optional[float] = None,
        query_frame_idx: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        temporal_threshold: Optional[int] = None,
    ) -> List[Tuple[int, float, float]]:
        """返回按角度筛选后、按平移距离升序排列的候选列表.

        与 find_best_match_by_angle_then_translation 逻辑一致，
        但返回全部候选而非仅最优，供外部多样性选择使用。

        Returns:
            按距离升序排列的 [(idx, angle_deg, dist), ...]
        """
        w2c_candidates = np.asarray(w2c_candidates, dtype=np.float64)
        w2c_query = np.asarray(w2c_query, dtype=np.float64)
        if w2c_query.ndim == 3:
            w2c_query = w2c_query[0]
        assert w2c_candidates.ndim == 3 and w2c_candidates.shape[1:] == (4, 4)
        assert w2c_query.shape == (4, 4)

        N = w2c_candidates.shape[0]
        if N == 0:
            return []

        # 提取光轴方向
        dirs_cand = w2c_candidates[:, 2, :3]  # (N, 3)
        dir_query = w2c_query[2, :3]  # (3,)
        dirs_cand = dirs_cand / (np.linalg.norm(dirs_cand, axis=1, keepdims=True) + 1e-12)
        dir_query = dir_query / (np.linalg.norm(dir_query) + 1e-12)

        # 夹角
        cos_angles = np.clip(dirs_cand @ dir_query, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(cos_angles))

        # 角度筛选
        if angle_threshold is not None:
            mask = angles_deg <= float(angle_threshold)
            if mask.sum() < min(angle_topk, N):
                topk_indices = np.argsort(angles_deg)[: min(angle_topk, N)]
            else:
                topk_indices = np.where(mask)[0]
        else:
            k = min(angle_topk, N)
            topk_indices = np.argsort(angles_deg)[:k]

        # 计算相机中心距离
        def _cam_center(w2c_mat: np.ndarray) -> np.ndarray:
            R = w2c_mat[:3, :3]
            t = w2c_mat[:3, 3]
            return -R.T @ t

        center_query = _cam_center(w2c_query)
        centers_cand = np.stack([_cam_center(w2c_candidates[i]) for i in topk_indices], axis=0)
        # Distances are in the same units as the extrinsics' translation vectors
        # (typically meters). The legacy `* 0.01` factor was an unexplained
        # scale that made `distance_threshold` operate in centimeters-ish units
        # without saying so; callers passing `distance_threshold=30` expected
        # 30 meters but were actually filtering at ~0.3 meters. Removed.
        dists = np.linalg.norm(centers_cand - center_query[None, :], axis=1)

        # 过滤1：空间距离阈值（超过阈值的候选直接丢弃）
        valid_mask = np.ones_like(dists, dtype=bool)
        if distance_threshold is not None:
            valid_mask &= dists <= float(distance_threshold)

        # 过滤2：时间阈值（离当前 query 太老的候选丢弃）
        # 仅过滤“更老”的候选：query_frame_idx - candidate_idx > temporal_threshold
        if query_frame_idx is not None and temporal_threshold is not None:
            valid_mask &= (int(query_frame_idx) - topk_indices) <= int(temporal_threshold)

        if not np.any(valid_mask):
            return []

        topk_indices = topk_indices[valid_mask]
        dists = dists[valid_mask]

        # 按距离升序排列
        order = np.argsort(dists)
        ranked: List[Tuple[int, float, float]] = []
        for o in order:
            gi = int(topk_indices[o])
            ranked.append((gi, float(angles_deg[gi]), float(dists[o])))
        return ranked
