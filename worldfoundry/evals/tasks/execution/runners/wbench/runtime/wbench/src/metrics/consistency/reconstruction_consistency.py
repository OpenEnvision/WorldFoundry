"""
Reconstruction consistency — DA3 depth estimation + cross-frame reprojection.

Two sub-metrics:
- geometric_consistency: depth reprojection relative error -> 1/(1+mean_rel_err)
- photometric_consistency: pixel-level reprojection PSNR

Requires pre-computed DA3 cache (depth.npy, extrinsics.npy, intrinsics.npy).
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
import torch

METRIC_NAME = "reconstruction_consistency"


def compute_case(
    video_path: str,
    da3_cache_dir: str,
    fps: float = 3.0,
    device: str = "cuda",
    max_pts_per_frame: int = 5000,
) -> dict[str, Any]:
    """Compute geometric + photometric consistency for a single video.

    Args:
        video_path: Path to video
        da3_cache_dir: Directory with depth.npy, extrinsics.npy, intrinsics.npy
        fps: Frame sampling rate (must match DA3 inference fps)
        device: CUDA device
        max_pts_per_frame: Max points per frame for point cloud

    Returns:
        Dict with score, geometric_consistency, photometric_psnr, etc.
    """
    if not os.path.isdir(da3_cache_dir):
        return {"score": None, "error": f"DA3 cache not found: {da3_cache_dir}"}

    depth_path = os.path.join(da3_cache_dir, "depth.npy")
    ext_path = os.path.join(da3_cache_dir, "extrinsics.npy")
    int_path = os.path.join(da3_cache_dir, "intrinsics.npy")

    for p in [depth_path, ext_path, int_path]:
        if not os.path.exists(p):
            return {"score": None, "error": f"Missing: {p}"}

    try:
        depth = np.load(depth_path)
        extrinsics = np.load(ext_path)
        intrinsics = np.load(int_path)

        frames_resized = _extract_frames_resized(video_path, depth.shape, fps)
        if frames_resized is None:
            return {"score": None, "error": "Failed to extract frames"}

        N = min(len(depth), len(extrinsics), len(intrinsics), len(frames_resized))
        depth = depth[:N]
        extrinsics = extrinsics[:N]
        intrinsics = intrinsics[:N]
        frames_resized = frames_resized[:N]

        world_pts, colors, source_ids = _build_point_cloud(
            depth, extrinsics, intrinsics, frames_resized, device=device, max_pts_per_frame=max_pts_per_frame
        )

        if world_pts is None:
            return {"score": None, "error": "Point cloud construction failed"}

        geo_result, photo_result = _compute_metrics(
            world_pts, colors, source_ids, depth, extrinsics, intrinsics, frames_resized, device
        )

        geo_score = geo_result.get("geometric_consistency", 0.0)
        psnr = photo_result.get("photo_mean_psnr", 0.0)

        return {
            "score": round(geo_score, 4),
            "details": {
                "geometric_consistency": round(geo_score, 4),
                "photometric_psnr": round(psnr, 2),
                "geo_score_100": round(geo_score * 100, 2),
                "psnr_score_100": round(min(100.0, max(0.0, (psnr - 10) / 15 * 100)), 2),
                "num_frames": N,
            },
            "params": {"fps": fps, "method": "da3_depth_reprojection", "max_pts_per_frame": max_pts_per_frame},
            "error": None,
        }
    except Exception as e:
        return {"score": None, "error": f"{type(e).__name__}: {e}"}


def _extract_frames_resized(video_path: str, depth_shape, fps: float) -> np.ndarray | None:
    """Extract frames and resize to match DA3 depth resolution."""
    N, H, W = depth_shape
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps <= 0 or total <= 0:
        cap.release()
        return None

    step = 1 if fps <= 0 or fps >= video_fps else max(1, round(video_fps / fps))
    indices = list(range(0, total, step))[:N]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (W, H))
        frames.append(frame_resized)
    cap.release()

    if len(frames) < 3:
        return None
    return np.array(frames, dtype=np.uint8)


def _build_point_cloud(
    depth, extrinsics, intrinsics, frames_resized, device="cuda", max_pts_per_frame=5000, conf=None, conf_threshold=0.5
):
    """Build world-space point cloud from depth maps."""
    N, H, W = depth.shape
    all_pts, all_colors, all_ids = [], [], []

    for i in range(N):
        d = torch.from_numpy(depth[i]).float().to(device)
        K = torch.from_numpy(intrinsics[i]).double().to(device)
        E = torch.from_numpy(extrinsics[i]).double().to(device)
        R, t = E[:3, :3], E[:3, 3]

        mask = d > 1e-6
        if conf is not None:
            c_mask = torch.from_numpy(conf[i]).float().to(device)
            mask = mask & (c_mask > conf_threshold)

        if mask.sum() == 0:
            continue

        ys, xs = torch.where(mask)
        ds = d[mask].double()

        if len(ds) > max_pts_per_frame:
            idx = torch.randperm(len(ds), device=device)[:max_pts_per_frame]
            xs, ys, ds = xs[idx], ys[idx], ds[idx]

        pixels_h = torch.stack(
            [xs.double(), ys.double(), torch.ones_like(xs, dtype=torch.float64, device=device)], dim=0
        )

        K_inv = torch.linalg.inv(K)
        cam_pts = ds.unsqueeze(0) * (K_inv @ pixels_h)
        world = R.T @ (cam_pts - t.unsqueeze(1))

        all_pts.append(world.T.float())
        colors_i = torch.from_numpy(frames_resized[i]).float().to(device)
        all_colors.append(colors_i[ys.long(), xs.long()])
        all_ids.append(torch.full((world.shape[1],), i, dtype=torch.long, device=device))

    if not all_pts:
        return None, None, None
    return torch.cat(all_pts), torch.cat(all_colors), torch.cat(all_ids)


def _compute_metrics(world_pts, colors, source_ids, depth, extrinsics, intrinsics, frames_resized, device="cuda"):
    """Compute geometric + photometric consistency via cross-frame reprojection."""
    N, H, W = depth.shape
    depth_t = torch.from_numpy(depth).float().to(device)
    frames_t = torch.from_numpy(frames_resized).float().to(device)
    ext_t = torch.from_numpy(extrinsics).double().to(device)
    int_t = torch.from_numpy(intrinsics).double().to(device)
    pts = world_pts.double().to(device)
    frame_masks = [source_ids == i for i in range(N)]

    geo_errors, photo_psnrs = [], []

    for i in range(N):
        other_mask = ~frame_masks[i]
        pts_other = pts[other_mask]
        col_other = colors[other_mask]

        if pts_other.shape[0] == 0:
            geo_errors.append(float("inf"))
            photo_psnrs.append(0.0)
            continue

        R, t, K = ext_t[i, :3, :3], ext_t[i, :3, 3], int_t[i]
        cam = R @ pts_other.T + t.unsqueeze(1)
        front = cam[2, :] > 1e-6
        cam_f, col_f = cam[:, front], col_other[front]

        if cam_f.shape[1] == 0:
            geo_errors.append(float("inf"))
            photo_psnrs.append(0.0)
            continue

        proj = K @ cam_f
        u = (proj[0] / proj[2]).float()
        v = (proj[1] / proj[2]).float()
        z = cam_f[2].float()

        valid = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        u_v, v_v, z_v, c_v = u[valid], v[valid], z[valid], col_f[valid]

        if u_v.shape[0] == 0:
            geo_errors.append(float("inf"))
            photo_psnrs.append(0.0)
            continue

        u_int = u_v.round().long().clamp(0, W - 1)
        v_int = v_v.round().long().clamp(0, H - 1)

        # Geometric: depth reprojection error
        d_pred = depth_t[i, v_int, u_int]
        d_valid = d_pred > 1e-6
        if d_valid.sum() > 0:
            rel_err = (z_v[d_valid] - d_pred[d_valid]).abs() / d_pred[d_valid]
            geo_errors.append(float(rel_err.median()))
        else:
            geo_errors.append(float("inf"))

        # Photometric: pixel reprojection
        sort_idx = z_v.argsort(descending=True)
        rendered = torch.zeros(H, W, 3, device=device)
        z_buf = torch.full((H, W), float("inf"), device=device)
        rendered[v_int[sort_idx], u_int[sort_idx]] = c_v[sort_idx].float()
        z_buf[v_int[sort_idx], u_int[sort_idx]] = z_v[sort_idx]

        has_render = z_buf < float("inf")
        if has_render.sum() < 100:
            photo_psnrs.append(0.0)
            continue

        diff = (rendered[has_render] - frames_t[i][has_render]) / 255.0
        mse = float((diff**2).mean())
        photo_psnrs.append(float(-10 * np.log10(mse + 1e-10)))

    # Aggregate
    geo_arr = np.array(geo_errors)
    valid_geo = geo_arr[np.isfinite(geo_arr)]
    if len(valid_geo) > 0:
        geo_result = {
            "geometric_consistency": 1.0 / (1.0 + float(np.mean(valid_geo))),
            "geo_mean_rel_error": float(np.mean(valid_geo)),
            "valid_frames": int(np.isfinite(geo_arr).sum()),
        }
    else:
        geo_result = {"geometric_consistency": 0.0, "error": "all frames invalid"}

    psnr_arr = np.array(photo_psnrs)
    valid_psnr = psnr_arr[psnr_arr > 0]
    photo_result = {
        "photo_mean_psnr": float(np.mean(valid_psnr)) if len(valid_psnr) > 0 else 0.0,
        "valid_frames": int((psnr_arr > 0).sum()),
    }

    return geo_result, photo_result
