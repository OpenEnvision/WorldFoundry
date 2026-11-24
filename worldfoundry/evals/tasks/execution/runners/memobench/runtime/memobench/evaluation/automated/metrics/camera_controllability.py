from __future__ import annotations

import os
import re
import tempfile

import cv2
import numpy as np
import pandas as pd

GT_SYN_PROCESSED = "data/Synthetic_processed"
GT_SYN_RAW = "data/Synthetic_Raw"
GT_REAL_POSES = "data/mapanything/outputs/real"


def _Rx_LH(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=np.float64)


def _Ry_LH(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=np.float64)


def _Rz_LH(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)


_FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])


def _csv_to_c2w(csv_path: str) -> np.ndarray:
    """
    Read a UE5 camera CSV and return (N, 4, 4) c2w_hybrid poses.
    Convention: c2w_ue @ flip_yz  (OpenCV camera frame, UE5 world units).
    """
    df = pd.read_csv(csv_path)
    N = len(df)
    poses = np.zeros((N, 4, 4), dtype=np.float64)
    for i, row in df.iterrows():
        R = _Rz_LH(row["rot_z"]) @ _Ry_LH(row["rot_y"]) @ _Rx_LH(row["rot_x"])
        c2w_ue = np.eye(4, dtype=np.float64)
        c2w_ue[:3, :3] = R
        c2w_ue[:3, 3] = [row["loc_x"], row["loc_y"], row["loc_z"]]
        poses[i] = c2w_ue @ _FLIP_YZ
    return poses


def _resample(poses: np.ndarray, n_gen: int) -> np.ndarray:
    """Uniformly resample (N, 4, 4) pose array to n_gen frames."""
    N = len(poses)
    if n_gen == N:
        return poses
    indices = [round(i * (N - 1) / (n_gen - 1)) for i in range(n_gen)]
    return poses[indices]


def _clip_num(clip_id: str) -> int:
    """Extract numeric suffix: 'Barnyard_007' → 7."""
    m = re.search(r"_(\d+)$", clip_id)
    if not m:
        raise ValueError(f"Cannot parse clip number from '{clip_id}'")
    return int(m.group(1))


def load_gt_poses_synthetic(scene: str, clip_id: str, n_gen: int) -> np.ndarray | None:
    """
    Load GT c2w poses for a synthetic clip, resampled to n_gen frames.

    Tries Synthetic_processed/poses.npy first (fast path for 81-frame models).
    Falls back to Synthetic_Raw CSV conversion for other models.
    Returns (n_gen, 4, 4) float64 or None.
    """
    # Fast path: pre-processed poses.npy
    proc_path = os.path.join(GT_SYN_PROCESSED, scene, clip_id, "poses.npy")
    if os.path.exists(proc_path):
        poses = np.load(proc_path).astype(np.float64)
        return _resample(poses, n_gen)

    # Fallback: raw UE5 CSV
    clip_n = _clip_num(clip_id)
    csv_path = os.path.join(GT_SYN_RAW, scene, "cameras", f"camera_full{clip_n}.csv")
    if not os.path.exists(csv_path):
        return None
    poses = _csv_to_c2w(csv_path)
    return _resample(poses, n_gen)


def load_gt_poses_real(clip_id: str, n_gen: int) -> np.ndarray | None:
    """
    Load mega-sam cam_c2w for a real clip, resampled to n_gen frames.
    Returns (n_gen, 4, 4) float64 or None.
    """
    npz_path = os.path.join(GT_REAL_POSES, f"{clip_id}_mapanything.npz")
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    poses = data["cam_c2w"].astype(np.float64)
    return _resample(poses, n_gen)


def load_intrinsics_synthetic(scene: str, clip_id: str) -> np.ndarray | None:
    """Load 3×3 intrinsics matrix for a synthetic clip."""
    path = os.path.join(GT_SYN_PROCESSED, scene, clip_id, "intrinsics.npy")
    if os.path.exists(path):
        K = np.load(path).astype(np.float64)
        if K.ndim == 2 and K.shape[1] == 4:
            fx, fy, cx, cy = K[0]
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        if K.ndim == 1 and K.shape[0] == 4:
            fx, fy, cx, cy = K
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return K  # already (3, 3)

    # Fallback: read fx, fy, cx, cy from first row of raw CSV
    clip_n = _clip_num(clip_id)
    csv_path = os.path.join(GT_SYN_RAW, scene, "cameras", f"camera_full{clip_n}.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, nrows=1)
    row = df.iloc[0]
    fx, fy = float(row["fx"]), float(row["fy"])
    cx, cy = float(row["cx"]), float(row["cy"])
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def load_intrinsics_real(clip_id: str) -> np.ndarray | None:
    """Load 3×3 intrinsics matrix from mega-sam output for a real clip."""
    npz_path = os.path.join(GT_REAL_POSES, f"{clip_id}_mapanything.npz")
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    return data["intrinsic"].astype(np.float64)


_mapanything_model = None
_mapanything_device = None


def _load_mapanything(device: str | None = None):
    global _mapanything_model, _mapanything_device
    if _mapanything_model is not None:
        return _mapanything_model, _mapanything_device
    import torch
    from mapanything.models import MapAnything

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _mapanything_device = device
    # Apache 2.0 licensed variant
    _mapanything_model = MapAnything.from_pretrained("facebook/map-anything-apache").to(device)
    _mapanything_model.eval()
    return _mapanything_model, _mapanything_device


def _estimate_poses_mapanything(
    frames,
    sample_indices: list[int],
    device: str,
) -> np.ndarray | None:
    """
    Run MapAnything on a list of sampled BGR frames.
    Returns (K, 4, 4) c2w poses in OpenCV convention, or None on failure.
    """
    import torch
    from mapanything.utils.image import load_images

    model, dev = _load_mapanything(device)

    # Write frames to a temp directory as PNG so load_images can read them
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for rank, idx in enumerate(sample_indices):
            bgr = frames.get(idx)
            path = os.path.join(tmp, f"{rank:05d}.png")
            cv2.imwrite(path, bgr)
            paths.append(path)

        views = load_images(paths)

    with torch.no_grad():
        outputs = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
        )

    poses = []
    for pred in outputs:
        p = pred["camera_poses"][0].cpu().numpy()  # (4, 4)
        poses.append(p)

    if not poses:
        return None
    return np.stack(poses, axis=0).astype(np.float64)  # (K, 4, 4)


def _relative_rotations(poses: np.ndarray) -> list[np.ndarray]:
    """Return (N-1) relative rotation matrices from absolute c2w sequence."""
    rels = []
    for i in range(len(poses) - 1):
        R0 = poses[i, :3, :3]
        R1 = poses[i + 1, :3, :3]
        rels.append(R0.T @ R1)
    return rels


def _geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angular distance between two rotation matrices (degrees)."""
    R_diff = R1.T @ R2
    trace = float(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace)))


def camera_controllability_score(
    frames,
    gt_poses: np.ndarray,
    K: np.ndarray,  # kept for API compatibility, not used by MapAnything
    device: str | None = None,
    tau_deg: float = 15.0,
) -> dict:
    """
    Camera Controllability: ATE-based rotation coverage score.

    Measures what fraction of the GT camera trajectory the generated video
    actually follows, using Absolute Trajectory Error (ATE) on rotations:

      coverage = max(0, 1 - ATE_rot / total_GT_rotation)

    where:
      ATE_rot          = RMSE of per-frame geodesic error after first-frame
                         alignment (degrees)
      total_GT_rotation = geodesic distance from GT frame-0 to GT frame-N
                         (total end-to-end rotation of the GT trajectory)

    Interpretation:
      1.0 (100%) — generated camera perfectly follows GT trajectory
      0.0 (  0%) — static camera or completely wrong trajectory

    All N generated frames are passed to MapAnything in a single forward pass.

    Parameters
    ----------
    frames    : FrameReader — generated video frames
    gt_poses  : (N, 4, 4) array of c2w poses aligned to gen frames
    K         : (3, 3) intrinsics (kept for API compat, unused)
    tau_deg   : unused, kept for API compatibility
    min_gt_rotation_deg : minimum GT rotation denominator (avoids div-by-zero
                          for near-static GT trajectories)

    Returns
    -------
    dict with keys:
      camera_controllability  : float in [0, 1],  higher = better
      ate_rot_deg             : ATE rotation RMSE in degrees (lower = better)
      total_gt_rotation_deg   : end-to-end GT rotation in degrees
      mean_rot_error_deg      : mean frame-to-frame geodesic error (diagnostic)
    """
    N = frames.num_frames
    sample_indices = list(range(N))

    # Estimate poses from generated frames
    est_poses = _estimate_poses_mapanything(frames, sample_indices, device)
    if est_poses is None or len(est_poses) < 2:
        return {
            "camera_controllability": None,
            "ate_rot_deg": None,
            "total_gt_rotation_deg": None,
            "mean_rot_error_deg": None,
        }

    # GT poses are already aligned to all N gen frames
    gt_sampled = gt_poses
    n = min(len(gt_sampled), len(est_poses))
    R_gt = [gt_sampled[i][:3, :3] for i in range(n)]
    R_est = [est_poses[i][:3, :3] for i in range(n)]

    # Align first frame: R_align maps estimated frame-0 onto GT frame-0
    R_align = R_gt[0] @ R_est[0].T
    ate_errors = [_geodesic_deg(R_align @ R_est[i], R_gt[i]) for i in range(n)]
    ate_rot = float(np.sqrt(np.mean(np.array(ate_errors) ** 2)))

    # Total GT rotation: end-to-end geodesic distance
    total_gt_rot = _geodesic_deg(R_gt[0].T @ R_gt[-1], np.eye(3))

    # Coverage score: fraction of GT trajectory followed
    min_gt_rotation_deg = 10.0
    denom = max(total_gt_rot, min_gt_rotation_deg)
    coverage = max(0.0, 1.0 - ate_rot / denom)

    gt_rels = _relative_rotations(gt_sampled[:n])
    est_rels = _relative_rotations(np.array(R_est))
    frame_errors = [_geodesic_deg(a, b) for a, b in zip(gt_rels, est_rels, strict=False)]
    mean_err = float(np.mean(frame_errors))

    return {
        "camera_controllability": round(coverage, 4),
        "ate_rot_deg": round(ate_rot, 3),
        "total_gt_rotation_deg": round(total_gt_rot, 3),
        "mean_rot_error_deg": round(mean_err, 3),
    }
