"""
Perspective consistency — SAM2 mask tracking + DA3 depth estimation.

Measures how stably a tracked subject maintains its position across frames.

Algorithm:
1. For all frames where target is present (mask area >= 10px):
     cx = mean(xs) / W, cy = mean(ys) / H (normalized centroid)
     d = mean(depth[mask])                 (average depth in mask region)
2. Whole-video statistics:
     c_std = sqrt(std(cx)^2 + std(cy)^2)
     d_cv  = std(d) / mean(d)
     presence = valid_frames / total_frames
3. Score computation:
     centroid_stability = max(0, 1 - c_std / 0.3) * presence
     depth_stability    = max(0, 1 - d_cv  / 0.5) * presence  (computed but not used in final score)
4. Final score = centroid_stability
"""

import os
from typing import Any

import cv2
import numpy as np

C_STD_THRESH = 0.3
D_CV_THRESH = 0.5
MIN_MASK_AREA_PX = 10


def compute_case(mask_dir: str, depth_path: str) -> dict[str, Any] | None:
    """Compute perspective_consistency for a single case.

    Args:
        mask_dir: SAM2 per-frame mask directory (contains {frame_id:05d}.png)
        depth_path: DA3 depth .npy file (shape: (N, H, W))

    Returns:
        Dict with score, details, params, error; or None if data missing
    """
    if not os.path.isdir(mask_dir) or not os.path.exists(depth_path):
        return None

    params = {"c_std_thresh": C_STD_THRESH, "d_cv_thresh": D_CV_THRESH, "min_mask_area_px": MIN_MASK_AREA_PX}

    try:
        depth = np.load(depth_path, mmap_mode="r")
        nd, hd, wd = depth.shape

        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
        if not mask_files:
            return None

        cx_l, cy_l, d_l = [], [], []
        total = 0
        for mf in mask_files:
            fid = int(mf.replace(".png", ""))
            didx = min(fid, nd - 1)
            m = cv2.imread(os.path.join(mask_dir, mf), cv2.IMREAD_GRAYSCALE)
            if m is None:
                total += 1
                continue
            total += 1
            mr = cv2.resize(m, (wd, hd), interpolation=cv2.INTER_NEAREST)
            b = mr > 127
            area = int(b.sum())
            if area < MIN_MASK_AREA_PX:
                continue
            ys, xs = np.where(b)
            cx_l.append(xs.mean() / wd)
            cy_l.append(ys.mean() / hd)
            d_l.append(float(np.asarray(depth[didx])[b].mean()))

        if total == 0:
            return None

        valid = len(d_l)
        presence = valid / total

        if valid < 2:
            return {
                "score": 0.0,
                "details": {
                    "presence": round(presence, 4),
                    "valid_frames": valid,
                    "total_frames": total,
                    "centroid_stability": 0.0,
                    "depth_stability": 0.0,
                },
                "params": params,
                "error": None,
            }

        c_std = float(np.sqrt(np.std(cx_l) ** 2 + np.std(cy_l) ** 2))
        d_mean = float(np.mean(d_l))
        d_cv = float(np.std(d_l)) / (d_mean + 1e-6)

        centroid_stab = max(0.0, 1.0 - c_std / C_STD_THRESH) * presence
        depth_stab = max(0.0, 1.0 - d_cv / D_CV_THRESH) * presence
        overall = centroid_stab

        return {
            "score": round(overall, 4),
            "details": {
                "presence": round(presence, 4),
                "valid_frames": valid,
                "total_frames": total,
                "centroid_std": round(c_std, 4),
                "depth_cv": round(d_cv, 4),
                "centroid_stability": round(centroid_stab, 4),
                "depth_stability": round(depth_stab, 4),
            },
            "params": params,
            "error": None,
        }
    except Exception as e:
        return {"score": None, "details": None, "params": params, "error": f"{type(e).__name__}: {e}"}
