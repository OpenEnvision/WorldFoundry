"""
Spatial consistency — MegaSAM pose + DreamSim perceptual similarity.

Evaluates loop-style navigation cases: measures whether the camera returns
to its starting viewpoint by comparing first and return frames.

Method:
1. Use MegaSAM poses to find the best return frame (closest rotation to start)
2. Compute DreamSim perceptual similarity between first frame and return frame
3. Gate by minimum similarity during trajectory (penalizes static videos)
4. Final score = ret_sim * gate, where gate = min(1, (1-min_sim)/tau)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TAU = 0.15
N_SAMPLE_FRAMES = 10
METRIC_NAME = "spatial_consistency"

NAVI_ACTIONS = {
    "W",
    "A",
    "S",
    "D",
    "left",
    "right",
    "up",
    "down",
    "W+up",
    "W+down",
    "W+A",
    "W+D",
    "W+left",
    "W+right",
}
SYMMETRIC_PAIRS = {
    ("left", "right"),
    ("right", "left"),
    ("A", "D"),
    ("D", "A"),
    ("W", "S"),
    ("S", "W"),
    ("up", "down"),
    ("down", "up"),
}


def find_loop_cases(cases_dir: str) -> list[tuple[str, dict]]:
    """Return loop-style navigation cases (symmetric action sequences)."""
    files = sorted(Path(cases_dir).glob("*.json"))
    loop_cases = []
    for file_path in files:
        with open(file_path) as handle:
            case_data = json.load(handle)
        actions = [t["action"] for t in case_data.get("interactions", [])]
        if not actions or actions[0] not in NAVI_ACTIONS:
            continue
        n = len(actions)
        if n < 2:
            continue
        is_loop = all((actions[i], actions[n - 1 - i]) in SYMMETRIC_PAIRS for i in range(n // 2))
        if is_loop:
            loop_cases.append((str(case_data["id"]), case_data))
    return loop_cases


def rot_angle_deg(rotation: np.ndarray) -> float:
    cos_val = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def compute_score(ret_sim: float, min_sim: float, tau: float = TAU) -> tuple[float, float]:
    """Compute gated spatial consistency score.

    gate = min(1, (1 - min_sim) / tau)  -- penalizes nearly-static videos
    score = ret_sim * gate
    """
    gate = min(1.0, (1.0 - min_sim) / tau)
    return float(ret_sim) * gate, gate


def evaluate_case(
    video_path: str,
    npz_path: str,
    n_turns: int,
    dreamsim_model,
    dreamsim_preprocess,
    device: str = "cuda",
) -> dict[str, Any] | None:
    """Evaluate spatial consistency for a single loop case.

    Args:
        video_path: Path to combined video
        npz_path: Path to MegaSAM .npz pose file
        n_turns: Number of interaction turns
        dreamsim_model: Loaded DreamSim model
        dreamsim_preprocess: DreamSim preprocessing function
        device: CUDA device

    Returns:
        Dict with ret_sim, min_sim, ret_rot, score, gate
    """
    from PIL import Image

    if not os.path.exists(video_path) or not os.path.exists(npz_path):
        return None

    npz_file = np.load(npz_path)
    poses = npz_file["cam_c2w"]
    stride = int(npz_file["stride"]) if "stride" in npz_file.files else 1
    n_poses = len(poses)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    per_turn = n_poses / n_turns
    last_turn_start = round((n_turns - 1) * per_turn)
    best_index, best_rot = last_turn_start, 999.0
    for pose_idx in range(last_turn_start, n_poses):
        rotation = rot_angle_deg(poses[0, :3, :3].T @ poses[pose_idx, :3, :3])
        if rotation < best_rot:
            best_rot = rotation
            best_index = pose_idx

    return_frame = min(best_index * stride, total_frames - 1)

    def get_ds_tensor(img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return dreamsim_preprocess(Image.fromarray(img_rgb)).to(device)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, frame0 = cap.read()
    if frame0 is None:
        cap.release()
        return None
    tensor0 = get_ds_tensor(frame0)

    sample_indices = np.linspace(0, return_frame, N_SAMPLE_FRAMES, dtype=int)
    similarities = []
    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        tensor_i = get_ds_tensor(frame)
        with torch.no_grad():
            distance = float(dreamsim_model(tensor0, tensor_i).cpu())
        similarities.append(1.0 / (1.0 + distance))

    if not similarities:
        cap.release()
        return None
    min_sim = min(similarities)

    cap.set(cv2.CAP_PROP_POS_FRAMES, return_frame)
    _, return_img = cap.read()
    cap.release()
    if return_img is None:
        return None

    tensor_ret = get_ds_tensor(return_img)
    with torch.no_grad():
        ret_dist = float(dreamsim_model(tensor0, tensor_ret).cpu())
    ret_sim = 1.0 / (1.0 + ret_dist)

    score, gate = compute_score(ret_sim, min_sim, TAU)
    return {
        "ret_sim": round(ret_sim, 4),
        "min_sim": round(min_sim, 4),
        "ret_rot": round(best_rot, 1),
        "score": round(score, 4),
        "gate": round(gate, 4),
    }


def summarize_model_results(case_records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in case_records if r.get("score") is not None]
    if not valid:
        return {"score": 0.0, "ret_sim": 0.0, "min_sim": 0.0, "n_cases": 0}
    return {
        "score": round(float(np.mean([r["score"] for r in valid])), 4),
        "ret_sim": round(float(np.mean([r["ret_sim"] for r in valid])), 4),
        "min_sim": round(float(np.mean([r["min_sim"] for r in valid])), 4),
        "n_cases": len(valid),
    }
