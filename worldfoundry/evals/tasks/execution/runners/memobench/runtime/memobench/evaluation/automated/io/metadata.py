import json

import cv2
import numpy as np


def load_intrinsics(path: str) -> dict[str, float]:
    """
    Loads intrinsics.npy with shape (N, 4) where each row is [fx, fy, cx, cy].
    Returns a dict with keys fx, fy, cx, cy from the first row.
    """
    data = np.load(path).astype(np.float64)
    if data.ndim == 2:
        row = data[0]
    elif data.ndim == 1:
        row = data
    else:
        raise ValueError(f"Unexpected intrinsics shape {data.shape} in {path}")
    return {"fx": float(row[0]), "fy": float(row[1]), "cx": float(row[2]), "cy": float(row[3])}


def load_gt_frame_count_synthetic(gt_video_path: str) -> int:
    """
    Returns the total frame count of the original UE5 GT video.
    Used as the denominator when mapping exits / re-enter annotations
    (which are in the original UE5 render frame space) to the generated
    video frame space.

    GT videos are NOT all 300 frames — they range from ~267 to 300 depending
    on the scene. We read the actual count directly from the video file.
    """
    cap = cv2.VideoCapture(gt_video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise RuntimeError(f"Cannot read frame count from {gt_video_path}")
    return n


def extract_gt_frame_at(gt_video_path: str, frame_idx: int) -> np.ndarray:
    """
    Extract a single frame by index from a GT video (0-based).
    Returns a BGR numpy array.
    """
    cap = cv2.VideoCapture(gt_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {gt_video_path}")
    return frame


def extract_gt_frames_batch(gt_video_path: str, indices: list) -> list:
    """
    Extract multiple frames from a GT video in a single open/close.
    indices: sorted list of 0-based frame indices.
    Returns a list of BGR numpy arrays in the same order as indices.
    """
    if not indices:
        return []
    cap = cv2.VideoCapture(gt_video_path)
    frames = []
    prev_idx = -1
    for idx in indices:
        if idx != prev_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Cannot read frame {idx} from {gt_video_path}")
        frames.append(frame)
        prev_idx = idx
    cap.release()
    return frames


def load_intrinsics_json(path: str) -> dict[str, float]:
    """
    Loads intrinsics from a Real GT JSON file.
    Expected format: {"camera": {"fx": ..., "fy": ..., "cx": ..., "cy": ..., ...}}
    Returns a dict with keys fx, fy, cx, cy.
    """
    with open(path) as f:
        data = json.load(f)
    cam = data.get("camera", data)  # handle both nested and flat formats
    return {
        "fx": float(cam["fx"]),
        "fy": float(cam["fy"]),
        "cx": float(cam["cx"]),
        "cy": float(cam["cy"]),
    }


def load_gt_frame_count_real(timestamps_path: str) -> int:
    """
    Returns the total GT frame count for a Real clip by counting non-empty
    lines in timestamps.txt (one timestamp per GT frame).
    """
    with open(timestamps_path) as f:
        return sum(1 for line in f if line.strip())


def extract_gt_frame0(mp4_path: str) -> np.ndarray:
    """
    Extract the first frame from a Real GT MP4 as a BGR numpy array.
    Used as the reference image for ReferenceFidelity.
    """
    cap = cv2.VideoCapture(mp4_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read first frame from {mp4_path}")
    return frame
