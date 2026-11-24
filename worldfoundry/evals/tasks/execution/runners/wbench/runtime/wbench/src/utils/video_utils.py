"""Video frame loading utilities."""

import cv2
import numpy as np
from PIL import Image


def load_video_frames(
    video_path: str,
    num_frames: int = 16,
    target_size: tuple[int, int] | None = None,
    sample_fps: float | None = None,
) -> list[Image.Image]:
    """
    Load frames from a video file.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample (-1 for all); ignored if sample_fps is set
        target_size: Target size (width, height)
        sample_fps: Sample at fixed FPS (e.g., 2.0 = 2 frames/sec), takes priority

    Returns:
        List of PIL Images
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return []

    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if vlen <= 0:
        cap.release()
        return []

    if sample_fps is not None:
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        step = max(1, round(video_fps / sample_fps))
        indices = set(range(0, vlen, step))
        read_all = False
    elif num_frames == -1 or num_frames >= vlen:
        indices = None
        read_all = True
    else:
        indices = set(np.linspace(0, vlen - 1, num_frames, dtype=int).tolist())
        read_all = False

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if read_all or idx in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            if target_size:
                img = img.resize(target_size, Image.LANCZOS)
            frames.append(img)
        idx += 1

    cap.release()
    return frames
