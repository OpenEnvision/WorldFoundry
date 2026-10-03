# SPDX-License-Identifier: Apache-2.0
"""Bounded InSpatio 1.5 condition codecs; uint16 depth and exact RGB scaling.

Adapted from inspatio-world-v1.5 dd3561f544053fe22d739b6b2f8461c9c97bf8cb.
Only inference codecs are retained, with atomic output publication.
"""
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MAX_UINT16 = (1 << 16) - 1
DEPTH_ENCODING = "uint16_minmax_v1"
def image_depth_paths(depth_dir: Path, views: int):
    if views == 1:
        return [depth_dir / "depth.png"]
    return [depth_dir / f"depth_{index:02d}.png" for index in range(views)]


def read_range(metadata_path: Path):
    values = np.loadtxt(metadata_path, ndmin=1)
    if values.size != 2 or not np.isfinite(values).all() or values[0] > values[1]:
        raise ValueError(f"Invalid depth min/max: {metadata_path}")
    return float(values[0]), float(values[1])


def write_range(metadata_path: Path, minimum: float, maximum: float):
    metadata_path.write_text(f"{minimum:.9g} {maximum:.9g}\n")


def encode_depth(depth, minimum: float, maximum: float):
    values = np.asarray(depth, dtype=np.float32)
    if maximum <= minimum:
        return np.zeros(values.shape, dtype=np.uint16)
    else:
        normalized = np.clip((values.astype(np.float64) - minimum) / (maximum - minimum), 0, 1)
        return np.rint(normalized * MAX_UINT16).astype(np.uint16)


def decode_depth(quantized, minimum: float, maximum: float):
    if quantized.ndim != 2 or quantized.dtype != np.uint16:
        raise ValueError("Linear uint16 depth must be one uint16 channel")
    return (minimum + quantized.astype(np.float64) / MAX_UINT16 *
            (maximum - minimum)).astype(np.float32)


def decode_image_depth(path: Path, metadata_path: Path):
    minimum, maximum = read_range(metadata_path)
    with Image.open(path) as image:
        if image.mode not in ("I;16", "I"):
            raise ValueError(f"Expected uint16 depth PNG: {path}")
        return decode_depth(np.asarray(image, dtype=np.uint16), minimum, maximum)


def decode_video_depth_frame(frame_bgr, minimum: float, maximum: float):
    if frame_bgr.ndim != 3 or frame_bgr.shape[-1] != 3 or frame_bgr.dtype != np.uint8:
        raise ValueError("Video depth must be three uint8 channels")
    quantized = ((frame_bgr[..., 2].astype(np.uint16) << 8) |
                 frame_bgr[..., 1].astype(np.uint16))
    return decode_depth(quantized, minimum, maximum)


def encode_video_frame(quantized):
    if quantized.ndim != 2 or quantized.dtype != np.uint16:
        raise ValueError("Expected one uint16 depth channel")
    rgb = np.zeros((*quantized.shape, 3), dtype=np.uint8)
    rgb[..., 0] = quantized >> 8
    rgb[..., 1] = quantized
    return rgb


def iter_video_chunks(path, valid_frames, padded_frames, video_size=(480, 832), repeat_first=False):
    """Yield uint8 TCHW: one initial frame, then four at a time.

    Padding repeats the final valid frame; image sources only decode frame zero.
    """
    if valid_frames < 1 or padded_frames < valid_frames or (padded_frames - 1) % 4:
        raise ValueError("Invalid valid/padded frame counts")
    import cv2

    reader = cv2.VideoCapture(str(path))
    if not reader.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        start = 0
        last = None
        while start < padded_frames:
            stop = min(start + (1 if start == 0 else 4), padded_frames)
            frames = []
            for index in range(start, stop):
                if last is None or (not repeat_first and index < valid_frames):
                    ok, image = reader.read()
                    if not ok:
                        raise ValueError(f"Video is shorter than {valid_frames} frames: {path}")
                    if image.shape[:2] != tuple(video_size):
                        raise ValueError(f"Video resolution must be {tuple(video_size)}: {path}")
                    last = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frames.append(last)
            yield torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
            start = stop
    finally:
        reader.release()


def normalize_rgb(frames, device, dtype):
    # Match float32 normalization before converting to model dtype.
    return (frames.float() / 255.0 * 2 - 1).to(device=device, dtype=dtype)


def iter_mask_latents(chunks, device, dtype):
    for index, frames in enumerate(chunks):
        mask = (frames[:, :1] > 127.5).float() * 2 - 1
        mask = mask.to(device=device, dtype=dtype)
        mask = F.interpolate(mask, size=(frames.shape[-2] // 8, frames.shape[-1] // 8),
                             mode="bilinear", align_corners=False)
        if index == 0:
            mask = mask.expand(4, -1, -1, -1)
        yield mask[:, 0][None, None]


@contextmanager
def video_writer(path, fps, width=832, height=480, lossless=False, depth=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "-", "-an", "-c:v", "libx264rgb" if depth else "libx264", "-preset", "ultrafast",
               "-crf", "0" if lossless or depth else "18", "-pix_fmt", "rgb24" if depth else "yuv420p", str(temporary)]
    process = None
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        yield process.stdin
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"Video encoding failed: {path}")
        temporary.replace(path)
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            if not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            process.wait()
        temporary.unlink(missing_ok=True)
