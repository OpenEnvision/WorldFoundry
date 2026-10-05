# SPDX-License-Identifier: Apache-2.0
"""Validate official v1.5 scenes and stage direct image/video requests."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .v15_io import DEPTH_ENCODING, image_depth_paths, read_range

RESOLUTION = (480, 832)
FRAMES_PER_BLOCK = 3
LATENT_CHANNELS = 16
LATENT_HEIGHT, LATENT_WIDTH = (value // 8 for value in RESOLUTION)
TOKENS_PER_FRAME = LATENT_HEIGHT * LATENT_WIDTH // 4


def source_type_for(meta):
    if meta["kind"] == "video":
        return "video"
    if meta["kind"] == "image" and meta["views"] in (1, 4):
        return "image" if meta["views"] == 1 else "multi-image"
    raise ValueError("InSpatio 1.5 requires one image, four images, or one video")


def latent_frame_groups(frames):
    yield range(1)
    for start in range(1, frames, 4):
        yield range(start, min(start + 4, frames))


def padded_frame_count(frames):
    if frames < 1:
        raise ValueError("A scene must contain at least one valid frame")
    latent_frames = (frames - 1 + 3) // 4 + 1
    latent_frames = ((latent_frames + 2) // 3) * 3
    return (latent_frames - 1) * 4 + 1


def matrices(path, size):
    values = np.loadtxt(path, ndmin=2).astype(np.float32)
    if values.shape[1] != size * size or not np.isfinite(values).all():
        raise ValueError(f"Expected finite row-major {size}x{size} matrices: {path}")
    result = values.reshape(-1, size, size)
    if size == 4 and not np.allclose(result[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError(f"Camera poses must be homogeneous OpenCV world-to-camera matrices: {path}")
    if np.any(np.abs(np.linalg.det(result)) < 1e-8):
        raise ValueError(f"Singular camera matrices: {path}")
    return result


def video_metadata(path):
    import cv2

    reader = cv2.VideoCapture(str(path))
    try:
        if not reader.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(reader.get(cv2.CAP_PROP_FPS))
        size = (int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)))
        if count < 1 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video frame count or fps: {path}")
        return count, fps, size
    finally:
        reader.release()


def validate_scene(record):
    scene = Path(record["path"])
    frames, views, valid = (int(record[key]) for key in ("frames", "views", "valid_frames"))
    if (not 0 < valid <= frames or tuple(record["resolution"]) != RESOLUTION
            or not math.isfinite(float(record["fps"])) or float(record["fps"]) <= 0):
        raise ValueError("InSpatio 1.5 requires positive frame counts/fps and resolution 480x832")
    source_type = source_type_for(record)
    if source_type == "video" and views != frames:
        raise ValueError("Video source views must equal frames")
    if source_type == "multi-image" and len(list(latent_frame_groups(frames))) % FRAMES_PER_BLOCK:
        raise ValueError("Four-image scenes must cover complete three-latent chunks")
    if record.get("depth_encoding") != DEPTH_ENCODING:
        raise ValueError(f"Expected depth_encoding={DEPTH_ENCODING}")
    expected_intrinsics = ("estimated_source_frame_000000_fixed" if source_type == "video"
                           else "estimated_source_view_00_fixed")
    if record.get("target_intrinsics") != expected_intrinsics:
        raise ValueError(f"Expected target_intrinsics={expected_intrinsics}")
    target = Path(record.get("target_traj_path", scene / "input/target_tcw.txt"))
    lengths = (len(matrices(scene / "depth/source_intrinsics.txt", 3)),
               len(matrices(scene / "depth/source_tcw.txt", 4)), len(matrices(target, 4)))
    if lengths != (views, views, frames):
        raise ValueError("RGB/depth/source camera/target trajectory count mismatch")
    read_range(scene / "depth/metadata.txt")
    if record["kind"] == "image":
        inputs = sorted((scene / "input").glob("view_*.png"))
        depths = image_depth_paths(scene / "depth", views)
        if len(inputs) != views:
            raise ValueError("Image count must match scene views")
        for path in inputs + depths:
            with Image.open(path) as image:
                if image.size != (832, 480):
                    raise ValueError(f"Scene images and depth must be 832x480: {path}")
                if path in depths and image.mode not in ("I;16", "I"):
                    raise ValueError(f"Depth PNG must have one uint16 channel: {path}")
    else:
        for path in (scene / "input/video.mp4", scene / "depth/depth.mp4"):
            count, fps, size = video_metadata(path)
            if count != frames or size != RESOLUTION or abs(fps - float(record["fps"])) > 0.05:
                raise ValueError(f"Video/depth resolution, frame count or fps mismatch: {path}")
    return record


def load_scene(scene_dir, *, prompt="", traj_txt_path=None):
    scene = Path(scene_dir).expanduser().resolve(strict=True)
    metadata = json.loads((scene / "scene.json").read_text())
    record = {**metadata, "id": "scene", "path": str(scene)}
    record["text"] = prompt or (scene / "input/prompt.txt").read_text().strip()
    if traj_txt_path is not None:
        record["target_traj_path"] = str(Path(traj_txt_path).expanduser().resolve(strict=True))
    return validate_scene(record)


def stage_direct_input(directory, *, images=None, videos=None, prompt="", traj_txt_path=None):
    if (images is None) == (videos is None):
        raise ValueError("Supply images or videos, with a prompt and target trajectory")
    if not prompt.strip() or traj_txt_path is None:
        raise ValueError("Direct input requires prompt and traj_txt_path (OpenCV Tcw per output frame)")
    trajectory = Path(traj_txt_path).expanduser().resolve(strict=True)
    target = matrices(trajectory, 4)
    directory = Path(directory)
    inputs = directory / "input"
    inputs.mkdir(parents=True)
    if images is not None:
        from worldfoundry.core.media.codecs.image import load_pil_image

        items = images if isinstance(images, (list, tuple)) else [images]
        if len(items) not in (1, 4):
            raise ValueError("InSpatio 1.5 image input must contain exactly one or four views")
        if len(items) == 4 and len(list(latent_frame_groups(len(target)))) % FRAMES_PER_BLOCK:
            raise ValueError("Four-image scenes must cover complete three-latent chunks")
        for index, item in enumerate(items):
            image = load_pil_image(item)
            if image.size != (832, 480):
                raise ValueError("Direct images must be 832x480; adjust cameras together when resizing")
            image.save(inputs / f"view_{index:02d}.png")
        kind, views, frames, fps = "image", len(items), len(target), 16.0
    else:
        items = videos if isinstance(videos, (list, tuple)) else [videos]
        if len(items) != 1:
            raise ValueError("InSpatio 1.5 accepts one input video per request")
        source = Path(items[0]).expanduser().resolve(strict=True)
        frames, fps, size = video_metadata(source)
        if len(target) != frames:
            raise ValueError("Target trajectory must contain one Tcw matrix per input video frame")
        if size == RESOLUTION:
            (inputs / "video.mp4").symlink_to(source)
        else:
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
                            "-vf", "scale=832:480:force_original_aspect_ratio=increase,crop=832:480",
                            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(inputs / "video.mp4")], check=True)
        kind, views = "video", frames
    (inputs / "target_tcw.txt").symlink_to(trajectory)
    (inputs / "prompt.txt").write_text(prompt.strip() + "\n")
    metadata = {"kind": kind, "views": views, "frames": frames, "valid_frames": frames,
                "fps": fps, "resolution": list(RESOLUTION),
                "target_intrinsics": ("estimated_source_frame_000000_fixed" if kind == "video"
                                      else "estimated_source_view_00_fixed"),
                "depth_encoding": DEPTH_ENCODING}
    (directory / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {**metadata, "id": "scene", "path": str(directory), "text": prompt.strip()}
