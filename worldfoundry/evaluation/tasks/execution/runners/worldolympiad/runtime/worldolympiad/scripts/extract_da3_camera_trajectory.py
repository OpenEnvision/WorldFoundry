#!/usr/bin/env python3
"""Extract a DA3-predicted camera trajectory from a video.

python worldeval/scripts/extract_da3_camera_trajectory.py \
    --video outputs/robotics_test/case1.mp4 \
    --model-name ./worldeval/weights/da3 \
    --gpus 5

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPTH_ANYTHING_3_SRC = REPO_ROOT / "Depth-Anything-3" / "src"

for path in (REPO_ROOT, DEPTH_ANYTHING_3_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


DEFAULT_MODEL_NAME = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DEFAULT_PROCESS_RES = 504
DEFAULT_MAX_FRAMES = 32


def sample_indices(total_frames: int, max_frames: int | None) -> list[int]:
    if total_frames <= 0:
        return []
    if max_frames is None or max_frames <= 0 or total_frames <= max_frames:
        return list(range(total_frames))
    if max_frames == 1:
        return [0]
    return [
        int(round(i * (total_frames - 1) / (max_frames - 1)))
        for i in range(max_frames)
    ]


def load_video_frames(video_path: str | os.PathLike[str], max_frames: int | None) -> tuple[list[np.ndarray], list[int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = set(sample_indices(total_frames, max_frames))
    frames: list[np.ndarray] = []
    frame_indices: list[int] = []
    frame_idx = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx in target_indices:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_indices.append(frame_idx)
        frame_idx += 1

    capture.release()

    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return frames, frame_indices


def resolve_model_source(model_name: str | os.PathLike[str]) -> str:
    model_name = os.fspath(model_name)
    candidate = Path(model_name).expanduser()

    looks_like_path = (
        candidate.is_absolute()
        or model_name.startswith(".")
        or model_name.startswith("~")
        or candidate.exists()
    )
    if not looks_like_path:
        return model_name

    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.exists():
        raise FileNotFoundError(f"Local DA3 model path does not exist: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"DA3 model path must be a directory: {candidate}")

    required_files = ("config.json", "model.safetensors")
    missing_files = [name for name in required_files if not (candidate / name).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Local DA3 model directory is missing required files {missing_files}: {candidate}"
        )

    return str(candidate)


def default_output_path(video_path: str | os.PathLike[str]) -> Path:
    video_path = Path(video_path).resolve()
    return video_path.parent / f"{video_path.stem}_da3_camera_trajectory.json"


def extract_camera_trajectory(
    video_path: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str] | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    max_frames: int = DEFAULT_MAX_FRAMES,
    process_res: int = DEFAULT_PROCESS_RES,
    gpus: str | None = None,
) -> tuple[dict[str, list[list[float]]], Path]:
    """Run DA3 on a video and save the predicted camera trajectory."""
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    import torch
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.geometry import affine_inverse_np, as_homogeneous

    video_path = Path(video_path).resolve()
    resolved_output_path = Path(output_path).resolve() if output_path else default_output_path(video_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    frames, frame_indices = load_video_frames(video_path, max_frames)
    model_source = resolve_model_source(model_name)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run DA3 trajectory extraction.")

    device = torch.device("cuda:0")
    print(f"Loading DA3 model from: {model_source}")
    print(f"Using device: {device} (CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')})")

    model = DepthAnything3.from_pretrained(model_source).to(device)
    model.eval()

    print(f"Running DA3 inference on {len(frames)} sampled frames...")
    prediction = model.inference(
        image=frames,
        extrinsics=None,
        intrinsics=None,
        process_res=process_res,
        align_to_input_ext_scale=True,
        infer_gs=False,
        export_dir=None,
    )

    extrinsics = as_homogeneous(np.asarray(prediction.extrinsics, dtype=np.float32))
    poses_c2w = affine_inverse_np(extrinsics)

    trajectory = {
        f"frame{frame_idx}": pose.tolist()
        for frame_idx, pose in zip(frame_indices, poses_c2w)
    }

    resolved_output_path.write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved DA3 camera trajectory to: {resolved_output_path}")
    print(f"Frames used: {len(frame_indices)}")
    print("This file can be passed to 3d_metrics/score_video_3d.py via --camera-trajectory")
    return trajectory, resolved_output_path


def main():
    parser = argparse.ArgumentParser(description="Extract a DA3 camera trajectory from a video")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument(
        "--output",
        help="Output JSON path. Defaults to <video_stem>_da3_camera_trajectory.json next to the video",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="DA3 reconstruction model name or local model directory",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=f"Maximum number of video frames to sample (default: {DEFAULT_MAX_FRAMES})",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=DEFAULT_PROCESS_RES,
        help=f"DA3 process resolution (default: {DEFAULT_PROCESS_RES})",
    )
    parser.add_argument("--gpus", help="Visible GPU ids, e.g. '0' or '2'")
    args = parser.parse_args()

    extract_camera_trajectory(
        args.video,
        output_path=args.output,
        model_name=args.model_name,
        max_frames=args.max_frames,
        process_res=args.process_res,
        gpus=args.gpus,
    )


if __name__ == "__main__":
    main()
