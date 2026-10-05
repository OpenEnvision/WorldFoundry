"""
Extract frames from a video to an output directory.

Usage:
python scripts/extract_frames.py \
    --video data/videos/buoyancy_self_forcing_2.mp4 \
    --output frames/buoyancy \
    --start 80 --end 120 --stride 1

python scripts/extract_frames.py \
    --video data/lemon.mp4 \
    --output frames/buoyancy \
    --frames 82,99,100

Options:
- --start: first frame index to save (inclusive, 0-based). Default: 0
- --end: last frame index to save (inclusive). Default: save to end
- --stride: save every Nth frame. Default: 1
- --frames: explicit frame indices (comma-separated), overrides start/end/stride
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import cv2


def parse_frames_arg(arg: Optional[str]) -> Optional[List[int]]:
    if not arg:
        return None
    parts = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parts.append(int(token))
        except ValueError:
            raise ValueError(f"Invalid frame index: {token}")
    return sorted(set(parts))


def extract_frames(
    video_path: str,
    output_dir: str,
    start: int = 0,
    end: Optional[int] = None,
    stride: int = 1,
    explicit_frames: Optional[List[int]] = None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_idx = total - 1 if end is None else min(end, total - 1)

    targets: List[int]
    if explicit_frames is not None:
        targets = [f for f in explicit_frames if 0 <= f < total]
    else:
        targets = list(range(start, last_idx + 1, max(1, stride)))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_set = set(targets)
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in target_set:
            out_path = out_dir / f"frame_{idx:05d}.png"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        idx += 1
        if idx > last_idx and explicit_frames is None:
            break
        if explicit_frames is not None and idx > max(targets, default=-1):
            break

    cap.release()
    print(f"Saved {saved} frames to {out_dir} (from video: {video_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Directory to save frames")
    parser.add_argument("--start", type=int, default=0, help="Start frame (inclusive, 0-based)")
    parser.add_argument("--end", type=int, help="End frame (inclusive)")
    parser.add_argument("--stride", type=int, default=1, help="Save every Nth frame")
    parser.add_argument("--frames", help="Comma-separated explicit frame indices (overrides start/end/stride)")
    args = parser.parse_args()

    frame_list = parse_frames_arg(args.frames)
    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        start=args.start,
        end=args.end,
        stride=args.stride,
        explicit_frames=frame_list,
    )
