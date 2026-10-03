#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI entry point for Depth-Anything-3 depth estimation.

Usage:
    # Single video
    python code/depth/depth_predict_da3_cli.py --input video.mp4 --output ./output

    # Video directory
    python code/depth/depth_predict_da3_cli.py --input ./videos/ --output ./output

    # With explicit config
    python depth/depth_predict_da3_cli.py --input video.mp4 --output ./output \
        --config-json '{"model_path": "./checkpoints/DA3"}'
"""

import argparse
import json
import logging
import os
import sys
import traceback
from typing import List

# Add code/ to path so sibling packages are importable
_code_dir = os.path.join(os.path.dirname(__file__), "..")
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

from depth.depth_predict_da3 import DepthPredictDA3
from worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_runtime.utils.reproducibility import (
    initialize_reproducibility,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_video_files(input_path: str, video_extensions: List[str]) -> List[str]:
    """Find video files from a path (file or directory)."""
    video_files = []
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in video_extensions:
            video_files.append(input_path)
        else:
            print(f"Warning: {input_path} is not a supported video format {video_extensions}")
    elif os.path.isdir(input_path):
        for filename in sorted(os.listdir(input_path)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in video_extensions:
                video_files.append(os.path.join(input_path, filename))
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    return video_files


def process_video(model, video_path, output_dir, video_idx, total_videos, flat_output=False):
    """Process a single video."""
    video_name = os.path.basename(video_path)
    video_stem = os.path.splitext(video_name)[0]

    print(f"\n{'=' * 60}")
    print(f"Processing [{video_idx}/{total_videos}]: {video_name}")
    print(f"{'=' * 60}")

    video_output_dir = output_dir if flat_output else os.path.join(output_dir, video_stem)
    input_files = {"videos": [video_path], "images": []}

    try:
        success = model.run(input_files, video_output_dir)
        if success:
            print(f"Done: {video_name} -> {video_output_dir}")
        else:
            print(f"Failed: {video_name}")
        return success
    except Exception as e:
        print(f"Error processing {video_name}: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Depth Predict DA3 CLI")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input video file or directory")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--video_ext", type=str, default=".mp4,.avi,.mov,.mkv",
                        help="Supported video extensions, comma-separated")
    parser.add_argument("--filter", "-f", type=str, default=None,
                        help="Filter videos by filename keyword")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip videos with existing output")
    parser.add_argument("--config-json", type=str, default=None,
                        help="JSON string with depth model config overrides")
    args = parser.parse_args()
    initialize_reproducibility()

    # Parse config
    config = {}
    if args.config_json:
        config = json.loads(args.config_json)

    # Parse extensions
    video_extensions = [ext.strip().lower() for ext in args.video_ext.split(",")]

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    video_files = get_video_files(args.input, video_extensions)
    if not video_files:
        print("Error: no video files found")
        sys.exit(1)

    if args.filter:
        video_files = [v for v in video_files if args.filter in os.path.basename(v)]

    if args.skip_existing:
        is_single = os.path.isfile(args.input)
        filtered = []
        for vp in video_files:
            check_dir = args.output if is_single else os.path.join(args.output, os.path.splitext(os.path.basename(vp))[0])
            if os.path.exists(os.path.join(check_dir, "intrinsic.txt")):
                print(f"Skipping (already processed): {os.path.basename(vp)}")
            else:
                filtered.append(vp)
        video_files = filtered

    if not video_files:
        print("All videos already processed")
        sys.exit(0)

    print(f"Found {len(video_files)} video(s) to process")
    os.makedirs(args.output, exist_ok=True)

    # Init model
    print("\nInitializing DepthPredictDA3...")
    model = DepthPredictDA3(config=config)

    is_single_file = os.path.isfile(args.input)
    success_count = 0
    fail_count = 0

    for idx, video_path in enumerate(video_files, 1):
        success = process_video(model, video_path, args.output, idx, len(video_files),
                                flat_output=is_single_file)
        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {success_count} success, {fail_count} failed, {len(video_files)} total")
    print(f"{'=' * 60}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
