#!/usr/bin/env python3
"""
Standalone VIPe worker — called via subprocess.Popen by unified_video_metrics._run_vipe().
Args: <videos_json_file> <vipe_output_dir> <process_index>
CUDA_VISIBLE_DEVICES must be set by the caller before launching.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent

POSE_DIR_NAME = "pose"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _is_labeled(video_path: str, vipe_output_dir: str) -> bool:
    stem = Path(video_path).stem
    return (Path(vipe_output_dir) / POSE_DIR_NAME / f"{stem}.npz").exists()


def main():
    videos_json_file = sys.argv[1]
    vipe_output_dir = sys.argv[2]
    proc_idx = int(sys.argv[3])

    logger = logging.getLogger(f"vipe.worker{proc_idx}")

    with open(videos_json_file) as f:
        video_paths = json.load(f)

    # Disable HuggingFace network calls and remove any proxy that might block loading
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for _proxy_key in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        os.environ.pop(_proxy_key, None)

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    logger.info(f"Worker {proc_idx} on GPU {gpu_id}: {len(video_paths)} videos assigned")

    from worldfoundry.base_models.three_dimensions.general_3d.vipe import infer_poses

    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES already restricts to one GPU
    torch.cuda.empty_cache()
    time.sleep(proc_idx * 2)  # stagger startup to reduce model-load contention

    unlabeled = [v for v in video_paths if not _is_labeled(v, vipe_output_dir)]
    logger.info(f"Worker {proc_idx}: {len(unlabeled)} unlabeled, {len(video_paths) - len(unlabeled)} already done")

    if not unlabeled:
        return
    try:
        results = infer_poses(unlabeled, vipe_output_dir)
    except Exception:
        logger.exception(f"Worker {proc_idx}: batch pose inference failed")
        raise
    for result in results:
        logger.info(f"Worker {proc_idx}: done {Path(result.input_video).name}")


if __name__ == "__main__":
    main()
