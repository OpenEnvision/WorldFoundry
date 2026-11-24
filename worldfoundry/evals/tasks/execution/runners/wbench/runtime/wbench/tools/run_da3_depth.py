"""
DA3 depth estimation — compute depth maps for videos.

Requires: DA3 conda environment (pycolmap, depth-anything-3)

Usage:
    # Single video
    python tools/run_da3_depth.py --video work_dirs/hunyuan/videos/case_1_combined.mp4 \
        --output_dir work_dirs/hunyuan/da3_cache/case_1

    # Batch: all videos in a directory
    python tools/run_da3_depth.py --video_dir work_dirs/hunyuan/videos \
        --output_base work_dirs/hunyuan/da3_cache --gpus 0,1,2,3

Output:
    {output_dir}/
        depth.npy        (N, H, W) float32
        extrinsics.npy   (N, 3, 4) float32  — world-to-camera
        c2w.npy          (N, 4, 4) float64   — camera-to-world
        intrinsics.npy   (N, 4) float32
        conf.npy         (N, H, W) float32
        meta.json        metadata
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from pathlib import Path

# isort: split
import src.compat  # noqa: F401 — stub optional deps (moviepy, trimesh, plyfile, etc.)

# isort: split
import cv2
import numpy as np

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES
from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.api import DepthAnything3

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DA3_MODEL_ID = os.environ.get("WBENCH_DA3_MODEL_ID") or "depth-anything/DA3-LARGE-1.1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("da3_depth")


def default_da3_model_id() -> str:
    """Resolve the shared DA3 asset, falling back to Hugging Face default loading."""
    asset = BASE_MODEL_CAPABILITIES["depth_anything_v3"].assets[0]
    status = asset.check()
    matched = status.get("matched_path")
    if matched:
        return str(matched)
    return asset.hf_repo_id or DA3_MODEL_ID


def extract_frames(video_path, fps=3):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps <= 0 or total_frames <= 0:
        cap.release()
        return None, None
    step = 1 if fps <= 0 or fps >= video_fps else max(1, round(video_fps / fps))
    indices = list(range(0, total_frames, step))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames), indices[: len(frames)]


def run_da3(model, frames_rgb, tmpdir):
    paths = []
    for i, frame in enumerate(frames_rgb):
        p = os.path.join(tmpdir, f"{i:04d}.png")
        cv2.imwrite(p, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        paths.append(p)
    return model.inference(paths)


def process_single(video_path, output_dir, model_dir=None, fps=3, device="cuda:0"):
    t0 = time.time()

    frames, indices = extract_frames(video_path, fps=fps)
    if frames is None or len(frames) < 3:
        raise ValueError(f"Insufficient frames: {len(frames) if frames is not None else 0}")

    da3_model = DepthAnything3.from_pretrained(model_dir or default_da3_model_id()).to(device=device)

    tmpdir = tempfile.mkdtemp()
    try:
        pred = run_da3(da3_model, frames, tmpdir)
    finally:
        shutil.rmtree(tmpdir)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "depth.npy"), pred.depth)
    np.save(os.path.join(output_dir, "extrinsics.npy"), pred.extrinsics)
    np.save(os.path.join(output_dir, "intrinsics.npy"), pred.intrinsics)
    if hasattr(pred, "conf") and pred.conf is not None:
        np.save(os.path.join(output_dir, "conf.npy"), pred.conf)

    ext = pred.extrinsics
    if ext.ndim == 3 and ext.shape[1:] == (3, 4):
        ext_44 = np.zeros((len(ext), 4, 4), dtype=np.float64)
        ext_44[:, :3, :4] = ext
        ext_44[:, 3, 3] = 1.0
    else:
        ext_44 = ext.reshape(-1, 4, 4).astype(np.float64)
    c2w = np.linalg.inv(ext_44)
    np.save(os.path.join(output_dir, "c2w.npy"), c2w)

    elapsed = time.time() - t0
    meta = {"num_frames": len(frames), "depth_shape": list(pred.depth.shape), "elapsed": round(elapsed, 1)}
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"{video_path} → {output_dir} ({len(frames)} frames, {elapsed:.1f}s)")
    return True


def _gpu_worker(args_tuple):
    gpu_id, tasks, model_dir, fps = args_tuple
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch

    torch.set_num_threads(4)
    device = "cuda:0"
    tag = f"[GPU{gpu_id}]"
    n_total = len(tasks)
    ok, fail = 0, 0
    for i, (video_path, output_dir) in enumerate(tasks):
        print(f"  {tag} [{i + 1}/{n_total}] {os.path.basename(video_path)}", flush=True)
        try:
            process_single(video_path, output_dir, model_dir, fps, device)
            ok += 1
        except Exception as e:
            logger.error(f"{tag} {video_path}: {e}")
            fail += 1
    print(f"  {tag} Done: {ok}/{n_total} ok, {fail} fail", flush=True)
    return ok, fail


def main():
    parser = argparse.ArgumentParser(description="DA3 depth estimation")
    parser.add_argument("--video", type=str, help="Single video path")
    parser.add_argument("--video_dir", type=str, help="Batch: video directory")
    parser.add_argument("--output_dir", type=str, help="Output directory (single mode)")
    parser.add_argument("--output_base", type=str, help="Output base (batch mode)")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--fps", type=float, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.video:
        output_dir = args.output_dir or "da3_output"
        process_single(args.video, output_dir, args.model_dir, args.fps)
        return

    if args.video_dir:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
        output_base = args.output_base or os.path.join(os.path.dirname(args.video_dir), "da3_cache")

        videos = sorted(Path(args.video_dir).glob("case_*_combined.mp4"))
        tasks = []
        for v in videos:
            case_id = v.stem.replace("case_", "").replace("_combined", "")
            out = os.path.join(output_base, f"case_{case_id}")
            if not args.force and os.path.exists(os.path.join(out, "depth.npy")):
                continue
            tasks.append((str(v), out))

        print(f"Found {len(videos)} videos, {len(tasks)} to process, {len(gpu_ids)} GPUs")
        if not tasks:
            return

        chunks = [[] for _ in gpu_ids]
        for i, t in enumerate(tasks):
            chunks[i % len(gpu_ids)].append(t)

        worker_args = [(gpu_ids[i], chunks[i], args.model_dir, args.fps) for i in range(len(gpu_ids)) if chunks[i]]

        ctx = mp.get_context("spawn")
        with ctx.Pool(len(worker_args)) as pool:
            results = pool.map(_gpu_worker, worker_args)

        total_ok = sum(r[0] for r in results)
        total_fail = sum(r[1] for r in results)
        print(f"Done: {total_ok} ok, {total_fail} fail")


if __name__ == "__main__":
    main()
