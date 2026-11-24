"""
SAM2 mask tracking — propagate initial mask across video frames (native SAM2 API).

Usage:
    # Single case
    python tools/run_sam2_track.py \
        --video work_dirs/hunyuan/videos/case_1_combined.mp4 \
        --mask data/masks/case_1_mask.png \
        --output_dir work_dirs/hunyuan/masks/case_1

    # Batch: all cases with masks
    python tools/run_sam2_track.py \
        --video_dir work_dirs/hunyuan/videos \
        --case_dir data/cases --mask_dir data/masks \
        --output_base work_dirs/hunyuan/masks

Output:
    {output_dir}/{frame_id:05d}.png — per-frame binary masks (0/255)
"""

import argparse
import contextlib
import glob
import json
import logging
import os
import tempfile
import time

import cv2
import numpy as np
from PIL import Image

from worldfoundry.base_models.perception_core.segment.sam2 import (
    checkpoint_path as sam2_checkpoint_path,
)
from worldfoundry.base_models.perception_core.segment.sam2 import (
    config_name as sam2_config_name,
)
from worldfoundry.base_models.perception_core.segment.sam2.build_sam import build_sam2_video_predictor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAM2_CONFIG = sam2_config_name()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_predictor(device="cuda"):
    ckpt_path = str(sam2_checkpoint_path())
    logger.info(f"Loading SAM2: {ckpt_path}")
    predictor = build_sam2_video_predictor(SAM2_CONFIG, ckpt_path, device=device)
    return predictor


def extract_frames_to_dir(video_path, output_dir, target_fps=5.0):
    """Extract frames at target_fps to a directory for SAM2 video predictor."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if 0 < target_fps < native_fps:
        step = native_fps / target_fps
        indices = sorted({round(i * step) for i in range(int(total / step)) if round(i * step) < total})
    else:
        indices = list(range(total))

    sample_set = set(indices)
    frame_indices = []
    fid = 0
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fid in sample_set:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(os.path.join(output_dir, f"{frame_count:05d}.jpg"))
            frame_indices.append(fid)
            frame_count += 1
        fid += 1
    cap.release()
    return frame_indices, native_fps


def track_video(
    predictor, video_path, init_mask_path, output_dir, device="cuda", target_fps=5.0, initial_image_path=None
):
    """Track subject mask through video using native SAM2 API."""
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    init_mask = (np.array(Image.open(init_mask_path).convert("L")) > 127).astype(np.uint8)

    with tempfile.TemporaryDirectory(prefix="sam2_frames_") as frames_dir:
        frame_indices, native_fps = extract_frames_to_dir(video_path, frames_dir, target_fps)
        n_frames = len(frame_indices)
        if n_frames == 0:
            raise ValueError(f"No frames: {video_path}")

        # If initial image provided, prepend as frame 0
        if initial_image_path and os.path.isfile(initial_image_path):
            # Shift all frames by 1
            for i in range(n_frames - 1, -1, -1):
                os.rename(
                    os.path.join(frames_dir, f"{i:05d}.jpg"),
                    os.path.join(frames_dir, f"{i + 1:05d}.jpg"),
                )
            # Read video frame size for resizing
            sample = Image.open(os.path.join(frames_dir, "00001.jpg"))
            video_w, video_h = sample.size
            init_img = Image.open(initial_image_path).convert("RGB").resize((video_w, video_h), Image.LANCZOS)
            init_img.save(os.path.join(frames_dir, "00000.jpg"))
            has_init_frame = True
        else:
            has_init_frame = False
            # Get video size from first frame
            sample = Image.open(os.path.join(frames_dir, "00000.jpg"))
            video_w, video_h = sample.size

        # Resize mask to video size
        mask_h, mask_w = init_mask.shape
        if mask_h != video_h or mask_w != video_w:
            mask_2d = np.array(Image.fromarray(init_mask * 255).resize((video_w, video_h), Image.NEAREST)) > 127
        else:
            mask_2d = init_mask > 0

        # Initialize SAM2 video state
        inference_state = predictor.init_state(video_path=frames_dir)

        # Add mask on frame 0
        _, obj_ids, mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            mask=mask_2d.astype(np.uint8),
        )

        # Propagate through all frames
        frame_masks = {}
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8)
            frame_masks[frame_idx] = mask

        # Save masks — skip frame 0 if it's the prepended initial image
        if has_init_frame:
            for sam_idx_offset, orig_fid in enumerate(frame_indices):
                sam_idx = sam_idx_offset + 1
                mask = frame_masks.get(sam_idx, np.zeros((video_h, video_w), dtype=np.uint8))
                Image.fromarray(mask * 255, mode="L").save(os.path.join(output_dir, f"{orig_fid:05d}.png"))
        else:
            for sam_idx, orig_fid in enumerate(frame_indices):
                mask = frame_masks.get(sam_idx, np.zeros((video_h, video_w), dtype=np.uint8))
                Image.fromarray(mask * 255, mode="L").save(os.path.join(output_dir, f"{orig_fid:05d}.png"))

        predictor.reset_state(inference_state)

    elapsed = time.time() - t0
    logger.info(f"Tracked {n_frames} frames in {elapsed:.1f}s → {output_dir}")
    return n_frames, elapsed


def _gpu_worker(gpu_id, worker_idx, n_workers, task_list, output_base, fps):
    """Multi-GPU worker: load SAM2 on assigned GPU, process cases sequentially."""
    # CPU affinity
    try:
        available = sorted(os.sched_getaffinity(0))
    except AttributeError:
        available = list(range(os.cpu_count() or 64))
    total_cpus = len(available)
    n_cores = max(1, total_cpus // n_workers)
    n_threads = max(1, total_cpus // (2 * n_workers))
    start = worker_idx * n_cores
    cpu_ids = available[start : start + n_cores]

    with contextlib.suppress(AttributeError, OSError):
        os.sched_setaffinity(0, cpu_ids)

    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch

    torch.set_num_threads(n_threads)

    device = "cuda:0"
    tag = f"[GPU{gpu_id}]"
    logger.info(
        f"{tag} Worker {worker_idx} started: {len(task_list)} cases, "
        f"threads={n_threads}, cpus={cpu_ids[0]}-{cpu_ids[-1]}"
    )

    predictor = build_predictor(device)

    n_total = len(task_list)
    done, fail = 0, 0
    for i, (case_id, mask_path, video_path, image_path) in enumerate(task_list):
        logger.info(f"{tag} [{i + 1}/{n_total}] case_{case_id}")
        out_dir = os.path.join(output_base, f"case_{case_id}")
        try:
            track_video(
                predictor, video_path, mask_path, out_dir, device=device, target_fps=fps, initial_image_path=image_path
            )
            done += 1
        except Exception as e:
            logger.error(f"{tag} case_{case_id}: {e}")
            fail += 1

    logger.info(f"{tag} Finished: {done}/{n_total} ok, {fail} fail")


def _collect_tasks(case_dir, mask_dir, video_dir, output_base, force=False):
    """Collect all trackable cases."""
    case_files = sorted(glob.glob(os.path.join(case_dir, "case_*.json")))
    tasks = []
    skipped = 0

    for cf in case_files:
        with open(cf) as f:
            data = json.load(f)
        case_id = data.get("id")
        mask_path = os.path.join(mask_dir, f"case_{case_id}_mask.png")
        if not os.path.isfile(mask_path):
            continue
        video_path = os.path.join(video_dir, f"case_{case_id}_combined.mp4")
        if not os.path.isfile(video_path):
            continue
        out_dir = os.path.join(output_base, f"case_{case_id}")
        if not force and os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
            skipped += 1
            continue
        image_path = os.path.join(os.path.dirname(case_dir), "images", f"case_{case_id}.jpg")
        if not os.path.isfile(image_path):
            image_path = None
        tasks.append((case_id, mask_path, video_path, image_path))

    return tasks, skipped


def main():
    parser = argparse.ArgumentParser(description="SAM2 video mask tracking (native)")
    parser.add_argument("--video", type=str, help="Single video path")
    parser.add_argument("--mask", type=str, help="Initial mask PNG (single mode)")
    parser.add_argument("--initial_image", type=str, help="Original first-frame image")
    parser.add_argument("--output_dir", type=str, help="Output mask directory (single mode)")
    parser.add_argument("--video_dir", type=str, help="Batch: video directory")
    parser.add_argument("--case_dir", type=str, help="Batch: case JSON directory")
    parser.add_argument("--mask_dir", type=str, help="Batch: initial mask directory")
    parser.add_argument("--output_base", type=str, help="Batch: output base directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpus", type=str, default=None, help="Multi-GPU: comma-separated GPU IDs (e.g. 0,1,2,3)")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # ── Single video mode ──
    if args.video and args.mask:
        predictor = build_predictor(args.device)
        output_dir = args.output_dir or "masks_output"
        track_video(
            predictor,
            args.video,
            args.mask,
            output_dir,
            device=args.device,
            target_fps=args.fps,
            initial_image_path=args.initial_image,
        )
        return

    # ── Batch mode ──
    if args.video_dir and args.case_dir and args.mask_dir:
        output_base = args.output_base or os.path.join(os.path.dirname(args.video_dir), "masks")
        tasks, skipped = _collect_tasks(args.case_dir, args.mask_dir, args.video_dir, output_base, args.force)

        logger.info(f"Tasks: {len(tasks)}, Skipped: {skipped}")

        if not tasks:
            logger.info("Nothing to do.")
            return

        # Multi-GPU mode
        if args.gpus:
            import multiprocessing as mp

            mp.set_start_method("spawn", force=True)

            gpu_ids = [int(g) for g in args.gpus.split(",")]
            n_workers = min(len(gpu_ids), len(tasks))

            # Distribute tasks
            worker_tasks = [[] for _ in range(n_workers)]
            for i, t in enumerate(tasks):
                worker_tasks[i % n_workers].append(t)

            logger.info(f"Multi-GPU: {n_workers} workers on GPUs {gpu_ids[:n_workers]}")

            processes = []
            for w in range(n_workers):
                p = mp.Process(
                    target=_gpu_worker,
                    args=(gpu_ids[w], w, n_workers, worker_tasks[w], output_base, args.fps),
                )
                p.start()
                processes.append(p)

            done, fail = 0, 0
            for p in processes:
                p.join()
                if p.exitcode == 0:
                    done += 1
                else:
                    fail += 1

            logger.info("All workers done.")
        else:
            # Single GPU mode
            predictor = build_predictor(args.device)
            done, fail = 0, 0
            for case_id, mask_path, video_path, image_path in tasks:
                out_dir = os.path.join(output_base, f"case_{case_id}")
                try:
                    track_video(
                        predictor,
                        video_path,
                        mask_path,
                        out_dir,
                        device=args.device,
                        target_fps=args.fps,
                        initial_image_path=image_path,
                    )
                    done += 1
                except Exception as e:
                    logger.error(f"case_{case_id}: {e}")
                    fail += 1
            logger.info(f"Done: {done}, Failed: {fail}")

        logger.info(f"Done: {done}, skipped: {skipped}, failed: {fail}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
