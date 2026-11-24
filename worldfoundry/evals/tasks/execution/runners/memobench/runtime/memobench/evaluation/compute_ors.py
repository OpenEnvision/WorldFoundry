import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from worldfoundry.evals.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# Add SAM-3 to path — update to your SAM-3 installation directory
sys.path.insert(0, os.environ.get("SAM3_DIR", "third_party/sam3"))
import cv2

GT_SYNTHETIC = "data/Synthetic_processed"
GT_REAL = "data/Real_Raw"
KF_SYN_XLS = "data/Synthetic_processed/Synthetic_ExitReenter.xlsx"
KF_REAL_XLS = "data/Real_Raw/Real_ExitReenter.xlsx"
SAM3_OUT_SYN = "data/sam3_output/synthetic"
SAM3_OUT_REAL = "data/sam3_output/real"
SAM3_DATA = os.environ.get(
    "WORLDFOUNDRY_MEMOBENCH_SAM3_METADATA_DIR",
    str(bundled_benchmark_asset("memobench", "data", "sam3_metadata")),
)
OUTPUT_DIR = "ors_results"

MODEL_CONFIGS = {
    "LingBot-World": {
        "synthetic": ["output/lingbot-world/Synthetic"],
        "real": ["output/lingbot-world/Real"],
    },
    "Wan2.2": {
        "synthetic": ["output/wan2.2/Synthetic"],
        "real": ["output/wan2.2/Real"],
    },
    "FantasyWorld": {
        "synthetic": ["output/fantasyworld/Synthetic"],
        "real": ["output/fantasyworld/Real"],
    },
    "Open-SoRA": {
        "synthetic": ["output/open-sora/Synthetic"],
        "real": ["output/open-sora/Real"],
    },
    "LTX-Video": {
        "synthetic": ["output/ltx-video/Synthetic"],
        "real": ["output/ltx-video/Real"],
    },
    "CogVideoX": {
        "synthetic": ["output/cogvideox/Synthetic"],
        "real": ["output/cogvideox/Real"],
    },
    "Matrix-Game2": {
        "synthetic": ["output/matrix-game2/Synthetic"],
        "real": ["output/matrix-game2/Real"],
    },
    "StableVirtualCamera": {
        "synthetic": ["output/stable-virtual-camera/Synthetic"],
        "real": ["output/stable-virtual-camera/Real"],
    },
}

_CLIP_RE = re.compile(r"^(.+)_(\d+)$")


def load_keyframe_map_synthetic() -> dict:
    """Returns {video_folder: {"h_start": int, "r_start": int}}."""
    if not os.path.exists(KF_SYN_XLS):
        return {}
    df = pd.read_excel(KF_SYN_XLS)
    skip_col = [c for c in df.columns if c.lower() == "skip"]
    if skip_col:
        df = df[df[skip_col[0]].astype(str).str.lower() != "yes"]
    result = {}
    for _, row in df.iterrows():
        vid = str(row["video_folder"]).strip()
        try:
            h = int(float(row["exits"]))
            r = int(float(row["re-enter"]))
        except (ValueError, TypeError, KeyError):
            continue
        result[vid] = {"h_start": h, "r_start": r}
    return result


def load_keyframe_map_real() -> dict:
    """Returns {"001": {"h_start": int, "r_start": int}}."""
    if not os.path.exists(KF_REAL_XLS):
        return {}
    df = pd.read_excel(KF_REAL_XLS)
    result = {}
    for _, row in df.iterrows():
        try:
            clip_id = str(int(float(row["id"]))).zfill(3)
            h = int(float(row["exit"]))
            r = int(float(row["reenter"]))
        except (ValueError, TypeError, KeyError):
            continue
        result[clip_id] = {"h_start": h, "r_start": r}
    return result


def map_keyframes(h_gt, r_gt, gt_total, gen_total):
    """Scale GT keyframe indices to generated video frame space."""
    h = round(h_gt / gt_total * gen_total)
    r = round(r_gt / gt_total * gen_total)
    h = max(2, min(h, gen_total - 4))
    r = max(h + 2, min(r, gen_total - 1))
    return int(h), int(r)


def gt_frame_count_synthetic(scene, clip_name):
    intr_path = os.path.join(GT_SYNTHETIC, scene, clip_name, "intrinsics.npy")
    if os.path.exists(intr_path):
        return np.load(intr_path).shape[0]
    return None


def gt_frame_count_real(clip_id):
    ts_path = os.path.join(GT_REAL, clip_id, "timestamps.txt")
    if os.path.exists(ts_path):
        with open(ts_path) as f:
            return sum(1 for line in f if line.strip())
    return None


def load_sam3_metadata():
    syn_meta, real_meta = {}, {}

    syn_csv = os.path.join(SAM3_DATA, "synthetic_metadata.csv")
    if os.path.exists(syn_csv):
        with open(syn_csv) as csv_file:
            for row in csv.DictReader(csv_file):
                scene = row["scene"]
                vid = int(row["video_id"])
                clip_id = f"{scene}_{vid:03d}"
                mask_path = os.path.join(SAM3_OUT_SYN, scene, f"video{vid}_mask.png")
                syn_meta[clip_id] = {
                    "subject": row["subject"],
                    "prompt": row.get("prompt", ""),
                    "mask": mask_path,
                }

    real_csv = os.path.join(SAM3_DATA, "real_metadata.csv")
    if os.path.exists(real_csv):
        with open(real_csv) as csv_file:
            for row in csv.DictReader(csv_file):
                clip_id = str(row["video_id"]).zfill(3)
                mask_path = os.path.join(SAM3_OUT_REAL, clip_id, "frame_mask.png")
                real_meta[clip_id] = {
                    "subject": row["subject"],
                    "prompt": row.get("prompt", ""),
                    "mask": mask_path,
                }

    return syn_meta, real_meta


def extract_subject_phrase(prompt_text):
    """Extract 'sees [text](subject) ...' for SAM-3 context."""
    m = re.search(r"sees (.*?\(subject\)[^.,]*)", prompt_text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".,")
    return None


def find_gen_frames(search_dirs, clip_name):
    """
    Find generated video frames for a clip across multiple search directories.
    Returns sorted list of frame paths, or None.
    Handles: nested (Scene/clip/frames/), flat (clip/frames/), PNG-direct, mp4.
    """
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        # Recursive search for the clip folder
        candidates = glob.glob(os.path.join(base, "**", clip_name), recursive=True)
        for cand in candidates:
            if not os.path.isdir(cand):
                continue
            # Check frames/ subdir
            frames_dir = os.path.join(cand, "frames")
            if os.path.isdir(frames_dir):
                pngs = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
                if pngs:
                    return pngs
            # Check samples-rgb/ subdir (Stable-Virtual-Camera)
            samples_dir = os.path.join(cand, "samples-rgb")
            if os.path.isdir(samples_dir):
                pngs = sorted(glob.glob(os.path.join(samples_dir, "*.png")))
                if pngs:
                    return pngs
            # Check PNG-direct
            pngs = sorted(glob.glob(os.path.join(cand, "*.png")))
            if pngs:
                return pngs
    return None


def find_gen_mp4(search_dirs, clip_name):
    """Fallback: find {clip_name}.mp4 in search dirs (CogVideo)."""
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        mp4 = os.path.join(base, f"{clip_name}.mp4")
        if os.path.isfile(mp4):
            return mp4
        # Also try video.mp4 inside clip dir (FantasyWorld)
        vid_mp4 = os.path.join(base, clip_name, "video.mp4")
        if os.path.isfile(vid_mp4):
            return vid_mp4
    return None


def extract_frames_from_mp4(mp4_path):
    """Extract all frames from an mp4 as numpy arrays. Returns list of BGR arrays."""
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def find_gen_frames_real(search_dirs, clip_id):
    """Find generated frames for a real clip. Handles {NNN}/ and real_{NNN}/ layouts."""
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        # {NNN}/frames/*.png
        frames_dir = os.path.join(base, clip_id, "frames")
        if os.path.isdir(frames_dir):
            pngs = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
            if pngs:
                return pngs
        # {NNN}/samples-rgb/*.png (Stable-Virtual-Camera)
        samples_dir = os.path.join(base, clip_id, "samples-rgb")
        if os.path.isdir(samples_dir):
            pngs = sorted(glob.glob(os.path.join(samples_dir, "*.png")))
            if pngs:
                return pngs
        # real_{NNN}/*.png (OpenSora)
        real_dir = os.path.join(base, f"real_{clip_id}")
        if os.path.isdir(real_dir):
            pngs = sorted(glob.glob(os.path.join(real_dir, "*.png")))
            if pngs:
                return pngs
        # PNG-direct in {NNN}/
        clip_dir = os.path.join(base, clip_id)
        if os.path.isdir(clip_dir):
            pngs = sorted(glob.glob(os.path.join(clip_dir, "*.png")))
            if pngs:
                return pngs
    return None


def find_gen_mp4_real(search_dirs, clip_id):
    """Fallback for real clips: {NNN}.mp4 or {NNN}/video.mp4."""
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        mp4 = os.path.join(base, f"{clip_id}.mp4")
        if os.path.isfile(mp4):
            return mp4
        vid_mp4 = os.path.join(base, clip_id, "video.mp4")
        if os.path.isfile(vid_mp4):
            return vid_mp4
    return None


def init_sam3():
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    return processor


def run_sam3_on_frame(processor, frame_bgr, prompt_str):
    """
    Run SAM-3 on a single frame to detect the target object.
    Returns (detected: bool, confidence: float).
    confidence = best mask score if detected, 0.0 otherwise.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    W, H = image.size

    processor.set_confidence_threshold(0.0)
    state = processor.set_image(image)
    state = processor.set_text_prompt(prompt_str, state)

    masks = state.get("masks", [])
    scores = state.get("scores", [])
    if len(masks) == 0:
        return False, 0.0

    img_area = W * H
    info = []
    for i in range(len(masks)):
        m = masks[i][0].cpu().numpy().astype(bool)
        area = int(m.sum())
        cov = area / img_area
        info.append((cov, i, float(scores[i].item())))

    # Filter by coverage: 0.05%–50%, fallback to 0.05%–70%
    t1 = [(cov, i, sc) for cov, i, sc in info if 0.0005 <= cov <= 0.50]
    t2 = [(cov, i, sc) for cov, i, sc in info if 0.0005 <= cov <= 0.70]
    pool = t1 if t1 else t2
    if not pool:
        return False, 0.0

    pool.sort(key=lambda t: -t[2])  # highest score first
    _, _, best_score = pool[0]
    return True, best_score


def compute_ors_for_clip(processor, frame_paths_or_arrays, r_start, prompt_str, is_bgr_arrays=False):
    """
    Detection-based ORS for a single clip.

    Runs SAM-3 on each R-phase frame to check if the target object is detected.
    ORS = detection_rate × mean_confidence.

    Args:
        processor: SAM-3 processor
        frame_paths_or_arrays: list of frame file paths, or list of BGR arrays
        r_start: R-phase start frame index
        prompt_str: text prompt for SAM-3
        is_bgr_arrays: True if frame_paths_or_arrays contains BGR arrays

    Returns:
        dict with: ors, detection_rate, mean_confidence, n_r_frames, n_detected
    """
    n_total = len(frame_paths_or_arrays)
    r_frames_indices = list(range(r_start, n_total))

    if not r_frames_indices:
        return {"ors": 0.0, "detection_rate": 0.0, "mean_confidence": 0.0, "n_r_frames": 0, "n_detected": 0}

    n_detected = 0
    confidences = []

    for idx in r_frames_indices:
        frame = frame_paths_or_arrays[idx] if is_bgr_arrays else cv2.imread(frame_paths_or_arrays[idx])
        if frame is None:
            continue

        detected, confidence = run_sam3_on_frame(processor, frame, prompt_str)
        if detected:
            n_detected += 1
            confidences.append(confidence)

    n_r = len(r_frames_indices)
    detection_rate = n_detected / n_r if n_r > 0 else 0.0
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    ors = detection_rate * mean_conf

    return {
        "ors": round(ors, 4),
        "detection_rate": round(detection_rate, 4),
        "mean_confidence": round(mean_conf, 4),
        "n_r_frames": n_r,
        "n_detected": n_detected,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Object Revisit Score (ORS)")
    parser.add_argument(
        "--model-name",
        required=True,
        choices=[*list(MODEL_CONFIGS.keys()), "all"],
        help="Model to evaluate, or 'all' for all models.",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help=f"Output directory. Default: {OUTPUT_DIR}")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip clips that already have a score in the output CSV."
    )
    return parser.parse_args()


def run_model(model_name, processor, syn_meta, real_meta, kf_syn, kf_real, args):
    config = MODEL_CONFIGS[model_name]
    out_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "ors_scores.csv")

    existing = set()
    if args.skip_existing and os.path.exists(out_csv):
        with open(out_csv) as csv_file:
            for row in csv.DictReader(csv_file):
                existing.add(row["clip_id"])

    results = []
    total = failed = skipped = 0

    syn_items = sorted(syn_meta.items())
    for clip_id, meta in tqdm(syn_items, desc=f"[{model_name}] Synthetic", unit="clip"):
        total += 1
        if clip_id in existing:
            skipped += 1
            continue

        m = _CLIP_RE.match(clip_id)
        if not m:
            continue
        scene = m.group(1)

        if clip_id not in kf_syn:
            failed += 1
            continue
        kf = kf_syn[clip_id]
        n_gt = gt_frame_count_synthetic(scene, clip_id)
        if n_gt is None:
            failed += 1
            continue

        frame_paths = find_gen_frames(config["synthetic"], clip_id)
        is_bgr = False
        if frame_paths is None:
            mp4 = find_gen_mp4(config["synthetic"], clip_id)
            if mp4 is None:
                failed += 1
                continue
            frame_paths = extract_frames_from_mp4(mp4)
            is_bgr = True

        n_gen = len(frame_paths)
        _, r_start = map_keyframes(kf["h_start"], kf["r_start"], n_gt, n_gen)

        phrase = extract_subject_phrase(meta["prompt"])
        prompt = phrase if phrase else meta["subject"]

        ors = compute_ors_for_clip(processor, frame_paths, r_start, prompt, is_bgr_arrays=is_bgr)

        results.append(
            {
                "clip_id": clip_id,
                "data_type": "synthetic",
                "scene": scene,
                "n_gen": n_gen,
                "r_start": r_start,
                "ors": ors["ors"],
                "detection_rate": ors["detection_rate"],
                "mean_confidence": ors["mean_confidence"],
                "n_r_frames": ors["n_r_frames"],
                "n_detected": ors["n_detected"],
            }
        )

    real_items = sorted(real_meta.items())
    for clip_id, meta in tqdm(real_items, desc=f"[{model_name}] Real", unit="clip"):
        total += 1
        if clip_id in existing:
            skipped += 1
            continue

        if clip_id not in kf_real:
            failed += 1
            continue
        kf = kf_real[clip_id]
        n_gt = gt_frame_count_real(clip_id)
        if n_gt is None:
            failed += 1
            continue

        frame_paths = find_gen_frames_real(config["real"], clip_id)
        is_bgr = False
        if frame_paths is None:
            mp4 = find_gen_mp4_real(config["real"], clip_id)
            if mp4 is None:
                failed += 1
                continue
            frame_paths = extract_frames_from_mp4(mp4)
            is_bgr = True

        n_gen = len(frame_paths)
        _, r_start = map_keyframes(kf["h_start"], kf["r_start"], n_gt, n_gen)

        prompt = meta["subject"]

        ors = compute_ors_for_clip(processor, frame_paths, r_start, prompt, is_bgr_arrays=is_bgr)

        results.append(
            {
                "clip_id": clip_id,
                "data_type": "real",
                "scene": "real",
                "n_gen": n_gen,
                "r_start": r_start,
                "ors": ors["ors"],
                "detection_rate": ors["detection_rate"],
                "mean_confidence": ors["mean_confidence"],
                "n_r_frames": ors["n_r_frames"],
                "n_detected": ors["n_detected"],
            }
        )

    if results:
        fieldnames = results[0].keys()

        if args.skip_existing and os.path.exists(out_csv):
            with open(out_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(results)
        else:
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

    if results:
        syn_scores = [r["ors"] for r in results if r["data_type"] == "synthetic"]
        real_scores = [r["ors"] for r in results if r["data_type"] == "real"]
        all_scores = [r["ors"] for r in results]
        print(f"\n{'=' * 60}")
        print(f"[{model_name}] SUMMARY")
        if syn_scores:
            print(f"  Synthetic  ORS mean: {np.mean(syn_scores):.4f}  ({len(syn_scores)} clips)")
        if real_scores:
            print(f"  Real       ORS mean: {np.mean(real_scores):.4f}  ({len(real_scores)} clips)")
        print(f"  Overall    ORS mean: {np.mean(all_scores):.4f}  ({len(all_scores)} clips)")
        print(f"  Saved → {out_csv}")

    print(f"\n  total={total}  processed={len(results)}  skipped={skipped}  failed={failed}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SAM-3 model...")
    processor = init_sam3()
    print("SAM-3 ready.\n")

    print("Loading metadata...")
    syn_meta, real_meta = load_sam3_metadata()
    kf_syn = load_keyframe_map_synthetic()
    kf_real = load_keyframe_map_real()
    print(f"  Synthetic: {len(syn_meta)} clips with prompts, {len(kf_syn)} with keyframes")
    print(f"  Real:      {len(real_meta)} clips with prompts, {len(kf_real)} with keyframes")

    models = list(MODEL_CONFIGS.keys()) if args.model_name == "all" else [args.model_name]

    for model in models:
        print(f"\n{'#' * 60}")
        print(f"# MODEL: {model}")
        print(f"{'#' * 60}")
        run_model(model, processor, syn_meta, real_meta, kf_syn, kf_real, args)

    print("\nAll done.")


if __name__ == "__main__":
    main()
