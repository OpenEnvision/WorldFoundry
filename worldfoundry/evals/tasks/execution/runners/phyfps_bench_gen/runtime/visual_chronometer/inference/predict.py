"""
Visual Chronometer — PhyFPS Inference Script

Predicts the Physical FPS (PhyFPS) of input videos using the Visual Chronometer model.
Outputs per-segment and average PhyFPS predictions, with an optional visualization table.

Usage:
    cd inference
    python predict.py --video_path demo_videos/gymnast_50fps.mp4
    python predict.py --video_dir demo_videos/ --output_csv results.csv
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.common_utils import instantiate_from_config

HF_REPO_ID = "xiangbog/Visual_Chronometer"
HF_CKPT_FILENAME = "vc_common_10_60fps.ckpt"


def download_checkpoint(ckpt_path):
    """Download checkpoint from HuggingFace if not present locally."""
    if os.path.exists(ckpt_path):
        return ckpt_path
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(  # noqa: B904 - Keeps the implicit exception context.
            "Checkpoint not found and `huggingface_hub` is not installed.\n"
            "Install the Visual Chronometer/PhyFPS benchmark profile, or stage the checkpoint at: "
            f"{ckpt_path}"
        )
    print(f"Downloading checkpoint from HuggingFace ({HF_REPO_ID})...")
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_CKPT_FILENAME,
        local_dir=os.path.dirname(ckpt_path),
    )
    print(f"Checkpoint saved to: {downloaded}")
    return downloaded


def load_model(config_path, ckpt_path, device):
    ckpt_path = download_checkpoint(ckpt_path)
    config = OmegaConf.load(config_path)
    config.model.params.freeze_encoder = False
    if "ckpt_path" in config.model.params:
        config.model.params.ckpt_path = None
    model = instantiate_from_config(config.model)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def extract_segments(video_path, clip_length=30, stride=4, resolution=216):
    """Extract overlapping video segments using a sliding window."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (resolution, resolution))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) < clip_length:
        raise RuntimeError(f"Video too short ({len(frames)} frames < {clip_length})")

    segments = []
    for start in range(0, len(frames) - clip_length + 1, stride):
        clip = np.stack(frames[start : start + clip_length])
        clip = clip.astype(np.float32) / 127.5 - 1.0  # normalize to [-1, 1]
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2)  # [C, T, H, W]
        segments.append((start, clip))

    return segments, len(frames)


@torch.no_grad()
def predict_video(model, video_path, device, clip_length=30, stride=4, resolution=216):
    """Predict PhyFPS for each segment of a video."""
    segments, total_frames = extract_segments(video_path, clip_length, stride, resolution)
    results = []

    for start_frame, clip in segments:
        clip = clip.unsqueeze(0).to(device)
        pred_log_fps = model(clip)
        fps = torch.exp(pred_log_fps).item()
        mid_frame = start_frame + clip_length // 2
        results.append(
            {
                "start_frame": start_frame,
                "mid_frame": mid_frame,
                "end_frame": start_frame + clip_length - 1,
                "predicted_phyfps": round(fps, 1),
            }
        )

    avg_fps = np.mean([r["predicted_phyfps"] for r in results])
    return results, round(avg_fps, 1), total_frames


def print_table(video_name, results, avg_fps, stride=4):
    """Print a formatted table of per-segment PhyFPS predictions."""
    print(f"\n{'=' * 60}")
    print(f"  Video: {video_name}")
    print(f"  Average PhyFPS: {avg_fps}")
    print(f"{'=' * 60}")
    print(f"  {'Segment':>8}  {'Frames':>12}  {'Mid Frame':>10}  {'PhyFPS':>8}")
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 10}  {'-' * 8}")
    for i, r in enumerate(results):
        print(
            f"  {i:>8d}  {r['start_frame']:>5d}-{r['end_frame']:<5d}  {r['mid_frame']:>10d}  {r['predicted_phyfps']:>8.1f}"
        )
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 10}  {'-' * 8}")
    print(f"  {'AVG':>8}  {'':>12}  {'':>10}  {avg_fps:>8.1f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Visual Chronometer: Predict PhyFPS from video")
    parser.add_argument("--video_path", type=str, help="Path to a single video file")
    parser.add_argument("--video_dir", type=str, help="Directory of videos to process")
    parser.add_argument("--config", type=str, default="configs/config_fps.yaml")
    parser.add_argument("--ckpt", type=str, default="ckpts/vc_common_10_60fps.ckpt")
    parser.add_argument("--clip_length", type=int, default=30, help="Frames per clip (default: 30)")
    parser.add_argument("--stride", type=int, default=4, help="Sliding window stride (default: 4)")
    parser.add_argument("--resolution", type=int, default=216, help="Resize resolution (default: 216)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_csv", type=str, default=None, help="Save results to CSV")
    args = parser.parse_args()

    if not args.video_path and not args.video_dir:
        parser.error("Provide either --video_path or --video_dir")

    print("Loading Visual Chronometer model...")
    model = load_model(args.config, args.ckpt, args.device)
    print("Model loaded.\n")

    videos = []
    if args.video_path:
        videos.append(args.video_path)
    else:
        exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        for f in sorted(os.listdir(args.video_dir)):
            if os.path.splitext(f)[1].lower() in exts:
                videos.append(os.path.join(args.video_dir, f))

    all_results = []
    for vpath in videos:
        vname = os.path.basename(vpath)
        try:
            results, avg_fps, total_frames = predict_video(
                model, vpath, args.device, args.clip_length, args.stride, args.resolution
            )
            print_table(vname, results, avg_fps, args.stride)
            all_results.append(
                {"video": vname, "avg_phyfps": avg_fps, "segments": results, "total_frames": total_frames}
            )
        except Exception as e:
            print(f"  Error processing {vname}: {e}")

    if args.output_csv and all_results:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video", "segment", "start_frame", "mid_frame", "end_frame", "predicted_phyfps"])
            for entry in all_results:
                for i, seg in enumerate(entry["segments"]):
                    writer.writerow(
                        [
                            entry["video"],
                            i,
                            seg["start_frame"],
                            seg["mid_frame"],
                            seg["end_frame"],
                            seg["predicted_phyfps"],
                        ]
                    )
                writer.writerow([entry["video"], "AVG", "", "", "", entry["avg_phyfps"]])
        print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
