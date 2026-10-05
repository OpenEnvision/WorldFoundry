#!/usr/bin/env python3
"""Single-video SAMA-14B inference using semantic pipeline."""

from __future__ import annotations

import argparse
import math
import os
from typing import List, Optional

import numpy as np
import torch
import imageio
from PIL import Image, ImageOps
from worldfoundry.core.io import VideoData, save_video
from worldfoundry.synthesis.visual_generation.sama.native_pipeline import (
    SamaWanPipeline,
    load_sama_pipeline,
)

DEFAULT_MODEL_ROOT = ""

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")


def find_closest_resolution(w: int, h: int, ratios, resolutions):
    input_ar = w / h
    best_idx = 0
    best_diff = float("inf")
    for idx, resolution_ar in enumerate(ratios):
        diff = abs(resolution_ar - input_ar)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx
    return resolutions[best_idx]


def get_all_resolution(target_pixels: int, factor: int = 32, min_ar: float = 0.5, max_ar: float = 2.0):
    h_min = math.sqrt(target_pixels / max_ar)
    h_max = math.sqrt(target_pixels / min_ar)
    h_min_aligned = math.floor(h_min / factor) * factor
    h_max_aligned = math.ceil(h_max / factor) * factor

    resolutions = []
    for height in range(h_min_aligned, h_max_aligned + 1, factor):
        width = round((target_pixels / height) / factor) * factor
        aspect_ratio = width / height
        if min_ar <= aspect_ratio <= max_ar:
            resolutions.append((width, height))
    return resolutions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-video Wan2.1 semantic inference")
    parser.add_argument("--src-video", required=True, help="Path to the source video file")
    parser.add_argument("--prompt", required=True, help="Text instruction / prompt")
    parser.add_argument("--output-dir", default="tmp", help="Output directory (default: tmp)")
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT, help="Root directory of Wan2.1-T2V-14B model weights")
    parser.add_argument("--device", default="cuda:0", help="Device string (default: cuda:0)")
    parser.add_argument("--lora-path", default="", help="LoRA checkpoint path (optional)")
    parser.add_argument("--state-dict", default="", help="Full DiT weight checkpoint path")
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt override (defaults to '[Video edit]' with --prompt-prefix, otherwise empty)",
    )
    parser.add_argument("--height", type=int, default=480, help="Target frame height (default: 480)")
    parser.add_argument("--width", type=int, default=832, help="Target frame width (default: 832)")
    parser.add_argument("--max-frames", type=int, default=81, help="Max frames to read from source video (default: 81)")
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS (default: 20)")
    parser.add_argument("--quality", type=int, default=5, help="save_video quality parameter (default: 5)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--tiled", action="store_true", help="Enable tiled inference")
    parser.add_argument("--prompt-prefix", action="store_true", help="Prepend '[Video edit]' to prompt")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists")

    args = parser.parse_args()

    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")

    args.src_video = os.path.abspath(args.src_video)
    if not os.path.exists(args.src_video):
        parser.error(f"--src-video not found: {args.src_video}")
    if not args.src_video.lower().endswith(VIDEO_EXTENSIONS):
        parser.error(f"Unsupported video extension: {args.src_video}")

    args.output_dir = os.path.abspath(args.output_dir)
    return args


# ---------- pipeline ----------

def init_pipeline(device: str, lora_path: str, state_dict: str, model_root: str) -> SamaWanPipeline:
    pipe = load_sama_pipeline(
        model_root,
        state_dict_path=state_dict.strip() or None,
        lora_path=lora_path.strip() or None,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_vram_management()
    return pipe


# ---------- video helpers ----------

def detect_video_resolution(video_path: str, fallback: tuple[int, int]) -> tuple[int, int]:
    width = height = None
    try:
        reader = imageio.get_reader(video_path)
        try:
            meta = reader.get_meta_data()
            size = meta.get("source_size") or meta.get("size")
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                width, height = int(size[0]), int(size[1])
        except Exception:
            pass
        finally:
            try:
                reader.close()
            except Exception:
                pass
    except Exception:
        pass
    if width and height:
        return width, height
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    except Exception:
        pass
    if width and height:
        return width, height
    return fallback


def detect_video_fps(video_path: str) -> Optional[float]:
    fps = None
    try:
        reader = imageio.get_reader(video_path)
        try:
            meta = reader.get_meta_data()
            fps = meta.get("fps") or meta.get("framerate")
        except Exception:
            pass
        finally:
            try:
                reader.close()
            except Exception:
                pass
    except Exception:
        pass
    if fps and fps > 0:
        return float(fps)
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or fps
        cap.release()
    except Exception:
        pass
    if fps and fps > 0:
        return float(fps)
    return None


def compute_letterbox_params(
    orig_w: int, orig_h: int, target_w: int, target_h: int
) -> tuple[int, int, int, int, int, int]:
    if orig_w <= 0 or orig_h <= 0:
        return target_w, target_h, 0, 0, 0, 0
    orig_ratio = orig_w / orig_h
    target_ratio = target_w / target_h
    if abs(orig_ratio - target_ratio) < 1e-3:
        return target_w, target_h, 0, 0, 0, 0
    if orig_ratio < target_ratio:
        resize_h = target_h
        resize_w = max(1, int(round(resize_h * orig_ratio)))
        pad_total = max(0, target_w - resize_w)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = pad_bottom = 0
    else:
        resize_w = target_w
        resize_h = max(1, int(round(resize_w / orig_ratio)))
        pad_total = max(0, target_h - resize_h)
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        pad_left = pad_right = 0
    return resize_w, resize_h, pad_left, pad_right, pad_top, pad_bottom


def _pad_numpy_array(arr: np.ndarray, pad_left: int, pad_right: int, pad_top: int, pad_bottom: int) -> np.ndarray:
    if arr.ndim == 2:
        pad_widths = ((pad_top, pad_bottom), (pad_left, pad_right))
    elif arr.ndim == 3:
        if arr.shape[0] <= 4:
            pad_widths = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
        else:
            pad_widths = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        raise ValueError(f"Unsupported array shape for padding: {arr.shape}")
    return np.pad(arr, pad_widths, mode="constant", constant_values=0)


def pad_frame(frame, pad_left: int, pad_right: int, pad_top: int, pad_bottom: int):
    if not (pad_left or pad_right or pad_top or pad_bottom):
        return frame
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
        padded = _pad_numpy_array(arr, pad_left, pad_right, pad_top, pad_bottom)
        return torch.from_numpy(padded).to(frame.dtype).to(frame.device)
    if isinstance(frame, Image.Image):
        return ImageOps.expand(frame, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
    if isinstance(frame, np.ndarray):
        return _pad_numpy_array(frame, pad_left, pad_right, pad_top, pad_bottom)
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def load_clip_frames(video_path: str, height: int, width: int, max_frames: Optional[int], resolutions=None) -> List[torch.Tensor]:
    if resolutions is None:
        orig_w, orig_h = detect_video_resolution(video_path, fallback=(width, height))
        resize_w, resize_h, pad_l, pad_r, pad_t, pad_b = compute_letterbox_params(orig_w, orig_h, width, height)
        reader = VideoData(video_file=video_path, height=resize_h, width=resize_w)
        total = len(reader)
        limit = min(total, max_frames) if max_frames is not None else total
        frames: List[torch.Tensor] = []
        for i in range(limit):
            frame = reader[i]
            frame = pad_frame(frame, pad_l, pad_r, pad_t, pad_b)
            frames.append(frame)
        return frames
    else:
        reader = VideoData(video_file=video_path)
        ratios = [res[0] / res[1] for res in resolutions]
        old_height, old_width = reader.shape()
        new_width, new_height = find_closest_resolution(old_width, old_height, ratios, resolutions)
        reader.set_shape(new_height, new_width)
        total = len(reader)
        limit = min(total, max_frames) if max_frames is not None else total
        frames: List[torch.Tensor] = []
        for i in range(limit):
            frames.append(reader[i])
        w, h = reader[0].size
        print(f"[Info] Resized: {old_height}x{old_width} -> {new_height}x{new_width}, actual: {h}x{w}", flush=True)
        return frames


def ensure_4k_plus_1_frames(frames: List[torch.Tensor]) -> tuple[List[torch.Tensor], bool]:
    """Pad frames until len % 4 == 1 (Wan requirement)."""
    if not frames:
        return frames, False
    remainder = len(frames) % 4
    if remainder == 1:
        return frames, False
    pad_count = (1 - remainder) % 4
    if pad_count <= 0:
        return frames, False
    last = frames[-1]
    for _ in range(pad_count):
        try:
            frames.append(last.clone())
        except AttributeError:
            frames.append(last)
    return frames, True


# ---------- main ----------

def main() -> None:
    args = parse_args()

    output_path = os.path.join(args.output_dir, os.path.basename(args.src_video))

    if os.path.exists(output_path) and not args.overwrite:
        print(f"[Skip] Output already exists: {output_path}. Use --overwrite to regenerate.", flush=True)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    pipe = init_pipeline(args.device, args.lora_path, args.state_dict, args.model_root)

    # Load frames
    resolutions = get_all_resolution(args.height * args.width, factor=32, min_ar=0.5, max_ar=2.0)
    frames = load_clip_frames(args.src_video, height=args.height, width=args.width,
                               max_frames=args.max_frames, resolutions=resolutions)
    if not frames:
        raise RuntimeError(f"No frames decoded from: {args.src_video}")

    frames, padded = ensure_4k_plus_1_frames(frames)
    if padded:
        print(f"[Info] Adjusted frame count to {len(frames)} (Wan requires 4k+1)", flush=True)
    case_fps = detect_video_fps(args.src_video) or float(args.fps)
    width, height = frames[0].size

    prompt = args.prompt
    negative_prompt = args.negative_prompt
    if args.prompt_prefix:
        prompt = f"[Video edit] {prompt}"
    if negative_prompt is None:
        negative_prompt = "[Video edit]" if args.prompt_prefix else ""
    
    print(f"[Run] prompt='{prompt}' src='{args.src_video}' frames={len(frames)} "
          f"size={width}x{height} fps={case_fps} seed={args.seed}", flush=True)
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        source_video=frames,
        num_frames=len(frames),
        seed=args.seed,
        height=height,
        width=width,
        tiled=args.tiled,
    )

    save_video(video, output_path, fps=case_fps, quality=args.quality)
    print(f"[Done] Saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
