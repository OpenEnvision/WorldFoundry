"""
SAM-based segmentation pipeline for physical events.

This script takes a video path and prompt, asks a VLM to list the key
moving/deforming objects (max 5), then runs SAM3 to segment them and
exports visualization videos, a binary dynamic mask video, and a background-inpainted video:
- *_mask.mp4: mask overlays only (no boxes)
- *_bbox.mp4: bounding boxes only (no masks)
- *_dynamic_binary_mask.mp4: white dynamic object regions on black background
- *_dynamic_inpainted.mp4: input video with moving/deforming regions filled from other video frames

Outputs are saved beside the input video.

python physical/sam_process.py --video path/to/video.mp4 


python physical/sam_process.py \
    --video data/real/motor/motor_gt.mp4 \
    --sam3-model ./weights/sam3/sam3.pt \
    --device cuda:1 \
    --max-frames 512
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import imageio_ffmpeg

import subprocess
import tempfile

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.vlm import (
    parse_json_content,
    resolve_model_name,
    resolve_vlm_backend,
    video_vlm_call,
)

try:
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
except ImportError as exc:  # pragma: no cover - depends on optional package
    raise ImportError(
        "ultralytics with SAM3 support is required. Install with: pip install ultralytics"
    ) from exc

# ── Monkey-patch Results.verbose() so it never crashes on missing/mistyped names ──
try:
    from ultralytics.engine.results import Results as _UltralyticsResults

    _orig_verbose = _UltralyticsResults.verbose

    def _safe_verbose(self):
        try:
            return _orig_verbose(self)
        except Exception:
            # Fallback: build a safe string ourselves
            try:
                import numpy as _np
                if self.boxes is None or self.boxes.cls is None:
                    return ""
                counts = _np.bincount(
                    self.boxes.cls.cpu().numpy().astype(int),
                    minlength=0,
                )
                parts = []
                for i, n in enumerate(counts):
                    if n > 0:
                        try:
                            name = self.names[i] if self.names else f"cls{i}"
                        except Exception:
                            name = f"cls{i}"
                        parts.append(f"{n} {name}{'s' if n > 1 else ''}")
                return ", ".join(parts) + (", " if parts else "")
            except Exception:
                return ""

    _UltralyticsResults.verbose = _safe_verbose
except Exception as _patch_err:
    print(f"[WARN] Could not patch Results.verbose: {_patch_err}")
# ──────────────────────────────────────────────────────────────────────────────────

# Simple BGR palette for consistent coloring
PALETTE = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
]
DEFAULT_MAX_FRAMES = 128
DEFAULT_MAX_OBJECTS = 3
DEFAULT_SAM_CONF = 0.7
DEFAULT_MIN_MASK_AREA_RATIO = 0.005
DEFAULT_MAX_MASKS_PER_FRAME = 3


def encode_video_to_data_url(video_path: str) -> Dict:
    with open(video_path, "rb") as video_file:
        base64_video = base64.b64encode(video_file.read()).decode("utf-8")
    return {
        "type": "video_url",
        "video_url": {"url": f"data:video/mp4;base64,{base64_video}"},
    }


def _make_safe_names(text_prompts: List[str]) -> defaultdict:
    """
    Return a defaultdict that maps any integer key to a prompt name.
    This prevents KeyError in ultralytics verbose() regardless of the
    class indices returned by the model.
    """
    class _SafeNames(defaultdict):
        def __missing__(self, key):
            try:
                idx = int(key)
            except Exception:
                return "object"
            if text_prompts:
                return text_prompts[idx % len(text_prompts)]
            return f"object_{idx}"

    return _SafeNames(str, {i: name for i, name in enumerate(text_prompts)})


def extract_dynamic_objects(
    video_path: str,
    video_prompt: str,
    model_name: str | None = None,
    backend: str | None = None,
    max_objects: int = DEFAULT_MAX_OBJECTS,
) -> List[str]:
    """
    Use VLM to list key moving/deforming objects in the video.
    """
    system_prompt = """You are an expert at spotting only the primary, visually dominant physical actors that visibly MOVE or DEFORM in a video.

Rules:
- Return the FEWEST distinct objects that visibly move or deform (max 3).
- Prefer the largest and most visually prominent moving foreground object(s).
- Include a smaller object only if it is the main manipulated object or central to the described action.
- Omit tiny incidental objects, background details, and uncertain detections even if they move.
- Do NOT include static background, scenery, floors, tables, walls, tools, supports, containers, or objects that are merely visible.
- If an object does not visibly change position, orientation, or shape, do not return it.
- Merge duplicates/synonyms into one concise noun (1-3 words).
- Output MUST be a JSON array only, no explanations."""

    video_data_url = encode_video_to_data_url(video_path)
    user_prompt = f"""Watch this video and list only the largest, most visually prominent objects that visibly move or deform.

Video description: {video_prompt}

Return 1-3 concise nouns in JSON array format only. Exclude static objects even if they are mentioned in the description. Prefer a large moving subject over tiny moving fragments.
    Example: ["person", "ball"]"""

    try:
        result = video_vlm_call(
            data_url=video_data_url,
            system_prompt=system_prompt,
            user_content=user_prompt,
            model_name=model_name,
            backend=backend,
        )
        print(f"VLM response: {result}")
        parsed = parse_json_content(result)
        seen = set()
        unique = []
        for obj in parsed:
            if not isinstance(obj, str):
                continue
            name = obj.strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(name)
            if len(unique) >= max_objects:
                break
        if unique:
            return unique
    except Exception as exc:
        print(f"[WARN] VLM extraction failed, fallback to generic prompt. Error: {exc}")

    return ["object"]


def _init_video_writer(video_path: str | Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc_alt = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(video_path), fourcc_alt, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for {video_path}.")
    return writer


def _remove_non_h264_intermediates(video_paths: List[str]) -> None:
    for video_path in video_paths:
        path = Path(video_path)
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            print(f"[WARN] Failed to remove non-H264 intermediate video {path}: {exc}")
        else:
            print(f"Removed non-H264 intermediate video: {path}")


def _probe_video(video_path: str) -> Dict[str, float | int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Failed to read video resolution: {video_path}")

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


def _compute_sample_indices(total_frames: int, max_frames: int) -> List[int]:
    if total_frames <= 0:
        return []
    if max_frames <= 0 or total_frames <= max_frames:
        return list(range(total_frames))
    sampled = np.linspace(0, total_frames - 1, num=max_frames, dtype=np.int32)
    indices = np.unique(sampled).tolist()
    if indices[-1] != total_frames - 1:
        indices[-1] = total_frames - 1
    return indices


def _build_sampled_video(
    *,
    video_path: str,
    work_dir: Path,
    max_frames: int,
) -> tuple[str, Dict[str, float | int | str]]:
    video_meta = _probe_video(video_path)
    total_frames = int(video_meta["frame_count"])
    selected_indices = _compute_sample_indices(total_frames, max_frames)

    if not selected_indices or len(selected_indices) == total_frames:
        return video_path, {
            "source": "original",
            "original_frame_count": total_frames,
            "processing_frame_count": total_frames,
            "original_fps": video_meta["fps"],
            "processing_fps": video_meta["fps"],
            "max_frames": max_frames,
        }

    output_path = work_dir / f"{Path(video_path).stem}_sampled.mp4"
    sampled_frame_count = len(selected_indices)
    sampled_fps = max(0.1, float(video_meta["fps"]) * sampled_frame_count / max(total_frames, 1))
    writer = _init_video_writer(
        output_path,
        sampled_fps,
        int(video_meta["width"]),
        int(video_meta["height"]),
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        writer.release()
        raise RuntimeError(f"Cannot open video for sampling: {video_path}")

    selected_set = set(selected_indices)
    frame_idx = 0
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in selected_set:
                writer.write(frame)
                written += 1
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    if written == 0:
        raise RuntimeError(f"Temporal sampling produced an empty video for: {video_path}")

    print(
        f"Built sampled proxy video: kept {written}/{total_frames} frames "
        f"({float(video_meta['fps']):.3f} fps -> {sampled_fps:.3f} fps)"
    )
    return str(output_path), {
        "source": "sampled",
        "original_frame_count": total_frames,
        "processing_frame_count": written,
        "original_fps": video_meta["fps"],
        "processing_fps": sampled_fps,
        "max_frames": max_frames,
        "sampled_video_path": str(output_path),
    }


def _init_writers(
    video_path: str,
    output_video_path: str | None = None,
) -> Tuple[cv2.VideoWriter, cv2.VideoWriter, cv2.VideoWriter, cv2.VideoWriter]:
    """Prepare mask, bbox, binary dynamic mask, and dynamic-object-masked writers."""
    video_meta = _probe_video(video_path)
    output_stem_video = output_video_path or video_path
    fps = float(video_meta["fps"])
    width = int(video_meta["width"])
    height = int(video_meta["height"])

    mask_out = _init_video_writer(_mask_path(output_stem_video), fps, width, height)
    bbox_out = _init_video_writer(_bbox_path(output_stem_video), fps, width, height)
    binary_mask_out = _init_video_writer(_dynamic_binary_mask_path(output_stem_video), fps, width, height)
    masked_out = _init_video_writer(_dynamic_inpainted_path(output_stem_video), fps, width, height)
    return mask_out, bbox_out, binary_mask_out, masked_out


def _mask_path(video_path: str, output_video_path: str | None = None) -> Path:
    path = Path(output_video_path or video_path)
    return path.with_name(f"{path.stem}_mask.mp4")


def _bbox_path(video_path: str, output_video_path: str | None = None) -> Path:
    path = Path(output_video_path or video_path)
    return path.with_name(f"{path.stem}_bbox.mp4")


def _dynamic_inpainted_path(video_path: str, output_video_path: str | None = None) -> Path:
    path = Path(output_video_path or video_path)
    return path.with_name(f"{path.stem}_dynamic_inpainted.mp4")


def _dynamic_binary_mask_path(video_path: str, output_video_path: str | None = None) -> Path:
    path = Path(output_video_path or video_path)
    return path.with_name(f"{path.stem}_dynamic_binary_mask.mp4")


def _dilate_dynamic_mask(dynamic_mask: np.ndarray) -> np.ndarray:
    if not dynamic_mask.any():
        return dynamic_mask

    mask_uint8 = dynamic_mask.astype(np.uint8)
    kernel = np.ones((5, 5), dtype=np.uint8)
    return cv2.dilate(mask_uint8, kernel, iterations=2).astype(bool)


def _mask_area_ratio(mask: np.ndarray, frame_area: int) -> float:
    if frame_area <= 0:
        return 0.0
    return float(mask.sum()) / float(frame_area)


def _inpaint_video_background(
    frames: List[np.ndarray],
    dynamic_masks: List[np.ndarray],
) -> List[np.ndarray]:
    """Fill dynamic regions using unmasked pixels from neighboring video frames first."""
    if len(frames) != len(dynamic_masks):
        raise ValueError(f"Frame/mask count mismatch: {len(frames)} vs {len(dynamic_masks)}")

    inpainted_frames: List[np.ndarray] = []
    frame_count = len(frames)
    valid_source_masks = [
        (~mask) & ~((frame <= 8).all(axis=2))
        for frame, mask in zip(frames, dynamic_masks)
    ]

    for frame_idx, (frame, dynamic_mask) in enumerate(zip(frames, dynamic_masks)):
        output_frame = frame.copy()
        remaining_mask = dynamic_mask.copy()

        if remaining_mask.any():
            for offset in range(1, frame_count):
                for neighbor_idx in (frame_idx - offset, frame_idx + offset):
                    if neighbor_idx < 0 or neighbor_idx >= frame_count:
                        continue
                    fill_mask = remaining_mask & valid_source_masks[neighbor_idx]
                    if fill_mask.any():
                        output_frame[fill_mask] = frames[neighbor_idx][fill_mask]
                        remaining_mask[fill_mask] = False
                if not remaining_mask.any():
                    break

        if remaining_mask.any():
            mask_uint8 = (remaining_mask.astype(np.uint8) * 255)
            output_frame = cv2.inpaint(output_frame, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        inpainted_frames.append(output_frame)

    return inpainted_frames


def build_sam3_predictor(
    *,
    sam3_model_path: str,
    device: str,
    conf: float = DEFAULT_SAM_CONF,
    iou: float = 0.7,
) -> Any:
    """Build a SAM3 predictor that can be reused across multiple videos."""
    overrides = dict(
        conf=conf,
        iou=iou,
        task="segment",
        mode="predict",
        imgsz=800,
        model=sam3_model_path,
        half=True,
        save=False,
        device=device,
        show_boxes=True,
        show_labels=False,
        show_conf=False,
    )
    return SAM3VideoSemanticPredictor(overrides=overrides)


def segment_with_sam3(
    video_path: str,
    text_prompts: List[str],
    sam3_model_path: str = "./weights/sam3/sam3.pt",
    device: str = "cuda:1",
    conf: float = DEFAULT_SAM_CONF,
    iou: float = 0.7,
    output_video_path: str | None = None,
    predictor: Any | None = None,
    min_mask_area_ratio: float = DEFAULT_MIN_MASK_AREA_RATIO,
    max_masks_per_frame: int = DEFAULT_MAX_MASKS_PER_FRAME,
) -> Tuple[str, str, str, str]:
    """
    Run SAM3 segmentation and write mask, bbox, binary dynamic mask, and video-inpainted outputs.
    """

    import time

    start = time.perf_counter()

    print("开始计算时间")
    mask_writer, bbox_writer, binary_mask_writer, masked_writer = _init_writers(
        video_path,
        output_video_path=output_video_path,
    )

    if predictor is None:
        predictor = build_sam3_predictor(
            sam3_model_path=sam3_model_path,
            device=device,
            conf=conf,
            iou=iou,
        )

    # Build a safe names mapping that handles any integer index without KeyError
    safe_names = _make_safe_names(text_prompts)

    # Patch names on both predictor and its model before inference starts
    model_obj = getattr(predictor, "model", None)
    if model_obj is not None:
        model_obj.names = safe_names
    predictor.names = safe_names

    print(f"Segmenting video with prompts: {text_prompts}")
    results = predictor(source=video_path, text=text_prompts, stream=True)
    source_frames: List[np.ndarray] = []
    dynamic_masks_for_inpainting: List[np.ndarray] = []

    for frame_idx, r in enumerate(results):
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}...")

        # Always enforce safe names on each result object so verbose() won't crash
        r.names = safe_names

        base_frame = getattr(r, "orig_img", None)
        if base_frame is None:
            base_frame = r.plot()  # fallback
        mask_frame = base_frame.copy()
        bbox_frame = base_frame.copy()
        dynamic_mask = np.zeros(base_frame.shape[:2], dtype=bool)

        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
            confs = r.boxes.conf.cpu().numpy() if r.boxes is not None and r.boxes.conf is not None else []
            cls_ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []

            raw_candidates: list[dict[str, Any]] = []
            frame_area = int(base_frame.shape[0] * base_frame.shape[1])
            for i, mask in enumerate(masks):
                score = float(confs[i]) if i < len(confs) else 1.0
                if score < conf:
                    continue

                mask_bool = mask > 0
                area_ratio = _mask_area_ratio(mask_bool, frame_area)
                raw_candidates.append(
                    {
                        "index": i,
                        "mask": mask_bool,
                        "score": score,
                        "area_ratio": area_ratio,
                        "cls_id": int(cls_ids[i]) if i < len(cls_ids) else i,
                    }
                )

            large_candidates = [
                candidate
                for candidate in raw_candidates
                if float(candidate["area_ratio"]) >= min_mask_area_ratio
            ]
            used_small_fallback = bool(raw_candidates) and not large_candidates
            candidates = large_candidates if large_candidates else raw_candidates
            candidates = sorted(
                candidates,
                key=lambda item: (float(item["area_ratio"]), float(item["score"])),
                reverse=True,
            )
            if max_masks_per_frame > 0 and len(candidates) > max_masks_per_frame:
                candidates = candidates[:max_masks_per_frame]

            if frame_idx % 10 == 0 and r.masks is not None:
                print(
                    f"Frame {frame_idx}: kept {len(candidates)}/{len(masks)} masks "
                    f"(eligible={len(raw_candidates)}, large={len(large_candidates)}, "
                    f"min_area_ratio={min_mask_area_ratio}, max_masks={max_masks_per_frame}, "
                    f"small_fallback={used_small_fallback})"
                )

            for candidate in candidates:
                i = int(candidate["index"])
                mask_bool = candidate["mask"]
                cls_id = int(candidate["cls_id"])
                color = PALETTE[cls_id % len(PALETTE)]

                if mask_bool.any():
                    dynamic_mask |= mask_bool
                    overlay = np.zeros_like(mask_frame, dtype=mask_frame.dtype)
                    overlay[mask_bool] = color
                    mask_frame = cv2.addWeighted(overlay, 0.6, mask_frame, 0.4, 0)

                bbox = boxes[i] if i < len(boxes) else None
                if bbox is None and mask_bool.any():
                    coords = np.argwhere(mask_bool)
                    x1, y1 = coords[:, 1].min(), coords[:, 0].min()
                    x2, y2 = coords[:, 1].max(), coords[:, 0].max()
                    bbox = [x1, y1, x2, y2]
                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(bbox_frame, (x1, y1), (x2, y2), color, 2)

        filtered_dynamic_mask = _dilate_dynamic_mask(dynamic_mask)
        binary_mask_frame = np.zeros_like(base_frame, dtype=base_frame.dtype)
        binary_mask_frame[filtered_dynamic_mask] = 255
        source_frames.append(base_frame.copy())
        dynamic_masks_for_inpainting.append(filtered_dynamic_mask)

        mask_writer.write(mask_frame)
        bbox_writer.write(bbox_frame)
        binary_mask_writer.write(binary_mask_frame)

    print("Inpainting dynamic regions from neighboring video frames...")
    for inpainted_frame in _inpaint_video_background(source_frames, dynamic_masks_for_inpainting):
        masked_writer.write(inpainted_frame)

    mask_writer.release()
    bbox_writer.release()
    binary_mask_writer.release()
    masked_writer.release()

    end = time.perf_counter()
    print("计算时间结束")

    print(f"SAM3 segmentation completed in {end - start:.6f} seconds.")

    def remux_h264(src: str, dst: str):
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([
            ffmpeg_exe, "-y", "-i", src,
            "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", dst
        ], check=True)


    mask_mp4_path = str(_mask_path(video_path, output_video_path=output_video_path))
    bbox_mp4_path = str(_bbox_path(video_path, output_video_path=output_video_path))
    binary_mask_mp4_path = str(_dynamic_binary_mask_path(video_path, output_video_path=output_video_path))
    masked_mp4_path = str(_dynamic_inpainted_path(video_path, output_video_path=output_video_path))
    mask_final = mask_mp4_path.replace(".mp4", "_h264.mp4")
    bbox_final = bbox_mp4_path.replace(".mp4", "_h264.mp4")
    binary_mask_final = binary_mask_mp4_path.replace(".mp4", "_h264.mp4")
    masked_final = masked_mp4_path.replace(".mp4", "_h264.mp4")


    non_h264_outputs = [
        mask_mp4_path,
        bbox_mp4_path,
        binary_mask_mp4_path,
        masked_mp4_path,
    ]
    remux_h264(mask_mp4_path, mask_final)
    remux_h264(bbox_mp4_path, bbox_final)
    remux_h264(binary_mask_mp4_path, binary_mask_final)
    remux_h264(masked_mp4_path, masked_final)
    _remove_non_h264_intermediates(non_h264_outputs)

    return mask_final, bbox_final, binary_mask_final, masked_final


def run_sam_process(
    video_path: str,
    video_prompt: str | None,
    model_name: str | None = None,
    backend: str | None = None,
    sam3_model_path: str = "./weights/sam3/sam3.pt",
    device: str = "cuda:1",
    max_frames: int = DEFAULT_MAX_FRAMES,
    sam3_predictor: Any | None = None,
    sam_conf: float = DEFAULT_SAM_CONF,
    sam_iou: float = 0.7,
    min_mask_area_ratio: float = DEFAULT_MIN_MASK_AREA_RATIO,
    max_masks_per_frame: int = DEFAULT_MAX_MASKS_PER_FRAME,
) -> Dict:
    print("=" * 80)
    print("PHYSICAL SAM PROCESS")
    print("=" * 80)
    print(f"Video: {video_path}")

    if not video_prompt:
        prompt_path = Path(video_path).parent / "prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"No prompt provided and {prompt_path} not found")
        video_prompt = prompt_path.read_text(encoding="utf-8").strip()
        print(f"Loaded prompt from {prompt_path}")
    print(f"Prompt: {video_prompt}")

    with tempfile.TemporaryDirectory(prefix="sam3_ultralytics_input_") as temp_dir:
        processing_video_path, sampling_info = _build_sampled_video(
            video_path=video_path,
            work_dir=Path(temp_dir),
            max_frames=max_frames,
        )
        print(
            f"Processing video source: {sampling_info['source']} "
            f"({sampling_info['processing_frame_count']} frames)"
        )

        print("\nStep 1: Extract moving/deforming objects via VLM...")
        vlm_start = time.perf_counter()
        text_prompts = extract_dynamic_objects(
            video_path=processing_video_path,
            video_prompt=video_prompt,
            model_name=model_name,
            backend=backend,
        )
        vlm_duration = time.perf_counter() - vlm_start
        print(f"VLM extraction took {vlm_duration:.2f} seconds")
        print(f"Objects to segment: {text_prompts}")

        print("\nStep 2: Segment objects with SAM3 and build visualizations...")
        seg_start = time.perf_counter()
        mask_video, bbox_video, dynamic_binary_mask_video, dynamic_masked_video = segment_with_sam3(
            video_path=processing_video_path,
            text_prompts=text_prompts,
            sam3_model_path=sam3_model_path,
            device=device,
            conf=sam_conf,
            iou=sam_iou,
            output_video_path=video_path,
            predictor=sam3_predictor,
            min_mask_area_ratio=min_mask_area_ratio,
            max_masks_per_frame=max_masks_per_frame,
        )
        seg_duration = time.perf_counter() - seg_start
        print(f"SAM3 segmentation took {seg_duration:.2f} seconds")

    print("\n" + "=" * 80)
    print("SAM PROCESS COMPLETE")
    print(f"Mask-only video: {mask_video}")
    print(f"BBox-only video: {bbox_video}")
    print(f"Dynamic binary mask video: {dynamic_binary_mask_video}")
    print(f"Dynamic inpainted video: {dynamic_masked_video}")
    print("=" * 80)

    return {
        "video_path": video_path,
        "video_prompt": video_prompt,
        "backend": resolve_vlm_backend(backend),
        "model_name": resolve_model_name(model_name, backend),
        "text_prompts": text_prompts,
        "outputs": {
            "mask_video": mask_video,
            "bbox_video": bbox_video,
            "dynamic_binary_mask_video": dynamic_binary_mask_video,
            "dynamic_masked_video": dynamic_masked_video,
        },
        "timings_seconds": {
            "vlm": round(vlm_duration, 2),
            "segmentation": round(seg_duration, 2),
        },
        "sampling": sampling_info,
        "max_frames": max_frames,
        "min_mask_area_ratio": min_mask_area_ratio,
        "max_masks_per_frame": max_masks_per_frame,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM-guided SAM3 segmentation for moving/deforming objects")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument(
        "--prompt",
        help="Text description of the video; if omitted, read prompt.txt in the same directory as the video",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="VLM backend: qwenvl_server, local/qwenvl, or api/openrouter. Defaults to qwenvl_server.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="VLM model name or local model path. Defaults to the selected backend's default model.",
    )
    parser.add_argument("--sam3-model", default="./weights/sam3/sam3.pt", help="Path to SAM3 weights")
    parser.add_argument("--device", default="cuda:1", help="Device for SAM3 (e.g., cuda:0 or cpu)")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Maximum number of frames kept for VLM and SAM3 processing; <=0 disables temporal sampling",
    )
    parser.add_argument(
        "--min-mask-area-ratio",
        type=float,
        default=DEFAULT_MIN_MASK_AREA_RATIO,
        help="Drop SAM masks smaller than this fraction of the frame area.",
    )
    parser.add_argument(
        "--max-masks-per-frame",
        type=int,
        default=DEFAULT_MAX_MASKS_PER_FRAME,
        help="Keep only the largest N masks per frame after confidence and area filtering; <=0 keeps all.",
    )

    args = parser.parse_args()
    vlm_backend = resolve_vlm_backend(args.backend)
    model_name = resolve_model_name(args.model, vlm_backend)

    result = run_sam_process(
        video_path=args.video,
        video_prompt=args.prompt,
        model_name=model_name,
        backend=vlm_backend,
        sam3_model_path=args.sam3_model,
        device=args.device,
        max_frames=args.max_frames,
        min_mask_area_ratio=args.min_mask_area_ratio,
        max_masks_per_frame=args.max_masks_per_frame,
    )

    # Save summary next to video
    summary_path = Path(args.video).with_name(f"{Path(args.video).stem}_sam_process.json")
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult summary saved to: {summary_path}")
