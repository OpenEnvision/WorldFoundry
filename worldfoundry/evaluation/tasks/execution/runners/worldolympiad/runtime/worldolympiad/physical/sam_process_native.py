"""
Native SAM3 segmentation pipeline for physical events.

This is a drop-in alternative to `physical/sam_process.py` that uses the
official SAM3 video predictor API directly instead of Ultralytics.

The script:
1. Uses a VLM to extract a few moving/deforming object prompts from the video.
2. Runs SAM3 native video prompting/tracking for each text prompt.
3. Exports two visualization videos beside the input video:
   - *_mask.mp4: mask overlays only (no boxes)
   - *_bbox.mp4: bounding boxes only (no masks)
4. Re-encodes them into *_h264.mp4 variants for downstream use.

Notes:
- This implementation uses the public SAM3 predictor API:
  `build_sam3_predictor(...).handle_request(...)`
- The native SAM3 video predictor in this checkout is CUDA-only.



python physical/sam_process_native.py \
    --video data/real/motor/motor_gt.mp4 \
    --sam3-model ./weights/sam3/sam3.pt \
    --device cuda:1 \
    --sam3-version sam3 \
    --max-frames 512
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.vlm import (
    parse_json_content,
    resolve_model_name,
    resolve_vlm_backend,
    video_vlm_call,
)

load_dotenv()


PALETTE = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
]

LOCAL_SAM3_PACKAGE_ROOT = REPO_ROOT / "sam3"


def encode_video_to_data_url(video_path: str) -> Dict[str, Any]:
    with open(video_path, "rb") as video_file:
        base64_video = base64.b64encode(video_file.read()).decode("utf-8")
    return {
        "type": "video_url",
        "video_url": {"url": f"data:video/mp4;base64,{base64_video}"},
    }


def extract_dynamic_objects(
    video_path: str,
    video_prompt: str,
    model_name: str | None = None,
    backend: str | None = None,
    max_objects: int = 4,
) -> List[str]:
    """
    Use a VLM to list a few moving/deforming objects in the video.
    """
    system_prompt = """You are an expert at spotting physical actors that MOVE or DEFORM in a video.

Rules:
- Return the FEWEST distinct objects involved in motion/shape change (max 5).
- Merge duplicates/synonyms into one concise noun (1-3 words).
- Ignore static background and scenery.
- Output MUST be a JSON array only, no explanations."""

    video_data_url = encode_video_to_data_url(video_path)
    user_prompt = f"""Watch this video and list the main objects that move or deform.

Video description: {video_prompt}

Return 1-5 concise nouns in JSON array format only.
    Example: ["person", "ball", "table"]"""

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

    return ["motor"]


def _mask_path(video_path: str) -> Path:
    path = Path(video_path)
    return path.with_name(f"{path.stem}_mask.mp4")


def _bbox_path(video_path: str) -> Path:
    path = Path(video_path)
    return path.with_name(f"{path.stem}_bbox.mp4")


def _probe_video(video_path: str) -> Dict[str, Any]:
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


def _init_video_writer(video_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc_alt = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(video_path), fourcc_alt, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for {video_path}")
    return writer


def _build_sampled_video(
    *,
    video_path: str,
    work_dir: Path,
    max_frames: int,
) -> tuple[str, Dict[str, Any]]:
    """
    Create a temporally subsampled proxy video for VLM and SAM3 processing.

    The output video keeps the original resolution but reduces frame count to stay
    within `max_frames`, which directly lowers SAM3 session memory.
    """
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
        fps=sampled_fps,
        width=int(video_meta["width"]),
        height=int(video_meta["height"]),
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
        f"({video_meta['fps']:.3f} fps -> {sampled_fps:.3f} fps)"
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


def _normalize_native_device(device: str) -> int:
    raw = str(device).strip().lower()
    if raw == "cpu":
        raise ValueError(
            "The native SAM3 video predictor in this repository is CUDA-only; cpu is not supported."
        )
    if raw.startswith("cuda:"):
        raw = raw.split(":", 1)[1]
    if raw.isdigit():
        return int(raw)
    raise ValueError(f"Unsupported --device value: {device}. Use 0/1/2 or cuda:0/cuda:1.")


def _load_build_sam3_predictor():
    # The repository contains an outer `sam3/` directory and an inner Python package
    # at `sam3/sam3/`. When running from repo root, Python may resolve the outer
    # directory as a namespace package with no `__file__`, which then breaks
    # `pkg_resources.resource_filename("sam3", ...)` inside SAM3 itself.
    if LOCAL_SAM3_PACKAGE_ROOT.is_dir():
        package_root_str = str(LOCAL_SAM3_PACKAGE_ROOT)
        if package_root_str not in sys.path:
            sys.path.insert(0, package_root_str)

    existing = sys.modules.get("sam3")
    if existing is not None and getattr(existing, "__file__", None) is None:
        del sys.modules["sam3"]

    try:
        from sam3 import build_sam3_predictor  # type: ignore
    except ImportError:
        raise ImportError(
            "Failed to import SAM3. Ensure the local `sam3/` package is importable and install "
            "its runtime dependencies (for example `setuptools` and `iopath`). "
            "A reliable setup is: `python -m pip install -e ./sam3`."
        )
    return build_sam3_predictor


def _require_cuda_device(device: str) -> int:
    import torch

    gpu_index = _normalize_native_device(device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but native SAM3 video inference requires CUDA.")
    if gpu_index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested CUDA device {gpu_index}, but only {torch.cuda.device_count()} device(s) are available."
        )
    torch.cuda.set_device(gpu_index)
    return gpu_index


def _extract_base_overlay_frames(
    video_path: str,
    mask_frame_dir: Path,
    bbox_frame_dir: Path,
) -> Tuple[int, int, int, List[Tuple[Path, Path]]]:
    """
    Decode the video once and materialize per-frame PNGs for mask/bbox overlays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_paths: List[Tuple[Path, Path]] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        mask_path = mask_frame_dir / f"{frame_idx:06d}.png"
        bbox_path = bbox_frame_dir / f"{frame_idx:06d}.png"
        if not cv2.imwrite(str(mask_path), frame):
            raise RuntimeError(f"Failed to write temporary frame: {mask_path}")
        if not cv2.imwrite(str(bbox_path), frame):
            raise RuntimeError(f"Failed to write temporary frame: {bbox_path}")
        frame_paths.append((mask_path, bbox_path))
        frame_idx += 1

    cap.release()

    if not frame_paths:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    return fps, width, height, frame_paths


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _normalize_masks(raw_masks: Any) -> np.ndarray:
    masks = _to_numpy(raw_masks)
    if masks.size == 0:
        return np.zeros((0, 0, 0), dtype=bool)
    if masks.ndim == 2:
        masks = masks[None, ...]
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, ...]
    elif masks.ndim != 3:
        raise ValueError(f"Unexpected SAM3 mask shape: {masks.shape}")
    return masks.astype(bool, copy=False)


def _normalize_boxes(raw_boxes: Any) -> np.ndarray:
    boxes = _to_numpy(raw_boxes)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes[None, ...]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Unexpected SAM3 box shape: {boxes.shape}")
    return boxes.astype(np.float32, copy=False)


def _mask_to_xyxy(mask_bool: np.ndarray) -> Tuple[int, int, int, int] | None:
    if not mask_bool.any():
        return None
    coords = np.argwhere(mask_bool)
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)
    return int(x1), int(y1), int(x2), int(y2)


def _normalized_xywh_to_xyxy(box_xywh: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = [float(value) for value in box_xywh]
    x1 = int(round(x * width))
    y1 = int(round(y * height))
    x2 = int(round((x + w) * width))
    y2 = int(round((y + h) * height))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2


def _apply_outputs_to_overlay_frames(
    *,
    mask_frame_path: Path,
    bbox_frame_path: Path,
    outputs: Dict[str, Any],
    color: Tuple[int, int, int],
    width: int,
    height: int,
) -> None:
    masks = _normalize_masks(outputs.get("out_binary_masks"))
    if masks.size == 0 or masks.shape[0] == 0:
        return

    boxes = _normalize_boxes(outputs.get("out_boxes_xywh"))

    mask_frame = cv2.imread(str(mask_frame_path), cv2.IMREAD_COLOR)
    bbox_frame = cv2.imread(str(bbox_frame_path), cv2.IMREAD_COLOR)
    if mask_frame is None or bbox_frame is None:
        raise RuntimeError(
            f"Failed to load temporary overlay frames: {mask_frame_path} / {bbox_frame_path}"
        )

    for mask_index, mask_bool in enumerate(masks):
        if mask_bool.shape != (height, width):
            raise ValueError(
                f"Unexpected mask resolution {mask_bool.shape}; expected {(height, width)}"
            )

        if mask_bool.any():
            overlay = np.zeros_like(mask_frame, dtype=mask_frame.dtype)
            overlay[mask_bool] = color
            mask_frame = cv2.addWeighted(overlay, 0.6, mask_frame, 0.4, 0)

        if mask_index < len(boxes):
            x1, y1, x2, y2 = _normalized_xywh_to_xyxy(boxes[mask_index], width, height)
        else:
            xyxy = _mask_to_xyxy(mask_bool)
            if xyxy is None:
                continue
            x1, y1, x2, y2 = xyxy

        cv2.rectangle(bbox_frame, (x1, y1), (x2, y2), color, 2)

    if not cv2.imwrite(str(mask_frame_path), mask_frame):
        raise RuntimeError(f"Failed to write overlay frame: {mask_frame_path}")
    if not cv2.imwrite(str(bbox_frame_path), bbox_frame):
        raise RuntimeError(f"Failed to write overlay frame: {bbox_frame_path}")


def _write_video_from_frames(
    frame_dir: Path,
    output_path: Path,
    fps: int,
    width: int,
    height: int,
) -> None:
    frame_paths = sorted(frame_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frame_dir}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc_alt = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_path), fourcc_alt, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for {output_path}")

    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read overlay frame: {frame_path}")
            writer.write(frame)
    finally:
        writer.release()


def _remux_h264(src: str, dst: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            src,
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            dst,
        ],
        check=True,
    )


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


def segment_with_sam3(
    video_path: str,
    text_prompts: List[str],
    sam3_model_path: str = "./weights/sam3/sam3.pt",
    device: str = "cuda:1",
    version: str = "sam3",
    compile: bool = False,
    async_loading_frames: bool = False,
    max_num_objects: int = 16,
    output_prob_thresh: float = 0.5,
    output_video_path: str | None = None,
) -> Tuple[str, str]:
    """
    Run native SAM3 video prompting/tracking and render mask/bbox videos.
    """
    gpu_index = _require_cuda_device(device)
    build_sam3_predictor = _load_build_sam3_predictor()

    predictor_kwargs: Dict[str, Any] = {
        "checkpoint_path": sam3_model_path,
        "version": version,
        "compile": compile,
        "async_loading_frames": async_loading_frames,
    }
    if version == "sam3":
        predictor_kwargs["gpus_to_use"] = [gpu_index]
    elif version == "sam3.1":
        predictor_kwargs["max_num_objects"] = max(max_num_objects, len(text_prompts))
    else:
        raise ValueError(f"Unsupported SAM3 version: {version}. Use 'sam3' or 'sam3.1'.")

    print(
        f"Building native SAM3 predictor: version={version}, device=cuda:{gpu_index}, "
        f"compile={compile}, async_loading_frames={async_loading_frames}"
    )
    predictor = build_sam3_predictor(**predictor_kwargs)
    render_video_path = output_video_path or video_path

    session_id: str | None = None
    try:
        session = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": video_path,
            }
        )
        session_id = session["session_id"]
        print(f"SAM3 session started: {session_id}")

        with tempfile.TemporaryDirectory(prefix="sam3_native_overlay_") as temp_dir:
            temp_root = Path(temp_dir)
            mask_frame_dir = temp_root / "mask_frames"
            bbox_frame_dir = temp_root / "bbox_frames"
            mask_frame_dir.mkdir(parents=True, exist_ok=True)
            bbox_frame_dir.mkdir(parents=True, exist_ok=True)

            fps, width, height, frame_paths = _extract_base_overlay_frames(
                video_path=video_path,
                mask_frame_dir=mask_frame_dir,
                bbox_frame_dir=bbox_frame_dir,
            )
            frame_count = len(frame_paths)
            print(f"Prepared {frame_count} base frames for overlay rendering")

            for prompt_index, prompt in enumerate(text_prompts):
                color = PALETTE[prompt_index % len(PALETTE)]
                print(f"\nTracking prompt {prompt_index + 1}/{len(text_prompts)}: {prompt}")
                predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": prompt,
                        "output_prob_thresh": output_prob_thresh,
                    }
                )

                processed = 0
                for response in predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "forward",
                        "start_frame_index": 0,
                        "max_frame_num_to_track": frame_count,
                        "output_prob_thresh": output_prob_thresh,
                    }
                ):
                    frame_index = response.get("frame_index")
                    if frame_index is None or not (0 <= frame_index < frame_count):
                        continue

                    outputs = response.get("outputs") or {}
                    mask_frame_path, bbox_frame_path = frame_paths[frame_index]
                    _apply_outputs_to_overlay_frames(
                        mask_frame_path=mask_frame_path,
                        bbox_frame_path=bbox_frame_path,
                        outputs=outputs,
                        color=color,
                        width=width,
                        height=height,
                    )
                    processed += 1
                    if processed % 20 == 0:
                        print(
                            f"  Prompt '{prompt}' processed {processed} propagated frames "
                            f"(current frame {frame_index})"
                        )

                predictor.handle_request(
                    {
                        "type": "reset_session",
                        "session_id": session_id,
                    }
                )

            mask_output = _mask_path(render_video_path)
            bbox_output = _bbox_path(render_video_path)
            _write_video_from_frames(mask_frame_dir, mask_output, fps, width, height)
            _write_video_from_frames(bbox_frame_dir, bbox_output, fps, width, height)

        mask_final = str(mask_output).replace(".mp4", "_h264.mp4")
        bbox_final = str(bbox_output).replace(".mp4", "_h264.mp4")
        _remux_h264(str(mask_output), mask_final)
        _remux_h264(str(bbox_output), bbox_final)
        _remove_non_h264_intermediates([str(mask_output), str(bbox_output)])
        return mask_final, bbox_final
    finally:
        if session_id is not None:
            try:
                predictor.handle_request(
                    {
                        "type": "close_session",
                        "session_id": session_id,
                    }
                )
            except Exception:
                pass
        if hasattr(predictor, "shutdown"):
            try:
                predictor.shutdown()
            except Exception:
                pass


def run_sam_process(
    video_path: str,
    video_prompt: str | None,
    model_name: str | None = None,
    backend: str | None = None,
    sam3_model_path: str = "./weights/sam3/sam3.pt",
    device: str = "cuda:1",
    sam3_version: str = "sam3",
    compile: bool = False,
    async_loading_frames: bool = False,
    max_num_objects: int = 16,
    output_prob_thresh: float = 0.5,
    max_frames: int = 128,
) -> Dict[str, Any]:
    print("=" * 80)
    print("PHYSICAL SAM PROCESS (NATIVE SAM3)")
    print("=" * 80)
    print(f"Video: {video_path}")

    if not video_prompt:
        prompt_path = Path(video_path).parent / "prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"No prompt provided and {prompt_path} not found")
        video_prompt = prompt_path.read_text(encoding="utf-8").strip()
        print(f"Loaded prompt from {prompt_path}")
    print(f"Prompt: {video_prompt}")

    with tempfile.TemporaryDirectory(prefix="sam3_native_input_") as temp_dir:
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

        print("\nStep 2: Segment objects with native SAM3 and build visualizations...")
        seg_start = time.perf_counter()
        mask_video, bbox_video = segment_with_sam3(
            video_path=processing_video_path,
            text_prompts=text_prompts,
            sam3_model_path=sam3_model_path,
            device=device,
            version=sam3_version,
            compile=compile,
            async_loading_frames=async_loading_frames,
            max_num_objects=max_num_objects,
            output_prob_thresh=output_prob_thresh,
            output_video_path=video_path,
        )
        seg_duration = time.perf_counter() - seg_start
        print(f"Native SAM3 segmentation took {seg_duration:.2f} seconds")

    print("\n" + "=" * 80)
    print("SAM PROCESS COMPLETE")
    print(f"Mask-only video: {mask_video}")
    print(f"BBox-only video: {bbox_video}")
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
        },
        "timings_seconds": {
            "vlm": round(vlm_duration, 2),
            "segmentation": round(seg_duration, 2),
        },
        "sampling": sampling_info,
        "sam3_backend": "native",
        "sam3_version": sam3_version,
        "compile": compile,
        "async_loading_frames": async_loading_frames,
        "max_num_objects": max_num_objects,
        "output_prob_thresh": output_prob_thresh,
        "max_frames": max_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLM-guided native SAM3 segmentation for moving/deforming objects"
    )
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument(
        "--prompt",
        help="Text description of the video; if omitted, read prompt.txt in the same directory as the video",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="VLM backend: api/openrouter or local/qwenvl. Defaults to VLM_BACKEND/openrouter.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="VLM model name or local model path. Defaults to the selected backend's default model.",
    )
    parser.add_argument("--sam3-model", default="./weights/sam3/sam3.pt", help="Path to SAM3 checkpoint")
    parser.add_argument("--device", default="cuda:1", help="CUDA device for SAM3, e.g. cuda:0 or 1")
    parser.add_argument(
        "--sam3-version",
        default="sam3",
        choices=["sam3", "sam3.1"],
        help="Native SAM3 predictor version",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for SAM3 when supported",
    )
    parser.add_argument(
        "--async-loading-frames",
        action="store_true",
        help="Enable SAM3 asynchronous frame loading",
    )
    parser.add_argument(
        "--max-num-objects",
        type=int,
        default=16,
        help="Maximum number of tracked objects for sam3.1 multiplex",
    )
    parser.add_argument(
        "--output-prob-thresh",
        type=float,
        default=0.5,
        help="SAM3 output probability threshold",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=128,
        help="Maximum number of frames kept for VLM and native SAM3 processing; <=0 disables temporal sampling",
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
        sam3_version=args.sam3_version,
        compile=args.compile,
        async_loading_frames=args.async_loading_frames,
        max_num_objects=args.max_num_objects,
        output_prob_thresh=args.output_prob_thresh,
        max_frames=args.max_frames,
    )

    summary_path = Path(args.video).with_name(f"{Path(args.video).stem}_sam_process_native.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(f"\nResult summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
