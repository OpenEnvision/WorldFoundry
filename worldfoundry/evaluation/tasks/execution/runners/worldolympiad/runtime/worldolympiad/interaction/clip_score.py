"""
CLIP semantic-adherence metric for chunk-generated long videos.

For each generated chunk, sample frames inside that chunk and compute CLIP
image-text similarity against the corresponding chunk prompt/caption. The final
score is the mean over chunks, weighted by the number of sampled frames.

Example:
python interaction/clip_score.py \
  --gen-video examples/02BEoux44n8_part3/matrix_game2_gen_02BEoux44n8_part3.mp4 \
  --prompt-json examples/02BEoux44n8_part3/prompt.json \
  --chunk-json examples/02BEoux44n8_part3/matrix_game2_gen_02BEoux44n8_part3_chunk_timestamps.json \
  --output clip_result.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def remove_script_dir_from_import_path() -> None:
    """
    Avoid shadowing the third-party `clip` package with this directory.

    When this file was named clip.py, running it as a script made `import clip`
    resolve to itself. The file is now clip_score.py, but removing the script
    directory keeps the import robust if users run from interaction/.
    """
    script_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != script_dir]


def load_clip_package():
    remove_script_dir_from_import_path()
    try:
        return importlib.import_module("clip")
    except ImportError as exc:
        raise ImportError(
            "CLIP scoring requires the OpenAI/Ultralytics CLIP package. "
            "Install it with: pip install git+https://github.com/ultralytics/CLIP.git"
        ) from exc


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_interval_seconds(interval: str) -> Tuple[Optional[float], Optional[float]]:
    cleaned = interval.strip().strip("[]()")
    if "," in cleaned:
        left, right = cleaned.split(",", 1)
    elif "-" in cleaned:
        left, right = cleaned.split("-", 1)
    else:
        return None, None
    return parse_timestamp(left), parse_timestamp(right)


def parse_timestamp(value: str) -> Optional[float]:
    text = value.strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return None
    return None


def normalize_prompts(prompt_data: Any) -> List[Dict[str, Any]]:
    if isinstance(prompt_data, dict):
        if "chunks" in prompt_data:
            raw_items = prompt_data["chunks"]
        elif "prompts" in prompt_data:
            raw_items = prompt_data["prompts"]
        else:
            raw_items = [prompt_data]
    elif isinstance(prompt_data, list):
        raw_items = prompt_data
    else:
        raise ValueError("prompt-json must be a JSON list or object.")

    prompts: List[Dict[str, Any]] = []
    for index, item in enumerate(raw_items):
        if isinstance(item, str):
            prompts.append({"chunk_index": index, "caption": item, "action": None, "interval": None})
            continue
        if not isinstance(item, dict):
            prompts.append({"chunk_index": index, "caption": str(item), "action": None, "interval": None})
            continue
        caption = item.get("caption") or item.get("prompt") or item.get("text") or item.get("description") or ""
        prompts.append(
            {
                "chunk_index": int(item.get("chunk_index", item.get("index", index))),
                "caption": str(caption),
                "action": item.get("action") or item.get("actions"),
                "interval": item.get("interval") or item.get("source_interval"),
            }
        )
    return prompts


def normalize_chunks(chunk_data: Any, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw_chunks = chunk_data.get("chunks", chunk_data) if isinstance(chunk_data, dict) else chunk_data
    if not isinstance(raw_chunks, list):
        raise ValueError("chunk-json must contain a list or a top-level 'chunks' list.")

    prompt_by_index = {int(item["chunk_index"]): item for item in prompts}
    normalized: List[Dict[str, Any]] = []
    for fallback_index, chunk in enumerate(raw_chunks):
        if not isinstance(chunk, dict):
            raise ValueError("Each chunk entry must be a JSON object.")

        chunk_index = int(chunk.get("chunk_index", chunk.get("index", fallback_index)))
        prompt = prompt_by_index.get(chunk_index, prompts[fallback_index] if fallback_index < len(prompts) else {})

        start_sec = chunk.get("generated_start_sec", chunk.get("start_sec", chunk.get("start")))
        end_sec = chunk.get("generated_end_sec", chunk.get("end_sec", chunk.get("end")))
        if start_sec is None or end_sec is None:
            interval = chunk.get("generated_interval") or chunk.get("interval") or prompt.get("interval")
            if isinstance(interval, str):
                parsed_start, parsed_end = parse_interval_seconds(interval)
                start_sec = parsed_start if start_sec is None else start_sec
                end_sec = parsed_end if end_sec is None else end_sec
        if start_sec is None or end_sec is None:
            raise ValueError(f"Chunk {chunk_index} is missing start/end seconds.")

        normalized.append(
            {
                **chunk,
                "chunk_index": chunk_index,
                "generated_start_sec": float(start_sec),
                "generated_end_sec": float(end_sec),
                "caption": str(prompt.get("caption", "")),
                "action": prompt.get("action"),
                "source_interval": chunk.get("source_interval") or prompt.get("interval"),
            }
        )

    return sorted(normalized, key=lambda item: item["chunk_index"])


def get_video_metadata(video_path: str | Path) -> Tuple[float, int, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0
    capture.release()
    return fps, total_frames, duration


def sample_chunk_frames(
    capture: cv2.VideoCapture,
    fps: float,
    total_frames: int,
    start_sec: float,
    end_sec: float,
    frames_per_chunk: int,
) -> List[Image.Image]:
    if frames_per_chunk <= 0:
        return []

    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec, float(end_sec))
    if end_sec <= start_sec:
        sample_times = [start_sec]
    elif frames_per_chunk == 1:
        sample_times = [(start_sec + end_sec) / 2.0]
    else:
        sample_times = [
            start_sec + (end_sec - start_sec) * idx / float(frames_per_chunk - 1)
            for idx in range(frames_per_chunk)
        ]

    images: List[Image.Image] = []
    for sample_time in sample_times:
        if fps > 0:
            frame_index = max(0, int(round(sample_time * fps)))
            if total_frames > 0:
                frame_index = min(frame_index, max(0, total_frames - 1))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        else:
            capture.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000.0)

        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(rgb))
    return images


def score_chunk(
    model: torch.nn.Module,
    preprocess,
    clip_module,
    device: torch.device,
    images: List[Image.Image],
    prompt: str,
) -> Dict[str, Any]:
    if not images:
        return {"score": None, "frame_scores": [], "num_frames": 0}
    if not prompt.strip():
        return {"score": None, "frame_scores": [], "num_frames": len(images), "error": "empty_prompt"}

    image_batch = torch.stack([preprocess(image) for image in images]).to(device)
    text_tokens = clip_module.tokenize([prompt], truncate=True).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_batch)
        text_features = model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze(-1)

    frame_scores = [float(value) for value in similarities.detach().cpu().tolist()]
    return {
        "score": float(np.mean(frame_scores)),
        "frame_scores": frame_scores,
        "num_frames": len(frame_scores),
    }


def evaluate_clip_semantic_adherence(
    gen_video: str,
    prompt_json: str,
    chunk_json: str,
    model_name: str = "ViT-B/32",
    device: str | None = None,
    frames_per_chunk: int = 8,
    download_root: str | None = None,
) -> Dict[str, Any]:
    clip_module = load_clip_package()
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    resolved_download_root = download_root or os.getenv("CLIP_DOWNLOAD_ROOT") or "/tmp/clip_cache"
    model, preprocess = clip_module.load(
        model_name,
        device=resolved_device,
        download_root=resolved_download_root,
    )
    model.eval()

    prompts = normalize_prompts(load_json(prompt_json))
    chunks = normalize_chunks(load_json(chunk_json), prompts)
    fps, total_frames, duration = get_video_metadata(gen_video)

    capture = cv2.VideoCapture(str(gen_video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {gen_video}")

    chunk_results: List[Dict[str, Any]] = []
    weighted_sum = 0.0
    weighted_count = 0

    try:
        for chunk in chunks:
            print(f"Scoring chunk {chunk['chunk_index']}...")
            images = sample_chunk_frames(
                capture,
                fps,
                total_frames,
                chunk["generated_start_sec"],
                chunk["generated_end_sec"],
                frames_per_chunk,
            )
            prompt = chunk.get("caption", "")
            result = score_chunk(model, preprocess, clip_module, resolved_device, images, prompt)
            score = result.get("score")
            if score is not None:
                weighted_sum += float(score) * int(result["num_frames"])
                weighted_count += int(result["num_frames"])
            chunk_results.append(
                {
                    "chunk_index": chunk["chunk_index"],
                    "source_interval": chunk.get("source_interval"),
                    "generated_start_sec": chunk["generated_start_sec"],
                    "generated_end_sec": chunk["generated_end_sec"],
                    "action": chunk.get("action"),
                    "caption": prompt,
                    **result,
                }
            )
    finally:
        capture.release()

    overall = weighted_sum / weighted_count if weighted_count > 0 else None
    return {
        "generated_video": gen_video,
        "prompt_json": prompt_json,
        "chunk_json": chunk_json,
        "clip_model": model_name,
        "device": str(resolved_device),
        "download_root": resolved_download_root,
        "video": {"fps": fps, "total_frames": total_frames, "duration_sec": duration},
        "frames_per_chunk": frames_per_chunk,
        "summary": {
            "semantic_adherence": overall,
            "num_scored_chunks": sum(1 for item in chunk_results if item.get("score") is not None),
            "num_chunks": len(chunk_results),
            "num_scored_frames": weighted_count,
        },
        "chunks": chunk_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute chunk-level CLIP semantic-adherence scores")
    parser.add_argument("--gen-video", required=True, help="Path to stitched generated video")
    parser.add_argument("--prompt-json", required=True, help="JSON file containing chunk prompts/actions/captions")
    parser.add_argument("--chunk-json", required=True, help="JSON file containing generated chunk timestamps")
    parser.add_argument("--model", default="ViT-B/32", help="CLIP model name, e.g. ViT-B/32 or ViT-L/14")
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda when available, else cpu")
    parser.add_argument("--frames-per-chunk", type=int, default=8, help="Number of frames to sample per chunk")
    parser.add_argument(
        "--download-root",
        default=None,
        help="CLIP weight cache directory. Defaults to CLIP_DOWNLOAD_ROOT or /tmp/clip_cache.",
    )
    parser.add_argument("--output", help="Optional path to save JSON results")
    args = parser.parse_args()

    result = evaluate_clip_semantic_adherence(
        gen_video=args.gen_video,
        prompt_json=args.prompt_json,
        chunk_json=args.chunk_json,
        model_name=args.model,
        device=args.device,
        frames_per_chunk=args.frames_per_chunk,
        download_root=args.download_root,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved results to {output_path}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
