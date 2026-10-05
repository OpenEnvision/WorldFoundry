"""
Qwen3-VL based interaction metrics for chunk-generated long videos.

The generated video is assumed to be stitched from multiple chunks. This
script evaluates:
1. per-chunk visual quality and text alignment,
2. transitions between adjacent chunks,
3. whole-video long-range consistency.

Example:
python interaction/vlm_inter.py \
  --gen-video examples/02BEoux44n8_part3/matrix_game2_gen_02BEoux44n8_part3.mp4 \
  --ref-video examples/02BEoux44n8_part3/ref_02BEoux44n8_part3.mp4 \
  --prompt-json examples/02BEoux44n8_part3/prompt.json \
  --chunk-json examples/02BEoux44n8_part3/matrix_game2_gen_02BEoux44n8_part3_chunk_timestamps.json \
  --backend qwenvl \
  --output interaction_result.json
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.vlm import chat_vlm_call, parse_json_content, resolve_model_name, resolve_vlm_backend

load_dotenv()


SYSTEM_PROMPT = """You are a strict evaluator for chunk-generated interactive videos.

You judge whether each generated video chunk follows its intended action/caption,
whether adjacent chunks transition smoothly, and whether the full stitched video
stays globally consistent. Use the reference video only as context for scene,
style, camera, and intended interaction, not as a demand for frame-exact matching.

Always respond with strict JSON only. Scores must be numbers from 0 to 5."""


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encode_image_to_data_url(image_path: str | Path) -> Dict[str, Dict[str, str] | str]:
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


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


def sample_video_frames(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    count: int,
    output_dir: str | Path,
    prefix: str,
) -> List[Path]:
    if count <= 0:
        return []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0
    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec, float(end_sec))
    if duration > 0:
        end_sec = min(end_sec, duration)

    if end_sec <= start_sec:
        sample_times = [start_sec]
    elif count == 1:
        sample_times = [(start_sec + end_sec) / 2.0]
    else:
        sample_times = [
            start_sec + (end_sec - start_sec) * idx / float(count - 1)
            for idx in range(count)
        ]

    output_paths: List[Path] = []
    output_root = Path(output_dir)
    for idx, sample_time in enumerate(sample_times):
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
        output_path = output_root / f"{prefix}_{idx:03d}.jpg"
        cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        output_paths.append(output_path)

    capture.release()
    if not output_paths:
        raise RuntimeError(f"Failed to sample frames from {video_path}.")
    return output_paths


def video_duration(video_path: str | Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0


def content_from_frames(label: str, frames: List[Path]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": label}]
    content.extend(encode_image_to_data_url(frame_path) for frame_path in frames)
    return content


def extract_json_response(response: Dict[str, Any]) -> Dict[str, Any]:
    try:
        parsed = parse_json_content(response)
    except Exception as exc:
        raw = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        raise ValueError(f"Failed to parse VLM JSON: {exc}. Raw content: {str(raw)[:500]}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object from VLM, got: {type(parsed)}")
    return parsed


def clip_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(5.0, max(0.0, score))


def build_chunk_prompt(chunk: Dict[str, Any]) -> str:
    return f"""Evaluate one generated video chunk.

Chunk metadata:
- chunk_index: {chunk["chunk_index"]}
- source_interval: {chunk.get("source_interval")}
- generated_interval_sec: [{chunk["generated_start_sec"]:.3f}, {chunk["generated_end_sec"]:.3f})
- intended_action: {chunk.get("action")}
- intended_caption: {chunk.get("caption")}

Use the generated chunk frames as the primary evidence. Use reference-video
frames only for context about the intended scene/style.

Score:
- visual_quality: clarity, realism, temporal stability, color/lighting consistency, lack of artifacts.
- text_alignment: whether visible content matches the intended action and caption.

Return strict JSON:
{{
  "chunk_index": {chunk["chunk_index"]},
  "visual_quality": 0.0,
  "text_alignment": 0.0,
  "overall": 0.0,
  "reason": "short evidence-based explanation"
}}"""


def build_transition_prompt(left: Dict[str, Any], right: Dict[str, Any]) -> str:
    return f"""Evaluate the transition between two adjacent generated video chunks.

Previous chunk:
- chunk_index: {left["chunk_index"]}
- generated_interval_sec: [{left["generated_start_sec"]:.3f}, {left["generated_end_sec"]:.3f})
- action: {left.get("action")}
- caption: {left.get("caption")}

Next chunk:
- chunk_index: {right["chunk_index"]}
- generated_interval_sec: [{right["generated_start_sec"]:.3f}, {right["generated_end_sec"]:.3f})
- action: {right.get("action")}
- caption: {right.get("caption")}

Judge whether the transition is smooth and continuous:
- scene, lighting, style, and subject identity should remain coherent.
- motion trajectory and camera movement should evolve naturally.
- penalize abrupt jumps, object identity resets, impossible camera jumps, and visible stitching artifacts.

Return strict JSON:
{{
  "from_chunk_index": {left["chunk_index"]},
  "to_chunk_index": {right["chunk_index"]},
  "transition_smoothness": 0.0,
  "overall": 0.0,
  "reason": "short evidence-based explanation"
}}"""


def build_global_prompt(chunks: List[Dict[str, Any]]) -> str:
    prompt_summary = [
        {
            "chunk_index": chunk["chunk_index"],
            "action": chunk.get("action"),
            "caption": chunk.get("caption"),
        }
        for chunk in chunks
    ]
    return f"""Evaluate the whole stitched long generated video.

All chunk prompts:
{json.dumps(prompt_summary, ensure_ascii=False, indent=2)}

Use the generated full-video frames as primary evidence and reference frames as
context. Judge long-range consistency:
- subject/character/object identity remains stable across the entire video.
- scene style, visual tone, lighting, and camera behavior remain coherent.
- global semantics align with the combined intent of all chunk prompts.

Return strict JSON:
{{
  "long_range_consistency": 0.0,
  "global_text_alignment": 0.0,
  "overall": 0.0,
  "reason": "short evidence-based explanation"
}}"""


def call_vlm_with_frames(
    user_prompt: str,
    frame_groups: List[Tuple[str, List[Path]]],
    model_name: str,
    backend: str,
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for label, frames in frame_groups:
        content.extend(content_from_frames(label, frames))

    response = chat_vlm_call(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        model_name=model_name,
        backend=backend,
        timeout=240,
        max_retries=3,
    )
    return extract_json_response(response)


def evaluate_interaction(
    gen_video: str,
    ref_video: str,
    prompt_json: str,
    chunk_json: str,
    model_name: str | None = None,
    backend: str | None = None,
    chunk_frames: int = 4,
    ref_frames: int = 4,
    transition_side_frames: int = 3,
    global_frames: int = 12,
    skip_chunk: bool = False,
    skip_transition: bool = False,
    skip_global: bool = False,
) -> Dict[str, Any]:
    vlm_backend = resolve_vlm_backend(backend)
    resolved_model_name = resolve_model_name(model_name, vlm_backend)
    prompts = normalize_prompts(load_json(prompt_json))
    chunks = normalize_chunks(load_json(chunk_json), prompts)

    with tempfile.TemporaryDirectory(prefix="interaction_vlm_") as temp_dir:
        ref_duration = video_duration(ref_video)
        ref_context_frames = sample_video_frames(
            ref_video,
            0.0,
            ref_duration,
            ref_frames,
            temp_dir,
            "reference_context",
        )

        chunk_results: List[Dict[str, Any]] = []
        if not skip_chunk:
            for chunk in chunks:
                print(f"Evaluating chunk {chunk['chunk_index']}...")
                frames = sample_video_frames(
                    gen_video,
                    chunk["generated_start_sec"],
                    chunk["generated_end_sec"],
                    chunk_frames,
                    temp_dir,
                    f"chunk_{chunk['chunk_index']}",
                )
                result = call_vlm_with_frames(
                    build_chunk_prompt(chunk),
                    [
                        ("Generated chunk frames:", frames),
                        ("Reference video context frames:", ref_context_frames),
                    ],
                    resolved_model_name,
                    vlm_backend,
                )
                result["visual_quality"] = clip_score(result.get("visual_quality"))
                result["text_alignment"] = clip_score(result.get("text_alignment"))
                overall = result.get("overall")
                if overall is None:
                    overall = (result["visual_quality"] + result["text_alignment"]) / 2.0
                result["overall"] = clip_score(overall)
                chunk_results.append(result)

        transition_results: List[Dict[str, Any]] = []
        if not skip_transition:
            for left, right in zip(chunks, chunks[1:]):
                print(f"Evaluating transition {left['chunk_index']} -> {right['chunk_index']}...")
                left_frames = sample_video_frames(
                    gen_video,
                    max(left["generated_start_sec"], left["generated_end_sec"] - 1.0),
                    left["generated_end_sec"],
                    transition_side_frames,
                    temp_dir,
                    f"transition_{left['chunk_index']}_left",
                )
                right_frames = sample_video_frames(
                    gen_video,
                    right["generated_start_sec"],
                    min(right["generated_end_sec"], right["generated_start_sec"] + 1.0),
                    transition_side_frames,
                    temp_dir,
                    f"transition_{right['chunk_index']}_right",
                )
                result = call_vlm_with_frames(
                    build_transition_prompt(left, right),
                    [
                        ("Last frames of previous generated chunk:", left_frames),
                        ("First frames of next generated chunk:", right_frames),
                        ("Reference video context frames:", ref_context_frames),
                    ],
                    resolved_model_name,
                    vlm_backend,
                )
                result["transition_smoothness"] = clip_score(result.get("transition_smoothness"))
                overall = result.get("overall")
                if overall is None:
                    overall = result["transition_smoothness"]
                result["overall"] = clip_score(overall)
                transition_results.append(result)

        global_result: Optional[Dict[str, Any]] = None
        if not skip_global:
            print("Evaluating global long-range consistency...")
            gen_duration = video_duration(gen_video)
            gen_global_frames = sample_video_frames(
                gen_video,
                0.0,
                gen_duration,
                global_frames,
                temp_dir,
                "global_generated",
            )
            global_result = call_vlm_with_frames(
                build_global_prompt(chunks),
                [
                    ("Generated full-video frames:", gen_global_frames),
                    ("Reference video context frames:", ref_context_frames),
                ],
                resolved_model_name,
                vlm_backend,
            )
            global_result["long_range_consistency"] = clip_score(global_result.get("long_range_consistency"))
            global_result["global_text_alignment"] = clip_score(global_result.get("global_text_alignment"))
            overall = global_result.get("overall")
            if overall is None:
                overall = (global_result["long_range_consistency"] + global_result["global_text_alignment"]) / 2.0
            global_result["overall"] = clip_score(overall)

    scored_parts: List[float] = []
    scored_parts.extend(result["overall"] for result in chunk_results)
    scored_parts.extend(result["overall"] for result in transition_results)
    if global_result is not None:
        scored_parts.append(global_result["overall"])

    summary = {
        "chunk_mean": mean([result["overall"] for result in chunk_results]),
        "transition_mean": mean([result["overall"] for result in transition_results]),
        "global_score": global_result["overall"] if global_result else None,
        "overall": mean(scored_parts),
    }

    return {
        "generated_video": gen_video,
        "reference_video": ref_video,
        "prompt_json": prompt_json,
        "chunk_json": chunk_json,
        "backend": vlm_backend,
        "model": resolved_model_name,
        "summary": summary,
        "chunks": chunk_results,
        "transitions": transition_results,
        "global": global_result,
    }


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chunk-level interaction metrics with Qwen3-VL/VLM")
    parser.add_argument("--gen-video", required=True, help="Path to stitched generated video")
    parser.add_argument("--ref-video", required=True, help="Path to reference/context video")
    parser.add_argument("--prompt-json", required=True, help="JSON file containing chunk prompts/actions/captions")
    parser.add_argument("--chunk-json", required=True, help="JSON file containing generated chunk timestamps")
    parser.add_argument(
        "--backend",
        default=None,
        help="VLM backend: qwenvl_server, qwenvl/local, or openrouter/api. Defaults to qwenvl_server.",
    )
    parser.add_argument("--model", default=None, help="Model name or local model path")
    parser.add_argument("--output", help="Optional path to save JSON results")
    parser.add_argument("--chunk-frames", type=int, default=4, help="Frames sampled per generated chunk")
    parser.add_argument("--ref-frames", type=int, default=4, help="Reference context frames")
    parser.add_argument("--transition-side-frames", type=int, default=3, help="Frames sampled on each side of a chunk boundary")
    parser.add_argument("--global-frames", type=int, default=12, help="Frames sampled from the full generated video")
    parser.add_argument("--skip-chunk", action="store_true", help="Skip per-chunk scoring")
    parser.add_argument("--skip-transition", action="store_true", help="Skip adjacent-transition scoring")
    parser.add_argument("--skip-global", action="store_true", help="Skip whole-video consistency scoring")
    args = parser.parse_args()

    result = evaluate_interaction(
        gen_video=args.gen_video,
        ref_video=args.ref_video,
        prompt_json=args.prompt_json,
        chunk_json=args.chunk_json,
        model_name=args.model,
        backend=args.backend,
        chunk_frames=args.chunk_frames,
        ref_frames=args.ref_frames,
        transition_side_frames=args.transition_side_frames,
        global_frames=args.global_frames,
        skip_chunk=args.skip_chunk,
        skip_transition=args.skip_transition,
        skip_global=args.skip_global,
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
