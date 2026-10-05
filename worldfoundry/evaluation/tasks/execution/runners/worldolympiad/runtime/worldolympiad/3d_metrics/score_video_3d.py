#!/usr/bin/env python3
"""Score a single video with the local 3D reward backend."""

from __future__ import annotations

from _bootstrap import setup_paths

setup_paths()

import argparse
import json
import os
from pathlib import Path

from model.vlm import resolve_model_name, resolve_vlm_backend

DEFAULT_MODEL_NAME = str(Path(__file__).resolve().parents[1] / "weights" / "da3")
DEFAULT_SCORING_MODEL: str | None = None


def resolve_prompt(video_path: str | os.PathLike[str], explicit_prompt: str | None) -> str:
    """Resolve the prompt from CLI or from a sibling prompt.txt file."""
    if explicit_prompt is not None:
        return explicit_prompt

    prompt_path = Path(video_path).resolve().parent / "prompt.txt"
    if not prompt_path.exists():
        return ""

    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    if prompt_text:
        print(f"Using prompt from {prompt_path}")
    return prompt_text


def main():
    parser = argparse.ArgumentParser(description="Score a video with 3D consistency metrics")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt associated with the video; if omitted, use prompt.txt next to the video when available",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="DA3 reconstruction model name or local model path",
    )
    parser.add_argument(
        "--vlm-backend",
        "--scorer",
        dest="scorer",
        default=os.getenv("REWARD_3D_SCORER"),
        help="VLM backend used for GS/meta scoring: api/openrouter or local/qwenvl.",
    )
    parser.add_argument(
        "--vlm-model",
        "--scoring-model",
        dest="scoring_model",
        default=os.getenv("REWARD_3D_SCORING_MODEL") or DEFAULT_SCORING_MODEL,
        help="VLM model name or local path used to score GS/meta renderings",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of local DA3 worker processes / GPUs to use",
    )
    parser.add_argument("--max-frames", type=int, default=32, help="Maximum number of frames to sample from the video")
    parser.add_argument("--gpus", help="Visible GPU ids, e.g. '0' or '0,1'")
    parser.add_argument(
        "--dynamic-masked-video",
        help="Precomputed video with dynamic objects removed; reconstruction uses this path when provided",
    )
    parser.add_argument(
        "--dynamic-mask-video",
        help="Precomputed binary dynamic-object mask video used to filter DA3 Gaussians after reconstruction",
    )
    parser.add_argument("--camera-trajectory", help="Path to a JSON camera trajectory file")
    parser.add_argument("--camera-trajectory-json", help="Raw camera trajectory JSON string")
    parser.add_argument("--lpips", dest="lpips", action="store_true", help="Use LPIPS for GS reconstruction scoring")
    parser.add_argument("--no-lpips", dest="lpips", action="store_false", help="Disable LPIPS scoring")
    parser.set_defaults(lpips=False)
    args = parser.parse_args()
    args.scorer = resolve_vlm_backend(args.scorer)
    args.scoring_model = resolve_model_name(args.scoring_model, args.scorer)

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    from reward_3d import load_camera_trajectory, score_video_file

    camera_trajectory = load_camera_trajectory(
        camera_trajectory_path=args.camera_trajectory,
        camera_trajectory_json=args.camera_trajectory_json,
    )
    prompt = resolve_prompt(args.video, args.prompt)

    reconstruction_video = args.dynamic_masked_video or args.video
    if args.dynamic_masked_video:
        print(f"Using dynamic-object-masked video for reconstruction: {args.dynamic_masked_video}")

    result = score_video_file(
        reconstruction_video,
        prompt=prompt,
        model_name=args.model_name,
        scorer_type=args.scorer,
        scoring_model_name=args.scoring_model,
        use_lpips=args.lpips,
        num_workers=args.num_workers,
        max_frames=args.max_frames,
        camera_trajectory=camera_trajectory,
        dynamic_mask_video_path=args.dynamic_mask_video,
    )
    result["source_video_path"] = str(Path(args.video).resolve())
    result["dynamic_masked_video_path"] = (
        str(Path(args.dynamic_masked_video).resolve()) if args.dynamic_masked_video else None
    )
    result["dynamic_mask_video_path"] = (
        str(Path(args.dynamic_mask_video).resolve()) if args.dynamic_mask_video else None
    )

    result_json = json.dumps(result, ensure_ascii=False, indent=2)
    print(result_json)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
