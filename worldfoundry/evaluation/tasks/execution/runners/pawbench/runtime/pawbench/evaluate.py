"""Evaluate one model's local PAWBench rollouts."""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Sequence
from pathlib import Path

from pawbench.evaluation import evaluate

VIDEO_RE = re.compile(r"^r(\d{3,})$", re.IGNORECASE)
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".avi"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a directory of PAWBench video rollouts with PAWEval."
    )
    parser.add_argument("--benchmark", type=Path, required=True, help="Downloaded PAWBench data")
    parser.add_argument(
        "--videos",
        type=Path,
        required=True,
        help="Rollouts arranged as <scene-id>/r000.mp4 ... r049.mp4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for checkpoints, judgment rows, and metrics",
    )
    parser.add_argument("--model", required=True, help="Model name recorded in the results")
    parser.add_argument(
        "--vlm-base-url", required=True, help="OpenAI-compatible PAWEval endpoint"
    )
    parser.add_argument("--vlm-model", required=True, help="VLM used by PAWEval")
    parser.add_argument(
        "--vlm-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the endpoint API key",
    )
    return parser.parse_args(argv)


def collect_video_items(results_dir: Path, model_or_lane: str) -> list[dict[str, object]]:
    """Discover ``<scene_id>/r###.<video extension>`` rollout files."""

    if not results_dir.is_dir():
        raise SystemExit(f"PAWBENCH_RESULTS_DIR does not exist: {results_dir}")
    items: list[dict[str, object]] = []
    for scene_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        for video in sorted(scene_dir.iterdir()):
            match = VIDEO_RE.fullmatch(video.stem)
            if not match or not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            items.append(
                {
                    "sample_id": f"{model_or_lane}::{scene_dir.name}::{video.stem}",
                    "scene_id": scene_dir.name,
                    "repeat_index": int(match.group(1)),
                    "video_path": str(video.resolve()),
                }
            )
    if not items:
        raise SystemExit(f"No <scene_id>/r###.mp4 rollouts found under {results_dir}")
    return items


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    benchmark_dir = args.benchmark.expanduser()
    results_dir = args.videos.expanduser()
    output_dir = args.output.expanduser()
    if not os.environ.get(args.vlm_api_key_env):
        raise SystemExit(f"Set {args.vlm_api_key_env} to the PAWEval endpoint API key.")
    if not (benchmark_dir / "manifest.json").is_file():
        raise SystemExit(f"Not a PAWBench data directory: {benchmark_dir}")

    videos = collect_video_items(results_dir, args.model)
    print(f"Collected {len(videos)} videos from {results_dir}")
    result = evaluate(
        benchmark_dir,
        videos,
        model_or_lane=args.model,
        vlm={
            "base_url": args.vlm_base_url,
            "model": args.vlm_model,
            "api_key_env": args.vlm_api_key_env,
        },
        output_dir=output_dir,
    )
    print(f"status: {result['status']}")
    for blocker in result["blockers"]:
        print(f"blocker: {blocker}")
    print(f"metrics: {result['metrics']}")
    for name, path in result.get("artifacts", {}).items():
        print(f"{name}: {path}")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
