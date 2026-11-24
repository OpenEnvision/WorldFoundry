#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Drive the native Self-Forcing or Causal-Forcing realtime session and compare it with the offline call.

``--model-id`` selects ``self-forcing`` (default) or a ``causal-forcing-*`` recipe.
One offline ``pipeline(prompt, seed, return_dict=True)`` run
produces the reference uint8 video. Each realtime run then calls
``prepare_realtime``, ``configure_realtime`` and ``stream_realtime`` once per
latent block, and the concatenated uint8 chunks must equal the reference bitwise.
The report records time to first frame (configure plus the first chunk), the
wall time of every chunk, and steady-state frames per second over the chunks
after the first, against the 16 fps playback clock. The offline call runs first,
so every realtime run is warm.

Example::

    python tests/stream/test_self_forcing_stream.py \\
        --checkpoint ckpts/gdhe17--Self-Forcing/checkpoints/self_forcing_dmd.pt \\
        --wan-root ckpts/Wan-AI--Wan2.1-T2V-1.3B \\
        --prompt-file prompts/MovieGenVideoBench_extended.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from worldfoundry.core.media.codecs.video import video_tensor_to_uint8_frames  # noqa: E402
from worldfoundry.pipelines.forcing.pipeline_native_forcing import (  # noqa: E402
    NativeCausalForcingPipeline,
    NativeSelfForcingPipeline,
)


def realtime_run(pipeline: NativeSelfForcingPipeline, prompt: str, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Stream one complete rollout; return its frames and timings."""

    pipeline.reset_realtime()
    started = time.perf_counter()
    configured = pipeline.configure_realtime(prompt=prompt, seed=seed)
    configure_ms = (time.perf_counter() - started) * 1000.0
    total = int(configured["realtime_spec"]["max_chunks"])
    chunks: list[np.ndarray] = []
    timings: list[dict[str, float]] = []
    for _ in range(total):
        chunk_started = time.perf_counter()
        result = pipeline.stream_realtime(prompt=prompt)
        wall_ms = (time.perf_counter() - chunk_started) * 1000.0
        chunks.append(result["video"])
        metrics = result["realtime_metrics"]
        timings.append(
            {
                "frames": int(result["video"].shape[0]),
                "wall_ms": round(wall_ms, 1),
                "model_ms": round(float(metrics["model_ms"]), 1),
                "decode_ms": round(float(metrics["decode_ms"]), 1),
            }
        )
    steady = timings[1:]
    steady_frames = sum(item["frames"] for item in steady)
    steady_seconds = sum(item["wall_ms"] for item in steady) / 1000.0
    return np.concatenate(chunks), {
        "configure_ms": round(configure_ms, 1),
        "time_to_first_frame_ms": round(configure_ms + timings[0]["wall_ms"], 1),
        "chunks": timings,
        "steady_state_fps": round(steady_frames / steady_seconds, 2),
    }


def main() -> None:
    """Load the native pipeline once, run the offline reference, then the realtime runs."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Generator checkpoint, e.g. self_forcing_dmd.pt")
    parser.add_argument("--model-id", default="self-forcing", help="self-forcing or a causal-forcing-* recipe")
    parser.add_argument("--wan-root", type=Path, required=True, help="Wan2.1-T2V-1.3B official layout directory")
    parser.add_argument("--prompt-file", type=Path, required=True, help="The first line is the prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3, help="Realtime rollouts after the offline call")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/self-forcing-stream.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The forcing realtime check runs the real checkpoint on a CUDA device")
    prompt = args.prompt_file.read_text(encoding="utf-8").splitlines()[0].strip()

    load_started = time.perf_counter()
    pipeline_class = NativeSelfForcingPipeline if args.model_id == "self-forcing" else NativeCausalForcingPipeline
    pipeline = cast(
        NativeSelfForcingPipeline,
        pipeline_class.from_pretrained(
            model_id=args.model_id,
            device=str(device),
            torch_dtype="bf16",
            offload_mode="none",
            checkpoint_overrides={
                "dit": str(args.checkpoint),
                "text-encoder": str(args.wan_root),
                "tokenizer": str(args.wan_root),
                "vae": str(args.wan_root),
            },
        ),
    )
    load_seconds = time.perf_counter() - load_started
    prepared = pipeline.prepare_realtime()["realtime_spec"]

    offline_started = time.perf_counter()
    offline = pipeline(prompt=prompt, seed=args.seed, return_dict=True)
    expected = video_tensor_to_uint8_frames(offline["video"], value_range="-1,1", layout="bcthw")
    offline_seconds = time.perf_counter() - offline_started

    runs = []
    for _ in range(args.repeats):
        frames, timings = realtime_run(pipeline, prompt, args.seed)
        mismatched = int(np.count_nonzero(frames != expected)) if frames.shape == expected.shape else None
        runs.append(
            {
                "frames_shape": list(frames.shape),
                "bitwise_equal": frames.shape == expected.shape and mismatched == 0,
                "mismatched_values": mismatched,
                **timings,
            }
        )
    pipeline.cleanup()

    report = {
        "evidence_level": "real checkpoint, realtime session versus offline call (one request)",
        "model_id": pipeline.model_id,
        "prompt": prompt,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "dtype": "bfloat16",
        "load_seconds": round(load_seconds, 2),
        "offline_seconds_cold": round(offline_seconds, 2),
        "offline_frames_shape": list(expected.shape),
        "prepare_realtime_spec": prepared,
        "playback_fps": prepared["fps"],
        "runs": runs,
        "all_bitwise_equal": all(run["bitwise_equal"] for run in runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_bitwise_equal"]:
        raise SystemExit("realtime frames differ from the offline call")


if __name__ == "__main__":
    main()
