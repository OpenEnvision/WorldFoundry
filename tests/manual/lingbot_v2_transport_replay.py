# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-backed LingBot-V2 CPU versus deferred frame delivery, excluding browser transport.

Run from the repository root in the prepared GPU environment. Uses local weights
only; compares identical RGB across routes, changed seeds and A/B/A resets.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

PROMPT = "A dusty abandoned room with wooden floor and sunlight through the window. The camera slowly moves forward."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shared-assets", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=21)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.chunks < 7 or args.rounds < 3:
        parser.error("use at least seven chunks (including KV eviction) and three measured rounds")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["WORLDFOUNDRY_LINGBOT_REALTIME_HEIGHT"] = "464"
    os.environ["WORLDFOUNDRY_LINGBOT_REALTIME_WIDTH"] = "832"
    os.environ["WORLDFOUNDRY_LINGBOT_V2_REALTIME_COMPILE"] = "0"
    os.environ["WORLDFOUNDRY_REALTIME_CONDITION_PREFETCH_CHUNKS"] = "0"

    import numpy as np
    import torch
    from PIL import Image
    from torch.version import cuda as cuda_version

    from worldfoundry.pipelines.lingbot_world_v2.pipeline_lingbot_world_v2 import LingBotWorldV2SmallPipeline
    from worldfoundry.studio.serving.realtime.media import realtime_frames_from_result
    from worldfoundry.synthesis.visual_generation.lingbot_world_v2.inference import LingBotWorldV2Inference
    from worldfoundry.synthesis.visual_generation.lingbot_world_v2.realtime import LingBotWorldV2SmallRealtimeSession

    torch.set_num_threads(8)
    image = Image.open(args.image).convert("RGB")
    report: dict[str, Any] = {
        "scope": "real_checkpoint_session_and_public_stream_realtime_to_CPU_frames_not_browser_latency",
        "settings": {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "height": 464,
            "width": 832,
            "fps": 16,
            "prompt": PROMPT,
            "dit_compile": False,
            "vae_compile": False,
            "attention": "torch",
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": cuda_version,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "worktree_dirty": bool(
                subprocess.check_output(["git", "diff", "--name-only", "--", "worldfoundry"], text=True).strip()
            ),
        },
        "warmup": {},
        "runs": {"cpu": [], "deferred": []},
    }

    def save() -> None:
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    started = time.perf_counter()
    core = LingBotWorldV2Inference(
        args.checkpoint,
        model_id="lingbot-world-v2-1.3b",
        shared_assets_path=args.shared_assets,
        attention_backend="torch",
        t5_cpu=False,
    )
    session = LingBotWorldV2SmallRealtimeSession(core)
    pipeline = object.__new__(LingBotWorldV2SmallPipeline)
    pipeline._realtime_session = session
    report["load_ms"] = (time.perf_counter() - started) * 1000
    save()
    print("loaded", report["load_ms"], flush=True)

    def rollout(mode: str, seed: int) -> tuple[dict[str, Any], list[np.ndarray]]:
        session.reset()
        configure_start = time.perf_counter()
        session.configure(image=image, prompt=PROMPT, seed=seed, fps=16)
        torch.cuda.synchronize(session.device)
        configure_ms = (time.perf_counter() - configure_start) * 1000
        torch.cuda.reset_peak_memory_stats(session.device)
        frames, timings = [], []
        for index in range(args.chunks):
            start = time.perf_counter()
            result = (
                pipeline.stream_realtime(interactions=["forward"])
                if mode == "deferred"
                else session.generate(interactions=["forward"])
            )
            generated = time.perf_counter()
            assert result is not None
            output = realtime_frames_from_result(result)
            delivered = time.perf_counter()
            assert len(output) == (13 if index == 0 else 16)
            frames.append(np.stack(output))
            timings.append(
                {
                    **result["realtime_metrics"],
                    "generation_wall_ms": (generated - start) * 1000,
                    "materialize_ms": (delivered - generated) * 1000,
                    "delivery_wall_ms": (delivered - start) * 1000,
                }
            )
        return {
            "seed": seed,
            "configure_ms": configure_ms,
            "chunks": timings,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(session.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(session.device),
        }, frames

    def equal(left: list[np.ndarray], right: list[np.ndarray]) -> None:
        assert len(left) == len(right)
        for index, (a, b) in enumerate(zip(left, right)):
            np.testing.assert_array_equal(a, b, err_msg=f"RGB differs at chunk {index}")

    with torch.no_grad():
        for mode in ("cpu", "deferred"):
            report["warmup"][mode], pixels = rollout(mode, 42)
            if mode == "cpu":
                reference = pixels
            else:
                equal(reference, pixels)
            print("warmup", mode, flush=True)
            save()
        first_seed_pixels = reference
        for repeat in range(args.rounds):
            seed = 43 if repeat % 2 else 42
            outputs = {}
            for mode in ("cpu", "deferred") if repeat % 2 == 0 else ("deferred", "cpu"):
                run, outputs[mode] = rollout(mode, seed)
                report["runs"][mode].append(run)
                if seed == 42:
                    equal(first_seed_pixels, outputs[mode])
                else:
                    assert any(not np.array_equal(a, b) for a, b in zip(first_seed_pixels, outputs[mode]))
                print(
                    "round",
                    repeat,
                    mode,
                    "mean delivery ms",
                    statistics.mean(chunk["delivery_wall_ms"] for chunk in run["chunks"][5:]),
                    flush=True,
                )
                save()
            equal(outputs["cpu"], outputs["deferred"])
            del outputs
        report["rgb_exact_across_routes_and_resets"] = True
        summary: dict[str, Any] = {
            mode: {
                key: {
                    "per_run_ms": values,
                    "mean_ms": statistics.mean(values),
                    "min_ms": min(values),
                    "max_ms": max(values),
                }
                for key in ("generation_wall_ms", "materialize_ms", "delivery_wall_ms")
                for values in [[statistics.mean(chunk[key] for chunk in run["chunks"][5:]) for run in runs]]
            }
            for mode, runs in report["runs"].items()
        }
        report["warm_summary"] = summary
        report["delivery_speedup_ratio"] = (
            summary["cpu"]["delivery_wall_ms"]["mean_ms"] / summary["deferred"]["delivery_wall_ms"]["mean_ms"]
        )
        save()
        # Attribution only; these timings do not enter the headline summaries.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True
        ) as profiler:
            result = pipeline.stream_realtime(interactions=["forward"])
            realtime_frames_from_result(result)
        profiler.export_chrome_trace(str(args.output / "profile.json"))
        (args.output / "profile.txt").write_text(
            profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
        )
    session.reset()
    print(json.dumps(report["warm_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
