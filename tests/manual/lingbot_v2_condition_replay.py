# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Matched 333-frame LingBot-V2 rollouts with settled I2V condition reuse."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tests.manual.lingbot_v2_vae_replay import PROMPT, compare_rgb


def summarize(runs: list[dict[str, Any]], start_chunk: int) -> dict[str, Any]:
    """Summarize matched warm stages without including numerical capture."""
    result = {}
    for key in ("condition_ms", "model_ms", "decode_ms", "total_ms"):
        per_run = [statistics.mean(chunk[key] for chunk in run["chunks"][start_chunk:]) for run in runs]
        all_chunks = [chunk[key] for run in runs for chunk in run["chunks"][start_chunk:]]
        result[key] = {
            "per_run": per_run,
            "mean": statistics.mean(per_run),
            "median": statistics.median(all_chunks),
            "p90": sorted(all_chunks)[int(0.9 * (len(all_chunks) - 1))],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("../ckpts/lingbot-world-v2-1.3b-causal-fast"))
    parser.add_argument("--shared-assets", type=Path, default=Path("../ckpts/Wan-AI--Wan2.1-T2V-1.3B"))
    parser.add_argument("--image", type=Path, default=Path("testcase/012_abandoned-room.png"))
    parser.add_argument("--output", type=Path, default=Path("tmp/i2v-condition-reuse/replay"))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["WORLDFOUNDRY_LINGBOT_REALTIME_HEIGHT"] = "464"
    os.environ["WORLDFOUNDRY_LINGBOT_REALTIME_WIDTH"] = "832"
    os.environ["WORLDFOUNDRY_LINGBOT_V2_REALTIME_COMPILE"] = "0"
    os.environ["WORLDFOUNDRY_REALTIME_CONDITION_PREFETCH_CHUNKS"] = "0"

    import numpy as np
    import torch
    from PIL import Image
    from torch.version import cuda as cuda_version

    from worldfoundry.core.attention.backends.dispatch import (
        attention_provider_runtime_report,
        reset_attention_provider_runtime,
    )
    from worldfoundry.core.media.codecs.video import save_image_or_video_tensor
    from worldfoundry.synthesis.visual_generation.lingbot_world_v2.inference import LingBotWorldV2Inference
    from worldfoundry.synthesis.visual_generation.lingbot_world_v2.realtime import LingBotWorldV2SmallRealtimeSession

    torch.set_num_threads(8)
    image = Image.open(args.image).convert("RGB")
    started = time.perf_counter()
    core = LingBotWorldV2Inference(
        args.checkpoint,
        model_id="lingbot-world-v2-1.3b",
        shared_assets_path=args.shared_assets,
        attention_backend="torch",
        t5_cpu=False,
    )
    session = LingBotWorldV2SmallRealtimeSession(core)
    factory = session._new_condition_cache
    decode = session._decode
    report: dict[str, Any] = {
        "load_ms": (time.perf_counter() - started) * 1000,
        "settings": {
            "checkpoint": str(args.checkpoint.resolve()),
            "shared_assets": str(args.shared_assets.resolve()),
            "image": str(args.image.resolve()),
            "prompt": PROMPT,
            "frames": 333,
            "chunks": 21,
            "height": 464,
            "width": 832,
            "fps": 16,
            "dit_compile": False,
            "decoder_compile": False,
            "attention": "torch",
            "dit_dtype": str(core.param_dtype),
            "vae_dtype": str(core.vae.dtype),
            "condition_replenish_chunks": session.condition_replenish_chunks,
            "world_scale": session.world_scale,
            "torch": torch.__version__,
            "cuda": cuda_version,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "native_sdpa_priority": os.environ.get("WORLDFOUNDRY_NATIVE_SDPA_PRIORITY", ""),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "instrumented": {},
        "comparisons": {},
        "warm": {"baseline": [], "cached": []},
    }

    def save_report() -> None:
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    def configure(mode: str, source: Image.Image, seed: int, prefetch: int) -> float:
        os.environ["WORLDFOUNDRY_REALTIME_CONDITION_PREFETCH_CHUNKS"] = str(prefetch)
        start = time.perf_counter()
        with patch.object(session, "_new_condition_cache", factory if mode == "cached" else lambda: None):
            session.configure(image=source, prompt=PROMPT, seed=seed, fps=16)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000

    @torch.no_grad()
    def rollout(mode: str, seed: int, name: str, prefetch: int, *, capture: bool) -> dict[str, Any]:
        configure_ms = configure(mode, image, seed, prefetch)
        reset_attention_provider_runtime()
        frames, metrics, latents, conditions = [], [], [], []
        encoder_calls = 0
        model_calls = 0

        def count_encode(_module: Any, _inputs: Any, _output: Any) -> None:
            nonlocal encoder_calls
            encoder_calls += 1

        def capture_model(_module: Any, _inputs: Any, kwargs: dict[str, Any]) -> None:
            nonlocal model_calls
            if model_calls % 5 == 0:
                conditions.append(kwargs["y"][0].detach().cpu().clone())
            model_calls += 1

        def capture_decode(value: torch.Tensor) -> np.ndarray:
            latents.append(value.detach().cpu().clone())
            return decode(value)

        if capture:
            assert session._pending_first_condition is not None
            assert session._timesteps is not None and session._sigmas is not None
            assert session._generator is not None
            torch.save(
                {
                    "context": [value.cpu() for value in session._context or []],
                    "first_condition": session._pending_first_condition.cpu(),
                    "timesteps": session._timesteps.cpu(),
                    "sigmas": session._sigmas.cpu(),
                    "rng": session._generator.get_state(),
                },
                out / f"{name}-configure.pt",
            )
        with ExitStack() as stack:
            if capture:
                stack.callback(core.vae.model.encoder.register_forward_hook(count_encode).remove)
                stack.callback(core.model.register_forward_pre_hook(capture_model, with_kwargs=True).remove)
                stack.enter_context(patch.object(session, "_decode", capture_decode))
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for chunk in range(21):
                result = session.generate(interactions=["forward"])
                assert result is not None
                frames.append(result["video"])
                metrics.append(result["realtime_metrics"])
                print(name, chunk, json.dumps(metrics[-1]), flush=True)
            wall_ms = (time.perf_counter() - start) * 1000
        pixels = np.concatenate(frames)
        assert pixels.shape == (333, 464, 832, 3)
        np.save(out / f"{name}.npy", pixels)
        if capture:
            assert len(latents) == len(conditions) == 21 and model_calls == 105
            assert all(torch.isfinite(value).all() for value in latents + conditions)
            torch.save({"latents": latents, "conditions": conditions}, out / f"{name}-boundaries.pt")
        providers = attention_provider_runtime_report()
        assert sum(value["successes"] for value in providers.values()) > 0
        assert sum(value["fallbacks"] + value["errors"] for value in providers.values()) == 0
        cache = session._condition_cache
        if mode == "cached":
            assert cache is not None and cache.reuse_calls > 0
        return {
            "seed": seed,
            "prefetch": prefetch,
            "configure_ms": configure_ms,
            "first_chunk_ms": metrics[0]["total_ms"],
            "wall_ms": wall_ms,
            "chunks": metrics,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "attention": providers,
            "encoder_calls_after_configure": encoder_calls if capture else None,
            "history_frames": cache.history_frames if cache is not None else None,
        }

    def compare_boundaries(left: str, right: str) -> dict[str, Any]:
        result = compare_rgb(out / f"{left}.npy", out / f"{right}.npy")
        a = torch.load(out / f"{left}-boundaries.pt", weights_only=True)
        b = torch.load(out / f"{right}-boundaries.pt", weights_only=True)
        for key in ("latents", "conditions"):
            result[f"{key}_exact"] = all(torch.equal(x, y) for x, y in zip(a[key], b[key], strict=True))
            assert result[f"{key}_exact"], (key, left, right)
        assert result["exact"], result
        return result

    with torch.no_grad():
        # A different image exercises encoder-history replacement before the A rollouts.
        changed_image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image_conditions = {}
        for mode in ("baseline", "cached"):
            configure(mode, changed_image, 43, 0)
            assert session._pending_first_condition is not None
            values = [session._pending_first_condition.detach().cpu().clone()]
            for _ in range(20):
                if not session._prefetched_conditions:
                    session._prefetch_steady_conditions(session.condition_replenish_chunks)
                values.append(session._prefetched_conditions.pop(0).detach().cpu().clone())
            image_conditions[mode] = values
        assert all(torch.equal(a, b) for a, b in zip(image_conditions["baseline"], image_conditions["cached"]))
        torch.save(image_conditions, out / "changed-image-conditions.pt")
        report["changed_image_conditions_exact"] = True
        for prefetch in (0, 8):
            for mode in ("baseline", "cached"):
                name = f"{mode}-prefetch{prefetch}"
                report["instrumented"][name] = rollout(mode, 42, name, prefetch, capture=True)
                save_report()
            report["comparisons"][f"prefetch{prefetch}"] = compare_boundaries(
                f"baseline-prefetch{prefetch}",
                f"cached-prefetch{prefetch}",
            )
            save_report()
        for repeat, seed in enumerate((42, 43, 42)):
            for mode in ("cached", "baseline") if repeat == 1 else ("baseline", "cached"):
                name = f"{mode}-warm{repeat}"
                report["warm"][mode].append(rollout(mode, seed, name, 0, capture=False))
                save_report()
            assert compare_rgb(out / f"baseline-warm{repeat}.npy", out / f"cached-warm{repeat}.npy")["exact"]
        for mode in ("baseline", "cached"):
            assert compare_rgb(out / f"{mode}-warm0.npy", out / f"{mode}-warm2.npy")["exact"]
            assert compare_rgb(out / f"{mode}-prefetch0.npy", out / f"{mode}-warm0.npy")["exact"]
            assert not compare_rgb(out / f"{mode}-warm0.npy", out / f"{mode}-warm1.npy")["exact"]
        report["aba_and_warm_rgb_exact"] = True
        # Report both the established measurement window and the fully settled tail.
        report["summaries"] = {
            str(start): {mode: summarize(runs, start) for mode, runs in report["warm"].items()} for start in (5, 13)
        }
        report["chunk_latency_reduction_percent"] = {
            start: 100 * (1 - values["cached"]["total_ms"]["mean"] / values["baseline"]["total_ms"]["mean"])
            for start, values in report["summaries"].items()
        }
        session.reset()
        save_report()
    for mode in ("baseline", "cached"):
        pixels = np.load(out / f"{mode}-prefetch0.npy")
        save_image_or_video_tensor(torch.from_numpy(pixels).permute(3, 0, 1, 2), str(out / f"{mode}.mp4"), fps=16)
    report["gpu_processes_at_completion"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    )
    save_report()
    print(json.dumps(report["summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()
