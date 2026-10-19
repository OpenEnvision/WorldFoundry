# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Full LingBot-V2 1.3B rollouts comparing eager and compiled Wan decoding."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

PROMPT = "A dusty abandoned room with wooden floor and sunlight through the window. The camera slowly moves forward."


def compare_rgb(reference: Path, candidate: Path) -> dict[str, Any]:
    import numpy as np

    left = np.load(reference, mmap_mode="r")
    right = np.load(candidate, mmap_mode="r")
    assert left.shape == right.shape
    total = 0
    squared = 0.0
    absolute = 0.0
    maximum = 0
    changed = 0
    for begin in range(0, len(left), 16):
        difference = left[begin : begin + 16].astype(np.int16) - right[begin : begin + 16].astype(np.int16)
        total += difference.size
        absolute += float(np.abs(difference).sum(dtype=np.float64))
        squared += float(np.square(difference.astype(np.float32)).sum(dtype=np.float64))
        maximum = max(maximum, int(np.abs(difference).max()))
        changed += int(np.count_nonzero(difference))
    return {
        "shape": list(left.shape),
        "exact": changed == 0,
        "max_pixel_error": maximum,
        "mae_255": absolute / total,
        "changed_component_fraction": changed / total,
        "psnr_db": 10 * math.log10(255**2 * total / squared) if squared else None,
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for key in ("condition_ms", "model_ms", "decode_ms", "total_ms"):
        values = [statistics.mean(chunk[key] for chunk in run["chunks"][5:]) for run in runs]
        result[key] = {
            "per_run": values,
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("../ckpts/lingbot-world-v2-1.3b-causal-fast"))
    parser.add_argument("--shared-assets", type=Path, default=Path("../ckpts/Wan-AI--Wan2.1-T2V-1.3B"))
    parser.add_argument("--image", type=Path, default=Path("testcase/012_abandoned-room.png"))
    parser.add_argument("--output", type=Path, default=Path("tmp/optimization-validation/vae-long"))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["WORLDFOUNDRY_LINGBOT_REALTIME_HEIGHT"] = "464"
    os.environ["WORLDFOUNDRY_LINGBOT_REALTIME_WIDTH"] = "832"
    os.environ["WORLDFOUNDRY_LINGBOT_V2_REALTIME_COMPILE"] = "0"
    os.environ["WORLDFOUNDRY_REALTIME_CONDITION_PREFETCH_CHUNKS"] = "0"

    from unittest.mock import patch

    import numpy as np
    import torch
    from PIL import Image
    from torch._dynamo.utils import counters
    from torch.version import cuda as cuda_version

    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.streaming import StreamingWanDecoder
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
    eager_decoder = session._decoder
    assert isinstance(eager_decoder, StreamingWanDecoder)
    decoders = {
        "eager": eager_decoder,
        "compiled": StreamingWanDecoder(core.vae, channels_last_3d=True, compile_decoder=True),
    }
    report: dict[str, Any] = {
        "load_ms": (time.perf_counter() - started) * 1000,
        "settings": {
            "model_id": "lingbot-world-v2-1.3b",
            "checkpoint_source": str(args.checkpoint.resolve()),
            "shared_assets": str(args.shared_assets.resolve()),
            "image": str(args.image.resolve()),
            "chunks": 21,
            "frames": 333,
            "height": 464,
            "width": 832,
            "fps": 16,
            "prompt": PROMPT,
            "attention": "torch",
            "dit_compile": False,
            "camera_reuse": True,
            "condition_prefetch_chunks": 0,
            "condition_replenish_chunks": session.condition_replenish_chunks,
            "world_scale": session.world_scale,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "native_sdpa_priority": os.environ.get("WORLDFOUNDRY_NATIVE_SDPA_PRIORITY", ""),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch": torch.__version__,
            "cuda": cuda_version,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(),
            "gpu_multiprocessors": torch.cuda.get_device_properties(core.device).multi_processor_count,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "instrumented": {},
        "runs": {"eager": [], "compiled": []},
    }
    print("loaded", report["load_ms"], flush=True)

    def save_report() -> None:
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    def rollout(mode: str, seed: int, name: str) -> tuple[dict[str, Any], Path]:
        session.reset()
        decoder = decoders[mode]
        session._decoder = decoder
        decoder.reset()
        reset_attention_provider_runtime()
        configure_start = time.perf_counter()
        session.configure(image=image, prompt=PROMPT, seed=seed, fps=16)
        torch.cuda.synchronize()
        configure_ms = (time.perf_counter() - configure_start) * 1000
        if name.endswith("-instrumented"):
            assert session._context is not None and session._pending_first_condition is not None
            assert session._timesteps is not None and session._sigmas is not None
            assert session._generator is not None
            torch.save(
                {
                    "context": [value.detach().cpu() for value in session._context],
                    "first_condition": session._pending_first_condition.detach().cpu(),
                    "timesteps": session._timesteps.cpu(),
                    "sigmas": session._sigmas.cpu(),
                    "generator_state": session._generator.get_state(),
                },
                out / f"{name}-configure.pt",
            )
        torch.cuda.reset_peak_memory_stats()
        path = out / f"{name}.npy"
        frames = []
        position = 0
        metrics = []
        started = time.perf_counter()
        for index in range(21):
            result = session.generate(interactions=["forward"])
            assert result is not None
            video = result["video"]
            assert len(video) == (13 if index == 0 else 16)
            frames.append(video)
            position += len(video)
            metrics.append(result["realtime_metrics"])
            print(name, index, json.dumps(metrics[-1]), flush=True)
        assert position == 333
        np.save(path, np.concatenate(frames))
        providers = attention_provider_runtime_report()
        assert sum(value["successes"] for value in providers.values()) > 0
        assert sum(value["errors"] for value in providers.values()) == 0
        assert sum(value["fallbacks"] for value in providers.values()) == 0
        run = {
            "seed": seed,
            "configure_ms": configure_ms,
            "wall_with_output_ms": (time.perf_counter() - started) * 1000,
            "chunks": metrics,
            "decoder": decoder.report(),
            "attention": providers,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "compile_stats": dict(counters["stats"]),
        }
        return run, path

    normalized = {"count": 0, "absolute": 0.0, "squared": 0.0, "reference_squared": 0.0, "max_abs": 0.0}
    latent_directory = out / "latents"
    latent_directory.mkdir(exist_ok=True)
    float_directory = out / "eager-normalized"
    float_directory.mkdir(exist_ok=True)
    with torch.no_grad():
        for mode in ("eager", "compiled"):
            decoder = decoders[mode]
            original_decode = decoder.decode

            def capture_decode(latents: torch.Tensor) -> torch.Tensor:
                index = session.autoregressive_index
                latent_path = latent_directory / f"{mode}-{index:02d}.pt"
                torch.save(latents.detach().cpu(), latent_path)
                if mode == "compiled":
                    expected_latents = torch.load(latent_directory / f"eager-{index:02d}.pt", weights_only=True)
                    assert torch.equal(latents.cpu(), expected_latents), f"denoising latents differ at chunk {index}"
                decoded = original_decode(latents)
                value = decoded.detach().cpu().numpy()
                assert np.isfinite(value).all()
                if mode == "eager":
                    np.save(float_directory / f"{index:02d}.npy", value)
                else:
                    expected = np.load(float_directory / f"{index:02d}.npy")
                    difference = value - expected
                    normalized["count"] += difference.size
                    normalized["absolute"] += float(np.abs(difference).sum(dtype=np.float64))
                    normalized["squared"] += float(np.square(difference).sum(dtype=np.float64))
                    normalized["reference_squared"] += float(np.square(expected).sum(dtype=np.float64))
                    normalized["max_abs"] = max(normalized["max_abs"], float(np.abs(difference).max()))
                return decoded

            with patch.object(decoder, "decode", capture_decode):
                run, pixels = rollout(mode, 42, f"{mode}-instrumented")
            report["instrumented"][mode] = run
            save_image_or_video_tensor(
                torch.from_numpy(np.load(pixels)).permute(3, 0, 1, 2),
                out / f"{mode}.mp4",
                fps=16,
                value_range=(0, 255),
            )
            save_report()
        count = normalized["count"]
        report["normalized_comparison"] = {
            "max_abs": normalized["max_abs"],
            "mae": normalized["absolute"] / count,
            "relative_l2": math.sqrt(normalized["squared"] / normalized["reference_squared"]),
            "psnr_db": 10 * math.log10(4 * count / normalized["squared"]) if normalized["squared"] else None,
            "denoising_latents_exact": True,
        }
        report["rgb_comparison"] = compare_rgb(out / "eager-instrumented.npy", out / "compiled-instrumented.npy")
        save_report()

        for repeat, seed in enumerate((42, 43, 42)):
            modes = ("eager", "compiled") if repeat != 1 else ("compiled", "eager")
            for mode in modes:
                run, pixels = rollout(mode, seed, f"{mode}-run-{repeat}")
                report["runs"][mode].append(run)
                if repeat == 0:
                    identical = compare_rgb(out / f"{mode}-instrumented.npy", pixels)
                    assert identical["exact"], f"warmup changed {mode} seed-42 output"
                if repeat == 1:
                    different = compare_rgb(out / f"{mode}-run-0.npy", pixels)
                    assert not different["exact"], f"{mode} ignored changed seed"
                if repeat == 2:
                    identical = compare_rgb(out / f"{mode}-run-0.npy", pixels)
                    assert identical["exact"], f"{mode} A/B/A reset changed output"
                save_report()
        report["warm_summary"] = {mode: summarize(runs) for mode, runs in report["runs"].items()}
        report["warmup_and_aba_rgb_exact"] = True
        report["decode_speedup"] = (
            report["warm_summary"]["eager"]["decode_ms"]["mean"]
            / report["warm_summary"]["compiled"]["decode_ms"]["mean"]
        )
        report["chunk_latency_reduction_percent"] = 100 * (
            1
            - report["warm_summary"]["compiled"]["total_ms"]["mean"]
            / report["warm_summary"]["eager"]["total_ms"]["mean"]
        )
        save_report()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as profiler:
            with torch.profiler.record_function("lingbot.full_chunk"):
                session.generate(interactions=["forward"])
        profiler.export_chrome_trace(str(out / "profile.json"))
        events = profiler.key_averages(group_by_input_shape=True)
        (out / "profile-cuda.txt").write_text(events.table(sort_by="self_device_time_total", row_limit=25))
        (out / "profile-cpu.txt").write_text(events.table(sort_by="self_cpu_time_total", row_limit=25))
        report["profile_compiled_regions"] = {
            event.key: event.count for event in profiler.key_averages() if "Torch-Compiled Region" in event.key
        }
        assert report["profile_compiled_regions"], "profile did not execute compiled decoder regions"
        session.reset()
        save_report()
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "normalized_comparison",
                    "rgb_comparison",
                    "warm_summary",
                    "decode_speedup",
                    "chunk_latency_reduction_percent",
                    "profile_compiled_regions",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
