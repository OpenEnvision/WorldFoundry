# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Matched 333-frame replays of static text projection and chunk-local RoPE."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tests.manual.lingbot_v2_condition_replay import summarize
from tests.manual.lingbot_v2_vae_replay import PROMPT, compare_rgb

MODES = ("baseline", "text", "rope", "both")
CHANGED_PROMPT = (
    "A dusty abandoned room with wooden floor and a bright window. The camera turns toward the wooden door."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("../ckpts/lingbot-world-v2-1.3b-causal-fast"))
    parser.add_argument("--shared-assets", type=Path, default=Path("../ckpts/Wan-AI--Wan2.1-T2V-1.3B"))
    parser.add_argument("--image", type=Path, default=Path("testcase/012_abandoned-room.png"))
    parser.add_argument("--output", type=Path, default=Path("tmp/lingbot-static-reuse/replay"))
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

    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import lingbot_world_v2 as network
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
    text_factory = session._new_text_condition
    rotary_factory = session._prepare_rotary_frequencies
    assemble = network.assemble_complex_rotary_frequencies_3d
    decode = session._decode
    report: dict[str, Any] = {
        "load_ms": (time.perf_counter() - started) * 1000,
        "settings": {
            "checkpoint": str(args.checkpoint.resolve()),
            "shared_assets": str(args.shared_assets.resolve()),
            "image": str(args.image.resolve()),
            "prompt": PROMPT,
            "changed_prompt": CHANGED_PROMPT,
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
            "condition_prefetch_chunks": 0,
            "condition_replenish_chunks": session.condition_replenish_chunks,
            "i2v_reuse": True,
            "host_cursors": True,
            "camera_reuse": True,
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
        "warm": {mode: [] for mode in MODES},
        "prompt_updates": {},
    }

    def save_report() -> None:
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    @torch.no_grad()
    def rollout(mode: str, seed: int, name: str, *, capture: bool, prompt_updates: bool = False) -> dict[str, Any]:
        frames, metrics = [], []
        boundaries: dict[str, list[torch.Tensor]] = {key: [] for key in ("latents", "conditions", "projected_text")}
        counts: Counter[str] = Counter()
        transition_states = []

        def capture_decode(value: torch.Tensor) -> np.ndarray:
            boundaries["latents"].append(value.detach().cpu().clone())
            return decode(value)

        def model_input(_module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
            if counts["model"] % 5 == 0:
                boundaries["conditions"].append(kwargs["y"][0].detach().cpu().clone())
            counts["model"] += 1

        def text_output(_module: Any, _args: Any, value: torch.Tensor) -> None:
            counts["text"] += 1
            boundaries["projected_text"].append(value.detach().cpu().clone())

        def counted_assemble(*values: Any, **kwargs: Any) -> torch.Tensor:
            counts["rope"] += 1
            return assemble(*values, **kwargs)

        def count(key: str) -> Any:
            def hook(*_args: Any) -> None:
                counts[key] += 1

            return hook

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    session, "_new_text_condition", text_factory if mode in ("text", "both") else lambda _context: None
                )
            )
            stack.enter_context(
                patch.object(
                    session,
                    "_prepare_rotary_frequencies",
                    rotary_factory if mode in ("rope", "both") else lambda *_a, **_k: None,
                )
            )
            started = time.perf_counter()
            session.configure(image=image, prompt=PROMPT, seed=seed, fps=16)
            torch.cuda.synchronize()
            configure_ms = (time.perf_counter() - started) * 1000
            assert session._condition_cache is not None
            assert session._text_condition is None or session._text_condition.nbytes == 0
            assert session._pending_first_condition is not None
            assert session._timesteps is not None and session._sigmas is not None and session._generator is not None
            if capture:
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
                stack.callback(core.model.register_forward_pre_hook(model_input, with_kwargs=True).remove)
                stack.callback(core.model.text_embedding.register_forward_hook(text_output).remove)
                for block in core.model.blocks:
                    for key in ("q", "k", "v"):
                        stack.callback(
                            getattr(block.cross_attn, key).register_forward_hook(count(f"cross_{key}")).remove
                        )
                stack.enter_context(patch.object(network, "assemble_complex_rotary_frequencies_3d", counted_assemble))
                stack.enter_context(patch.object(session, "_decode", capture_decode))
            reset_attention_provider_runtime()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            chunks = 5 if prompt_updates else 21
            for chunk in range(chunks):
                if prompt_updates and chunk in (2, 3):
                    old_text, old_cross = session._text_condition, session._cross_cache
                    video = (session._self_cache, session._vae_state, session._generator, session._condition_cache)
                    rng = session._generator.get_state().clone()
                    positions = [
                        (layer["global_end_index"].clone(), layer["local_end_index"].clone())
                        for layer in session._self_cache or []
                    ]
                    index = session.autoregressive_index
                    assert session.update_prompt(CHANGED_PROMPT if chunk == 2 else PROMPT)
                    assert session._cross_cache is not old_cross
                    if old_text is not None:
                        assert session._text_condition is not old_text and session._text_condition is not None
                        assert session._text_condition.nbytes == 0
                    assert all(
                        a is b
                        for a, b in zip(
                            video,
                            (session._self_cache, session._vae_state, session._generator, session._condition_cache),
                        )
                    )
                    assert torch.equal(rng, session._generator.get_state()) and session.autoregressive_index == index
                    assert all(
                        torch.equal(a, layer["global_end_index"]) and torch.equal(b, layer["local_end_index"])
                        for (a, b), layer in zip(positions, session._self_cache or [])
                    )
                    transition_states.append({"chunk": chunk, "ar_index": index, "video_state_preserved": True})
                result = session.generate(interactions=["forward"])
                assert result is not None
                frames.append(result["video"])
                metrics.append(result["realtime_metrics"])
                print(name, chunk, json.dumps(metrics[-1]), flush=True)
            wall_ms = (time.perf_counter() - started) * 1000
            peak_bytes = torch.cuda.max_memory_allocated()
        pixels = np.concatenate(frames)
        assert pixels.shape == (13 + 16 * (chunks - 1), 464, 832, 3)
        np.save(out / f"{name}.npy", pixels)
        if capture:
            assert counts["model"] == chunks * 5
            assert counts["cross_q"] == chunks * 5 * 30
            assert counts["cross_k"] == counts["cross_v"] == (90 if prompt_updates else 30)
            assert counts["rope"] == (chunks if mode in ("rope", "both") else chunks * 5 * 30 * 2)
            assert counts["text"] == ((3 if prompt_updates else 1) if mode in ("text", "both") else chunks * 5)
            assert len(boundaries["latents"]) == len(boundaries["conditions"]) == chunks
            assert all(torch.isfinite(value).all() for values in boundaries.values() for value in values)
            if not prompt_updates:
                assert all(
                    torch.equal(boundaries["projected_text"][0], value) for value in boundaries["projected_text"]
                )
            torch.save(boundaries, out / f"{name}-boundaries.pt")
        providers = attention_provider_runtime_report()
        assert sum(value["successes"] for value in providers.values()) > 0
        assert sum(value["errors"] + value["fallbacks"] for value in providers.values()) == 0
        if session._text_condition is not None:
            assert session._text_condition.store.condition_hits > 0
            assert not session._text_condition.store._entries
        return {
            "seed": seed,
            "configure_ms": configure_ms,
            "first_chunk_ms": metrics[0]["total_ms"],
            "resident_request_ms": configure_ms + sum(value["total_ms"] for value in metrics),
            "wall_ms_including_prints": wall_ms,
            "chunks": metrics,
            "peak_allocated_bytes": peak_bytes,
            "calls": dict(counts) if capture else None,
            "attention": providers,
            "prompt_transitions": transition_states,
        }

    def compare(left: str, right: str, *, prompt_updates: bool = False) -> dict[str, Any]:
        result = compare_rgb(out / f"{left}.npy", out / f"{right}.npy")
        a = torch.load(out / f"{left}-boundaries.pt", weights_only=True)
        b = torch.load(out / f"{right}-boundaries.pt", weights_only=True)
        for key in ("latents", "conditions"):
            result[f"{key}_exact"] = all(torch.equal(x, y) for x, y in zip(a[key], b[key], strict=True))
            assert result[f"{key}_exact"], (key, left, right)
        if prompt_updates:
            assert all(
                torch.equal(a["projected_text"][index], value)
                for index, value in zip((0, 10, 15), b["projected_text"], strict=True)
            )
            assert torch.equal(b["projected_text"][0], b["projected_text"][2])
            assert not torch.equal(b["projected_text"][0], b["projected_text"][1])
        else:
            assert torch.equal(a["projected_text"][0], b["projected_text"][0])
        result["projected_text_exact"] = True
        assert result["exact"], result
        return result

    with torch.no_grad():
        for mode in MODES:
            name = f"{mode}-instrumented"
            report["instrumented"][mode] = rollout(mode, 42, name, capture=True)
            if mode != "baseline":
                report["comparisons"][mode] = compare("baseline-instrumented", name)
            save_report()
        for repeat, seed in enumerate((42, 43, 42)):
            for mode in reversed(MODES) if repeat == 1 else MODES:
                name = f"{mode}-warm{repeat}"
                report["warm"][mode].append(rollout(mode, seed, name, capture=False))
                save_report()
            for mode in MODES[1:]:
                assert compare_rgb(out / f"baseline-warm{repeat}.npy", out / f"{mode}-warm{repeat}.npy")["exact"]
        for mode in MODES:
            assert compare_rgb(out / f"{mode}-warm0.npy", out / f"{mode}-warm2.npy")["exact"]
            assert compare_rgb(out / f"{mode}-instrumented.npy", out / f"{mode}-warm0.npy")["exact"]
            assert not compare_rgb(out / f"{mode}-warm0.npy", out / f"{mode}-warm1.npy")["exact"]
        report["aba_and_instrumented_warm_exact"] = True
        report["summaries"] = {
            str(start): {mode: summarize(runs, start) for mode, runs in report["warm"].items()} for start in (5, 13)
        }
        report["resident_request_ms"] = {
            mode: {
                "per_run": [run["resident_request_ms"] for run in runs],
                "mean": statistics.mean(run["resident_request_ms"] for run in runs),
            }
            for mode, runs in report["warm"].items()
        }
        for mode in ("baseline", "both"):
            report["prompt_updates"][mode] = rollout(mode, 42, f"{mode}-prompt", capture=True, prompt_updates=True)
            save_report()
        report["comparisons"]["prompt_updates"] = compare("baseline-prompt", "both-prompt", prompt_updates=True)
        unchanged = np.load(out / "baseline-instrumented.npy", mmap_mode="r")
        changed = np.load(out / "both-prompt.npy", mmap_mode="r")
        assert np.array_equal(unchanged[:29], changed[:29]) and not np.array_equal(unchanged[29:77], changed[29:])
        session.reset()
        assert session._text_condition is None
        save_report()
    for mode in ("baseline", "both"):
        pixels = np.load(out / f"{mode}-instrumented.npy")
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
