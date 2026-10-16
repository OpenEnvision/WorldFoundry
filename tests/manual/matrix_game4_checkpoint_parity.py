#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare the public Egypt renderer with pinned upstream on local checkpoints.

Requires CUDA and an explicit RGB input video. Native request reuse and failure
recovery execute on resident weights. Reference imports are validation-only;
this short-case comparison is separate from accepted-reference release replay.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from fractions import Fraction
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REVISION = "5446525c80da2a4a2f87801addbbc7faa8d2486e"
sys.path.insert(0, str(ROOT))


def snapshot(result: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Own finite CPU copies of the generated latent and signed RGB tensors."""
    tensors = {key: result[key].detach().cpu().clone() for key in ("latents", "video")}
    for value in tensors.values():
        assert torch.isfinite(value).all()
    return tensors


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """Require exact values while retaining each public boundary's dtype."""
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
    return {
        "shape": list(actual.shape),
        "native_dtype": str(actual.dtype),
        "reference_dtype": str(expected.dtype),
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
    }


def native_case(
    args: argparse.Namespace, pixels: np.ndarray, fps: float
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load publicly, export A, exercise B/failure, then require exact A reuse."""
    from worldfoundry.pipelines.matrix_game.pipeline_matrix_game4 import MatrixGame4EgyptPipeline

    pipeline = cast(
        MatrixGame4EgyptPipeline,
        MatrixGame4EgyptPipeline.from_pretrained(
            device="cuda",
            torch_dtype="bfloat16",
            attention_backend="sdpa",
            offload_mode="none",
            checkpoint_overrides={
                "dit": args.egypt_root / "fps-egypt/model.safetensors",
                "sink": args.egypt_root / "fps-egypt/sink.png",
                "text-encoder": args.base_root / "models_t5_umt5-xxl-enc-bf16.pth",
                "tokenizer": args.base_root,
                "vae": args.base_root / "Wan2.2_VAE.pth",
            },
        ),
    )
    cpu_rng, cuda_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    print("Native A: public file input and MP4 export", flush=True)
    first_result = pipeline(
        input_path=args.input,
        output_path=args.output_dir / "native.mp4",
        seed=args.seed,
        return_dict=True,
    )
    first = snapshot(first_result)
    metadata = dict(first_result["metadata"])
    del first_result
    print("Native B: changed control input", flush=True)
    changed_result = pipeline(video=255 - pixels, fps=fps, seed=args.seed, return_dict=True)
    changed_latents = changed_result["latents"].detach().cpu().clone()
    assert not torch.equal(first["latents"], changed_latents)
    del changed_result, changed_latents

    class InterruptedRender(RuntimeError):
        """Expected interruption after partial request-local cache writes."""

    calls = 0

    def interrupt(_module: torch.nn.Module, _inputs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise InterruptedRender("Interrupted denoising")

    denoiser = pipeline.native_pipeline.components.denoiser
    handle = denoiser.model.head.register_forward_pre_hook(interrupt)
    print("Native failed request: interrupt after partial KV writes", flush=True)
    try:
        try:
            pipeline(input_path=args.input, seed=args.seed, return_dict=True)
        except InterruptedRender:
            pass
        else:
            raise AssertionError("Expected the injected denoising interruption")
    finally:
        handle.remove()
    print("Native A: exact reuse after changed input and failed request", flush=True)
    repeated_result = pipeline(input_path=args.input, seed=args.seed, return_dict=True)
    repeated = snapshot(repeated_result)
    for key in first:
        compare(repeated[key], first[key])
    torch.testing.assert_close(torch.random.get_rng_state(), cpu_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
    native_peak = torch.cuda.max_memory_allocated() / 1024**3
    del repeated_result, repeated, pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return first, {
        "metadata": metadata,
        "request_reuse": "exact",
        "changed_control": "different_latents",
        "failure_recovery": "exact",
        "global_rng": "unchanged",
        "peak_allocated_gib": native_peak,
    }


def reference_case(args: argparse.Namespace, pixels: np.ndarray) -> dict[str, torch.Tensor]:
    """Execute the pinned quality preset with shared decoded RGB input."""
    sys.path.insert(0, str(args.reference_root.resolve()))
    attention: Any = import_module("matrix_game.inference.wan.modules.attention")
    attention.FLASH_ATTN_2_AVAILABLE = False
    attention.FLASH_ATTN_3_AVAILABLE = False
    scenes = import_module("matrix_game.scenes")
    parser = argparse.ArgumentParser()
    scenes.add_model_arguments(parser)
    options = scenes.resolve_model_arguments(
        parser.parse_args(
            [
                "--scene",
                "ancient-egypt",
                "--preset",
                "quality",
                "--dit-runtime",
                "native",
                "--t5-runtime",
                "native",
                "--model-dir",
                str(args.base_root),
                "--checkpoint",
                str(args.egypt_root / "fps-egypt/model.safetensors"),
                "--sink-image",
                str(args.egypt_root / "fps-egypt/sink.png"),
            ]
        )
    )
    components = import_module("matrix_game.inference.loader").ModelComponents(options)
    components.load()
    torch.set_num_threads(args.threads)
    preprocessing = import_module("matrix_game.inference.preprocessing")
    frame_blocks = import_module("matrix_game.inference.offline").frame_blocks
    stream = components.stream(seed=args.seed)
    chunks, latents = [], []
    written = 0
    try:
        for raw, valid in frame_blocks(iter(pixels)):
            frames = [preprocessing.crop_and_resize(Image.fromarray(value), 480, 864) for value in raw]
            result = stream.push(preprocessing.frames_to_tensor(frames), frame_start=written)
            chunks.append(result.pixels[:valid].clone())
            latents.append(result.latent.detach().cpu().clone())
            written += valid
    finally:
        stream.close()
    return {
        "latents": torch.cat(latents, dim=1).permute(0, 2, 1, 3, 4).contiguous(),
        "video": torch.cat(chunks, dim=0).permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
    }


def main() -> None:
    """Run an explicit checkpoint case without changing production imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--egypt-root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tmp/matrix-game4-checkpoint-parity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative for deterministic request-reuse comparisons")
    revision = subprocess.check_output(["git", "-C", str(args.reference_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Reference must be pinned at {REVISION}, got {revision}")
    if not torch.cuda.is_available():
        raise RuntimeError("Matrix-Game 4 checkpoint parity requires CUDA")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from worldfoundry.core.media.codecs.video import read_video

    pixels, source_metadata = read_video(args.input)
    fps = float(source_metadata["fps"])
    actual, lifecycle = native_case(args, pixels, fps)
    torch.save(actual, args.output_dir / "native.pt")
    print("Pinned upstream quality preset", flush=True)
    expected = reference_case(args, pixels)
    torch.save(expected, args.output_dir / "upstream.pt")
    metrics = {key: compare(actual[key], expected[key]) for key in actual}
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "stream=codec_type,width,height,nb_read_frames,r_frame_rate",
                "-of",
                "json",
                str(args.output_dir / "native.mp4"),
            ],
            text=True,
        )
    )["streams"][0]
    assert probe["codec_type"] == "video"
    assert int(probe["nb_read_frames"]) == len(pixels)
    assert (probe["width"], probe["height"]) == (864, 480)
    assert abs(float(Fraction(probe["r_frame_rate"])) - fps) < 0.001
    report = {
        "evidence_level": "public-loader-released-checkpoint-upstream-comparison",
        "reference_revision": revision,
        "input": str(args.input),
        "input_frames": len(pixels),
        "input_decoder": "shared-worldfoundry-rgb-reader",
        "seed": args.seed,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "dtype": "bfloat16",
        "attention": "sdpa",
        "offload": "none",
        "metrics": metrics,
        "lifecycle": lifecycle,
        "ffprobe": probe,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
