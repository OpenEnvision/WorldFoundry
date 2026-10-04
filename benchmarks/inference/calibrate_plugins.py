"""Explicit offline calibration of checkpoint-bound native acceleration plugins."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path

import torch

from benchmarks.inference.native_plugins import _assets
from benchmarks.inference.plugin_diagnostics import file_metadata
from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import load_wan_video_codec
from worldfoundry.base_models.diffusion_model.optimizations.lightvae_fp8 import lightvae_encoder_convolutions
from worldfoundry.base_models.diffusion_model.optimizations.projection_selection import (
    ProjectionSelection,
    select_native_projections,
)
from worldfoundry.base_models.diffusion_model.pipeline import NativeDiffusionPipeline
from worldfoundry.core.acceleration.quantization.calibration import ChannelObserver, save_calibration
from worldfoundry.core.acceleration.quantization.fp8_conv import calibrate_fp8_convolution
from worldfoundry.core.acceleration.quantization.svdquant import calibrate_svdquant
from worldfoundry.core.model_loading.policy import RuntimePolicy
from worldfoundry.runtime.performance import capture_runtime_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["svdquant", "lightvae_fp8"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path, nargs="+", default=[])
    parser.add_argument("--model", default="wan2.1-t2v-1.3b", choices=["wan2.1-t2v-1.3b", "sana-video-2b-480p"])
    parser.add_argument("--include", nargs="+", required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--save-references", type=Path, help="new directory for calibration RGB/latents (seeds 11/12)")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("calibration output must be a new path")
    if args.save_references is not None:
        if args.kind != "svdquant":
            parser.error("--save-references is only supported for generated SVDQuant calibration")
        args.save_references.mkdir(parents=True, exist_ok=False)
    if min(args.height, args.width, args.frames, args.steps) <= 0 or args.frames < 5:
        parser.error("positive geometry and at least five frames required")
    policy = RuntimePolicy(device="cuda", dtype=torch.bfloat16, options={"dit_weight_dtype": "bf16"})
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": args.kind,
        "runtime": capture_runtime_fingerprint(device_index=0).to_dict(),
        "geometry": [args.frames, args.height, args.width],
    }
    if args.kind == "svdquant":
        if args.assets is None:
            parser.error("SVDQuant calibration requires --assets")
        overrides, weights = _assets(args.model, args.assets.resolve())
        inputs = file_metadata(weights, args.assets.resolve())
        pipeline = NativeDiffusionPipeline.from_pretrained(args.model, policy=policy, checkpoint_overrides=overrides)
        model = pipeline.components.denoiser.model
        _, selected = select_native_projections(model, ProjectionSelection(args.include, min_features=16), policy)
        observer = ChannelObserver(dict(selected))
        prompts = [
            "A sailing boat moves across a calm lake at sunset.",
            "People walk through a city street in the rain.",
        ]
        with observer, torch.inference_mode():
            for seed, prompt in zip((11, 12), prompts):
                output = pipeline(
                    DiffusionRequest(
                        prompt=prompt,
                        negative_prompt="",
                        height=args.height,
                        width=args.width,
                        num_frames=args.frames,
                        sampling=SamplingConfig(
                            num_inference_steps=args.steps,
                            guidance_scale=6.0,
                            seed=seed,
                            scheduler_options={"shift": 7},
                        ),
                    )
                )
                if args.save_references is not None:
                    torch.save(
                        {"sample": output.sample.cpu(), "latents": output.latents.cpu()},
                        args.save_references / f"calibration_{seed}.pt",
                    )
        observer.validate()
        states = {name: calibrate_svdquant(module, observer.maxima[name], rank=args.rank) for name, module in selected}
        metadata.update(model=args.model, seeds=[11, 12], prompts=prompts, rank=args.rank)
        if inputs != file_metadata(weights, args.assets.resolve()):
            raise RuntimeError("checkpoint files changed during calibration")
    else:
        if args.checkpoint is None or not args.reference:
            parser.error("codec calibration requires --checkpoint and --reference")
        paths = [args.checkpoint, *args.reference]
        inputs = file_metadata(paths, Path("/"))
        codec = load_wan_video_codec(args.checkpoint, variant="lightvae-wan21")
        available = lightvae_encoder_convolutions(codec.vae)
        selected = {
            name: module
            for name, module in available.items()
            if any(fnmatchcase(name, pattern) for pattern in args.include)
        }
        if any(not any(fnmatchcase(name, pattern) for name in selected) for pattern in args.include):
            raise ValueError("each include pattern must match encoder convolutions")
        observer = ChannelObserver(selected, channel_dim=1)
        with observer, torch.inference_mode():
            for path in args.reference:
                pixels = torch.load(path, map_location="cpu", weights_only=True)["sample"]
                if pixels.ndim != 5 or pixels.shape[1] != 3 or min(pixels.shape[2:]) <= 0:
                    raise ValueError("calibration reference must contain BCTHW RGB")
                codec.encode(pixels[:, :, : args.frames, : args.height, : args.width].to("cuda", torch.float32))
        observer.validate()
        states = {name: calibrate_fp8_convolution(module, observer.maxima[name]) for name, module in selected.items()}
        if inputs != file_metadata(paths, Path("/")):
            raise RuntimeError("codec calibration input files changed")
    metadata.update(inputs=inputs, inputs_unchanged=True, observed_calls=observer.calls)
    save_calibration(args.out, kind=args.kind, states=states, metadata=metadata)
    print(f"Saved {len(states)} checkpoint-bound {args.kind} module states to {args.out}", flush=True)


if __name__ == "__main__":
    main()
