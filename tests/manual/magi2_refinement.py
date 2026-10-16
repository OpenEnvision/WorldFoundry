# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Replay the native MAGI-2 Refiner on a saved checkpoint-generated Preview latent."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--latent", required=True, type=Path)
    parser.add_argument("--audio", type=Path, help="Optional decoded Preview audio tensor at 44.1 kHz")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import torch

    from worldfoundry.base_models.diffusion_model.models.encoders.magi2_qwen35 import (
        Magi2Qwen35SubprocessTextEncoder,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.magi2.sampler import DEFAULT_NEGATIVE_PROMPT
    from worldfoundry.core.io import write_json
    from worldfoundry.pipelines.magi2 import NativeMagi2Pipeline

    pipeline = NativeMagi2Pipeline.from_pretrained(str(args.checkpoint_root), device="cuda")
    pipeline.text_encoder = Magi2Qwen35SubprocessTextEncoder(
        str(args.checkpoint_root / "text_encoder"), device="cuda", skip_layer=2
    )
    context = pipeline._encode_prompt(args.prompt)
    null_context = pipeline._encode_prompt(DEFAULT_NEGATIVE_PROMPT)
    torch.save({"context": context.cpu(), "null_context": null_context.cpu()}, args.output.with_suffix(".context.pt"))
    pipeline._ensure_components(load_preview=False, load_decoders=False, load_refiner=True)
    latent = torch.load(args.latent, map_location="cuda", weights_only=True)
    if not isinstance(latent, torch.Tensor) or latent.ndim != 5 or latent.shape[:2] != (1, 48):
        raise ValueError("Expected a saved Preview latent with shape [1,48,T,H,W]")
    refined = pipeline._run_refiner(
        latent,
        context,
        null_context,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    )
    if not torch.isfinite(refined).all():
        raise ValueError("Refiner produced nonfinite latents")
    pipeline._ensure_components(load_preview=False, load_decoders=True)
    video = pipeline.turbo_decoder.decode(refined)
    if not torch.isfinite(video).all():
        raise ValueError("Refiner decode produced nonfinite pixels")
    audio = torch.load(args.audio, map_location="cpu", weights_only=True) if args.audio is not None else None
    pipeline._write_mp4_with_audio(video=video, audio=audio, output_path=args.output, fps=25.0)
    latent_path = args.output.with_suffix(".latents.pt")
    torch.save(refined.cpu(), latent_path)
    report = {
        "scope": "checkpoint_backed_refiner_stage",
        "source_latent": str(args.latent),
        "latent_shape": list(refined.shape),
        "video_shape": list(video.shape),
        "fps": 25.0,
        "steps": args.steps,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "audio_preserved": args.audio is not None,
        "output": str(args.output),
        "latent_output": str(latent_path),
    }
    write_json(args.output.with_suffix(".json"), report)
    print(report)


if __name__ == "__main__":
    main()
