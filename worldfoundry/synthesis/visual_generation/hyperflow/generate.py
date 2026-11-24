# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
# Inference entry derived from Video-Rebirth/hyperflow at 1dd2f342.
"""Package worker for HyperFlow's canonical MiniMax-H3 inference."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--memory-reserve-margin", default="24GB")
    parser.add_argument("--no-offload", action="store_true")
    parser.add_argument("--sol-attn", action="store_true")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--last-image", type=Path)
    parser.add_argument("--ref", action="append", default=[])
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.hyperflow.attention import (
        resolve_attention_backend,
    )

    from .media import prepare_media, write_video_audio
    from .pipeline import run_inference
    from .runtime import _reference

    rank = int(os.environ.get("RANK", "0"))
    degree = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if degree != args.gpus:
        parser.error("Multi-GPU workers must be launched through HyperFlowRuntime or package torchrun.")
    device = torch.device(f"cuda:{local_rank}" if degree > 1 else args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("HyperFlow inference requires CUDA.")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    backend = resolve_attention_backend(args.attention_backend, gpus=degree)
    torch.cuda.set_device(device)
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING)
    workflow = "ref2va" if args.ref else "fl2va" if args.image or args.last_image else "t2va"
    media = None
    if rank == 0:
        media = prepare_media(
            image=args.image,
            last_image=args.last_image,
            references=[_reference(value) for value in args.ref],
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )
    if degree > 1:
        dist.init_process_group("nccl", device_id=device)
    try:
        video, audio = run_inference(
            args.model,
            args.weights,
            media,
            workflow=workflow,
            prompt=args.prompt,
            seed=args.seed,
            device=device,
            attention_backend=backend,
            sol_attn=args.sol_attn,
            offload=not args.no_offload,
            memory_reserve_margin=args.memory_reserve_margin,
            rank=rank,
            degree=degree,
        )
        if degree > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if rank == 0:
        write_video_audio(video, audio, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
