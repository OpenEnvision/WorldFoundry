#!/usr/bin/env python3
"""Precompute complete Echo R2V conditioning with independent torchrun workers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from ..inference import DEFAULT_CONFIG, InferenceConfig, load_request_files, load_requests
from ..ltx_distillation.r2v_conditioning import (
    encode_r2v_requests,
    load_r2v_conditioning,
    r2v_conditioning_cache_path,
    save_r2v_conditioning,
)
from ..ltx_distillation.release_checkpoint import resolve_release_checkpoint
from ..ltx_distillation.text_conditioning import artifact_fingerprint
from ..r2v_schema import R2VRequest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute complete Echo 1.5 R2V conditions")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--request")
    parser.add_argument("--checkpoint")
    parser.add_argument("--gemma-path")
    parser.add_argument("--requests-dir")
    parser.add_argument("--requests-glob")
    parser.add_argument("--output-dir", "--conditioning-cache-dir", dest="output_dir")
    parser.add_argument("--text-batch-size", type=int)
    parser.add_argument("--image-batch-size", type=int)
    parser.add_argument("--audio-batch-size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    overrides = {
        key: value
        for key, value in {
            "checkpoint": args.checkpoint,
            "gemma_path": args.gemma_path,
            "requests_dir": args.requests_dir,
            "requests_glob": args.requests_glob,
            "conditioning_cache_dir": args.output_dir,
            "text_batch_size": args.text_batch_size,
            "image_batch_size": args.image_batch_size,
            "audio_batch_size": args.audio_batch_size,
        }.items()
        if value is not None
    }
    for path_key in ("checkpoint", "gemma_path", "requests_dir", "conditioning_cache_dir"):
        if path_key in overrides:
            overrides[path_key] = str(Path(overrides[path_key]).expanduser().resolve())
    config = InferenceConfig(Path(args.config).expanduser().resolve(), **overrides)
    if not config.conditioning_cache_dir:
        raise ValueError(
            "paths.conditioning_cache_dir is required for --condition-encode "
            "(or override it with --conditioning-cache-dir)"
        )
    request_files = load_request_files(config, args.request)
    requests = load_requests(config, request_files)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R2V conditioning precomputation")
    if local_rank >= torch.cuda.device_count():
        raise ValueError(f"LOCAL_RANK={local_rank} is outside {torch.cuda.device_count()} visible CUDA devices")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    release = resolve_release_checkpoint(config.checkpoint)
    checkpoint_id = artifact_fingerprint(release.root)
    gemma_id = artifact_fingerprint(config.gemma_path)
    assigned = list(zip(request_files, requests, strict=True))[rank::world_size]
    pending: list[tuple[Path, R2VRequest]] = []
    for request_file, request in assigned:
        cache_path = r2v_conditioning_cache_path(config.conditioning_cache_dir, config.requests_dir, request_file)
        if cache_path.is_file() and not args.overwrite:
            try:
                load_r2v_conditioning(
                    cache_path,
                    request=request,
                    checkpoint_fingerprint=checkpoint_id,
                    gemma_fingerprint=gemma_id,
                )
                print(f"[rank {rank}] cached {cache_path}", flush=True)
                continue
            except ValueError:
                pass
        pending.append((request_file, request))
    if not pending:
        print(f"[rank {rank}] nothing to encode", flush=True)
        return

    print(
        f"[rank {rank}/{world_size}] encoding {len(pending)} of {len(assigned)} requests on {device}",
        flush=True,
    )
    bundles = encode_r2v_requests(
        [request for _, request in pending],
        checkpoint_path=str(release.model_path),
        gemma_path=str(config.gemma_path),
        device=device,
        voice_filter_config=config.voice_filter,
        dtype=torch.bfloat16,
        text_batch_size=config.text_batch_size,
        image_batch_size=config.image_batch_size,
        audio_batch_size=config.audio_batch_size,
        enable_audio_memory=config.enable_audio_memory,
        memory_position_mode=config.memory_position_mode,
        memory_position_offset=config.memory_position_offset,
        memory_position_slot_stride=config.memory_position_slot_stride,
    )
    for (request_file, request), bundle in zip(pending, bundles, strict=True):
        cache_path = r2v_conditioning_cache_path(config.conditioning_cache_dir, config.requests_dir, request_file)
        save_r2v_conditioning(
            cache_path,
            bundle,
            request=request,
            checkpoint_fingerprint=checkpoint_id,
            gemma_fingerprint=gemma_id,
        )
        print(f"[rank {rank}] wrote {cache_path}", flush=True)
    print(f"[rank {rank}] complete", flush=True)


if __name__ == "__main__":
    main()
