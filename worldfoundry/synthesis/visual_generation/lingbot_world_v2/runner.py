"""Distributed CLI entry point for LingBot-World-V2 inference."""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys

import torch
import torch.distributed as dist
from PIL import Image

from worldfoundry.base_models.diffusion_model.recipes.wan_configs.lingbot_world_v2 import (
    LINGBOT_WORLD_V2_CONFIG,
    SUPPORTED_SIZES,
)
from worldfoundry.core.media.codecs.video import save_image_or_video_tensor

from .inference import LingBotWorldV2Inference


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WorldFoundry LingBot-World-V2 causal-fast runner")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--action-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", choices=sorted(SUPPORTED_SIZES), default="480*832")
    parser.add_argument("--frame-num", type=int, default=361)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-shift", type=float, default=10.0)
    parser.add_argument("--local-attn-size", type=int, default=18)
    parser.add_argument("--sink-size", type=int, default=6)
    parser.add_argument("--max-attention-size", type=int)
    parser.add_argument("--t5-fsdp", action="store_true")
    parser.add_argument("--dit-fsdp", action="store_true")
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--offload-model", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--convert-model-dtype", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        stream=sys.stdout,
    )
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", init_method="env://")
    elif args.t5_fsdp or args.dit_fsdp:
        raise ValueError("FSDP requires a multi-process launch.")
    if LINGBOT_WORLD_V2_CONFIG.num_heads % world_size:
        raise ValueError(f"num_heads={LINGBOT_WORLD_V2_CONFIG.num_heads} is not divisible by world_size={world_size}.")
    if args.frame_num < 5 or (args.frame_num - 1) % 4:
        raise ValueError("frame_num must be at least 5 and follow the 4n+1 layout.")
    latent_frames = (args.frame_num - 1) // 4 + 1
    if args.chunk_size < 1 or latent_frames < args.chunk_size:
        raise ValueError("chunk_size must be positive and fit within the latent frame count.")
    if args.local_attn_size != -1 and args.local_attn_size < args.chunk_size:
        raise ValueError("local_attn_size must be -1 or at least chunk_size.")
    if args.sink_size < 0 or (args.local_attn_size != -1 and args.sink_size + args.chunk_size > args.local_attn_size):
        raise ValueError("sink_size must leave room for one chunk in the local attention window.")
    if args.t5_cpu and args.t5_fsdp:
        raise ValueError("t5_cpu and t5_fsdp are mutually exclusive.")
    if args.max_attention_size is not None and args.max_attention_size < 1:
        raise ValueError("max_attention_size must be positive when provided.")

    seed_payload = [args.seed if args.seed >= 0 else random.randint(0, sys.maxsize)]
    if dist.is_initialized():
        dist.broadcast_object_list(seed_payload, src=0)
    offload_model = args.offload_model if args.offload_model is not None else world_size == 1
    if offload_model and args.dit_fsdp:
        raise ValueError("offload_model is not supported together with DiT FSDP.")
    logging.info("Loading checkpoint: %s", args.checkpoint_dir)
    pipeline = LingBotWorldV2Inference(
        args.checkpoint_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=world_size > 1,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
    )
    video = pipeline.generate(
        args.prompt,
        Image.open(args.image).convert("RGB"),
        args.action_path,
        chunk_size=args.chunk_size,
        max_area=math.prod(int(part) for part in args.size.split("*")),
        frame_num=args.frame_num,
        shift=args.sample_shift,
        seed=seed_payload[0],
        offload_model=offload_model,
        max_attention_size=args.max_attention_size,
    )
    if rank == 0:
        save_image_or_video_tensor(
            video,
            args.output,
            fps=LINGBOT_WORLD_V2_CONFIG.sample_fps,
            value_range=(-1.0, 1.0),
        )
        logging.info("Saved video: %s", args.output)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
