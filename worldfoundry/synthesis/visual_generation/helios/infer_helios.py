"""Inference-only Helios entry point vendored for WorldFoundry."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "8"

import torch
import torch.distributed as dist


def _parse_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean switch, got {value!r}.")


def _add_bool_argument(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(name, nargs="?", const=True, default=False, type=_parse_bool)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a video with Helios")
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--transformer_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_variant", choices=["base", "mid", "distilled"], default="distilled")
    parser.add_argument("--sample_type", choices=["auto", "t2v", "i2v", "v2v"], default="auto")
    parser.add_argument("--weight_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", default="auto")
    parser.add_argument("--zero_steps", type=int, default=1)
    parser.add_argument("--num_latent_frames_per_chunk", type=int, default=9)
    parser.add_argument("--pyramid_num_inference_steps_list", nargs="+", default=None,
                        help="Three positive step counts, or 'auto' for the model variant defaults.")
    parser.add_argument("--image_path")
    parser.add_argument("--video_path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--image_noise_sigma_min", type=float, default=0.111)
    parser.add_argument("--image_noise_sigma_max", type=float, default=0.135)
    parser.add_argument("--video_noise_sigma_min", type=float, default=0.111)
    parser.add_argument("--video_noise_sigma_max", type=float, default=0.135)
    parser.add_argument(
        "--cp_backend",
        choices=["ring", "ulysses", "unified", "ulysses_anything"],
        default="ulysses",
    )
    parser.add_argument(
        "--group_offloading_type",
        choices=["leaf_level", "block_level"],
        default="leaf_level",
    )
    parser.add_argument("--num_blocks_per_group", type=int, default=4)
    for name in (
        "--enable_compile",
        "--use_zero_init",
        "--is_enable_stage2",
        "--is_skip_first_chunk",
        "--is_amplify_first_chunk",
        "--enable_parallelism",
        "--enable_low_vram_mode",
        "--disable_flash_attention",
    ):
        _add_bool_argument(parser, name)
    return parser.parse_args()


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.image_path = (args.image_path or "").strip() or None
    args.video_path = (args.video_path or "").strip() or None
    if args.sample_type == "auto":
        args.sample_type = "v2v" if args.video_path else "i2v" if args.image_path else "t2v"
    if args.sample_type == "i2v" and not args.image_path:
        raise ValueError("Helios i2v inference requires --image_path.")
    if args.sample_type == "v2v" and not args.video_path:
        raise ValueError("Helios v2v inference requires --video_path.")
    if args.enable_compile and args.enable_low_vram_mode:
        raise ValueError("--enable_compile and --enable_low_vram_mode are mutually exclusive.")
    if min(args.height, args.width, args.num_frames, args.fps, args.num_inference_steps) <= 0:
        raise ValueError("Dimensions, frame count, FPS, and inference steps must be positive.")

    if args.guidance_scale == "auto":
        args.guidance_scale = 1.0 if args.model_variant == "distilled" else 5.0
    else:
        args.guidance_scale = float(args.guidance_scale)
    if args.pyramid_num_inference_steps_list is None or args.pyramid_num_inference_steps_list == ["auto"]:
        args.pyramid_num_inference_steps_list = [2, 2, 2] if args.model_variant == "distilled" else [20, 20, 20]
    else:
        args.pyramid_num_inference_steps_list = [int(value) for value in args.pyramid_num_inference_steps_list]
        if len(args.pyramid_num_inference_steps_list) != 3 or min(args.pyramid_num_inference_steps_list) <= 0:
            raise ValueError("pyramid_num_inference_steps_list requires three positive integers or 'auto'.")
    if args.model_variant == "distilled":
        args.is_enable_stage2 = True
        args.is_amplify_first_chunk = True
    elif args.model_variant == "mid":
        args.is_enable_stage2 = True
        args.use_zero_init = True
    return args


def _distributed_device(args: argparse.Namespace) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("Helios inference requires CUDA.")
    if dist.is_available() and "RANK" in os.environ:
        backend = "cpu:gloo,cuda:nccl" if args.cp_backend == "ulysses_anything" else "nccl"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("cuda", rank % torch.cuda.device_count())
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda")
    if world_size > 1:
        args.enable_parallelism = True
    if world_size > 1 and args.enable_low_vram_mode:
        raise ValueError("Low-VRAM group offloading only supports one GPU.")
    return rank, world_size, device


def _enable_context_parallelism(pipe: object, backend: str, world_size: int) -> None:
    from diffusers import ContextParallelConfig

    if backend == "ring":
        config = ContextParallelConfig(ring_degree=world_size)
    elif backend == "unified":
        if world_size % 2:
            raise ValueError("The unified context-parallel backend requires an even GPU count.")
        config = ContextParallelConfig(ring_degree=2, ulysses_degree=world_size // 2)
    elif backend == "ulysses":
        config = ContextParallelConfig(ulysses_degree=world_size)
    else:
        config = ContextParallelConfig(ulysses_degree=world_size, ulysses_anything=True)
    pipe.transformer.enable_parallelism(config=config)


def _run(args: argparse.Namespace) -> None:
    from diffusers.models import AutoencoderKLWan
    from diffusers.utils import export_to_video, load_image, load_video

    from .kernels import (
        replace_all_norms_with_flash_norms,
        replace_rmsnorm_with_fp32,
        replace_rope_with_flash_rope,
    )
    from .pipeline_helios_diffusers import HeliosPipeline
    from .scheduling_helios_diffusers import HeliosScheduler
    from .transformer_helios_diffusers import HeliosTransformer3DModel

    rank, world_size, device = _distributed_device(args)
    weight_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.weight_dtype]

    transformer = HeliosTransformer3DModel.from_pretrained(
        args.transformer_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )
    # The vendored Triton norm/RoPE replacements are part of the same fused
    # kernel path as flash attention.  Applying them while
    # --disable_flash_attention is set silently changes the supposedly pure
    # PyTorch reference path and has produced dark, nearly constant A100
    # outputs.  Keep the entire fused-kernel stack behind the one opt-in.
    if not args.enable_compile and not args.disable_flash_attention:
        transformer = replace_rmsnorm_with_fp32(transformer)
        transformer = replace_all_norms_with_flash_norms(transformer)
        replace_rope_with_flash_rope()
    if not args.disable_flash_attention:
        backend = "_flash_3_hub" if torch.cuda.get_device_capability()[0] >= 9 else "flash_hub"
        try:
            transformer.set_attention_backend(backend)
        except Exception:
            if backend != "_flash_3_hub":
                raise
            transformer.set_attention_backend("flash_hub")

    vae = AutoencoderKLWan.from_pretrained(
        args.base_model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    scheduler = HeliosScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    pipe = HeliosPipeline.from_pretrained(
        args.base_model_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=weight_dtype,
    )
    if args.enable_compile:
        torch.backends.cudnn.benchmark = True
        pipe.text_encoder.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.vae.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
    if args.enable_low_vram_mode:
        pipe.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type=args.group_offloading_type,
            num_blocks_per_group=(args.num_blocks_per_group if args.group_offloading_type == "block_level" else None),
            use_stream=True,
            record_stream=True,
        )
    else:
        pipe = pipe.to(device)
    if world_size > 1:
        _enable_context_parallelism(pipe, args.cp_backend, world_size)

    image = load_image(args.image_path).resize((args.width, args.height)) if args.image_path else None
    video = load_video(args.video_path) if args.video_path else None
    with torch.no_grad():
        frames = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            history_sizes=[16, 2, 1],
            num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
            keep_first_frame=True,
            is_enable_stage2=args.is_enable_stage2,
            pyramid_num_inference_steps_list=args.pyramid_num_inference_steps_list,
            is_skip_first_chunk=args.is_skip_first_chunk,
            is_amplify_first_chunk=args.is_amplify_first_chunk,
            use_zero_init=args.use_zero_init,
            zero_steps=args.zero_steps,
            image=image,
            image_noise_sigma_min=args.image_noise_sigma_min,
            image_noise_sigma_max=args.image_noise_sigma_max,
            video=video,
            video_noise_sigma_min=args.video_noise_sigma_min,
            video_noise_sigma_max=args.video_noise_sigma_max,
            use_interpolate_prompt=False,
            interpolation_steps=3,
            interpolate_time_list=None,
        ).frames[0]

    if rank == 0:
        output_path = Path(args.output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(output_path), fps=args.fps)
        print(f"Saved Helios video to {output_path}")
    if torch.cuda.is_available():
        print(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")


def main() -> None:
    args = _normalize_args(parse_args())
    try:
        _run(args)
    finally:
        # torchrun sets RANK even for a single worker, so _distributed_device()
        # initializes a process group for the common one-GPU Studio path too.
        # Always release it, including when model loading or generation fails.
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
