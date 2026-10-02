"""Inference-only in-tree runner for TencentARC RollingForcing.

Adapted from upstream ``inference.py`` at revision
``a1477d09e85dc759a6a6728f55f77f59342ce388``.  WorldFoundry removes the
training and LMDB data paths, reuses its shared forcing/Wan components, and
uses the common video writer.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    import torch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the in-tree RollingForcing inference path.")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--default_config_path", default=None)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument(
        "--wan_models_root",
        default=None,
        help="Parent directory containing Wan2.1-T2V-1.3B (no cwd-relative symlink required).",
    )
    parser.add_argument("--data_path", required=True, help="UTF-8 file containing one prompt per line.")
    parser.add_argument("--extended_prompt_path", default=None)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--num_output_frames", type=int, default=126)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--save_with_index", action="store_true")
    parser.add_argument("--report_timing", action="store_true")
    # Kept only so a generic forcing command receives a precise error rather
    # than an argparse failure.  The released RollingForcing route is T2V.
    parser.add_argument("--i2v", action="store_true")
    return parser


def _read_prompts(path: str | Path) -> list[str]:
    prompts = [line.strip() for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines()]
    prompts = [prompt for prompt in prompts if prompt]
    if not prompts:
        raise ValueError(f"No non-empty prompts found in {path}")
    return prompts


def _extended_prompts(path: str | Path | None, count: int) -> list[str | None]:
    if path is None:
        return [None] * count
    values = _read_prompts(path)
    if len(values) != count:
        raise ValueError(
            f"Extended prompt count ({len(values)}) must match prompt count ({count})."
        )
    return values


def _generator_state(checkpoint: object, *, use_ema: bool) -> OrderedDict[str, torch.Tensor]:
    from worldfoundry.core.model_loading.checkpoints import tensor_state_dict

    if not isinstance(checkpoint, Mapping):
        raise TypeError("RollingForcing checkpoint must be a mapping.")
    key = "generator_ema" if use_ema else "generator"
    state = checkpoint.get(key)
    if not isinstance(state, Mapping):
        available = ", ".join(sorted(str(item) for item in checkpoint))
        raise KeyError(f"Checkpoint has no {key!r} state (available: {available}).")
    state = tensor_state_dict(
        state,
        source=f"RollingForcing {key}",
        wrapper_keys=(),
    )
    normalized: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in state.items():
        normalized[str(name).replace("_fsdp_wrapped_module.", "")] = value
    return normalized


def _safe_stem(prompt: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", prompt).strip("._-")
    return (stem or "prompt")[:80]


def _distributed_device(seed: int) -> tuple[torch.device, int, int]:
    import torch
    import torch.distributed as dist

    from worldfoundry.core.utils.tensors.torch import set_seed_everywhere

    if "LOCAL_RANK" not in os.environ:
        set_seed_everywhere(seed)
        return torch.device("cuda"), 0, 1
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    set_seed_everywhere(seed + local_rank)
    return torch.device(f"cuda:{local_rank}"), dist.get_rank(), dist.get_world_size()


def main() -> None:
    args = _parser().parse_args()
    if args.wan_models_root:
        os.environ["WORLDFOUNDRY_WAN_MODELS_ROOT"] = str(
            Path(args.wan_models_root).expanduser().resolve()
        )
    import torch
    import torch.distributed as dist
    from einops import rearrange
    from omegaconf import OmegaConf

    from worldfoundry.core.model_loading.checkpoints import load_weights_only
    from worldfoundry.core.media.codecs.video import write_video_torchvision

    # The causal Wan variant is placed on PYTHONPATH by RollingForcingRuntime.
    # Delaying this import keeps module discovery and ``--help`` lightweight.
    from worldfoundry.synthesis.visual_generation.forcing.causal_forcing_runtime.long_video.pipeline.rolling_forcing_inference import (
        CausalInferencePipeline,
    )

    if args.i2v:
        raise ValueError(
            "The public RollingForcing checkpoint and official runner support text-to-video only."
        )
    if args.num_samples < 1:
        raise ValueError("num_samples must be positive.")

    device, rank, world_size = _distributed_device(args.seed)
    torch.set_grad_enabled(False)

    config_path = Path(args.config_path).expanduser().resolve()
    default_path = (
        Path(args.default_config_path).expanduser().resolve()
        if args.default_config_path
        else config_path.with_name("default_config.yaml")
    )
    config = OmegaConf.merge(OmegaConf.load(default_path), OmegaConf.load(config_path))
    block_size = int(config.num_frame_per_block)
    if args.num_output_frames < block_size or args.num_output_frames % block_size:
        raise ValueError(
            f"num_output_frames must be a positive multiple of num_frame_per_block={block_size}."
        )

    pipeline = CausalInferencePipeline(config, device=device)
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    try:
        checkpoint = load_weights_only(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
        )
    except RuntimeError as exc:
        if "mmap" not in str(exc).lower():
            raise
        checkpoint = load_weights_only(checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(
        _generator_state(checkpoint, use_ema=args.use_ema),
        strict=True,
        assign=True,
    )
    del checkpoint
    pipeline = pipeline.to(device=device, dtype=torch.bfloat16).eval()

    prompts = _read_prompts(args.data_path)
    extended = _extended_prompts(args.extended_prompt_path, len(prompts))
    output_folder = Path(args.output_folder).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    with torch.inference_mode():
        for prompt_index in range(rank, len(prompts), world_size):
            prompt = prompts[prompt_index]
            conditioning = extended[prompt_index] or prompt
            noise = torch.randn(
                [args.num_samples, args.num_output_frames, 16, 60, 104],
                device=device,
                dtype=torch.bfloat16,
            )
            video, _ = pipeline.inference_rolling_forcing(
                noise=noise,
                text_prompts=[conditioning] * args.num_samples,
                return_latents=True,
                initial_latent=None,
                profile=args.report_timing,
            )
            video = 255.0 * rearrange(video, "b t c h w -> b t h w c").cpu()
            pipeline.vae.model.clear_cache()
            model = "ema" if args.use_ema else "regular"
            for sample_index in range(args.num_samples):
                stem = (
                    f"{prompt_index}-{sample_index}_{model}"
                    if args.save_with_index
                    else f"{_safe_stem(prompt)}-{sample_index}"
                )
                write_video_torchvision(
                    output_folder / f"{stem}.mp4",
                    video[sample_index].clamp_(0, 255).to(torch.uint8),
                    fps=16,
                )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
