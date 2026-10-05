import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from worldfoundry.core.media.codecs.video import write_video_torchvision as write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from worldfoundry.core.geometry.trajectory import camera_trajectory_tensors
from worldfoundry.core.io import TextPromptDataset
from worldfoundry.core.utils.tensors.torch import set_seed_everywhere
from worldfoundry.core.vram import DynamicSwapInstaller, get_cuda_free_memory_gb, gpu

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True, help="Path to the inference config file")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the inference checkpoint")
parser.add_argument("--data_path", type=str, required=True, help="Path to the prompt file")
parser.add_argument("--output_folder", type=str, required=True, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=20, help="Number of overlap frames between sliding windows")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--sp_size", type=int, default=1, help="Sequence parallel size (1=disabled)")
parser.add_argument("--trajectory", type=str, default=None, help="Camera trajectory string (e.g., 'w*19' for camera control)")
parser.add_argument("--trajectory_path", type=str, default=None, help="Path to trajectory file (one trajectory string per line, aligned with data_path)")
args = parser.parse_args()

# Initialize distributed inference
# IMPORTANT: distributed init MUST happen before importing pipeline modules,
# because causal_model.py checks for CleanCode SP infra at import time.
if args.sp_size > 1:
    # SP mode requires torchrun with nproc_per_node >= sp_size
    world_size_env = int(os.environ.get("WORLD_SIZE", 1))
    assert world_size_env >= args.sp_size, (
        f"SP requires at least {args.sp_size} processes, but WORLD_SIZE={world_size_env}. "
        f"Launch with: torchrun --nproc_per_node={args.sp_size} inference.py ... --sp_size {args.sp_size}"
    )
    from .wan_utils.distributed import get_sp_seed_offset, launch_distributed_job
    launch_distributed_job(backend="nccl", sp_size=args.sp_size)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
elif "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1

# Seed: under SP, ranks in the same SP group must share the same seed
if args.sp_size > 1:
    set_seed_everywhere(args.seed + get_sp_seed_offset())
else:
    set_seed_everywhere(args.seed)

# Refresh the shared VRAM helper after distributed init selects the local GPU.
from worldfoundry.core.vram import memory as _mem
_mem.gpu = torch.device(f'cuda:{local_rank}')
gpu = _mem.gpu

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

# Import pipeline AFTER distributed init so causal_model.py sees CleanCode SP infra
from .pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
    BidirectionalDiffusionInferencePipeline,
    BidirectionalInferencePipeline,
)

# Initialize pipeline
is_causal = config.get('causal', True)

if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    if is_causal:
        pipeline = CausalInferencePipeline(config, device=device)
    else:
        pipeline = BidirectionalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    if is_causal:
        pipeline = CausalDiffusionInferencePipeline(config, device=device)
    else:
        pipeline = BidirectionalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)
    key = 'generator_ema'
    try:
        gen_sd = state_dict[key]
    except:
        key = 'generator'
        gen_sd = state_dict[key]

    try:
        pipeline.generator.load_state_dict(gen_sd)
    except RuntimeError:
        fixed = {}
        for k, v in gen_sd.items():
            if k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[k] = v
        pipeline.generator.load_state_dict(fixed, strict=False)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# Create dataset
dataset = TextPromptDataset(prompt_path=args.data_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized() and args.sp_size <= 1:
    # Standard DP: split prompts across ranks
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
elif dist.is_initialized() and args.sp_size > 1:
    # SP mode: use SP-aware sampler so ranks in the same SP group get the same data
    from .wan_utils.distributed import get_sp_data_sampler
    sampler = get_sp_data_sampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

# Load per-prompt trajectory list if provided
trajectory_list = None
if args.trajectory_path:
    with open(args.trajectory_path, encoding="utf-8") as _f:
        trajectory_list = [line.strip() for line in _f if line.strip()]
    assert len(trajectory_list) >= num_prompts, (
        f"trajectory_path has {len(trajectory_list)} lines but need >= {num_prompts} prompts"
    )

# Latency bookkeeping (rank 0 only; first prompt is recorded as None to skip warmup).
# All pipelines expose `last_chunk0_latency`: time from sampling start to the first
# denoised latent ready, EXCLUDING VAE decode — matches HY15 latency definition.
chunk0_latencies = []


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames


    prompt = batch['prompts'][0]
    output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
    if os.path.exists(output_path):
        print('Video has been generated. Pass!')
        continue
    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    prompts = [extended_prompt if extended_prompt is not None else prompt]
    initial_latent = None
    sampled_noise = torch.randn(
        [1, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
    )

    # Parse camera trajectory if provided
    viewmats = None
    Ks = None
    traj_str = None
    if trajectory_list:
        traj_str = trajectory_list[idx]
    elif args.trajectory:
        traj_str = args.trajectory
    if traj_str:
        viewmats, Ks = camera_trajectory_tensors(
            traj_str,
            fx=0.5,
            fy=0.5,
            cx=0.5,
            cy=0.5,
            rotation_step_degrees=float(os.environ.get("WORLDFOUNDRY_MINWM_ROT_STEP_DEG", "3.0")),
            device=device,
            dtype=torch.bfloat16,
        )

    # Generate frames
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        viewmats=viewmats,
        Ks=Ks
    )

    # Record latency on rank 0; first prompt is warmup → None.
    # All pipelines stop the timer before VAE decode (see pipeline.last_chunk0_latency).
    if local_rank == 0:
        sample_lat = getattr(pipeline, "last_chunk0_latency", None)
        if len(chunk0_latencies) >= 1:
            chunk0_latencies.append(sample_lat)
        else:
            chunk0_latencies.append(None)

    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    clean_latent = latents[0].cpu()
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    traj_suffix = "_" + traj_str.replace("*", "").replace(",", "") if traj_str else ""
    output_path = os.path.join(args.output_folder, f'{prompt[:100]}{traj_suffix}.mp4')
    if not (args.sp_size > 1 and local_rank != 0):
        write_video(output_path, video[0], fps=16)
    if dist.is_initialized():
        dist.barrier()


# Aggregate latency on rank 0 (drop the first prompt's warmup).
if local_rank == 0:
    valid = [v for v in chunk0_latencies[1:] if v is not None]
    if valid:
        print(f"[timing] rank0 chunk0 latency excl. decode (from 2nd prompt): "
              f"avg={sum(valid)/len(valid):.3f}s over {len(valid)} samples")
