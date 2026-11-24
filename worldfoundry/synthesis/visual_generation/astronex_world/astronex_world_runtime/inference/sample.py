# Adapted from Astronex-Robotics/Astronex-World (Apache-2.0) at 27584bf1, inference/sample.py.
"""Sample one Astronex-World rollout per prompt and write it as an mp4.

Started by :mod:`.generate` with ``CUDA_VISIBLE_DEVICES`` already narrowed to one GPU.
"""

import argparse
import os
import re


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="sampling config (resolved by generate.py)")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="released weights directory")
    parser.add_argument("--data_path", type=str, required=True, help="prompt file, one prompt per line")
    parser.add_argument("--image_path", type=str, default=None, help="reference image for I2V")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--num_output_frames", type=int, default=20, help="latent frames to generate")
    parser.add_argument("--sampling_steps", type=int, default=None, help="override the config's sampler steps")
    parser.add_argument("--latent_height", type=int, default=None, help="latent output height (44 = 704 pixels)")
    parser.add_argument("--latent_width", type=int, default=None, help="latent output width (80 = 1280 pixels)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--i2v", action="store_true", help="image-to-video (text-to-video by default)")
    parser.add_argument("--event_prompt", type=str, default=None, help="event text merged into the caption")
    parser.add_argument(
        "--event_start_frame", type=int, default=0, help="latent frame the event caption takes effect from"
    )
    parser.add_argument(
        "--action_tokens",
        type=str,
        default=None,
        help="per-latent-frame action names with '*N' repeats, e.g. 'W*8,A*8'",
    )
    parser.add_argument("--action_scale", type=float, default=1.0, help="stick deflection of --action_tokens")
    parser.add_argument("--embodiment_id", type=int, default=None, help="row of action_embedder.embodiment")
    parser.add_argument(
        "--action_keys", action="store_true", help="read --action_tokens as keyboard keys instead of WBench actions"
    )
    parser.add_argument("--trajectory", type=str, default=None, help="camera trajectory string, e.g. 'w*19'")
    parser.add_argument("--fps", type=int, default=16, help="frame rate written into the mp4")
    return parser.parse_args(argv)


def load_prompts(args) -> list[str]:
    """The I2V reference takes the whole prompt file as one caption; T2V samples every line."""
    with open(args.data_path, encoding="utf-8") as f:
        if args.i2v:
            return [f.read().strip()]
        return [line.rstrip() for line in f]


def camera_inputs(traj_str, trajectory_speed, device):
    """``(viewmats [1, T, 4, 4], Ks [1, T, 3, 3])`` in bf16 for a trajectory string."""
    import numpy as np
    import torch

    from ..utils.camera_trajectory import parse_trajectory

    viewmats_np = parse_trajectory(traj_str)
    if trajectory_speed != 1.0:
        if trajectory_speed <= 0:
            raise ValueError("trajectory_speed must be positive")
        # Scale the camera-centre displacement; orientations and the first-frame gauge stay.
        c2w = np.linalg.inv(viewmats_np)
        origin = c2w[:1, :3, 3].copy()
        c2w[:, :3, 3] = origin + trajectory_speed * (c2w[:, :3, 3] - origin)
        viewmats_np = np.linalg.inv(c2w).astype(np.float32)
    fx, fy, cx, cy = 0.5050505, 0.89786756, 0.5, 0.5
    Ks_np = np.array([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]] * len(viewmats_np), dtype=np.float32)
    viewmats = torch.from_numpy(viewmats_np).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    Ks = torch.from_numpy(Ks_np).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    return viewmats, Ks


def action_inputs(args, device):
    """``(actions [1, F, 64], embodiment_id [1] or None)`` from ``--action_tokens``."""
    import torch

    from ..utils.action_encoding import KEYBOARD, WBENCH_ACTION, from_tokens

    tokens = []
    for segment in args.action_tokens.split(","):
        segment = segment.strip()
        if not segment:
            continue
        match = re.fullmatch(r"([A-Za-z+]+)(?:\*(\d+))?", segment)
        if match is None:
            raise ValueError(f"cannot parse action segment '{segment}'")
        tokens += [match.group(1)] * int(match.group(2) or 1)
    table = KEYBOARD if args.action_keys else WBENCH_ACTION
    actions = torch.from_numpy(from_tokens(tokens, table=table, magnitude=args.action_scale))
    actions = actions.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    embodiment = torch.tensor([args.embodiment_id], device=device).long() if args.embodiment_id is not None else None
    return actions, embodiment


def correct_opening_direction(config, traj_str, video):
    """Apply the commanded forward/backward zoom over the cold-start frames of a pure w/s opening.

    The first decoded frames of a causal I2V rollout are driven almost entirely by the
    reference image; this ramps a bounded affine zoom over ``opening_direction_frames``.
    """
    import torch
    import torch.nn.functional as F

    match = re.match(r"\s*(w|s)(?:\*|,|$)", traj_str.lower())
    if not match:
        return video
    direction = match.group(1)
    opening_frames = max(2, int(config.get("opening_direction_frames", 12)))
    strength_key = "opening_forward_strength" if direction == "w" else "opening_backward_strength"
    strength = float(config.get(strength_key, 0.06 if direction == "w" else 0.12))
    t_count = video.shape[1]
    ramp = torch.arange(t_count, device=video.device, dtype=torch.float32).div(float(opening_frames)).clamp_(0, 1)
    # Forward affine scale: >1 zooms in. grid_sample consumes the inverse mapping.
    scale = 1.0 + (strength if direction == "w" else -strength) * ramp
    theta = torch.zeros(t_count, 2, 3, device=video.device, dtype=torch.float32)
    theta[:, 0, 0] = 1.0 / scale
    theta[:, 1, 1] = 1.0 / scale
    frames = video[0].float()
    grid = F.affine_grid(theta, frames.shape, align_corners=False)
    corrected = F.grid_sample(frames, grid, mode="bicubic", padding_mode="border", align_corners=False)
    video = video.clone()
    video[0] = corrected.to(video.dtype).clamp_(0, 1)
    print(f"[camera-opening] {direction=} frames={opening_frames} strength={strength:.3f}", flush=True)
    return video


def load_generator_weights(generator, weights_dir):
    """Load the released ``model-*.safetensors``; the untrained action-output head is not built."""
    from worldfoundry.core.model_loading import load_state_dict

    state = load_state_dict(os.path.join(weights_dir, "model.safetensors.index.json"))
    incompatible = generator.load_state_dict(state, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("model.action_head.")]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {incompatible.missing_keys[:5]}, unexpected {unexpected[:5]}")
    print(f"[ckpt] loaded {len(state)} tensors from {weights_dir}")


def main(argv=None):
    args = parse_args(argv)

    import random

    import imageio
    import numpy as np
    import torch
    from einops import rearrange
    from omegaconf import DictConfig, OmegaConf
    from PIL import Image
    from torchvision import transforms
    from tqdm import tqdm

    from worldfoundry.core.vram.memory import DynamicSwapInstaller, get_cuda_free_memory_gb

    from .pipelines import BidirectionalDiffusionInferencePipeline, CausalDiffusionInferencePipeline

    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    gpu = torch.device("cuda:0")
    # ASTRONEX_VRAM_LIMIT_GB caps this process, so a larger card behaves like a smaller one.
    vram_limit = os.environ.get("ASTRONEX_VRAM_LIMIT_GB")
    if vram_limit:
        total = torch.cuda.get_device_properties(gpu).total_memory / (1024**3)
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(vram_limit) / total), gpu)
        print(f"[vram] capped at {float(vram_limit):.1f} GB of {total:.1f} GB")
    free_vram = get_cuda_free_memory_gb(gpu)
    if vram_limit:
        free_vram = min(free_vram, float(vram_limit))
    print(f"Free VRAM {free_vram} GB")
    low_memory = os.environ.get("ASTRONEX_DISABLE_LOW_MEMORY") != "1" and free_vram < 40

    torch.set_grad_enabled(False)

    config = OmegaConf.load(args.config_path)
    assert isinstance(config, DictConfig)
    if args.sampling_steps is not None:
        config.sampling_steps = args.sampling_steps
    if args.latent_height is not None:
        config.image_or_video_shape[3] = args.latent_height
    if args.latent_width is not None:
        config.image_or_video_shape[4] = args.latent_width
    trajectory_speed = float(config.get("trajectory_speed", 1.0))

    is_causal = bool(config.get("causal", True))
    pipeline = (
        CausalDiffusionInferencePipeline(config) if is_causal else BidirectionalDiffusionInferencePipeline(config)
    )
    load_generator_weights(pipeline.generator, args.checkpoint_path)

    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.text_encoder.execution_device = gpu
    pipeline.generator.to(device=gpu)
    vae_device = torch.device(os.environ.get("WAN_VAE_DEVICE", str(gpu)))
    pipeline.vae.to(device=vae_device)
    pipeline.vae_device = vae_device if vae_device != gpu else None
    if pipeline.vae_device is not None:
        print(f"[device] generator={gpu}, vae={pipeline.vae_device}")

    image = None
    if args.i2v:
        # The reference image goes on the requested latent grid.
        target_height = int(config.image_or_video_shape[3]) * 16
        target_width = int(config.image_or_video_shape[4]) * 16
        transform = transforms.Compose(
            [
                transforms.Resize((target_height, target_width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        image = transform(Image.open(args.image_path).convert("RGB"))
    prompts_to_sample = load_prompts(args)
    print(f"Number of prompts: {len(prompts_to_sample)}")
    os.makedirs(args.output_folder, exist_ok=True)

    latent_shape = config.image_or_video_shape
    for prompt in tqdm(prompts_to_sample):
        initial_latent = None
        if image is not None:
            pixels = image.unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
            initial_latent = pipeline.vae.encode_to_latent(pixels.to(device=vae_device)).to(
                device=device, dtype=torch.bfloat16
            )
        sampled_noise = torch.randn(
            [1, args.num_output_frames, latent_shape[2], latent_shape[3], latent_shape[4]],
            device=device,
            dtype=torch.bfloat16,
        )
        viewmats = Ks = None
        traj_str = args.trajectory
        if traj_str:
            viewmats, Ks = camera_inputs(traj_str, trajectory_speed, device)
        inference_kwargs = {}
        if args.action_tokens and is_causal:
            actions, embodiment = action_inputs(args, device)
            inference_kwargs["actions"] = actions
            if embodiment is not None:
                inference_kwargs["embodiment_id"] = embodiment

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=[prompt],
            return_latents=True,
            initial_latent=initial_latent,
            viewmats=viewmats,
            Ks=Ks,
            event_prompts=args.event_prompt,
            event_start_frame=args.event_start_frame,
            **inference_kwargs,
        )
        if bool(config.get("opening_direction_correction", False)) and traj_str:
            video = correct_opening_direction(config, traj_str, video)

        video = 255.0 * rearrange(video, "b t c h w -> b t h w c").cpu()
        traj_suffix = "_" + traj_str.replace("*", "").replace(",", "") if traj_str else ""
        output_path = os.path.join(args.output_folder, f"{prompt[:100]}{traj_suffix}.mp4")
        writer = imageio.get_writer(output_path, fps=args.fps)
        for frame in video[0]:
            writer.append_data(frame.numpy().astype("uint8"))
        writer.close()


if __name__ == "__main__":
    main()
