"""Headless Gaussian-conditioned video inference pipeline."""

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Path setup ──────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# Add scripts/ to path for traj_to_extrinsics import
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from omegaconf import OmegaConf
from PIL import Image

from rendering.gaussian_renderer import GaussianRenderer, load_gaussians
from pipeline import CausalInferencePipeline
from traj_to_extrinsics import load_trajectory, trajectory_to_w2cs


def parse_args():
    p = argparse.ArgumentParser(
        description="Headless video generation pipeline (reuses server pipeline)"
    )
    # Pipeline config
    p.add_argument("--config_path", type=str, required=True,
                   help="Inference config file (YAML)")
    p.add_argument("--default_config_path", type=str, required=True,
                   help="Base inference config file (YAML)")
    p.add_argument("--checkpoint_path", type=str, default=None,
                   help="Model checkpoint path (overrides config)")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--fp8", action="store_true")
    p.add_argument("--taehv", action="store_true", default=True,
                   help="Use TAEHV lightweight VAE decoder (default: True)")
    p.add_argument("--taehv_checkpoint", type=str,
                   default="checkpoints/taew2_1.pth",
                   help="TAEHV checkpoint path")

    # Scene
    p.add_argument("--scene", type=str, required=True,
                   help="Path to 3DGS PLY or .pt file")
    p.add_argument("--prompt", type=str, required=True,
                   help="Video generation text prompt")
    p.add_argument("--radius", type=float, default=1.0,
                   help="Scene radius for trajectory scaling")

    # Trajectory
    p.add_argument("--traj", type=str, required=True,
                   help="Trajectory file path")
    p.add_argument("--num_frames", type=int, default=161,
                   help="Number of output frames")

    # Render settings
    p.add_argument("--render_fov_deg", type=float, default=70.0,
                   help="Render horizontal FOV (degrees)")
    p.add_argument("--near_plane", type=float, default=0.01)
    p.add_argument("--far_plane", type=float, default=1000.0)
    p.add_argument("--gaussian_scale", type=float, default=1.0,
                   help="Gaussian scale factor (default 1.0)")

    # Output
    p.add_argument("--output", type=str, default="output/video.mp4",
                   help="Output video path")
    p.add_argument("--fps", type=int, default=16,
                   help="Output video FPS")
    p.add_argument("--save_frames_dir", type=str, default=None,
                   help="Directory to save individual frames (default: temp)")

    return p.parse_args()


def load_pipeline(args, config):
    """Load inference pipeline (replicates server's _load_pipeline logic).

    Returns:
        (pipeline, device, dtype)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    print("[init] Loading CausalInferencePipeline ...")
    t0 = time.perf_counter()
    pipeline = CausalInferencePipeline(config, device=device)
    print(f"[init] Pipeline loaded ({(time.perf_counter()-t0)*1000:.0f} ms)")

    # Load checkpoint
    ckpt_path = args.checkpoint_path
    if ckpt_path is None and hasattr(config, "checkpoint_paths"):
        ckpt_path = list(config.checkpoint_paths)[0]
        print(f"[init] Reading checkpoint from config.checkpoint_paths: {ckpt_path}")

    if ckpt_path:
        print(f"[init] Loading checkpoint: {ckpt_path}")
        t0 = time.perf_counter()
        raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        def _strip_fsdp(sd):
            return {k.replace("._fsdp_wrapped_module", ""): v for k, v in sd.items()}

        if args.use_ema and "generator_ema" in raw_ckpt:
            ema_sd = _strip_fsdp(raw_ckpt["generator_ema"])
            try:
                pipeline.generator.load_state_dict(ema_sd)
                print("[init] EMA weights loaded.")
            except RuntimeError:
                remapped = {"model." + k: v for k, v in ema_sd.items()}
                pipeline.generator.load_state_dict(remapped)
                print("[init] EMA weights loaded (after remap model. prefix).")
        elif "generator" in raw_ckpt:
            pipeline.generator.load_state_dict(_strip_fsdp(raw_ckpt["generator"]))
            print("[init] Generator weights loaded.")
        else:
            sd = _strip_fsdp(raw_ckpt)
            if not any(k.startswith("model.") for k in sd):
                sd = {"model." + k: v for k, v in sd.items()}
            pipeline.generator.load_state_dict(sd)
            print("[init] Flat format weights loaded.")

        print(f"[init] Checkpoint loaded ({(time.perf_counter()-t0)*1000:.0f} ms)")

    pipeline = pipeline.to(dtype=dtype)
    pipeline.generator.eval()
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    # FP8 quantization
    if args.fp8:
        try:
            from torchao.quantization.quant_api import (
                quantize_, Float8DynamicActivationFloat8WeightConfig)
            quantize_(pipeline.generator, Float8DynamicActivationFloat8WeightConfig())
            print("[init] FP8 quantization enabled")
        except Exception as e:
            print(f"[init] FP8 quantization failed: {e}")

    # TAEHV lightweight VAE decoder
    if args.taehv:
        import types
        from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.taehv import TAEHV

        taehv_ckpt = args.taehv_checkpoint
        print(f"[init] Loading TAEHV lightweight VAE decoder: {taehv_ckpt} ...")
        t0 = time.perf_counter()
        taehv_model = (
            TAEHV(checkpoint_path=taehv_ckpt)
            .to(dtype=torch.float16)
            .eval()
            .requires_grad_(False)
            .to(device)
        )

        taehv_latent_cache = []
        _TAEHV_MAX_CACHE = 3

        def _taehv_decode_to_pixel(self_vae, latent, use_cache=False):
            nonlocal taehv_latent_cache

            if not use_cache:
                pixels = taehv_model.decode_video(
                    latent.to(dtype=torch.float16), parallel=False)
                return pixels.mul(2).sub(1).float()

            T_new = latent.shape[1]

            if len(taehv_latent_cache) == 0:
                decode_input = latent
                skip_frames = taehv_model.frames_to_trim
            else:
                cache_tensor = torch.cat(taehv_latent_cache, dim=1)
                decode_input = torch.cat([cache_tensor, latent], dim=1)
                skip_frames = cache_tensor.shape[1] * 4

            all_pixels = taehv_model.decode_video(
                decode_input.to(dtype=torch.float16), parallel=False)
            new_pixels = all_pixels[:, skip_frames:]

            for t in range(T_new):
                taehv_latent_cache.append(latent[:, t:t + 1].detach())
            if len(taehv_latent_cache) > _TAEHV_MAX_CACHE:
                taehv_latent_cache = taehv_latent_cache[-_TAEHV_MAX_CACHE:]

            return new_pixels.mul(2).sub(1).float()

        pipeline.vae.decode_to_pixel = types.MethodType(
            _taehv_decode_to_pixel, pipeline.vae)

        _original_clear_cache = pipeline.vae.model.clear_cache

        def _taehv_aware_clear_cache():
            _original_clear_cache()
            nonlocal taehv_latent_cache
            taehv_latent_cache = []

        pipeline.vae.model.clear_cache = _taehv_aware_clear_cache

        print(f"[init] TAEHV VAE decoder activated (streaming cache mode) "
              f"({(time.perf_counter()-t0)*1000:.0f} ms)")

    return pipeline, device, dtype


def encode_prompt(pipeline, prompt, device, dtype):
    """Encode text prompt to conditional dict.

    Temporarily moves text encoder to GPU, encodes, then offloads to CPU.
    """
    print(f"[encode] Encoding prompt: {prompt[:80]}...")
    pipeline.text_encoder.to(device=device)
    with torch.no_grad():
        cond_dict = pipeline.text_encoder([prompt])
    cond_dict = {
        k: v.to(dtype=dtype, device=device).clone()
        for k, v in cond_dict.items()
    }
    pipeline.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    print("[encode] Prompt encoding complete")
    return cond_dict


def run_pipeline_step_headless(
    frame_w2cs,
    intrinsics,
    renderer,
    pipeline,
    device,
    dtype,
):
    """Execute one pipeline step with absolute w2c matrices.

    Mirrors _run_pipeline_step from server_pano_gs.py, but uses pre-computed
    w2c matrices instead of incremental camera updates.

    Args:
        frame_w2cs: list of (4, 4) w2c tensors on GPU
        intrinsics: (3, 3) intrinsics tensor on GPU
        renderer: GaussianRenderer instance
        pipeline: CausalInferencePipeline instance
        device: torch.device
        dtype: torch.dtype

    Returns:
        gen_pixel: (1, N, 3, H, W) generated frames in [0, 1]
        render_pixel: (1, N, 3, H, W) rendered condition frames in [-1, 1]
    """
    frames_per_step = len(frame_w2cs)

    # ── 1. Gaussian rendering ─────────────────────────────────────────────
    with torch.no_grad():
        viewmats = torch.stack(frame_w2cs)                                   # (T, 4, 4)
        Ks = intrinsics.unsqueeze(0).expand(frames_per_step, -1, -1)       # (T, 3, 3)
        rendered_rgb = renderer.render_batch(viewmats, Ks)                  # (T, H, W, 3)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # (T, H, W, 3) → (1, T, 3, H, W), [0,1] → [-1,1]
    render_pixel = (rendered_rgb * 2.0 - 1.0).permute(0, 3, 1, 2).unsqueeze(0).to(device)

    # ── 2. VAE encode ─────────────────────────────────────────────────────
    pixel_bct = render_pixel.permute(0, 2, 1, 3, 4).to(device=device, dtype=dtype)
    with torch.no_grad():
        cond_latent = pipeline.vae.encode_to_latent_cached(pixel_bct)
    cond_latent = cond_latent.to(dtype=dtype)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ── 3. DiT inference + VAE decode ────────────────────────────────────
    _, T_lat, C_lat, H_lat, W_lat = cond_latent.shape
    noise = torch.randn(1, T_lat, C_lat, H_lat, W_lat, device=device, dtype=dtype)

    _step_timing = {}
    with torch.no_grad():
        gen_pixel = pipeline.streaming_step(
            noise=noise,
            cond_latent=cond_latent,
            _timing=_step_timing,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dit_ms = _step_timing.get("dit_ms", 0.0)
    vae_dec_ms = _step_timing.get("vae_dec_ms", 0.0)
    print(f"  [pipeline] DiT: {dit_ms:.0f} ms, VAE dec: {vae_dec_ms:.0f} ms, "
          f"frames out: {gen_pixel.shape[1]}")

    return gen_pixel, render_pixel


def save_frames(gen_pixel, frame_idx, output_dir):
    """Save generated frames from gen_pixel tensor to PNG files.

    Args:
        gen_pixel: (1, N, 3, H, W) tensor in [0, 1]
        frame_idx: starting frame index
        output_dir: directory to save frames

    Returns:
        updated frame_idx
    """
    for t in range(gen_pixel.shape[1]):
        frame = gen_pixel[0, t]  # (3, H, W)
        frame = frame.clamp(0, 1)
        frame = (frame * 255).byte().cpu().numpy()
        frame = np.transpose(frame, (1, 2, 0))  # (H, W, 3)
        img = Image.fromarray(frame)
        img.save(os.path.join(output_dir, f"frame_{frame_idx:05d}.png"))
        frame_idx += 1
    return frame_idx


def save_render_frames(render_pixel, frame_idx, output_dir):
    """Save rendered condition frames from render_pixel tensor to PNG files.

    Args:
        render_pixel: (1, N, 3, H, W) tensor in [-1, 1]
        frame_idx: starting frame index
        output_dir: directory to save frames

    Returns:
        updated frame_idx
    """
    for t in range(render_pixel.shape[1]):
        frame = render_pixel[0, t]  # (3, H, W)
        frame = frame.clamp(-1, 1)
        frame = ((frame + 1.0) / 2.0 * 255).byte().cpu().numpy()
        frame = np.transpose(frame, (1, 2, 0))  # (H, W, 3)
        img = Image.fromarray(frame)
        img.save(os.path.join(output_dir, f"frame_{frame_idx:05d}.png"))
        frame_idx += 1
    return frame_idx


def frames_to_video(frames_dir, output_path, fps=16):
    """Compile frames into a video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_path,
    ]
    print(f"[ffmpeg] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ffmpeg] Error: {result.stderr}")
        return False
    print(f"[ffmpeg] Video saved to: {output_path}")
    return True


def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    print(f"[init] ROOT_DIR = {ROOT_DIR}")
    print(f"[init] Loading config: {args.config_path}")
    config = OmegaConf.load(args.config_path)
    default_cfg = OmegaConf.load(args.default_config_path)
    config = OmegaConf.merge(default_cfg, config)

    if hasattr(config, "denoising_step_lists") and not hasattr(config, "denoising_step_list"):
        OmegaConf.update(config, "denoising_step_list",
                         list(config.denoising_step_lists[0]))

    # ── Load pipeline ─────────────────────────────────────────────────────
    pipeline, device, dtype = load_pipeline(args, config)

    # ── Load Gaussian data ────────────────────────────────────────────────
    if not os.path.exists(args.scene):
        raise FileNotFoundError(f"Scene file not found: {args.scene}")

    print(f"[init] Loading Gaussian data: {args.scene}")
    t0 = time.perf_counter()
    with torch.no_grad():
        gs_data = load_gaussians(args.scene, str(device))
    dt = (time.perf_counter() - t0) * 1000
    print(f"[init] Gaussian load complete ({dt:.0f} ms, "
          f"{gs_data['means'].shape[0]:,} Gaussians)")

    # ── Create GaussianRenderer ───────────────────────────────────────────
    target_h, target_w = 480, 832
    gs_scales = gs_data["scales"]
    if args.gaussian_scale != 1.0:
        gs_scales = gs_scales * args.gaussian_scale
        print(f"[init] Applying Gaussian scale factor: {args.gaussian_scale}")

    renderer = GaussianRenderer(
        means=gs_data["means"],
        quats=gs_data["quats"],
        scales=gs_scales,
        opacities=gs_data["opacities"],
        colors=gs_data["colors"],
        sh_degree=gs_data["sh_degree"],
        width=target_w,
        height=target_h,
        device=str(device),
        near_plane=args.near_plane,
        far_plane=args.far_plane,
    )
    print(f"[init] GaussianRenderer created ({target_w}x{target_h})")

    # ── Compute intrinsics ───────────────────────────────────────────────
    fov_h_deg = args.render_fov_deg
    fx = (target_w / 2.0) / math.tan(math.radians(fov_h_deg / 2.0))
    fy = fx
    cx = target_w / 2.0
    cy = target_h / 2.0
    intrinsics = torch.tensor([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)
    print(f"[init] Intrinsics: FOV={fov_h_deg:.0f}°, fx={fx:.1f}")

    # ── Load trajectory → w2c matrices ────────────────────────────────────
    theta_seq, phi_seq, r_seq = load_trajectory(args.traj)
    print(f"[traj] Loaded {len(theta_seq)} keypoints from {args.traj}")

    # VAE cached_encode only supports T=1 (step 1) or T=4 (subsequent steps).
    # Round num_frames up to nearest valid count: 1 + 4*N.
    if args.num_frames == 1:
        total_frames_padded = 1
    else:
        total_frames_padded = 1 + 4 * ((args.num_frames - 1 + 3) // 4)
    if total_frames_padded != args.num_frames:
        print(f"[traj] Padding {args.num_frames} → {total_frames_padded} frames "
              f"(VAE requires 1+4N pattern; extra frames will be discarded)")

    w2c_matrices = trajectory_to_w2cs(
        theta_seq, phi_seq, r_seq,
        num_frames=total_frames_padded,
        radius=args.radius,
    )
    w2c_matrices = w2c_matrices.to(device)
    print(f"[traj] Generated {w2c_matrices.shape[0]} w2c matrices")

    # ── Encode text prompt ────────────────────────────────────────────────
    cond_dict = encode_prompt(pipeline, args.prompt, device, dtype)

    # ── Initialize streaming ─────────────────────────────────────────────
    num_frames_for_cache = getattr(config, "num_frames", 21)
    print(f"[init] Initializing streaming (cache={num_frames_for_cache} frames)...")
    pipeline.initialize_streaming(
        batch_size=1,
        device=device,
        dtype=dtype,
        conditional_dict=cond_dict,
        num_frames_for_cache=num_frames_for_cache,
    )

    # ── Output directory ─────────────────────────────────────────────────
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    frames_dir = args.save_frames_dir or os.path.join(
        output_dir or ".", "_frames_tmp"
    )
    os.makedirs(frames_dir, exist_ok=True)

    # Render frames directory (for side-by-side comparison)
    render_frames_dir = os.path.join(
        output_dir or ".", "_render_frames_tmp"
    )
    os.makedirs(render_frames_dir, exist_ok=True)

    # ── Run pipeline ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Headless Video Generation")
    print(f"  Scene:    {args.scene}")
    print(f"  Prompt:   {args.prompt[:60]}...")
    print(f"  Frames:   {args.num_frames}")
    print(f"  Output:   {args.output}")
    print(f"{'=' * 60}\n")

    frame_idx = 0
    step_idx = 0

    while frame_idx < total_frames_padded:
        step_idx += 1

        # VAE cached_encode requires exactly 1 frame (step 1) or 4 frames (step 2+).
        # Never use 2 or 3 — the 3D temporal conv (kernel=3) will fail.
        frames_per_step = 1 if step_idx == 1 else 4

        # Get w2c matrices for this step (padded total guarantees enough frames)
        step_w2cs = [
            w2c_matrices[frame_idx + i]
            for i in range(frames_per_step)
        ]

        print(f"[step {step_idx}] Rendering {frames_per_step} frame(s) "
              f"(frames {frame_idx}-{frame_idx + frames_per_step - 1})...")

        t0 = time.perf_counter()
        gen_pixel, render_pixel = run_pipeline_step_headless(
            frame_w2cs=step_w2cs,
            intrinsics=intrinsics,
            renderer=renderer,
            pipeline=pipeline,
            device=device,
            dtype=dtype,
        )
        dt = (time.perf_counter() - t0) * 1000
        print(f"[step {step_idx}] Done ({dt:.0f} ms, "
              f"{gen_pixel.shape[1]} frames generated)")

        # Save generated frames
        frame_idx = save_frames(gen_pixel, frame_idx, frames_dir)
        # Save rendered condition frames
        save_render_frames(render_pixel, frame_idx - gen_pixel.shape[1], render_frames_dir)

    # Truncate to requested num_frames (discard padded extra frames)
    actual_frames = min(frame_idx, args.num_frames)
    print(f"\n[done] Generated {frame_idx} frames in {step_idx} steps "
          f"(keeping {actual_frames}, requested {args.num_frames})")

    # Remove excess frames beyond requested count
    if frame_idx > args.num_frames:
        for d in [frames_dir, render_frames_dir]:
            dir_path = Path(d)
            for f in sorted(dir_path.glob("frame_*.png"), reverse=True)[:frame_idx - args.num_frames]:
                f.unlink()
                print(f"  [cleanup] Removed excess frame: {f.name}")

    # ── Compile gen video ──────────────────────────────────────────────────
    print(f"[ffmpeg] Compiling gen video...")
    gen_success = frames_to_video(frames_dir, args.output, fps=args.fps)

    # ── Compile render video ───────────────────────────────────────────────
    render_video = args.output.replace(".mp4", "_render.mp4")
    print(f"[ffmpeg] Compiling render video...")
    render_success = frames_to_video(render_frames_dir, render_video, fps=args.fps)

    # ── Create side-by-side comparison (numpy + imageio, same as UI) ───────
    combined_video = args.output.replace(".mp4", "_side_by_side.mp4")
    if gen_success and render_success:
        print(f"[combine] Creating side-by-side comparison...")
        try:
            import imageio
            render_imgs = [imageio.imread(p) for p in sorted(Path(render_frames_dir).glob("frame_*.png"))]
            gen_imgs = [imageio.imread(p) for p in sorted(Path(frames_dir).glob("frame_*.png"))]

            # Match frame counts: pad the shorter one by repeating its last frame
            n = max(len(render_imgs), len(gen_imgs))
            while len(render_imgs) < n:
                render_imgs.append(render_imgs[-1])
            while len(gen_imgs) < n:
                gen_imgs.append(gen_imgs[-1])

            # Resize gen frames to match render frame height (if different)
            target_h = render_imgs[0].shape[0]
            target_w_render = render_imgs[0].shape[1]
            target_w_gen = gen_imgs[0].shape[1]
            if gen_imgs[0].shape[0] != target_h:
                scale = target_h / gen_imgs[0].shape[0]
                target_w_gen = int(gen_imgs[0].shape[1] * scale)
                import cv2 as _cv2
                gen_imgs = [_cv2.resize(img, (target_w_gen, target_h)) for img in gen_imgs]

            # Concatenate horizontally: [render | gen]
            combined_imgs = []
            for r_img, g_img in zip(render_imgs, gen_imgs):
                combined = np.zeros((target_h, target_w_render + target_w_gen, 3),
                                    dtype=np.uint8)
                combined[:, :target_w_render] = r_img
                combined[:, target_w_render:] = g_img
                combined_imgs.append(combined)

            imageio.mimsave(combined_video, combined_imgs, fps=args.fps,
                            codec="libx264", quality=8, macro_block_size=None,
                            ffmpeg_params=['-movflags', 'faststart', '-pix_fmt', 'yuv420p'])
            print(f"[combine] Side-by-side saved: {combined_video} ({len(combined_imgs)} frames)")
        except Exception as e:
            print(f"[combine] Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[warning] Skipping side-by-side (gen_success={gen_success}, render_success={render_success})")

    # ── Clean up temporary frames ───────────────────────────────────────────
    if gen_success and args.save_frames_dir is None:
        shutil.rmtree(frames_dir)
        print(f"[cleanup] Removed gen frames directory")
    if render_success:
        shutil.rmtree(render_frames_dir)
        print(f"[cleanup] Removed render frames directory")

    print(f"\n[complete] Outputs:")
    print(f"  Gen video:       {args.output}")
    print(f"  Render video:    {render_video}")
    if os.path.exists(combined_video):
        print(f"  Side-by-side:    {combined_video}")


if __name__ == "__main__":
    main()
