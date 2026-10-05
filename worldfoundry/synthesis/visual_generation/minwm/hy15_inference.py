"""Camera-conditioned minWM HY15 inference.

Supports bidirectional denoising and cached autoregressive rollout using local
checkpoints. Invoke with --use_camera and --trajectory, for example w*19.
"""

import argparse
import json
import os
import re
import time
from types import SimpleNamespace

import imageio
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from worldfoundry.core.geometry.trajectory import (
    camera_trajectory_action_labels,
    camera_trajectory_tensors,
)
from worldfoundry.base_models.diffusion_model.schedulers.hunyuan_compat import (
    HunyuanVideoFlowMatchDiscreteScheduler as FlowMatchDiscreteScheduler,
)


# ---------------------------------------------------------------------------
# Camera trajectory (ProPE) utilities
# ---------------------------------------------------------------------------

def make_camera_tensors(traj_str, fx=0.5050505, fy=0.89786756, cx=0.5, cy=0.5):
    return camera_trajectory_tensors(
        traj_str,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        rotation_step_degrees=float(os.environ.get("WORLDFOUNDRY_MINWM_ROT_STEP_DEG", "3.0")),
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["bidirectional", "ar_rollout"])

    # Model paths
    parser.add_argument("--transformer_dir", required=True,
                        help="Transformer checkpoint dir (contains config.json + diffusion_pytorch_model.safetensors)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="HunyuanVideo-1.5 base model dir (contains vae/, text_encoder/, vision_encoder/). "
                             "Auto-detected from HF cache if not specified.")
    parser.add_argument("--action_ckpt", type=str, default=None,
                        help="Path to action model safetensors (overrides transformer_dir weights)")

    # Data
    parser.add_argument("--example_json", required=True,
                        help="JSON file with list of {image, caption} entries (relative paths resolved from JSON dir)")
    parser.add_argument("--output_dir", required=True)

    # Camera trajectory (ProPE) - optional
    parser.add_argument("--trajectory", type=str, default=None,
                        help="Camera trajectory string, e.g. 'w*19', 'd*10,w*9'. "
                             "Overrides per-sample trajectory from JSON.")
    parser.add_argument("--use_camera", action="store_true",
                        help="Enable camera mode: read trajectory from JSON per sample. "
                             "Samples without trajectory field are skipped.")

    # Inference hyperparameters
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=6.0,
                        help="CFG guidance scale. 1.0 disables CFG.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--video_length", type=int, default=77,
                        help="Number of output video frames (must satisfy (n-1)//4+1 divisible by 4)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed used by NumPy and PyTorch")

    # Discrete action conditioning
    parser.add_argument("--use_discrete_action", action="store_true",
                        help="Pass discrete action labels to model (requires action_in module in ckpt)")

    # AR-specific
    parser.add_argument("--stabilization_level", type=int, default=1,
                        help="Timestep for clean context frame modulation (ar_rollout only)")
    parser.add_argument("--chunk_latent_frames", type=int, default=4,
                        help="Frames per chunk for ar_rollout")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def setup_dist(mode):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method="env://",
                                world_size=world_size, rank=rank)

    if mode == "bidirectional":
        from worldfoundry.core.distributed.sequence_parallel import parallel_state as ps

        _orig_init_mp = ps.init_model_parallel_group
        def _patched_init_mp(group_ranks, local_rank, backend, **kwargs):
            return ps.GroupCoordinator(
                group_ranks=group_ranks,
                local_rank=local_rank,
                torch_distributed_backend=backend,
                use_device_communicator=False,
                group_name=kwargs.get("group_name"),
            )
        ps.init_model_parallel_group = _patched_init_mp

        if ps._WORLD is None:
            ps._WORLD = ps.GroupCoordinator(
                group_ranks=[list(range(world_size))],
                local_rank=local_rank,
                torch_distributed_backend="gloo",
                use_device_communicator=False,
                group_name="world",
            )
        ps.initialize_model_parallel(tensor_model_parallel_size=1,
                                     sequence_model_parallel_size=1)
        ps.init_model_parallel_group = _orig_init_mp

    return rank, world_size, local_rank


# ---------------------------------------------------------------------------
# Shared utilities: on-the-fly encoding
# ---------------------------------------------------------------------------

def load_examples(example_json):
    """Load example list from JSON file. Returns list of {image, caption} dicts with absolute paths."""
    base_dir = os.path.dirname(os.path.abspath(example_json))
    with open(example_json) as f:
        examples = json.load(f)
    for ex in examples:
        img_path = ex["image"]
        if not os.path.isabs(img_path):
            ex["image"] = os.path.join(base_dir, img_path)
    return examples


def find_hunyuanvideo_model_path():
    """Auto-detect HunyuanVideo-1.5 model path under ./ckpts/."""
    c = "./ckpts/HunyuanVideo-1.5"
    if os.path.isdir(c):
        return c
    return None


def encode_image_to_cond_latent(vae, image_path, height, width, device):
    """Load image, resize, encode through VAE to get conditional latent."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_tensor = transform(image).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
    img_tensor = img_tensor.to(device, dtype=torch.float16)
    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.sample()
    scaling_factor = vae.config.scaling_factor
    shift_factor = getattr(vae.config, "shift_factor", None)
    if shift_factor:
        latent = (latent - shift_factor) * scaling_factor
    else:
        latent = latent * scaling_factor
    return latent.to(dtype=torch.bfloat16)


def encode_text(text_encoder, caption, device):
    """Encode text using the LLM text encoder."""
    text_encoder = text_encoder.to(device)
    with torch.no_grad():
        outputs = text_encoder([caption])
    text_encoder.cpu()
    prompt_embeds = outputs.hidden_state.to(device)
    prompt_mask = outputs.attention_mask.to(device) if outputs.attention_mask is not None else None
    return prompt_embeds, prompt_mask


def encode_vision(vision_encoder, image_path, height, width, device):
    """Encode image using SigLIP vision encoder."""
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image.resize((width, height)))  # [H, W, 3] uint8
    image_np = image_np[np.newaxis, ...]  # [1, H, W, 3]
    vision_encoder = vision_encoder.to(device)
    with torch.no_grad():
        outputs = vision_encoder.encode_images(image_np)
    vision_encoder.cpu()
    return outputs.last_hidden_state


def encode_byt5(byt5_model, byt5_tokenizer, caption, byt5_max_length, device):
    """Encode text using byT5 model."""
    if byt5_model is None:
        if re.search(r'"[^"\n]+"', caption):
            raise ValueError("Quoted glyph text requires local Glyph-SDXL-v2 / ByT5 checkpoints.")
        return torch.zeros((1, byt5_max_length, 1472), device=device), \
               torch.zeros((1, byt5_max_length), device=device, dtype=torch.int64)

    with torch.no_grad():
        byt5_text_inputs = byt5_tokenizer(
            caption, return_tensors="pt", padding="max_length",
            max_length=byt5_max_length, truncation=True,
        )
        text_ids = byt5_text_inputs.input_ids.to(device)
        text_mask = byt5_text_inputs.attention_mask.to(device)
        byt5_outputs = byt5_model(text_ids, attention_mask=text_mask.float())
        byt5_embeddings = byt5_outputs[0]
        byt5_mask = text_mask

    return byt5_embeddings, byt5_mask


def encode_negative_prompt(text_encoder, byt5_model, byt5_tokenizer, byt5_max_length, device):
    """Encode empty/negative prompt for CFG."""
    neg_prompt = ""
    neg_embeds, neg_mask = encode_text(text_encoder, neg_prompt, device)
    neg_byt5_states = torch.zeros((1, byt5_max_length, 1472), device=device)
    neg_byt5_mask = torch.zeros((1, byt5_max_length), device=device, dtype=torch.int64)
    return {
        "prompt_embeds": neg_embeds,
        "prompt_mask": neg_mask,
        "byt5_text_states": neg_byt5_states,
        "byt5_text_mask": neg_byt5_mask,
    }


def prepare_sample_data(vae, text_encoder, vision_encoder, byt5_model, byt5_tokenizer,
                        example, height, width, video_length, device):
    """
    Encode a single example (image + caption) into the data dict expected by inference functions.
    Returns dict with keys: image_cond, prompt_embeds, prompt_mask, vision_states,
                            byt5_text_states, byt5_text_mask, latent_shape
    """
    image_path = example["image"]
    caption = example["caption"]

    # Encode image -> VAE conditional latent
    vae = vae.to(device)
    image_cond = encode_image_to_cond_latent(vae, image_path, height, width, device)
    vae.cpu()
    # image_cond: [1, C, 1, h, w]

    # Compute latent spatial dims
    C = image_cond.shape[1]
    h = image_cond.shape[3]
    w = image_cond.shape[4]
    T = (video_length - 1) // 4 + 1  # temporal compression factor = 4

    # Encode text
    prompt_embeds, prompt_mask = encode_text(text_encoder, caption, device)

    # Encode vision (SigLIP)
    vision_states = encode_vision(vision_encoder, image_path, height, width, device)

    # Encode byT5
    byt5_max_length = 256
    byt5_states, byt5_mask = encode_byt5(byt5_model, byt5_tokenizer, caption, byt5_max_length, device)

    return {
        "image_cond": image_cond.cpu(),  # [1, C, 1, h, w]
        "prompt_embeds": prompt_embeds.cpu(),
        "prompt_mask": prompt_mask.cpu(),
        "vision_states": vision_states.cpu(),
        "byt5_text_states": byt5_states.cpu(),
        "byt5_text_mask": byt5_mask.cpu(),
        "latent_shape": (1, C, T, h, w),
    }


def decode_and_save(x, vae, device, output_path, fps):
    vae = vae.to(device)
    vae.set_tile_sample_min_size(256, tile_overlap_factor=0.25)
    vae.enable_tiling()
    scaling_factor = vae.config.scaling_factor
    shift_factor = getattr(vae.config, "shift_factor", None)
    x_decoded = x / scaling_factor + shift_factor if shift_factor else x / scaling_factor
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        video = vae.decode(x_decoded).sample
    video = (video.float().clamp(-1, 1) + 1) / 2
    frames = rearrange(video[0], "c t h w -> t h w c")
    frames = (frames.cpu().numpy() * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    vae.cpu()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_bidirectional(transformer_dir):
    raise ValueError("The public minWM HY runtime requires camera conditioning (--use_camera).")


def load_model_ar_rollout(transformer_dir):
    raise ValueError("The public minWM HY runtime requires camera conditioning (--use_camera).")


def load_model_prope(transformer_dir, use_discrete_action=False):
    """Load ProPE camera-conditioned transformer (used for both modes when trajectory is provided)."""
    from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.h15.action_camera import (
        ARHunyuanVideo_1_5_DiffusionTransformer,
    )
    config = ARHunyuanVideo_1_5_DiffusionTransformer.load_config(transformer_dir)
    model = ARHunyuanVideo_1_5_DiffusionTransformer.from_config(config)
    model.add_prope_parameters()
    if use_discrete_action:
        model.add_discrete_action_parameters()
    ckpt_path = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors")
    state_dict = load_file(ckpt_path, device="cpu")
    if any(key.startswith("action_in.") for key in state_dict):
        model.add_discrete_action_parameters()
    model.load_state_dict(state_dict, strict=True)
    return model


# ---------------------------------------------------------------------------
# Bidirectional inference
# ---------------------------------------------------------------------------

def run_inference_bidirectional(model, data, neg_prompts, device, num_steps, shift,
                                guidance_scale, viewmats=None, Ks=None, action=None):
    """
    Bidirectional full-sequence denoising.
    When viewmats/Ks are provided, passes them to model (ProPE mode).
    """
    image_cond = data["image_cond"].to(device, dtype=torch.bfloat16)
    prompt_embed = data["prompt_embeds"].to(device, dtype=torch.bfloat16)
    prompt_mask = data["prompt_mask"].to(device, dtype=torch.bfloat16)
    vision_states = data["vision_states"].to(device, dtype=torch.bfloat16)
    byt5_text_states = data["byt5_text_states"].to(device, dtype=torch.bfloat16)
    byt5_text_mask = data["byt5_text_mask"].to(device, dtype=torch.bfloat16)

    B, C, T, H, W = data["latent_shape"]
    x = torch.randn(B, C, T, H, W, device=device, dtype=torch.bfloat16)

    # i2v conditioning: [B, C+1, T, H, W]
    cond_latents = image_cond.repeat(1, 1, T, 1, 1)
    cond_latents[:, :, 1:, :, :] = 0.0
    mask = torch.zeros(B, 1, T, H, W, device=device)
    mask[:, :, 0, :, :] = 1.0
    cond_input = torch.cat([cond_latents, mask], dim=1)

    # ProPE camera kwargs (empty dict if no trajectory)
    prope_kwargs = {}
    if viewmats is not None:
        prope_kwargs = {"viewmats": viewmats.to(device, dtype=torch.bfloat16),
                        "Ks": Ks.to(device, dtype=torch.bfloat16)}
    if action is not None:
        prope_kwargs["action"] = action.to(device, dtype=torch.int64)

    use_cfg = guidance_scale > 1.0
    if use_cfg:
        neg_embed = neg_prompts["prompt_embeds"].to(device, dtype=torch.bfloat16)
        neg_mask = neg_prompts["prompt_mask"].to(device, dtype=torch.bfloat16)
        neg_byt5_states = neg_prompts["byt5_text_states"].to(device, dtype=torch.bfloat16)
        neg_byt5_mask = neg_prompts["byt5_text_mask"].to(device, dtype=torch.bfloat16)

    scheduler = FlowMatchDiscreteScheduler(shift=shift)
    scheduler.set_timesteps(num_steps, device=device)

    extra_kwargs = {"byt5_text_states": byt5_text_states, "byt5_text_mask": byt5_text_mask}
    timestep_txt = torch.tensor(0).unsqueeze(0).to(device, dtype=torch.bfloat16)

    total_steps = len(scheduler.timesteps)
    for i, t in enumerate(scheduler.timesteps):
        if dist.get_rank() == 0:
            print(f"  denoising {i+1}/{total_steps}, t={t.item():.1f}", flush=True)
        timesteps_in = t.unsqueeze(0).expand(B * T).to(device, dtype=torch.bfloat16)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cond_pred = model(
                hidden_states=torch.cat([x, cond_input], dim=1),
                timestep=timesteps_in,
                timestep_txt=timestep_txt,
                text_states=prompt_embed,
                text_states_2=None,
                encoder_attention_mask=prompt_mask,
                timestep_r=None,
                vision_states=vision_states,
                mask_type="i2v",
                guidance=None,
                extra_kwargs=extra_kwargs,
                return_dict=False,
                **prope_kwargs,
            )[0]

            if use_cfg:
                uncond_pred = model(
                    hidden_states=torch.cat([x, cond_input], dim=1),
                    timestep=timesteps_in,
                    timestep_txt=timestep_txt,
                    text_states=neg_embed,
                    text_states_2=None,
                    encoder_attention_mask=neg_mask,
                    timestep_r=None,
                    vision_states=vision_states,
                    mask_type="i2v",
                    guidance=None,
                    extra_kwargs={"byt5_text_states": neg_byt5_states, "byt5_text_mask": neg_byt5_mask},
                    return_dict=False,
                    **prope_kwargs,
                )[0]
                pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                pred = cond_pred

        x = scheduler.step(pred, t, x).prev_sample

    return x


# ---------------------------------------------------------------------------
# AR rollout inference (chunk-by-chunk with KV cache)
# ---------------------------------------------------------------------------

def _init_kv_cache(num_layers):
    return [{"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None}
            for _ in range(num_layers)]


def run_inference_rollout(model, data, neg_prompts, device, num_steps, shift,
                          guidance_scale, stabilization_level, chunk_latent_frames=4,
                          viewmats=None, Ks=None, action=None):
    """
    AR rollout chunk-by-chunk denoising with KV cache.
    When viewmats/Ks are provided, passes per-chunk camera tensors to model (ProPE mode).
    """
    torch.cuda.synchronize()
    _chunk0_t0 = time.perf_counter()
    chunk0_latency = None

    image_cond = data["image_cond"].to(device, dtype=torch.bfloat16)
    prompt_embed = data["prompt_embeds"].to(device, dtype=torch.bfloat16)
    prompt_mask = data["prompt_mask"].to(device, dtype=torch.bfloat16)
    vision_states = data["vision_states"].to(device, dtype=torch.bfloat16)
    byt5_text_states = data["byt5_text_states"].to(device, dtype=torch.bfloat16)
    byt5_text_mask = data["byt5_text_mask"].to(device, dtype=torch.bfloat16)

    B, C, T, H, W = data["latent_shape"]
    use_cfg = guidance_scale > 1.0
    chunk_num = T // chunk_latent_frames

    # i2v conditioning
    cond_latents = image_cond.repeat(1, 1, T, 1, 1)
    cond_latents[:, :, 1:, :, :] = 0.0
    mask = torch.zeros(B, 1, T, H, W, device=device)
    mask[:, :, 0, :, :] = 1.0
    cond_input = torch.cat([cond_latents, mask], dim=1)

    latents = torch.randn(B, C, T, H, W, device=device, dtype=torch.bfloat16)

    num_layers = len(model.double_blocks)
    kv_cache = _init_kv_cache(num_layers)
    kv_cache_neg = _init_kv_cache(num_layers) if use_cfg else None

    extra_kwargs = {"byt5_text_states": byt5_text_states, "byt5_text_mask": byt5_text_mask}
    t_txt = torch.tensor([0]).to(device, dtype=torch.bfloat16)

    # Phase 1: cache text KV
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        kv_cache = model(
            bi_inference=False, ar_txt_inference=True, ar_vision_inference=False,
            timestep_txt=t_txt, text_states=prompt_embed,
            encoder_attention_mask=prompt_mask, vision_states=vision_states,
            mask_type="i2v", extra_kwargs=extra_kwargs,
            kv_cache=kv_cache, cache_txt=True,
        )
        if use_cfg:
            neg_embed = neg_prompts["prompt_embeds"].to(device, dtype=torch.bfloat16)
            neg_mask = neg_prompts["prompt_mask"].to(device, dtype=torch.bfloat16)
            neg_byt5 = neg_prompts["byt5_text_states"].to(device, dtype=torch.bfloat16)
            neg_byt5_mask = neg_prompts["byt5_text_mask"].to(device, dtype=torch.bfloat16)
            neg_extra = {"byt5_text_states": neg_byt5, "byt5_text_mask": neg_byt5_mask}
            kv_cache_neg = model(
                bi_inference=False, ar_txt_inference=True, ar_vision_inference=False,
                timestep_txt=t_txt, text_states=neg_embed,
                encoder_attention_mask=neg_mask, vision_states=vision_states,
                mask_type="i2v", extra_kwargs=neg_extra,
                kv_cache=kv_cache_neg, cache_txt=True,
            )

    # Phase 2: chunk-by-chunk denoising
    scheduler = FlowMatchDiscreteScheduler(shift=shift, reverse=True, solver="euler")

    for chunk_i in range(chunk_num):
        start_idx = chunk_i * chunk_latent_frames
        end_idx = start_idx + chunk_latent_frames
        rope_total = end_idx

        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps

        if dist.get_rank() == 0:
            print(f"  Chunk {chunk_i+1}/{chunk_num} frames[{start_idx}:{end_idx})", flush=True)

        # ProPE: per-chunk camera tensors
        prope_kwargs = {}
        if viewmats is not None:
            vm_chunk = viewmats[:, start_idx:end_idx].to(device, dtype=torch.bfloat16)
            Ks_chunk = Ks[:, start_idx:end_idx].to(device, dtype=torch.bfloat16)
            prope_kwargs = {"viewmats": vm_chunk, "Ks": Ks_chunk}
        if action is not None:
            prope_kwargs["action"] = action[start_idx:end_idx].to(device, dtype=torch.int64)

        for i, t in enumerate(timesteps):
            if dist.get_rank() == 0:
                print("timesteps", t.item(), flush=True)
            ts_in = torch.full((chunk_latent_frames,), t, device=device, dtype=timesteps.dtype)
            latent_chunk = latents[:, :, start_idx:end_idx]
            cond_chunk = cond_input[:, :, start_idx:end_idx]
            hidden = torch.cat([latent_chunk, cond_chunk], dim=1)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                cond_pred = model(
                    bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                    hidden_states=hidden, timestep=ts_in, timestep_r=None,
                    mask_type="i2v", return_dict=False,
                    kv_cache=kv_cache, cache_vision=False,
                    rope_temporal_size=rope_total, start_rope_start_idx=start_idx,
                    **prope_kwargs,
                )[0]
                if use_cfg:
                    uncond_pred = model(
                        bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                        hidden_states=hidden, timestep=ts_in, timestep_r=None,
                        mask_type="i2v", return_dict=False,
                        kv_cache=kv_cache_neg, cache_vision=False,
                        rope_temporal_size=rope_total, start_rope_start_idx=start_idx,
                        **prope_kwargs,
                    )[0]

            if use_cfg:
                pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                pred = cond_pred

            latent_chunk = scheduler.step(pred, t, latent_chunk, return_dict=False)[0]
            latents[:, :, start_idx:end_idx] = latent_chunk

        if chunk_i == 0:
            torch.cuda.synchronize()
            chunk0_latency = time.perf_counter() - _chunk0_t0

        # Phase 3: cache denoised chunk vision KV
        denoised_chunk = latents[:, :, start_idx:end_idx]
        denoised_cond = cond_input[:, :, start_idx:end_idx]
        denoised_input = torch.cat([denoised_chunk, denoised_cond], dim=1)
        ctx_ts = torch.full((chunk_latent_frames,), stabilization_level - 1,
                            device=device, dtype=torch.bfloat16)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            new_kv = model(
                bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                hidden_states=denoised_input, timestep=ctx_ts, timestep_r=None,
                mask_type="i2v", return_dict=False,
                kv_cache=kv_cache, cache_vision=True,
                rope_temporal_size=rope_total, start_rope_start_idx=start_idx,
                **prope_kwargs,
            )
            for j in range(num_layers):
                if kv_cache[j]["k_vision"] is None:
                    kv_cache[j]["k_vision"] = new_kv[j]["k_vision"]
                    kv_cache[j]["v_vision"] = new_kv[j]["v_vision"]
                else:
                    kv_cache[j]["k_vision"] = torch.cat(
                        [kv_cache[j]["k_vision"], new_kv[j]["k_vision"]], dim=2)
                    kv_cache[j]["v_vision"] = torch.cat(
                        [kv_cache[j]["v_vision"], new_kv[j]["v_vision"]], dim=2)

            if use_cfg:
                new_kv_neg = model(
                    bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                    hidden_states=denoised_input, timestep=ctx_ts, timestep_r=None,
                    mask_type="i2v", return_dict=False,
                    kv_cache=kv_cache_neg, cache_vision=True,
                    rope_temporal_size=rope_total, start_rope_start_idx=start_idx,
                    **prope_kwargs,
                )
                for j in range(num_layers):
                    if kv_cache_neg[j]["k_vision"] is None:
                        kv_cache_neg[j]["k_vision"] = new_kv_neg[j]["k_vision"]
                        kv_cache_neg[j]["v_vision"] = new_kv_neg[j]["v_vision"]
                    else:
                        kv_cache_neg[j]["k_vision"] = torch.cat(
                            [kv_cache_neg[j]["k_vision"], new_kv_neg[j]["k_vision"]], dim=2)
                        kv_cache_neg[j]["v_vision"] = torch.cat(
                            [kv_cache_neg[j]["v_vision"], new_kv_neg[j]["v_vision"]], dim=2)

    return latents, chunk0_latency


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rank, world_size, local_rank = setup_dist(args.mode)
    device = torch.device(f"cuda:{local_rank}")
    process_seed = args.seed + rank
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)

    if rank == 0:
        print(f"Mode: {args.mode}")
        print(f"Transformer: {args.transformer_dir}")
        print(f"CFG guidance_scale={args.guidance_scale}, shift={args.shift}")
        print(f"World size: {world_size}")

    # Auto-detect model_path if not provided
    model_path = args.model_path
    if model_path is None:
        model_path = find_hunyuanvideo_model_path()
        if model_path is None:
            raise RuntimeError(
                "Cannot auto-detect HunyuanVideo-1.5 model path. "
                "Please specify --model_path explicitly.")
    if rank == 0:
        print(f"Model path: {model_path}")

    # Load all examples and assign to this rank (round-robin)
    all_examples = load_examples(args.example_json)

    # Camera mode: --use_camera or --trajectory CLI arg
    camera_mode = args.use_camera or args.trajectory is not None

    if camera_mode and args.trajectory is None:
        # Camera mode reading per-sample trajectory: skip samples without trajectory
        target_examples = [(i, ex) for i, ex in enumerate(all_examples)
                           if ex.get("trajectory")]
    elif not camera_mode:
        # Ti2v mode: skip samples that have trajectory (they're camera-only)
        target_examples = [(i, ex) for i, ex in enumerate(all_examples)
                           if not ex.get("trajectory")]
        if not target_examples:
            # All samples have trajectory but we're in ti2v mode — use all, ignore trajectory
            target_examples = list(enumerate(all_examples))
    else:
        # --trajectory CLI override: apply same trajectory to all samples
        target_examples = list(enumerate(all_examples))

    # Round-robin assignment across ranks
    my_examples = [(idx, ex) for j, (idx, ex) in enumerate(target_examples)
                   if j % world_size == rank]

    if rank == 0:
        print(f"Total examples: {len(all_examples)}, assigned to rank {rank}: {len(my_examples)}")

    if not my_examples:
        print(f"[rank {rank}] No examples assigned, idle.")
        dist.destroy_process_group()
        return

    # ── Load encoders ONCE ──
    from worldfoundry.base_models.diffusion_model.models.autoencoders.hunyuan_video.h15 import (
        AutoencoderKLConv3D,
    )
    vae_path = os.path.join(model_path, "vae")
    vae = AutoencoderKLConv3D.from_pretrained(vae_path, torch_dtype=torch.float16).cpu()

    from worldfoundry.base_models.diffusion_model.models.encoders.hunyuan_video.h15_resources import (
        HunyuanVideo15Resources as HunyuanVideo_1_5_Pipeline,
    )
    text_encoder, _ = HunyuanVideo_1_5_Pipeline._load_text_encoders(model_path, device="cpu")
    vision_encoder = HunyuanVideo_1_5_Pipeline._load_vision_encoder(model_path, device="cpu")
    byt5_kwargs, _ = HunyuanVideo_1_5_Pipeline._load_byt5(model_path, True, 256, device="cpu")
    byt5_model = byt5_kwargs["byt5_model"]
    byt5_tokenizer = byt5_kwargs["byt5_tokenizer"]

    # ── Load diffusion model ONCE ──
    use_prope = camera_mode

    if use_prope:
        model = load_model_prope(args.transformer_dir, use_discrete_action=args.use_discrete_action)
    elif args.mode == "bidirectional":
        model = load_model_bidirectional(args.transformer_dir)
    else:
        model = load_model_ar_rollout(args.transformer_dir)

    if args.action_ckpt and not use_prope:
        state_dict = load_file(args.action_ckpt)
        model.load_state_dict(state_dict, strict=False)
        if rank == 0:
            print(f"Loaded action ckpt: {args.action_ckpt}")

    model = model.to(device, dtype=torch.bfloat16)
    model.eval()

    if args.mode == "bidirectional" and hasattr(model, "set_attn_mode"):
        model.set_attn_mode("flash")

    if rank == 0:
        print(f"Model loaded. Processing {len(my_examples)} samples...")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Process each assigned sample ──
    chunk0_latencies = []  # rank 0: collect from 2nd prompt onward
    for task_idx, (sample_idx, example) in enumerate(my_examples):
        trajectory = (example.get("trajectory") or args.trajectory) if camera_mode else None

        # Determine output path
        if trajectory:
            traj_safe = trajectory.replace("*", "").replace(",", "_")
            output_path = os.path.join(args.output_dir, f"sample_{sample_idx:03d}_{traj_safe}.mp4")
        else:
            output_path = os.path.join(args.output_dir, f"sample_{sample_idx:03d}.mp4")

        # Skip if already exists
        if os.path.isfile(output_path):
            print(f"[rank {rank}] Already exists, skipping: {output_path}")
            continue

        print(f"[rank {rank}] ({task_idx+1}/{len(my_examples)}) "
              f"sample {sample_idx}: {example['caption'][:60]}...")

        # Encode inputs
        text_encoder.to(device)
        vision_encoder.to(device)
        if byt5_model is not None:
            byt5_model.to(device)

        data = prepare_sample_data(
            vae, text_encoder, vision_encoder, byt5_model, byt5_tokenizer,
            example, args.height, args.width, args.video_length, device,
        )

        neg_prompts = None
        if args.guidance_scale > 1.0:
            neg_prompts = encode_negative_prompt(text_encoder, byt5_model, byt5_tokenizer, 256, device)

        # Free encoders from GPU
        text_encoder.cpu()
        vision_encoder.cpu()
        if byt5_model is not None:
            byt5_model.cpu()
        torch.cuda.empty_cache()

        # Build camera tensors if ProPE
        viewmats, Ks = None, None
        if trajectory:
            T_lat = data["latent_shape"][2]
            viewmats, Ks = make_camera_tensors(trajectory)
            if viewmats.shape[1] > T_lat:
                viewmats = viewmats[:, :T_lat]
                Ks = Ks[:, :T_lat]
            elif viewmats.shape[1] < T_lat:
                pad_n = T_lat - viewmats.shape[1]
                viewmats = torch.cat([viewmats, viewmats[:, -1:].expand(-1, pad_n, -1, -1)], dim=1)
                Ks = torch.cat([Ks, Ks[:, -1:].expand(-1, pad_n, -1, -1)], dim=1)

        # Build discrete action labels if requested
        action = None
        if args.use_discrete_action and trajectory:
            T_lat = data["latent_shape"][2]
            action = camera_trajectory_action_labels(trajectory, T_lat)

        # Run inference
        t0 = time.perf_counter()
        if args.mode == "bidirectional":
            torch.cuda.synchronize()
            _t0 = time.perf_counter()
            x = run_inference_bidirectional(
                model, data, neg_prompts, device,
                args.num_inference_steps, args.shift, args.guidance_scale,
                viewmats, Ks, action,
            )
            torch.cuda.synchronize()
            infer_lat = time.perf_counter() - _t0
            if rank == 0:
                if len(chunk0_latencies) >= 1:
                    chunk0_latencies.append(infer_lat)
                else:
                    chunk0_latencies.append(None)
        else:
            x, chunk0_lat = run_inference_rollout(
                model, data, neg_prompts, device,
                args.num_inference_steps, args.shift, args.guidance_scale,
                args.stabilization_level, args.chunk_latent_frames,
                viewmats, Ks, action,
            )
            if rank == 0:
                if len(chunk0_latencies) >= 1:
                    chunk0_latencies.append(chunk0_lat)
                else:
                    chunk0_latencies.append(None)

        elapsed = time.perf_counter() - t0
        print(f"[rank {rank}] Inference done in {elapsed:.1f}s, decoding...")

        # Decode and save
        decode_and_save(x, vae, device, output_path, args.fps)
        print(f"[rank {rank}] Saved {output_path}")

        # Clean up for next iteration
        del x, data, neg_prompts, viewmats, Ks
        torch.cuda.empty_cache()

    # All ranks done
    if rank == 0:
        valid = [v for v in chunk0_latencies[1:] if v is not None]
        if valid:
            label = "full inference" if args.mode == "bidirectional" else "chunk0"
            print(f"[timing] rank0 {label} latency (from 2nd prompt): avg={sum(valid)/len(valid):.3f}s over {len(valid)} samples")

    dist.destroy_process_group()
    if rank == 0:
        print("All done.")


if __name__ == "__main__":
    main()
