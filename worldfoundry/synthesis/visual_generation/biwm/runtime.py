"""Strict local checkpoint loading and shared Wan component orchestration."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.biwm import BiWMWanModel
from .camera import ACTION_TEXTS, action_labels
from .sampling import sample_stage1, sample_stage2


def checkpoint_files(generator_ckpt, wan_base):
    checkpoint = Path(generator_ckpt).expanduser().resolve()
    if checkpoint.is_dir():
        checkpoint /= "diffusion_pytorch_model.safetensors"
    base = Path(wan_base).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix != ".safetensors":
        raise FileNotFoundError(f"BiWM requires a trained generator .safetensors checkpoint: {checkpoint}")
    if checkpoint.parent == base:
        raise ValueError("Use a trained BiWM generator checkpoint, separate from the Wan base directory")
    config = json.loads((base / "config.json").read_text())
    channels = config.get("in_dim")
    if channels not in (16, 48) or config.get("out_dim") != channels:
        raise ValueError("BiWM supports Wan2.1 T2V (16 channels) and Wan2.2 TI2V (48 channels)")
    version = "2.1" if channels == 16 else "2.2"
    required = [f"Wan{version}_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl"]
    missing = [p for p in required if not (base / p).exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete native Wan base directory {base}: {missing}")
    return checkpoint, base, config, version


def load_transformer(checkpoint, config, *, device):
    state = load_file(str(checkpoint), device="cpu")
    defaults = dict(patch_size=(1, 2, 2), text_dim=4096, eps=1e-6, text_len=512,
                    cross_attn_norm=True)
    keys = {*defaults, "dim", "ffn_dim", "freq_dim", "in_dim", "out_dim", "num_heads", "num_layers"}
    defaults.update({k: v for k, v in config.items() if k in keys})
    if config.get("qk_norm", True) is not True or tuple(config.get("window_size", (-1, -1))) != (-1, -1):
        raise ValueError("BiWM public checkpoints require QK norm and bidirectional global attention")
    defaults["action_embedder"] = any(k.startswith("hycam_action_embedder.") for k in state)
    with torch.device("meta"):
        model = BiWMWanModel(**defaults)
    # Native upstream keys are identical to canonical Wan; no second converter.
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model = model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    # The base constructs these nonpersistent lookup tables in the meta context.
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import precompute_freqs_cis_3d
    model.freqs = precompute_freqs_cis_3d(model.dim // defaults["num_heads"])
    return model


class BiWMRuntime:
    def __init__(self, generator_ckpt, wan_base, *, device="cuda"):
        checkpoint, base, config, version = checkpoint_files(generator_ckpt, wan_base)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Full BiWM video inference requires a CUDA device")
        self.dtype = torch.bfloat16
        self.version = version
        self.spatial_stride = 8 if version == "2.1" else 16
        self.model = load_transformer(checkpoint, config, device=self.device)
        from worldfoundry.base_models.diffusion_model.models.encoders.wan.reference import T5EncoderModel
        self.text_encoder = T5EncoderModel(
            text_len=self.model.text_len, dtype=self.dtype, device="cpu",
            checkpoint_path=str(base / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(base / "google/umt5-xxl"), load_with_mmap=True)
        if version == "2.1":
            from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_21 import WanVAE
            vae_class = WanVAE
        else:
            from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_22 import Wan2_2_VAE
            vae_class = Wan2_2_VAE
        self.vae = vae_class(vae_pth=str(base / f"Wan{version}_VAE.pth"),
                             dtype=self.dtype, device=self.device)
        self.text_encoder.to(self.device)
        try:
            with torch.no_grad():
                # Bound T5 activation memory for the 81 variable-length texts.
                camera = [self.text_encoder([text], self.device)[0].to(self.dtype) for text in ACTION_TEXTS]
                self.model.set_camera_text(camera)
        finally:
            self.text_encoder.to("cpu")

    @torch.no_grad()
    def generate(self, *, prompt, output_path, stage=2, image=None, actions=None,
                 action_label=None, height=480, width=832, num_chunks=4,
                 chunk_size=4, max_chunks=5, sink_chunks=1, num_frames=77,
                 num_inference_steps=None, guidance_scale=None, negative_prompt=None,
                 sigma_shift=5., sigmas=(1., .75, .5, .25), seed=42, fps=24.):
        from worldfoundry.core.media.codecs.video import save_video_h264
        from worldfoundry.core.media.processing.image_utils import load_pil_image
        if stage not in (1, 2) or (stage == 1 and image is not None):
            raise ValueError("Stage 1 supports T2V; stage 2 supports T2V and I2V")
        multiple = self.spatial_stride * 2
        if height < multiple or width < multiple or height % multiple or width % multiple:
            raise ValueError(f"height and width must be positive multiples of {multiple}")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        if stage == 1 and (num_frames < 1 or (num_frames - 1) % 4):
            raise ValueError("Stage-1 num_frames must be 4k+1")
        if stage == 2:
            if num_inference_steps not in (None, len(sigmas)) or guidance_scale not in (None, 1.):
                raise ValueError("Stage 2 uses the supplied distilled sigma schedule without CFG")
            if negative_prompt not in (None, ""):
                raise ValueError("Stage 2 does not use a negative prompt")
        else:
            num_inference_steps = 50 if num_inference_steps is None else num_inference_steps
            guidance_scale = 5. if guidance_scale is None else guidance_scale
            if negative_prompt is None:
                from worldfoundry.pipelines.wan.pipeline_wan_2p1_t2v import WAN21_NEGATIVE_PROMPT
                negative_prompt = WAN21_NEGATIVE_PROMPT
        total = (num_frames - 1) // 4 + 1 if stage == 1 else int(image is not None) + num_chunks * chunk_size
        labels = action_labels(total, actions=actions, action_label=action_label, seed=seed).to(self.device)
        self.text_encoder.to(self.device)
        try:
            context = self.text_encoder([prompt], self.device)[0].unsqueeze(0).to(self.dtype)
            negative = (self.text_encoder([negative_prompt], self.device)[0].unsqueeze(0).to(self.dtype)
                        if stage == 1 else None)
        finally:
            self.text_encoder.to("cpu")
        prefix = None
        with torch.autocast("cuda", dtype=self.dtype):
            if image is not None:
                import numpy as np
                pixels = torch.from_numpy(np.array(load_pil_image(image).convert("RGB"), copy=True))
                pixels = pixels.permute(2, 0, 1).unsqueeze(0).float() / 255
                pixels = torch.nn.functional.interpolate(pixels, (height, width), mode="bicubic",
                                                        align_corners=False, antialias=True)
                pixels = (pixels[0] * 2 - 1).unsqueeze(1).to(self.device, self.dtype)
                prefix = self.vae.encode([pixels])[0].unsqueeze(0).to(self.dtype)
            shape = (self.model.in_dim, height // self.spatial_stride, width // self.spatial_stride)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            if stage == 2:
                latents = sample_stage2(self.model, context, labels, latent_shape=shape,
                    num_chunks=num_chunks, chunk_size=chunk_size, max_chunks=max_chunks,
                    sink_chunks=sink_chunks, prefix=prefix, sigmas=sigmas,
                    sigma_shift=sigma_shift, generator=generator)
            else:
                latents = sample_stage1(self.model, context, labels, latent_shape=shape,
                    num_latent_frames=total, negative_context=negative,
                    num_steps=num_inference_steps, guidance_scale=guidance_scale,
                    sigma_shift=sigma_shift, generator=generator)
            if not torch.isfinite(latents).all():
                raise RuntimeError("BiWM generated non-finite latents")
            # One decode maintains the causal cache across every chunk boundary.
            raw = self.vae.model.decode(latents, self.vae.scale).float()
        if not torch.isfinite(raw).all():
            raise RuntimeError("BiWM VAE produced non-finite pixels")
        stats = {"min": raw.min().item(), "max": raw.max().item(), "std": raw.std().item()}
        frames = ((raw[0].clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        save_video_h264(frames, output_path, fps=fps)
        return dict(video_path=str(Path(output_path).resolve()), num_frames=len(frames),
                    latent_frames=total, stage=stage, wan_version=self.version,
                    raw_decode_stats=stats, action_labels=labels[0].tolist())
