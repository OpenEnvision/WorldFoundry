"""Single-device, sequentially offloaded JING-Flash inference.

The transformer and Qwen conditioner reuse WorldFoundry's H3 implementation;
Diffusers supplies its existing H3 codecs. No SGLang service or training code.
"""
from __future__ import annotations

import gc
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.loading import _shard_paths
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.jing.loading import load_transformer
from worldfoundry.base_models.diffusion_model.schedulers.minimax_h3 import (
    minimax_h3_euler_eta0_step, minimax_h3_rf_v_to_x0,
)
from worldfoundry.pipelines.minimax._minimax_h3.packed_tokens import minimax_h3_unpatchify_video_tokens
from .packing import audio_endpoint, chunk_lengths, expand_slices, pack


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def checkpoint_paths(checkpoint, base):
    checkpoint, base = Path(checkpoint).expanduser().resolve(), Path(base).expanduser().resolve()
    if (checkpoint / "jing_flash_v1").is_dir():
        checkpoint = checkpoint / "jing_flash_v1"
    for folder in (checkpoint, *(base / name for name in ("text_encoder", "vae", "audio_vae"))):
        if not (folder / "config.json").is_file():
            raise FileNotFoundError(folder / "config.json")
        for shard in _shard_paths(folder):
            if not shard.is_file():
                raise FileNotFoundError(shard)
    for name in ("tokenizer/tokenizer_config.json", "processor/preprocessor_config.json",
                 "scheduler/scheduler_config.json", "audio_scheduler/scheduler_config.json"):
        if not (base / name).is_file():
            raise FileNotFoundError(base / name)
    return checkpoint, base


def sigma_schedule(shift, device):
    if not np.isfinite(shift) or shift <= 0:
        raise ValueError("H3 scheduler shift must be finite and positive")
    base = torch.tensor([1., .7, .4, .15, 0.], dtype=torch.float32, device=device)
    return shift * base / (1 + (shift - 1) * base)


def denoise(model, packed, *, video_shift, audio_shift, reference_timestep=0., callback=None):
    from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.jing.model import attention_groups

    groups = attention_groups(packed.owners, packed.is_media)
    vs, aus = (sigma_schedule(x, packed.video.device) for x in (video_shift, audio_shift))
    start = packed.output_video_offset
    for index in range(4):
        vt, at = 1 - vs[index], 1 - aus[index]
        times, indices = packed.timesteps(vt, at, reference_timestep)
        vp, ap = model(packed, times, indices, groups)
        if not torch.isfinite(vp).all() or not torch.isfinite(ap).all():
            raise FloatingPointError(f"Non-finite velocity at JING step {index}")
        for state, prediction, t, sigmas in (
            (packed.video[:, start:], vp[:, start:], vt, vs), (packed.audio, ap, at, aus),
        ):
            x0 = minimax_h3_rf_v_to_x0(state.float(), prediction.float(), t)
            state.copy_(minimax_h3_euler_eta0_step(
                state.float(), x0, sigma_curr=float(sigmas[index]), sigma_next=float(sigmas[index + 1])))
        if callback is not None:
            callback(index, packed)
    return packed.video[0, start:].cpu(), packed.audio_channel_major().cpu()


def _codec_classes():
    try:
        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
    except ImportError as exc:
        raise ImportError(
            "XGEN-JING requires Diffusers >= 0.40.0 with the MiniMax H3 video and audio codecs"
        ) from exc
    return AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio


def _load_codec(base, name, device):
    AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio = _codec_classes()
    cls = AutoencoderKLMiniMaxH3 if name == "vae" else AutoencoderKLMiniMaxH3Audio
    model, info = cls.from_pretrained(base / name, torch_dtype=torch.float32,
                                      local_files_only=True, output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete H3 {name} checkpoint: {info}")
    if name == "vae":
        model.enable_tiling()
    return model.eval().requires_grad_(False).to(device)


class JINGRuntime:
    def __init__(self, checkpoint, base, *, device="cuda", cpu_offload=True):
        self.checkpoint, self.base = checkpoint_paths(checkpoint, base)
        self.device = torch.device(device)
        self.cpu_offload = bool(cpu_offload)
        self.architecture = json.loads((self.checkpoint / "config.json").read_text())

    @torch.inference_mode()
    def generate(self, *, prompt="", reference_images=(), chunks=None, slices=None, control="",
                 height=544, width=960, num_frames=None, first_chunk_size=2, seed=42,
                 reference_timestep=0., text_encoder_reference_images=True, output_path,
                 save_latents=False):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        from transformers import AutoProcessor, AutoTokenizer
        from worldfoundry.base_models.diffusion_model.models.encoders.minimax_h3_qwen3vl import MiniMaxH3Qwen3VLEncoder
        from worldfoundry.core.media.codecs.video import save_video_h264
        from worldfoundry.core.media.codecs.audio import write_audio, mux_audio_video

        for name, value in (("height", height), ("width", width)):
            if type(value) is not int or value <= 0 or value % 32:
                raise ValueError(f"{name} must be a positive multiple of 32")
        if type(first_chunk_size) is not int or first_chunk_size not in (0, 2):
            raise ValueError("first_chunk_size must be 0 or 2")
        if not np.isfinite(reference_timestep) or not 0 <= reference_timestep <= 1:
            raise ValueError("reference_timestep must be in [0, 1]")
        if len(reference_images) > 5:
            raise ValueError("JING accepts at most five ordered reference images")
        case = {"prompt": prompt}
        if chunks is not None:
            case["chunks"] = chunks
        if slices is not None:
            case["slices"] = slices
        if control:
            case["control"] = control
        if num_frames is None:
            count = sum(c.get("repeat", 1) for c in chunks) if chunks else len(slices) if slices else 7
            num_frames = 17 * count + 5 if first_chunk_size == 2 else 17 * (count - 1) + 5
        if type(num_frames) is not int or num_frames < 22:
            raise ValueError("num_frames must be an integer >= 22")
        _codec_classes()
        frames, lengths = chunk_lengths(num_frames, first_chunk_size)
        expanded = expand_slices(case, len(lengths))
        images = []
        for index, value in enumerate(reference_images):
            image = value.copy() if isinstance(value, Image.Image) else Image.open(value)
            image = ImageOps.exif_transpose(image).convert("RGB")
            if image.size != (width, height):
                if index == 0:
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                else:
                    scale = max(width / image.width, height / image.height)
                    size = max(width, round(image.width * scale)), max(height, round(image.height * scale))
                    left, top = (size[0] - width) // 2, (size[1] - height) // 2
                    image = image.resize(size, Image.Resampling.LANCZOS).crop((left, top, left + width, top + height))
            images.append(image)
        tokenizer = AutoTokenizer.from_pretrained(self.base / "tokenizer", local_files_only=True)
        processor = AutoProcessor.from_pretrained(self.base / "processor", local_files_only=True)
        encoder = MiniMaxH3Qwen3VLEncoder.from_pretrained(str(self.base / "text_encoder"), local_files_only=True,
                                                       strict=True, preserve_rotary_precision=True, low_cpu_mem_usage=True)
        encoder = encoder.eval().requires_grad_(False).to(self.device)
        text_loading = encoder.loading_report
        bank = {}
        for text in dict.fromkeys(s["prompt"] for s in expanded):
            hidden, tags = encoder.encode_presentation(text, tokenizer=tokenizer, processor=processor,
                                                       images=images if text_encoder_reference_images else ())
            if not torch.isfinite(hidden).all():
                raise FloatingPointError("Non-finite Qwen conditioning")
            bank[text] = hidden.cpu(), tags.cpu()
        del encoder
        release_memory()
        references = []
        pixel_mean, pixel_std = (.485, .456, .406), (.229, .224, .225)
        if images:
            vae = _load_codec(self.base, "vae", self.device)
            mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
            for image in images:
                pixels = torch.from_numpy(np.array(image)).to(self.device).permute(2, 0, 1)[None, :, None].float() / 255
                pixels = (pixels - pixels.new_tensor(pixel_mean)[None, :, None, None, None]) / pixels.new_tensor(pixel_std)[None, :, None, None, None]
                moments = vae._encode_clip(pixels)
                latent = DiagonalGaussianDistribution(moments).sample(generator=torch.Generator().manual_seed(42))
                references.append(((latent.half().float().cpu() - mean) / std)[0])
            del vae
            release_memory()
        channels, audio_channels = self.architecture["in_channels"], self.architecture["audio_in_channels"]
        payload = dict(lengths=lengths, slices=expanded, text_bank=bank, references=references,
                       video=torch.randn(sum(lengths), channels, height // 16, width // 16,
                                         generator=torch.Generator().manual_seed(seed), dtype=torch.float32).permute(1, 0, 2, 3).contiguous(),
                       audio=torch.randn(audio_endpoint(sum(lengths)), 2, audio_channels,
                                         generator=torch.Generator().manual_seed(seed + 1), dtype=torch.float32).permute(1, 0, 2).contiguous())
        packed = pack(payload, self.architecture, {"reserved_slots": 15}, self.device)
        model = load_transformer(self.checkpoint)
        loading = model.loading_report
        if self.cpu_offload and self.device.type == "cuda":
            from accelerate import cpu_offload
            cpu_offload(model, execution_device=self.device, offload_buffers=True)
        else:
            model.to(self.device)
        shifts = [float(json.loads((self.base / folder / "scheduler_config.json").read_text())["shift"])
                  for folder in ("scheduler", "audio_scheduler")]
        video_rows, audio_rows = denoise(model, packed, video_shift=shifts[0], audio_shift=shifts[1],
                                         reference_timestep=reference_timestep)
        del model, packed
        release_memory()
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if save_latents:
            from safetensors.torch import save_file
            save_file({"video": video_rows.contiguous(), "audio": audio_rows.contiguous()}, str(target.with_suffix(".safetensors")))
        patch = self.architecture["patch_size"]
        latent = minimax_h3_unpatchify_video_tokens(video_rows, latent_shape=(sum(lengths), height // 16 // patch[1], width // 16 // patch[2], channels), patch_size=patch).to(self.device)
        vae = _load_codec(self.base, "vae", self.device)
        mean = latent.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = latent.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            raw_video = vae.decode(latent * std + mean, return_dict=False)[0].float()
        if not torch.isfinite(raw_video).all():
            raise FloatingPointError("Non-finite decoded video before clamp")
        video = (raw_video * raw_video.new_tensor(pixel_std)[None, :, None, None, None]
                 + raw_video.new_tensor(pixel_mean)[None, :, None, None, None]).clamp(0, 1)[0].permute(1, 2, 3, 0).cpu()
        if tuple(video.shape) != (frames, height, width, 3):
            raise ValueError(f"Unexpected decoded video shape: {tuple(video.shape)}")
        del vae, raw_video, latent
        release_memory()
        audio_vae = _load_codec(self.base, "audio_vae", self.device)
        latent = audio_rows.reshape(2, -1, audio_channels).transpose(1, 2).to(self.device)
        mean = latent.new_tensor(audio_vae.config.latents_mean).view(1, -1, 1)
        std = latent.new_tensor(audio_vae.config.latents_std).view(1, -1, 1)
        audio = audio_vae.decode(latent * std + mean, return_dict=False)[0].float()[:, 0].cpu()
        if not torch.isfinite(audio).all():
            raise FloatingPointError("Non-finite decoded audio")
        sample_rate = int(audio_vae.config.sampling_rate)
        del audio_vae, latent
        release_memory()
        with tempfile.TemporaryDirectory() as directory:
            silent, wav = Path(directory) / "video.mp4", Path(directory) / "audio.wav"
            save_video_h264(video, silent, fps=24)
            write_audio(audio, wav, sample_rate=sample_rate)
            mux_audio_video(silent, wav, output_path=target, shortest=False)
        result = dict(status="success", model_id="xgen-jing", artifact_path=str(target),
                      num_frames=frames, height=height, width=width, fps=24, sample_rate=sample_rate,
                      audio_samples=audio.shape[-1], seed=seed, slices=expanded, slice_lengths=lengths,
                      loading=loading, text_loading=text_loading, inference_steps=4, reference_images=len(images))
        target.with_suffix(".settings.json").write_text(json.dumps(result, indent=2))
        return result
