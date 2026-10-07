from __future__ import annotations

import logging
from functools import wraps
from pathlib import Path

import torch

from worldfoundry.base_models.diffusion_model.runners.kandinsky6_algorithms.denoise_loop import denoise_loop
from worldfoundry.base_models.diffusion_model.runners.kandinsky6_algorithms.mux import mux_video_audio
from worldfoundry.base_models.diffusion_model.runners.kandinsky6_algorithms.piflow_sampler import piflow_denoise_loop
from worldfoundry.base_models.diffusion_model.runners.kandinsky6_algorithms.postprocess_audio import postprocess_audio
from worldfoundry.base_models.diffusion_model.runners.kandinsky6_algorithms.postprocess_video import postprocess_video
from worldfoundry.base_models.diffusion_model.runners.kandinsky6_algorithms.prepare_latents import append_i2va_tail_condition, audio_latent_duration, encode_i2va_first_frame, prepare_audio_latents, prepare_video_latents
from worldfoundry.base_models.diffusion_model.runners.kandinsky6_algorithms.prepare_ropes import VisualRopeCache, compute_rope1d
from worldfoundry.base_models.diffusion_model.models.encoders.kandinsky6.beautifier import NullBeautifier
from worldfoundry.base_models.diffusion_model.kandinsky6_types import Kandinsky6PipelineOutput, LatentBundle
from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.cache import CacheDiT, CacheMode
from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.offload import NoOpOffload, OffloadHandle, OffloadStrategy
from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.profile import NoOpProfile
from worldfoundry.base_models.diffusion_model.recipes.kandinsky6_config import CacheConfig, MagCacheConfig, NaviCacheConfig, PipelineConfig
from worldfoundry.base_models.diffusion_model.runners.kandinsky6_progress import denoising_progress

logger = logging.getLogger("kandinsky")

_DEFAULT_NEG = (
    "Static, 2D cartoon, cartoon, 2d animation, paintings, images, "
    "worst quality, low quality, ugly, deformed, walking backwards"
)


def write_expanded_prompt(video_path: str | Path, prompt: str) -> Path:
    """Write the caption encoded for ``video_path`` next to that file."""
    text_path = Path(video_path).with_suffix(".txt")
    text_path.write_text(prompt.rstrip() + "\n", encoding="utf-8")
    logger.info("expanded prompt saved path=%s", text_path)
    return text_path


def _batch_values(value, batch_size: int, name: str) -> list:
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise ValueError(f"{name} must not be empty")
    if len(values) == 1:
        return values * batch_size
    if len(values) != batch_size:
        raise ValueError(f"{name} has batch size {len(values)}, expected {batch_size}")
    return values


def _batch_size(*values) -> int:
    sizes = [len(value) for value in values if isinstance(value, (list, tuple))]
    return max(sizes, default=1)


def _release_modules_after_call(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        try:
            return function(self, *args, **kwargs)
        finally:
            self.offload.release("text_embedder", "dit", "vae", "audio_vae", "vocoder")

    return wrapped


def _profile_call(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        # Block offload refuses a stage while autograd is enabled.
        with self.profile.pipeline(), torch.no_grad():
            return function(self, *args, **kwargs)

    return wrapped


class Kandinsky6Sampler:
    def __init__(
        self,
        dit,
        text_embedder,
        vae,
        device: torch.device,
        audio_vae=None,
        vocoder=None,
        num_steps: int = 50,
        guidance_weight: float = 5.0,
        scheduler_scale: float = 10.0,
        scale_factor: tuple = (1.0, 2.0, 2.0),
        height: int = 512,
        width: int = 768,
        latent_frames: int | None = None,
        sample_frames: int | None = None,
        cache_conf: CacheConfig | dict | None = None,
        fps: float = 24.0,
        audio_fps: int = 44100,
        offload: OffloadHandle | None = None,
        visual_cond_scheme: str = "pretrain",
        max_area: int | None = None,
        image_divisibility: int = 16,
        piflow_conf=None,
    ):
        self.dit = dit
        self.text_embedder = text_embedder
        self.vae = vae
        self.audio_vae = audio_vae
        self.vocoder = vocoder
        self.device = device
        self.num_steps = num_steps
        self.guidance_weight = guidance_weight
        self.scheduler_scale = scheduler_scale
        self.scale_factor = scale_factor
        self.height = height
        self.width = width
        self.latent_frames = latent_frames
        self.sample_frames = sample_frames
        self.fps = fps
        self.audio_fps = audio_fps
        self.visual_cond_scheme = visual_cond_scheme
        self.max_area = max_area if max_area is not None else height * width
        self.image_divisibility = image_divisibility
        self.piflow_conf = piflow_conf
        self.offload: OffloadHandle = offload if offload is not None else NoOpOffload()
        self.profile = NoOpProfile()
        if cache_conf is None:
            self.cache_conf = CacheConfig()
        elif isinstance(cache_conf, CacheConfig):
            self.cache_conf = cache_conf
        else:
            self.cache_conf = CacheConfig.model_validate(cache_conf)
        self.beautifier = NullBeautifier()
        self._visual_rope_cache = VisualRopeCache()

    @property
    def raw_dit(self):
        """Bare DiffusionTransformer3D — target for torch.export / torch.compile."""
        return self.dit.module if isinstance(self.dit, CacheDiT) else self.dit

    def _materialize_ropes(
        self,
        *,
        T_rope: int,
        H_patches: int,
        W_patches: int,
        text_len: int,
        null_text_len: int,
        audio_len: int | None,
    ):
        """Precompute RoPE tensors for DiT.forward (max-canvas visual cache + slice)."""
        dit = self.raw_dit
        scale = (
            float(self.scale_factor[0]),
            float(self.scale_factor[1]),
            float(self.scale_factor[2]),
        )
        # Prefer pipeline default canvas as max table when call is smaller.
        pT, pH, pW = dit.patch_size
        T_def = (self.latent_frames // pT) if self.latent_frames is not None else T_rope
        H_def = (self.height // 8) // pH
        W_def = (self.width // 8) // pW
        cache_shape = (max(T_rope, T_def), max(H_patches, H_def), max(W_patches, W_def))
        self._visual_rope_cache.get(dit.visual_rope, cache_shape, scale)
        visual_rope = self._visual_rope_cache.get(dit.visual_rope, (T_rope, H_patches, W_patches), scale)

        if dit.is_multimodal:
            text_rope = [
                compute_rope1d(dit.video_text_rope, text_len),
                compute_rope1d(dit.audio_text_rope, text_len),
            ]
            null_text_rope = (
                [compute_rope1d(dit.video_text_rope, null_text_len), compute_rope1d(dit.audio_text_rope, null_text_len)]
                if null_text_len > 0
                else None
            )
            audio_rope = compute_rope1d(dit.audio_rope, audio_len) if audio_len is not None else None
        else:
            text_rope = compute_rope1d(dit.text_rope, text_len)
            null_text_rope = compute_rope1d(dit.text_rope, null_text_len) if null_text_len > 0 else None
            audio_rope = None

        return visual_rope, audio_rope, text_rope, null_text_rope

    def set_cache(self, mode: CacheMode | None, mode_name: str | None = None, **params) -> None:
        """Switch MagCache / NaviCache / none without recreating the pipeline.

        Extra Mag/Navi hyperparameters fall back to values from the loaded YAML
        (`cache_conf`) when not passed explicitly.
        """
        if not isinstance(self.dit, CacheDiT):
            self.dit = CacheDiT(self.dit)

        if mode is None or mode.lower() == "none":
            self.dit.set_cache("none")
            return

        num_steps = params.pop("num_steps", self.num_steps)
        no_cfg = params.pop("no_cfg", abs(self.guidance_weight - 1.0) < 1e-6)

        if mode == "magcache":
            defaults: MagCacheConfig | None = self.cache_conf.magcache
            mag_ratios = params.pop(
                "mag_ratios",
                None if defaults is None else defaults.mag_ratios,
            )
            if mag_ratios is None:
                raise ValueError("set_cache('magcache') requires mag_ratios (or cache.magcache in config)")
            self.dit.set_cache(
                "magcache",
                num_steps=num_steps,
                no_cfg=no_cfg,
                mag_ratios=mag_ratios,
                mode_name=mode_name,
                thresh=params.pop("thresh", 0.12 if defaults is None else defaults.thresh),
                K=params.pop("K", 2 if defaults is None else defaults.K),
                retention_ratio=params.pop(
                    "retention_ratio",
                    0.2 if defaults is None else defaults.retention_ratio,
                ),
            )
            if params:
                raise TypeError(f"Unexpected MagCache params: {sorted(params)}")
            return

        if mode == "navicache":
            defaults_n: NaviCacheConfig | None = self.cache_conf.navicache
            self.dit.set_cache(
                "navicache",
                num_steps=num_steps,
                no_cfg=no_cfg,
                thresh=params.pop("thresh", 0.05 if defaults_n is None else defaults_n.thresh),
                align_steps=params.pop("align_steps", 10 if defaults_n is None else defaults_n.align_steps),
                process_noise=params.pop(
                    "process_noise",
                    0.05 if defaults_n is None else defaults_n.process_noise,
                ),
                measurement_noise=params.pop(
                    "measurement_noise",
                    0.05 if defaults_n is None else defaults_n.measurement_noise,
                ),
            )
            if params:
                raise TypeError(f"Unexpected NaviCache params: {sorted(params)}")
            return

        raise ValueError(f"Unknown cache mode: {mode!r}")


    @staticmethod
    def _generation_mode(*, do_audio: bool, image: object | None) -> str:
        """Same coeffs for t2v-t2va, i2v-i2va for now."""
        prefix = "i2v" if image is not None else "t2v"
        return f"{prefix}a" if do_audio else prefix

    def _expand_captions(self, captions, *, images, do_audio, seed):
        beautifier = self.beautifier
        if beautifier.name == "none":
            return list(captions)
        image_items = images if images is not None else [None] * len(captions)
        expanded = []
        for index, (caption, image_item) in enumerate(zip(captions, image_items, strict=True)):
            expanded.append(
                beautifier.expand(
                    caption,
                    image=image_item,
                    audio=do_audio,
                    seed=seed if index == 0 else None,
                    text_embedder=self.text_embedder,
                )
            )
        return expanded

    @_profile_call
    @_release_modules_after_call
    def __call__(
        self,
        text: str | list[str],
        height: int | None = None,
        width: int | None = None,
        time_length: int | None = None,
        latent_frames: int | None = None,
        seed: int | None = None,
        num_steps: int | None = None,
        guidance_weight: float | None = None,
        scheduler_scale: float | None = None,
        negative_text: str | list[str] = _DEFAULT_NEG,
        sample_audio: bool | None = None,
        save_path: str | Path | list[str | Path] | None = None,
        show_progress: bool = False,
        image: str | object | list[str | object] | None = None,
        visual_cond_scheme: str | None = None,
    ) -> Kandinsky6PipelineOutput:
        batch_size = _batch_size(text, negative_text, image, save_path)
        texts = _batch_values(text, batch_size, "text")
        negative_texts = _batch_values(negative_text, batch_size, "negative_text")
        images = None if image is None else _batch_values(image, batch_size, "image")
        if save_path is not None and batch_size > 1 and not isinstance(save_path, (list, tuple)):
            raise ValueError("save_path must be a list when generating a batch")
        save_paths = None if save_path is None else _batch_values(save_path, batch_size, "save_path")

        if seed is None:
            seed = torch.randint(0, 2**31, (1,)).item()
        self.profile.note_call(
            seed=seed,
            height=height,
            width=width,
            time_length=time_length,
            latent_frames=latent_frames,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
            scheduler_scale=scheduler_scale,
            sample_audio=sample_audio,
            visual_cond_scheme=visual_cond_scheme,
            image=image,
        )

        explicit_size = height is not None or width is not None
        height = self.height if height is None else height
        width = self.width if width is None else width
        num_steps = self.num_steps if num_steps is None else num_steps
        guidance_weight = self.guidance_weight if guidance_weight is None else guidance_weight
        scheduler_scale = self.scheduler_scale if scheduler_scale is None else scheduler_scale
        scheme = visual_cond_scheme or self.visual_cond_scheme

        dit = self.raw_dit
        do_audio = (
            sample_audio
            if sample_audio is not None
            else bool(getattr(dit, "is_multimodal", False) and self.audio_vae is not None and self.vocoder is not None)
        )
        if do_audio and (self.audio_vae is None or self.vocoder is None):
            raise ValueError("sample_audio=True requires audio_vae and vocoder")

        piflow_enabled = bool(getattr(self.piflow_conf, "enabled", False))
        if piflow_enabled:
            if not do_audio:
                raise ValueError("piflow.enabled requires sample_audio=True")
            if abs(guidance_weight - 1.0) > 1e-6:
                raise ValueError("piflow.enabled requires guidance_weight=1.0")

        generation_mode = self._generation_mode(do_audio=do_audio, image=images)

        if images is not None and scheme == "pretrain" and not dit.visual_cond:
            raise ValueError("image conditioning requires dit.visual_cond=True")
        if images is not None and scheme == "tail_cond_first_frame":
            if getattr(dit, "visual_token_type_num_embeddings", 0) < 2:
                raise ValueError("tail_cond_first_frame requires dit.visual_token_type_num_embeddings >= 2")

        if isinstance(self.dit, CacheDiT) and self.dit.mode != "none":
            # Keep cache step counters aligned with this generation's schedule.
            if self.dit.mode == "magcache" and self.dit._mag is not None:
                mag_steps_changed = bool(self.dit._mag["num_steps"] != num_steps * 2)
                mag_gen_mode_changed = bool(
                    self.dit._mag["gen_mode"] is not None and generation_mode != self.dit._mag["gen_mode"]
                )
                if mag_gen_mode_changed:
                    logger.warning(
                        "MagCache uses different ratios per mode for this model. Generation mode changed: %s",
                        generation_mode,
                    )
                if mag_steps_changed or mag_gen_mode_changed:
                    self.set_cache(
                        "magcache",
                        num_steps=num_steps,
                        mag_ratios=self.dit._mag["mag_ratios"],
                        mode_name=generation_mode,
                        thresh=self.dit._mag["thresh"],
                        K=self.dit._mag["K"],
                        retention_ratio=self.dit._mag["retention_ratio"],
                    )
            elif self.dit.mode == "navicache" and self.dit._navi is not None:
                if self.dit._navi["num_steps"] != num_steps:
                    self.set_cache(
                        "navicache",
                        num_steps=num_steps,
                        thresh=self.dit._navi["thresh"],
                        align_steps=self.dit._navi["align_forwards"] // 2,
                        process_noise=self.dit._navi["process_noise"],
                        measurement_noise=self.dit._navi["measurement_noise"],
                    )
            self.dit.reset_cache_state()

        raw = self.raw_dit
        if hasattr(raw, "clear_text_proj_cache"):
            raw.clear_text_proj_cache()

        # --- optional I2VA first-frame encode (may override height/width) ---
        first_frames = None
        if images is not None:
            with self.offload.use("vae", prefetch="text_embedder"):
                encoded_frames = []
                for image_item in images:
                    if explicit_size:
                        encoded, item_height, item_width = encode_i2va_first_frame(
                            image_item, self.vae, self.device, height=height, width=width
                        )
                    else:
                        encoded, item_height, item_width = encode_i2va_first_frame(
                            image_item,
                            self.vae,
                            self.device,
                            max_area=self.max_area,
                            divisibility=self.image_divisibility,
                        )
                    encoded_frames.append(encoded)
                    if len(encoded_frames) == 1:
                        height, width = item_height, item_width
                    elif (item_height, item_width) != (height, width):
                        raise ValueError("all batched images must resolve to the same height and width")
                first_frames = torch.cat(encoded_frames, dim=0)

        # Latent grid dimensions
        H_lat = height // 8
        W_lat = width // 8
        if latent_frames is not None:
            num_frames = latent_frames
        elif self.latent_frames is not None and time_length is None:
            num_frames = self.latent_frames
        else:
            tl = 5 if time_length is None else time_length
            num_frames = tl * 24 // 4 + 1  # temporal compression = 4
        H_patches = H_lat // dit.patch_size[1]
        W_patches = W_lat // dit.patch_size[2]

        decode_names = ("vae", "audio_vae", "vocoder") if do_audio else ("vae",)

        with self.offload.use("text_embedder", prefetch="dit"):
            # Qwen stays on CPU until this stage. Expanding outside it runs the
            # beautifier on the host.
            captions = self._expand_captions(
                texts,
                images=images,
                do_audio=do_audio,
                seed=seed,
            )
            self.profile.note_prompt(captions[0] if len(captions) == 1 else list(captions))
            text_embeds, text_cu_seqlens, attention_mask = self.text_embedder.encode(captions)
            # NOTE: piflow use implicit CFG, no need for null entities
            if piflow_enabled:
                null_embeds, null_cu_seqlens, null_attention_mask = {}, None, None
            else:
                null_embeds, null_cu_seqlens, null_attention_mask = self.text_embedder.encode(negative_texts)

        # Move embeddings / ropes to compute device for DiT (bf16 — no autocast).
        text_embeds = {k: v.to(device=self.device, dtype=torch.bfloat16) for k, v in text_embeds.items()}
        null_embeds = {k: v.to(device=self.device, dtype=torch.bfloat16) for k, v in null_embeds.items()}
        text_cu_seqlens = text_cu_seqlens.to(self.device)
        if null_cu_seqlens is not None:
            null_cu_seqlens = null_cu_seqlens.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if null_attention_mask is not None:
            null_attention_mask = null_attention_mask.to(self.device)

        # --- latents ---
        video_duration = num_frames
        bundle = prepare_video_latents(
            bs=batch_size,
            duration=num_frames,
            H_lat=H_lat,
            W_lat=W_lat,
            C=dit.in_visual_dim,
            seed=seed,
            device=self.device,
        )
        bundle = LatentBundle(
            video=bundle.video.reshape(batch_size, num_frames, H_lat, W_lat, dit.in_visual_dim),
            audio=bundle.audio,
            video_cu_seqlens=bundle.video_cu_seqlens,
            audio_cu_seqlens=bundle.audio_cu_seqlens,
        )

        visual_token_type_ids = None
        generated_visual_mask = None
        if first_frames is not None and scheme == "tail_cond_first_frame":
            video, visual_token_type_ids, generated_visual_mask = append_i2va_tail_condition(
                bundle.video, first_frames, batch_size=batch_size, video_duration=video_duration
            )
            duration_with_ref = video_duration + 1
            cu = duration_with_ref * torch.arange(batch_size + 1, dtype=torch.int32, device=self.device)
            bundle = LatentBundle(
                video=video,
                audio=None,
                video_cu_seqlens=cu,
                audio_cu_seqlens=None,
            )
            num_frames = duration_with_ref
        elif first_frames is not None and scheme in ("pretrain", "i2v"):
            pass  # first_frames injected inside denoise_loop via visual_cond_scheme

        audio_len = None
        if do_audio:
            downsample = int(getattr(self.audio_vae, "downsample_factor", 1024))
            # Audio length follows generated frames only (exclude tail reference).
            audio_dur = audio_latent_duration(
                video_duration,
                fps=self.fps,
                audio_fps=self.audio_fps,
                downsample_factor=downsample,
            )
            bundle = prepare_audio_latents(
                bundle,
                audio_duration=audio_dur,
                audio_dim=dit.in_audio_dim,
                seed=seed,
                device=self.device,
            )
            bundle = LatentBundle(
                video=bundle.video,
                audio=bundle.audio.reshape(batch_size, audio_dur, dit.in_audio_dim),
                video_cu_seqlens=bundle.video_cu_seqlens,
                audio_cu_seqlens=bundle.audio_cu_seqlens,
            )
            audio_len = int(audio_dur)

        # --- denoise ---
        progress_bar = denoising_progress(num_steps) if show_progress else None

        try:
            with self.offload.use("dit", prefetch=decode_names):
                # RoPE: generated frames use 0..T-1; tail reference reuses position 0 (K5).
                T_rope = video_duration // dit.patch_size[0]
                visual_rope, audio_rope, text_rope, null_text_rope = self._materialize_ropes(
                    T_rope=T_rope,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    text_len=(
                        text_embeds["text_embeds"].shape[1]
                        if text_embeds["text_embeds"].ndim == 3
                        else int(text_cu_seqlens[-1].item())
                    ),
                    null_text_len=(
                        0
                        if null_cu_seqlens is None
                        else (
                            null_embeds["text_embeds"].shape[1]
                            if null_embeds["text_embeds"].ndim == 3
                            else int(null_cu_seqlens[-1].item())
                        )
                    ),
                    audio_len=audio_len,
                )
                if scheme == "tail_cond_first_frame" and images is not None:
                    visual_rope = torch.cat([visual_rope, visual_rope[:1]], dim=0)

                if piflow_enabled:
                    result_bundle = piflow_denoise_loop(
                        bundle=bundle,
                        dit=self.dit,
                        text_embeds=text_embeds,
                        visual_rope=visual_rope,
                        audio_rope=audio_rope,
                        text_rope=text_rope,
                        num_steps=num_steps,
                        scheduler_scale=scheduler_scale,
                        first_frames=first_frames,
                        visual_cond_scheme=scheme,
                        sample_audio=do_audio,
                        attention_mask=attention_mask,
                        visual_token_type_ids=visual_token_type_ids,
                        scale_factor=self.scale_factor,
                        eps=self.piflow_conf.eps,
                        final_step_size_scale=self.piflow_conf.final_step_size_scale,
                        shift=self.piflow_conf.shift,
                        num_policy_substeps=self.piflow_conf.num_policy_substeps,
                        progress_callback=progress_bar.update if progress_bar is not None else None,
                    )
                else:
                    result_bundle = denoise_loop(
                        bundle=bundle,
                        dit=self.dit,
                        text_embeds=text_embeds,
                        null_text_embeds=null_embeds,
                        visual_rope=visual_rope,
                        audio_rope=audio_rope,
                        text_rope=text_rope,
                        null_text_rope=null_text_rope,
                        num_steps=num_steps,
                        guidance_weight=guidance_weight,
                        scheduler_scale=scheduler_scale,
                        first_frames=first_frames,
                        visual_cond_scheme=scheme,
                        sample_audio=do_audio,
                        attention_mask=attention_mask,
                        null_attention_mask=null_attention_mask,
                        visual_token_type_ids=visual_token_type_ids,
                        scale_factor=self.scale_factor,
                        progress_callback=progress_bar.update if progress_bar is not None else None,
                    )
        finally:
            if progress_bar is not None:
                progress_bar.close()

        if generated_visual_mask is not None and result_bundle.video is not None:
            generated_video = (
                result_bundle.video[:, generated_visual_mask[0]]
                if result_bundle.video.ndim == 5
                else result_bundle.video[generated_visual_mask]
            )
            result_bundle = LatentBundle(
                video=generated_video,
                audio=result_bundle.audio,
                video_cu_seqlens=video_duration * torch.arange(batch_size + 1, dtype=torch.int32, device=self.device),
                audio_cu_seqlens=result_bundle.audio_cu_seqlens,
            )

        # --- decode ---
        with self.offload.use(*decode_names):
            frames = postprocess_video(result_bundle, self.vae, bs=batch_size)
            audio = postprocess_audio(result_bundle, self.audio_vae, self.vocoder) if do_audio else None

        path = None
        if save_paths is not None:
            paths = []
            for index, output_path in enumerate(save_paths):
                saved = mux_video_audio(
                    frames[index],
                    None if audio is None else audio[index],
                    output_path,
                    fps=int(self.fps),
                    audio_sample_rate=self.audio_fps,
                )
                paths.append(str(saved))
                write_expanded_prompt(saved, captions[index])
            path = paths[0] if batch_size == 1 else paths

        return Kandinsky6PipelineOutput(frames=frames, audio=audio, path=path, prompts=list(captions), latents=result_bundle)
