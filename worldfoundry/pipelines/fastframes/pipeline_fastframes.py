# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
# Adapted from apple-aiml-research/ml-fastframes, revision
# a65c59240e01cf1de86197fa771ca09b38cf592a (Apple Sample Code License).
"""WorldFoundry pipeline for FastFrames-S (1.3B) and FastFrames-L (14B) T2V inference."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar

import torch

from worldfoundry.base_models.diffusion_model.models.denoisers.fastframes import (
    load_fastframes_components,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.fastframes import (
    BridgedPixelDecoder,
    FastFramesCausalWanModel,
    FastFramesFastOps,
    KVRefinementNetwork,
    PixelDecoder,
    align_frames,
    allocate_fastframes_kv_cache,
)
from worldfoundry.base_models.diffusion_model.schedulers.fastframes import (
    FASTFRAMES_CONTEXT_MODES,
    FASTFRAMES_NOISE_MODES,
    FF_S,
    FastFramesFlowMatchScheduler,
    FastFramesVariantConfig,
    resolve_fastframes_variant,
)
from worldfoundry.core.media.codecs.video import save_image_or_video_tensor
from worldfoundry.core.media.processing.postprocess import VideoChunk
from worldfoundry.operators.fastframes_operator import FastFramesOperator

from ..pipeline_utils import PipelineABC

__all__ = [
    "FastFramesPipeline",
    "configure_fastframes_backends",
    "pixels_to_uint8",
]


def configure_fastframes_backends(config: FastFramesVariantConfig) -> None:
    """Apply the CUDA matmul/cuDNN TF32 settings that a FastFrames recipe was evaluated with."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(config.allow_tf32)
        torch.backends.cudnn.allow_tf32 = True


def pixels_to_uint8(pixels: torch.Tensor) -> torch.Tensor:
    """Convert ``[T, 3, H, W]`` pixels in ``[-1, 1]`` to ``[T, H, W, 3]`` uint8."""
    return ((pixels.float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()


class FastFramesPipeline(PipelineABC):
    """Interactive-latency two-step causal video diffusion pipeline (FastFrames-S / FastFrames-L).

    Each latent frame is denoised by ``FastFramesCausalWanModel`` attending to the
    persistent KV cache of earlier frames. After frame 0 receives a clean ``t = 0``
    DiT pass, each subsequent finished frame repairs its cache slot via
    ``KVRefinementNetwork`` (``context="refiner"``) or a clean ``t = 0`` DiT pass
    (``context="dit"``), and ``PixelDecoder`` / ``BridgedPixelDecoder`` decodes
    the pre-head hidden states directly to RGB frames without a VAE decoder.
    """

    MODEL_ID: ClassVar[str | None] = "fastframes"
    MODEL_PATH_OPTION: ClassVar[str] = "checkpoint_dir"
    GENERATION_TYPE: ClassVar[str] = "t2v"
    DEFAULT_FPS: ClassVar[int] = 16
    DEFAULT_HEIGHT: ClassVar[int] = 480
    DEFAULT_WIDTH: ClassVar[int] = 832
    DEFAULT_NUM_FRAMES: ClassVar[int] = 81

    def __init__(
        self,
        *,
        config: FastFramesVariantConfig | None = None,
        dit: FastFramesCausalWanModel | None = None,
        refiner: KVRefinementNetwork | None = None,
        decoder: PixelDecoder | BridgedPixelDecoder | None = None,
        text_encoder: Any = None,
        operator: FastFramesOperator | None = None,
        device: str | torch.device | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_config = resolve_fastframes_variant(config if config is not None else FF_S)
        resolved_device = (
            torch.device(device)
            if device is not None
            else (dit.patch_embedding.weight.device if dit is not None else torch.device("cpu"))
        )
        super().__init__(
            model_id=model_id or self.MODEL_ID,
            operator=operator or FastFramesOperator(default_variant=resolved_config),
            device=str(resolved_device),
            **kwargs,
        )
        self.config = resolved_config
        self.variant = resolved_config.variant_id
        self.dit = dit
        self.refiner = refiner
        self.decoder = decoder
        self.text_encoder = text_encoder
        self._device = resolved_device
        self.scheduler = FastFramesFlowMatchScheduler(shift=5.0)
        self._fast_cache: dict[tuple[int, int, int], FastFramesFastOps] = {}
        self._time_cache: dict[
            tuple[float, torch.dtype, torch.device], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._refine_embed_cache: dict[tuple[float, torch.dtype, torch.device], torch.Tensor] = {}

    @property
    def torch_device(self) -> torch.device:
        if self.dit is not None:
            return self.dit.patch_embedding.weight.device
        return self._device

    @property
    def _compute_dtype(self) -> torch.dtype:
        if self.dit is not None:
            return self.dit.patch_embedding.weight.dtype
        return torch.bfloat16

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        variant: str | FastFramesVariantConfig | None = None,
        **kwargs: Any,
    ) -> FastFramesPipeline:
        """Load FastFrames-S or FastFrames-L from a checkpoint directory or explicit paths."""
        options = cls._runtime_options(
            dict(model_path) if isinstance(model_path, Mapping) else model_path,
            None if required_components is None else dict(required_components),
            dict(kwargs),
        )
        cls._strip_framework_loading_options(options)

        requested_variant = variant if variant is not None else options.pop("variant", None)
        if requested_variant is None and model_id is not None and model_id != cls.MODEL_ID:
            requested_variant = model_id
        variant_cfg = resolve_fastframes_variant(requested_variant)

        checkpoint_overrides = dict(options.pop("checkpoint_overrides", {}) or {})
        checkpoint_dir = options.pop("checkpoint_dir", options.pop("model_path", None))
        wan_dir = options.pop(
            "wan_dir",
            checkpoint_overrides.get("wan_dir") or checkpoint_overrides.get("text-encoder"),
        )
        dit_path = options.pop("dit_path", checkpoint_overrides.get("dit"))
        refiner_path = options.pop("refiner_path", checkpoint_overrides.get("refiner"))
        decoder_path = options.pop("decoder_path", checkpoint_overrides.get("decoder"))
        dit_dtype = options.pop("dit_dtype", torch.bfloat16)
        decoder_dtype = options.pop("decoder_dtype", None)
        load_text_encoder = bool(options.pop("load_text_encoder", True))
        wan_config = options.pop("wan_config", None)
        stem_n_heads = options.pop("stem_n_heads", None)
        apply_backends = bool(options.pop("configure_backends", True))

        if apply_backends:
            configure_fastframes_backends(variant_cfg)

        components = load_fastframes_components(
            variant=variant_cfg,
            checkpoint_dir=checkpoint_dir,
            wan_dir=wan_dir,
            dit_path=dit_path,
            refiner_path=refiner_path,
            decoder_path=decoder_path,
            wan_config=wan_config,
            device=device,
            dit_dtype=dit_dtype,
            decoder_dtype=decoder_dtype,
            load_text_encoder=load_text_encoder,
            stem_n_heads=stem_n_heads,
        )
        return cls(
            config=components.config,
            dit=components.dit,
            refiner=components.refiner,
            decoder=components.decoder,
            text_encoder=components.text_encoder,
            device=device,
            model_id=model_id or cls.MODEL_ID,
        )

    def _autocast_context(self) -> contextlib.AbstractContextManager[Any]:
        if self.config.autocast and self.torch_device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
        return contextlib.nullcontext()

    def _get_fast_ops(self, latent_h: int, latent_w: int, num_latent_frames: int) -> FastFramesFastOps | None:
        if not self.config.fast_ops or self.dit is None:
            return None
        key = (latent_h, latent_w, num_latent_frames)
        if key not in self._fast_cache:
            self._fast_cache[key] = FastFramesFastOps(self.dit, latent_h, latent_w, num_latent_frames)
        return self._fast_cache[key]

    @torch.inference_mode()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode one prompt string into padded UMT5-XXL embeddings ``[1, L, C]``."""
        if self.text_encoder is None:
            raise RuntimeError("FastFrames text_encoder is not loaded; pass prompt_embeds or load_text_encoder=True.")
        with self._autocast_context():
            encoded = self.text_encoder([prompt])
        embeds = encoded["prompt_embeds"] if isinstance(encoded, Mapping) else encoded
        return embeds.to(device=self.torch_device)

    def _noise_schedule(
        self,
        seed: int,
        *,
        num_latent_frames: int,
        latent_size: tuple[int, int],
        noise_mode: str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, Callable[[int, int], torch.Tensor]]:
        if self.dit is None:
            raise RuntimeError("FastFrames DiT is not loaded.")
        device = self.torch_device
        channels = self.dit.config.in_dim
        shape = (1, num_latent_frames, channels, *latent_size)
        generator = torch.Generator(device=device).manual_seed(int(seed))
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        if noise_mode == "pregenerated":
            transition = torch.randn((3, *shape), generator=generator, device=device, dtype=dtype)
            return noise, lambda step, frame: transition[step, :, frame]
        if noise_mode == "reseeded":
            generator = torch.Generator(device=device).manual_seed(int(seed))
        elif noise_mode != "sequential":
            raise ValueError(
                f"Unknown FastFrames noise convention {noise_mode!r}; expected one of {FASTFRAMES_NOISE_MODES}"
            )
        frame_shape = (1, channels, *latent_size)
        return noise, lambda step, frame: torch.randn(frame_shape, generator=generator, device=device, dtype=dtype)

    def _time_conditioning(
        self,
        timestep: float,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dit is None:
            raise RuntimeError("FastFrames DiT is not loaded.")
        device = self.torch_device
        key = (float(timestep), dtype, device)
        if key not in self._time_cache:
            t = torch.full((1, 1), float(timestep), device=device, dtype=torch.float32)
            e0, e_head = self.dit.time_conditioning(t, dtype)
            self._time_cache[key] = (t, e0, e_head)
        return self._time_cache[key]

    def _predict(
        self,
        noisy: torch.Tensor,
        timestep: float,
        frame: int,
        text_cache: list[dict[str, torch.Tensor]],
        kv_cache: list[dict[str, torch.Tensor]],
        *,
        fast: FastFramesFastOps | None,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dit is None:
            raise RuntimeError("FastFrames DiT is not loaded.")
        t, e0, e_head = self._time_conditioning(timestep, dtype)
        flow, features = self.dit.forward_frame(noisy, e0, e_head, text_cache, kv_cache, frame, fast=fast)
        x0 = self.scheduler.flow_to_x0(flow.flatten(0, 1), noisy.flatten(0, 1), t.flatten())
        return x0.unflatten(0, flow.shape[:2]), features, e_head

    def _refine(
        self,
        x0: torch.Tensor,
        frame: int,
        kv_cache: list[dict[str, torch.Tensor]],
        *,
        frame_tokens: int,
        dtype: torch.dtype,
    ) -> None:
        if self.dit is None or self.refiner is None:
            raise RuntimeError("FastFrames DiT and KVRefinementNetwork are required when context='refiner'.")
        device = self.torch_device
        embed_key = (float(self.config.refine_timestep), dtype, device)
        if embed_key not in self._refine_embed_cache:
            self._refine_embed_cache[embed_key] = self.refiner.embed_timestep(
                self.config.refine_timestep,
                1,
                device,
                dtype,
            )
        emb = self._refine_embed_cache[embed_key]
        start, end = frame * frame_tokens, (frame + 1) * frame_tokens
        stem = self.dit.patch_embedding(x0.permute(0, 2, 1, 3, 4).contiguous()).flatten(2).transpose(1, 2)
        if self.config.contiguous_stem:
            stem = stem.contiguous()
        phi = self.refiner.compute_phi(stem.to(dtype), emb)
        keys = torch.stack([cache["k"][:, start:end].reshape(frame_tokens, -1) for cache in kv_cache])
        values = torch.stack([cache["v"][:, start:end].reshape(frame_tokens, -1) for cache in kv_cache])
        keys, values = self.refiner.repair_all(keys, values, phi, emb)
        for index in self.refiner.repaired_block_indices:
            slot = kv_cache[index]["k"][:, start:end]
            kv_cache[index]["k"][:, start:end].copy_(keys[index].reshape(slot.shape).to(slot.dtype))
            kv_cache[index]["v"][:, start:end].copy_(values[index].reshape(slot.shape).to(slot.dtype))

    @torch.inference_mode()
    def rollout(
        self,
        prompt_embeds: torch.Tensor,
        seed: int = 0,
        context: str = "refiner",
        *,
        num_latent_frames: int | None = None,
        latent_size: tuple[int, int] | None = None,
        noise_mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run autoregressive denoising returning ``(latents, features, modulation)``.

        * ``context="refiner"`` (FastFrames): clean ``t = 0`` pass on frame 0, then
          ``KVRefinementNetwork`` cache repair on frames ``1..F-1``.
        * ``context="dit"`` (Causal Forcing++): clean ``t = 0`` DiT pass on every frame.
        """
        if self.dit is None:
            raise RuntimeError("FastFrames DiT is not loaded. Use FastFramesPipeline.from_pretrained() first.")
        if context not in FASTFRAMES_CONTEXT_MODES:
            raise ValueError(f"context must be one of {FASTFRAMES_CONTEXT_MODES}, got {context!r}")
        if context == "refiner" and self.refiner is None:
            raise RuntimeError("FastFrames KVRefinementNetwork is required when context='refiner'.")

        cfg = self.config
        frames = cfg.num_frames if num_latent_frames is None else int(num_latent_frames)
        lat_h, lat_w = cfg.latent_size if latent_size is None else (int(latent_size[0]), int(latent_size[1]))
        resolved_noise = cfg.noise if noise_mode is None else noise_mode
        p_h, p_w = self.dit.patch_size[1], self.dit.patch_size[2]
        frame_tokens = (lat_h // p_h) * (lat_w // p_w)
        compute_dtype = self._compute_dtype
        device = self.torch_device
        fast = self._get_fast_ops(lat_h, lat_w, frames)

        with self._autocast_context():
            noise, transition = self._noise_schedule(
                seed,
                num_latent_frames=frames,
                latent_size=(lat_h, lat_w),
                noise_mode=resolved_noise,
                dtype=compute_dtype,
            )
            embeds = prompt_embeds.to(device=device, dtype=compute_dtype)
            text_cache = self.dit.text_cache(embeds)
            kv_cache = allocate_fastframes_kv_cache(
                self.dit,
                frames,
                frame_tokens,
                dtype=compute_dtype,
                device=device,
            )
            latents = torch.empty_like(noise)
            features_list: list[torch.Tensor] = []
            modulation_list: list[torch.Tensor] = []
            for frame in range(frames):
                schedule = cfg.first_frame_timesteps if frame == 0 else cfg.timesteps
                x = noise[:, frame : frame + 1]
                x0 = x
                feature = torch.empty(0, device=device)
                mod = torch.empty(0, device=device)
                for step, timestep in enumerate(schedule):
                    x0, feature, mod = self._predict(
                        x,
                        timestep,
                        frame,
                        text_cache,
                        kv_cache,
                        fast=fast,
                        dtype=compute_dtype,
                    )
                    if step + 1 < len(schedule):
                        next_t = torch.full((1,), schedule[step + 1], device=device, dtype=torch.float32)
                        eps = transition(step, frame)
                        x = self.scheduler.add_noise(x0.flatten(0, 1), eps, next_t).unflatten(0, x0.shape[:2])
                latents[:, frame : frame + 1].copy_(x0)
                features_list.append(feature)
                modulation_list.append(mod)
                if context == "dit" or frame == 0:
                    self._predict(x0, 0.0, frame, text_cache, kv_cache, fast=fast, dtype=compute_dtype)
                elif context == "refiner":
                    self._refine(x0, frame, kv_cache, frame_tokens=frame_tokens, dtype=compute_dtype)

        return latents, torch.cat(features_list, dim=1), torch.cat(modulation_list, dim=1)

    @torch.inference_mode()
    def decode(
        self,
        features: torch.Tensor,
        modulation: torch.Tensor,
        *,
        latent_size: tuple[int, int] | None = None,
        num_pixel_frames: int | None = None,
    ) -> torch.Tensor:
        """Decode pre-head hidden states to ``[1, T, 3, H, W]`` RGB pixels in ``[-1, 1]``."""
        if self.decoder is None:
            raise RuntimeError("FastFrames pixel decoder is not loaded.")
        device = self.torch_device
        lat_h, lat_w = self.config.latent_size if latent_size is None else latent_size
        p_h, p_w = self.dit.patch_size[1:] if self.dit is not None else (2, 2)
        grid = torch.tensor([[modulation.shape[1], lat_h // p_h, lat_w // p_w]], device=device, dtype=torch.long)
        target_frames = self.config.num_pixel_frames if num_pixel_frames is None else int(num_pixel_frames)

        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pixels = self.decoder(features, modulation, grid)
        else:
            decoder_param = next(self.decoder.parameters(), None)
            target_dtype = decoder_param.dtype if decoder_param is not None else features.dtype
            pixels = self.decoder(features.to(target_dtype), modulation.to(target_dtype), grid)
        return align_frames(pixels, target_frames)

    def __call__(
        self,
        prompt: str | None = None,
        *,
        seed: int = 0,
        context: str = "refiner",
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        return_dict: bool = True,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: int | None = None,
        noise: str | None = None,
        prompt_embeds: torch.Tensor | None = None,
        images: Any = None,
        video: Any = None,
        return_latents: bool = False,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run FastFrames text-to-video generation and optionally export an MP4 artifact."""
        extra = dict(operator_kwargs or {})
        extra.update(kwargs)
        for framework_key in ("sample_id", "task_name", "task"):
            extra.pop(framework_key, None)

        requested_variant = extra.pop("variant", None)
        if requested_variant is not None:
            resolved_requested = resolve_fastframes_variant(requested_variant)
            if resolved_requested.variant_id != self.config.variant_id:
                raise ValueError(
                    f"Pipeline is loaded with variant {self.config.variant_id!r}, "
                    f"cannot serve variant {resolved_requested.variant_id!r} without reloading."
                )

        if extra:
            raise TypeError(f"Unexpected FastFramesPipeline.__call__ keyword argument(s): {sorted(extra)}")

        request = self.operator.prepare_request(
            prompt=prompt,
            variant=self.config,
            context=context,
            seed=seed,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            noise=noise,
            images=images,
            video=video,
            allow_empty_prompt=prompt_embeds is not None,
        )

        embeds = prompt_embeds if prompt_embeds is not None else self.encode_prompt(request["prompt"])
        latents, features, modulation = self.rollout(
            embeds,
            request["seed"],
            context=request["context"],
            num_latent_frames=request["num_latent_frames"],
            latent_size=request["latent_size"],
            noise_mode=request["noise"],
        )
        pixels = self.decode(
            features,
            modulation,
            latent_size=request["latent_size"],
            num_pixel_frames=request["num_frames"],
        )
        video_tensor = pixels_to_uint8(pixels[0]).cpu()
        video_np = video_tensor.numpy()

        resolved_output_path: Path | None = None
        if output_path is not None:
            resolved_output_path = Path(output_path)
        elif output_dir is not None:
            resolved_output_path = Path(output_dir) / f"{self.config.variant_id}_seed{request['seed']}.mp4"

        saved_path: str | None = None
        if resolved_output_path is not None:
            saved_path = save_image_or_video_tensor(
                video_tensor,
                resolved_output_path,
                fps=request["fps"],
                value_range="0,255",
                layout="thwc",
            )

        result: dict[str, Any] = {
            "model_id": self.model_id,
            "variant": self.config.variant_id,
            "prompt": request["prompt"],
            "seed": request["seed"],
            "context": request["context"],
            "noise": request["noise"],
            "fps": request["fps"],
            "num_frames": int(video_np.shape[0]),
            "height": int(video_np.shape[1]),
            "width": int(video_np.shape[2]),
            "video": video_np,
            "video_chunk": VideoChunk(frames=video_np, layout="thwc", value_range="0,255"),
        }
        if saved_path is not None:
            result["artifact_path"] = saved_path
            result["output_path"] = saved_path
        if return_latents:
            result["latents"] = latents

        if return_dict:
            return result
        return saved_path if saved_path is not None else video_np
