# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Danze Chen. Adapted from Danzer1xxxxChan/H3-World, revision
# f0c7be2acbbde8256b2473f6e7b088f58272acae (Apache-2.0).
"""H3-World interactive world-model pipeline over WorldFoundry's shared MiniMax-H3 runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.h3_world import (
    H3WorldActionMaskSpec,
    h3_world_directed_attention_context,
    install_h3_world_directed_attention,
    load_h3_world_adapter,
)
from worldfoundry.base_models.diffusion_model.schedulers.minimax_h3 import minimax_h3_video_latent_t
from worldfoundry.core.io.paths import hfd_root_path, resolve_local_checkpoint_file
from worldfoundry.pipelines.minimax._minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
from worldfoundry.pipelines.minimax.pipeline_minimax_h3 import NativeMiniMaxH3Pipeline

from .action_script import H3_WORLD_ACTION_PRESETS, null_action_script, resolve_action_script

H3_WORLD_DEFAULT_ADAPTER_FILENAME = "step-10000.safetensors"


class H3WorldPipeline(NativeMiniMaxH3Pipeline):
    """Action-conditioned H3-World pipeline reusing canonical MiniMax-H3 components."""

    MODEL_ID = "h3-world"
    GENERATION_TYPE = "i2v"
    MODEL_PATH_OPTION = "base_model_path"
    DEFAULT_HEIGHT = 480
    DEFAULT_WIDTH = 832
    DEFAULT_NUM_FRAMES = 124
    DEFAULT_NUM_INFERENCE_STEPS = 50
    DEFAULT_CFG_SCALE = 1.0
    DEFAULT_SEED = 2
    DEFAULT_ACTION_PRESET = "forward"

    def __init__(
        self,
        *,
        adapter_path: str | Path | None = None,
        base_model_path: str | Path | None = None,
        use_flex_attention: bool = True,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id=model_id or self.MODEL_ID, **kwargs)
        self.adapter_path = None if adapter_path is None else Path(adapter_path).expanduser().resolve()
        self.base_model_path = None if base_model_path is None else Path(base_model_path).expanduser().resolve()
        self.use_flex_attention = bool(use_flex_attention)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        component_devices: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> H3WorldPipeline:
        """Load the H3-World LoRA adapter over the shared MiniMax-H3 FL2VA backbone."""

        options = cls._runtime_options(
            model_path,
            None if required_components is None else dict(required_components),
            kwargs,
        )
        cls._strip_framework_loading_options(options)
        adapter_source = options.pop("adapter_path", options.pop("lora_path", None))
        base_model_path = options.pop("base_model_path", options.pop("model_path", None))
        partition = str(options.pop("partition", "FL2VA"))
        if partition != "FL2VA":
            raise ValueError(f"H3-World requires the MiniMax-H3 FL2VA partition, got {partition!r}")
        transformer_subfolder = options.pop("transformer_subfolder", "transformer")
        adapter_rank = int(options.pop("adapter_rank", 32))
        adapter_alpha = options.pop("adapter_alpha", None)
        if adapter_alpha is not None:
            adapter_alpha = int(adapter_alpha)
        adapter_strict = bool(options.pop("adapter_strict", True))
        use_flex_attention = bool(options.pop("use_flex_attention", True))
        processor_path = options.pop("processor_path", None)

        if base_model_path is None:
            base_model_path = hfd_root_path("MiniMaxAI--MiniMax-H3")
        base_root = Path(base_model_path).expanduser().resolve()
        adapter_path = cls._resolve_adapter_path(adapter_source)

        base_options = dict(options)
        base_options.update(partition="FL2VA", transformer_subfolder=transformer_subfolder)
        base = NativeMiniMaxH3Pipeline.from_pretrained(
            model_path=str(base_root),
            required_components=base_options,
            device=device,
            model_id=cls.MODEL_ID,
            component_devices=component_devices,
        )
        if base.transformer is None:
            raise RuntimeError("H3-World base MiniMax-H3 transformer was not initialized")
        load_h3_world_adapter(
            base.transformer,
            adapter_path,
            rank=adapter_rank,
            alpha=adapter_alpha,
            strict=adapter_strict,
        )
        install_h3_world_directed_attention(base.transformer, use_flex=use_flex_attention)

        processor_root = (
            Path(processor_path).expanduser() if processor_path is not None else cls._processor_root(base_root, "FL2VA")
        )
        processor = cls._load_processor(processor_root)
        return cls(
            transformer=base.transformer,
            video_vae=base.video_vae,
            audio_vae=base.audio_vae,
            text_encoder=base.text_encoder,
            tokenizer=base.tokenizer,
            processor=processor,
            device=base.device,
            component_devices=base.component_devices,
            adapter_path=adapter_path,
            base_model_path=base_root,
            use_flex_attention=use_flex_attention,
            model_id=model_id or cls.MODEL_ID,
        )

    @staticmethod
    def _resolve_adapter_path(adapter_source: Any) -> Path:
        if adapter_source is not None:
            candidate = Path(str(adapter_source)).expanduser()
            if candidate.is_dir():
                candidate = candidate / H3_WORLD_DEFAULT_ADAPTER_FILENAME
            return candidate
        try:
            return resolve_local_checkpoint_file("DANNY621/H3-World", H3_WORLD_DEFAULT_ADAPTER_FILENAME)
        except FileNotFoundError:
            return hfd_root_path("DANNY621--H3-World") / H3_WORLD_DEFAULT_ADAPTER_FILENAME

    def _encode_action_presentation(
        self,
        prompt: str,
        first_frame: Any,
        script: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        """Encode the first-frame head followed by independently encoded per-latent clauses."""

        head_hidden, head_tags = self._encode_presentation(prompt, [first_frame])
        if head_tags is None:
            head_tags = torch.ones(int(head_hidden.shape[0]), dtype=torch.long)

        cache: dict[str, torch.Tensor] = {}
        rows: list[torch.Tensor] = [head_hidden]
        spans: list[tuple[int, int]] = []
        cursor = int(head_hidden.shape[0])
        for line in script:
            if line not in cache:
                cache[line] = self._encode_prompt(line).to(device=head_hidden.device, dtype=head_hidden.dtype)
            emb = cache[line]
            length = int(emb.shape[0])
            spans.append((cursor, cursor + length))
            cursor += length
            rows.append(emb)

        extra = cursor - int(head_hidden.shape[0])
        full_hidden = torch.cat(rows, dim=0)
        full_tags = torch.cat([head_tags.to(dtype=torch.long), torch.ones(extra, dtype=torch.long)])
        return full_hidden, full_tags, spans

    def _branch_attention_context(self, branch: MiniMaxH3DenoiseBranch) -> AbstractContextManager[None]:
        raw_spec = branch.action_mask_spec
        if raw_spec is None or self.transformer is None:
            return nullcontext()
        spec = H3WorldActionMaskSpec(
            action_text_rows=raw_spec["action_text_rows"],
            video_start=raw_spec["video_start"],
            frame_rows=raw_spec["frame_rows"],
        )
        return h3_world_directed_attention_context(self.transformer, spec)

    @staticmethod
    def _extract_first_frame(images: Any) -> Any:
        if images is None:
            return None
        if isinstance(images, (list, tuple)):
            if len(images) != 1:
                raise ValueError(f"H3-World requires exactly one first-frame image, got {len(images)}")
            return H3WorldPipeline._as_image(images[0])
        return H3WorldPipeline._as_image(images)

    def __call__(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        return_dict: bool = False,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate an action-conditioned H3-World video+audio clip from a starting frame."""

        if video is not None:
            raise ValueError("H3-World accepts a single starting image, not a video input")
        request = dict(operator_kwargs or {})
        request.update(kwargs)
        request.pop("task_name", None)
        request.pop("task", None)

        prompt_value = prompt if prompt is not None else request.pop("prompt", request.pop("scene_prompt", None))
        if not isinstance(prompt_value, str) or not prompt_value.strip():
            raise ValueError("H3-World requires a non-empty scene prompt")

        frame_input = images if images is not None else request.pop("first_frame", request.pop("image", None))
        first_frame = self._extract_first_frame(frame_input)
        if first_frame is None:
            raise ValueError("H3-World requires a first-frame image (images or first_frame)")

        num_frames = request.pop("num_frames", self.DEFAULT_NUM_FRAMES)
        if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 5 or (num_frames - 5) % 17:
            raise ValueError(f"H3-World num_frames must satisfy 17k+5 (e.g. 124, 243, 481), got {num_frames!r}")
        height = int(request.pop("height", self.DEFAULT_HEIGHT))
        width = int(request.pop("width", self.DEFAULT_WIDTH))
        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError(f"H3-World height and width must be positive multiples of 32, got {width}x{height}")
        fps = int(request.pop("fps", self.DEFAULT_FPS))
        if fps != self.DEFAULT_FPS:
            raise ValueError(f"H3-World uses {self.DEFAULT_FPS} fps, got {fps}")
        steps = int(request.pop("num_inference_steps", request.pop("steps", self.DEFAULT_NUM_INFERENCE_STEPS)))
        if steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {steps}")
        seed = int(request.pop("seed", self.DEFAULT_SEED))
        if seed < 0:
            raise ValueError("seed must be non-negative")
        cfg_scale = float(request.pop("cfg_scale", request.pop("guidance_scale", self.DEFAULT_CFG_SCALE)))
        if cfg_scale <= 0.0:
            raise ValueError(f"cfg_scale must be positive, got {cfg_scale}")

        action_preset = request.pop("action_preset", None)
        action_script_arg = request.pop("action_script", None)
        interactions_arg = interactions if interactions is not None else request.pop("interactions", None)
        negative_prompt = str(request.pop("negative_prompt", " "))
        negative_action_script_arg = request.pop("negative_action_script", None)
        flow_shift = request.pop("flow_shift", None)
        audio_flow_shift = request.pop("audio_flow_shift", None)
        return_latent_rows = bool(request.pop("return_latent_rows", False))
        if request:
            raise TypeError(f"Unsupported H3-World call options: {sorted(request)}")

        latent_t = minimax_h3_video_latent_t(num_frames)
        script = resolve_action_script(
            latent_t=latent_t,
            action_preset=action_preset,
            action_script=action_script_arg,
            interactions=interactions_arg,
            default_preset=self.DEFAULT_ACTION_PRESET,
        )

        fitted_frame = self._fit_image(first_frame, height=height, width=width)
        prompt_embeds, text_tags, action_spans = self._encode_action_presentation(prompt_value, fitted_frame, script)

        negative_embeds = None
        negative_tags = None
        negative_spans = None
        if cfg_scale != 1.0:
            neg_script = (
                resolve_action_script(latent_t=latent_t, action_script=negative_action_script_arg)
                if negative_action_script_arg is not None
                else null_action_script(latent_t)
            )
            negative_embeds, negative_tags, negative_spans = self._encode_action_presentation(
                negative_prompt, fitted_frame, neg_script
            )

        encoded_first = self._encode_image(fitted_frame, deterministic=True)
        condition_rows = self._patchify_condition(encoded_first, seed=seed, noise_aug=0.999, seed_offset=0)
        # Match DiffSynth FlowMatchScheduler("MiniMax-H3"): N steps over N+1 knots ending at 0.0.
        sigma_knots = tuple(float(v) for v in torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32).tolist())

        result = self.generate(
            prompt_embeds=prompt_embeds,
            task="fl2va",
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            seed=seed,
            flow_shift=flow_shift,
            audio_flow_shift=audio_flow_shift,
            keyframe_frame_indices=(0,),
            keyframe_cond_rows=condition_rows,
            text_token_tags=text_tags,
            action_text_spans=action_spans,
            negative_prompt_embeds=negative_embeds,
            negative_text_token_tags=negative_tags,
            negative_action_text_spans=negative_spans,
            cfg_scale=cfg_scale,
            branch_context=self._branch_attention_context,
            sigma_knots=sigma_knots,
            return_latent_rows=return_latent_rows,
        )
        result.update(
            model_id=self.model_id,
            action_script=list(script),
            fps=self.DEFAULT_FPS,
            inference_steps=steps,
            cfg_scale=cfg_scale,
        )
        destination = Path(output_path or Path(output_dir or ".") / "h3-world.mp4").expanduser()
        if "video" in result:
            result["output_path"] = self._write_mp4_with_audio(
                video=result["video"], audio=result.get("audio"), output_path=destination, fps=self.DEFAULT_FPS
            )
            result["artifact_path"] = result["output_path"]
        return result if return_dict else result.get("artifact_path", result)


__all__ = [
    "H3_WORLD_ACTION_PRESETS",
    "H3_WORLD_DEFAULT_ADAPTER_FILENAME",
    "H3WorldPipeline",
]
