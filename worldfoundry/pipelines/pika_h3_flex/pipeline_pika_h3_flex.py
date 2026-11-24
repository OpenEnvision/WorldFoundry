# SPDX-License-Identifier: Apache-2.0
# Inference contract adapted from Pika-Labs/Pika-H3-Flex, revision
# 0ddf55ad7067ca1e76812253bd7b4600df87f759.
"""Pika H3 Flex inference on WorldFoundry's shared MiniMax-H3 components."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.pika import (
    load_pika_h3_flex_adapter,
)
from worldfoundry.core.io.paths import hfd_root_path
from worldfoundry.pipelines.minimax.pipeline_minimax_h3 import NativeMiniMaxH3Pipeline

PIKA_H3_FLEX_KNOTS = (1.0, 0.75, 0.5, 0.25, 0.1, 0.04, 0.0)
PIKA_H3_FLEX_TASKS = ("t2va", "i2va", "l2va", "fl2va", "ref2va")
PIKA_H3_FLEX_ADAPTERS = ("fl2va", "ref2va")


class PikaH3FlexPipeline(NativeMiniMaxH3Pipeline):
    """Six-step Pika H3 Flex adapter sharing the native MiniMax-H3 pipeline."""

    MODEL_ID = "pika-h3-flex"
    MODEL_PATH_OPTION = "base_model_path"
    DEFAULT_NUM_INFERENCE_STEPS = 6
    DEFAULT_SHORT_EDGE = 768
    DEFAULT_DURATION_SECONDS = 5.0

    def __init__(
        self,
        *,
        adapter_task: str = "fl2va",
        adapter_path: str | Path | None = None,
        processor: Any = None,
        base_model_path: str | Path | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        if adapter_task not in PIKA_H3_FLEX_ADAPTERS:
            raise ValueError(f"Pika H3 Flex adapter_task must be one of {PIKA_H3_FLEX_ADAPTERS}")
        super().__init__(model_id=model_id or self.MODEL_ID, processor=processor, **kwargs)
        self.adapter_task = adapter_task
        self.adapter_path = None if adapter_path is None else Path(adapter_path).expanduser().resolve()
        self.base_model_path = None if base_model_path is None else Path(base_model_path).expanduser().resolve()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        component_devices: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> PikaH3FlexPipeline:
        """Load one Pika adapter and the shared MiniMax-H3 base components."""

        options = cls._runtime_options(
            model_path,
            None if required_components is None else dict(required_components),
            kwargs,
        )
        cls._strip_framework_loading_options(options)
        adapter_task = str(options.pop("adapter_task", "fl2va")).lower()
        if adapter_task not in PIKA_H3_FLEX_ADAPTERS:
            raise ValueError(f"Pika H3 Flex adapter_task must be one of {PIKA_H3_FLEX_ADAPTERS}")
        adapter_path = options.pop("adapter_path", None)
        base_model_path = options.pop("base_model_path", options.pop("model_path", None))
        partition = str(options.pop("partition", "Ref2VA" if adapter_task == "ref2va" else "FL2VA"))
        transformer_subfolder = options.pop("transformer_subfolder", "transformer")
        adapter_rank = int(options.pop("adapter_rank", 128))
        adapter_alpha = int(options.pop("adapter_alpha", 128))
        adapter_strict = bool(options.pop("adapter_strict", True))
        processor_path = options.pop("processor_path", None)

        if base_model_path is None:
            base_model_path = hfd_root_path("MiniMaxAI--MiniMax-H3")
        base_root = Path(base_model_path).expanduser().resolve()
        adapter_path = (
            Path(adapter_path).expanduser()
            if adapter_path is not None
            else (hfd_root_path("Pika-Labs--Pika-H3-Flex") / adapter_task / "adapter.safetensors")
        )

        base_options = dict(options)
        base_options.update(partition=partition, transformer_subfolder=transformer_subfolder)
        base = NativeMiniMaxH3Pipeline.from_pretrained(
            model_path=str(base_root),
            required_components=base_options,
            device=device,
            model_id=cls.MODEL_ID,
            component_devices=component_devices,
        )
        if base.transformer is None:
            raise RuntimeError("Pika H3 Flex base transformer was not initialized")
        load_pika_h3_flex_adapter(
            base.transformer,
            adapter_path,
            rank=adapter_rank,
            alpha=adapter_alpha,
            strict=adapter_strict,
        )

        processor_root = (
            Path(processor_path).expanduser()
            if processor_path is not None
            else cls._processor_root(base_root, partition)
        )
        processor = cls._load_processor(processor_root)
        return cls(
            transformer=base.transformer,
            video_vae=base.video_vae,
            audio_vae=base.audio_vae,
            text_encoder=base.text_encoder,
            tokenizer=base.tokenizer,
            device=base.device,
            component_devices=base.component_devices,
            adapter_task=adapter_task,
            adapter_path=adapter_path,
            processor=processor,
            base_model_path=base_root,
            model_id=model_id or cls.MODEL_ID,
        )

    @staticmethod
    def _normalise_images(values: Any) -> list[Any]:
        if values is None:
            return []
        if isinstance(values, (list, tuple)):
            return [PikaH3FlexPipeline._as_image(value) for value in values]
        return [PikaH3FlexPipeline._as_image(values)]

    @staticmethod
    def _reference_images(values: Any) -> list[Any]:
        if values is None:
            return []
        values = values if isinstance(values, (list, tuple)) else [values]
        images = []
        for value in values:
            if isinstance(value, Mapping):
                kind = str(value.get("kind", "image")).lower()
                if kind != "image":
                    raise ValueError("Pika H3 Flex in-tree reference conditioning currently accepts image references")
                value = value.get("path")
            elif isinstance(value, str) and ":" in value:
                kind, value = value.split(":", 1)
                if kind.lower() != "image":
                    raise ValueError("Pika H3 Flex in-tree reference conditioning currently accepts image references")
            images.append(PikaH3FlexPipeline._as_image(value))
        return images

    def __call__(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        return_dict: bool = False,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a six-step Pika H3 Flex clip and optionally write its MP4 artifact."""

        if video is not None:
            raise ValueError("Pika H3 Flex currently accepts image references, not a video argument")
        request = dict(operator_kwargs or {})
        request.update(kwargs)
        request.pop("task_name", None)
        prompt = prompt if prompt is not None else request.pop("prompt", None)
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
            raise ValueError("Pika H3 Flex prompt must contain 1–8000 characters")
        task = str(request.pop("task", request.pop("workflow", "auto"))).lower()
        references = request.pop("references", request.pop("reference_images", None))
        reference_images = self._reference_images(references)
        keyframes = self._normalise_images(images)
        if reference_images and keyframes:
            raise ValueError("Use references or endpoint images, not both")
        if reference_images:
            task = "ref2va" if task in ("auto", "ref2va") else task
        elif task == "auto":
            task = {0: "t2va", 1: "i2va", 2: "fl2va"}.get(len(keyframes), "")
        if task not in PIKA_H3_FLEX_TASKS:
            raise ValueError(f"Unsupported Pika H3 Flex task: {task!r}")
        if task == "ref2va" and not reference_images:
            raise ValueError("ref2va requires at least one image reference")
        if task == "ref2va" and self.adapter_task != "ref2va":
            raise ValueError("Load the Pika H3 Flex ref2va adapter for reference conditioning")
        if task != "ref2va" and self.adapter_task != "fl2va":
            raise ValueError("Load the Pika H3 Flex fl2va adapter for text/keyframe conditioning")
        if task != "ref2va" and reference_images:
            raise ValueError("image references require task=ref2va")
        expected_keyframes = {"t2va": 0, "i2va": 1, "l2va": 1, "fl2va": 2}[task]
        if len(keyframes) != expected_keyframes:
            raise ValueError(f"task {task} requires {expected_keyframes} endpoint image(s)")

        num_frames = request.pop("num_frames", 124)
        if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
            raise ValueError("num_frames must be a positive integer")
        fps = int(request.pop("fps", 24))
        if fps != 24:
            raise ValueError("Pika H3 Flex uses 24 fps")
        steps = int(request.pop("num_inference_steps", 6))
        if steps != 6:
            raise ValueError("Pika H3 Flex requires six inference steps")
        seed = int(request.pop("seed", 0))
        if seed < 0:
            raise ValueError("seed must be non-negative")
        short_edge = int(request.pop("short_edge", self.DEFAULT_SHORT_EDGE))
        aspect_ratio = str(request.pop("aspect_ratio", "16:9"))
        duration_seconds = float(request.pop("duration_seconds", num_frames / 24.0))
        if not 5.0 <= duration_seconds <= 15.0:
            raise ValueError("Pika H3 Flex duration must be between 5 and 15 seconds")
        if request:
            raise TypeError(f"Unsupported Pika H3 Flex call options: {sorted(request)}")

        model_task = "ref2va" if task == "ref2va" else "fl2va" if keyframes else "t2va"
        plan = self._resolve_plan(
            task=model_task,
            short_edge=short_edge,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
        )
        presentation = reference_images or keyframes
        presentation = [self._fit_image(image, height=plan["height"], width=plan["width"]) for image in presentation]
        prompt_embeds, text_tags = self._encode_presentation(prompt, presentation)

        condition_rows = None
        keyframe_indices = None
        ref_blocks = None
        if keyframes:
            encoded = [self._encode_image(image) for image in presentation]
            condition_rows = torch.cat([self._patchify_condition(latent, seed=seed) for latent in encoded], dim=0)
            keyframe_indices = {"i2va": (0,), "l2va": (-1,), "fl2va": (0, -1)}[task]
        elif reference_images:
            encoded = [self._encode_image(image) for image in presentation]
            condition_rows = torch.cat([self._patchify_condition(latent, seed=seed) for latent in encoded], dim=0)
            ref_blocks = [
                {
                    "kind": "image",
                    "latent_h": int(latent.shape[-2]),
                    "latent_w": int(latent.shape[-1]),
                }
                for latent in encoded
            ]

        result = self.generate(
            prompt_embeds=prompt_embeds,
            task=model_task,
            short_edge=short_edge,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            num_inference_steps=steps,
            seed=seed,
            keyframe_frame_indices=keyframe_indices,
            keyframe_cond_rows=condition_rows,
            ref_blocks=ref_blocks,
            text_token_tags=text_tags,
            sigma_knots=PIKA_H3_FLEX_KNOTS,
            renoise=True,
            shared_initial_noise=True,
        )
        result.update(model_id=self.MODEL_ID, adapter_task=self.adapter_task, fps=24, inference_steps=6)
        destination = Path(output_path or Path(output_dir or ".") / "pika-h3-flex.mp4").expanduser()
        if "video" in result:
            result["output_path"] = self._write_mp4_with_audio(
                video=result["video"], audio=result.get("audio"), output_path=destination, fps=24
            )
            result["artifact_path"] = result["output_path"]
        return result if return_dict else result.get("artifact_path", result)


__all__ = ["PIKA_H3_FLEX_KNOTS", "PikaH3FlexPipeline"]
