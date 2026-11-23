# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from fuyi25/H3-Tiny at a3f17f0ef85e9af5d07b2007c4b2dbd6d035e69c (Apache-2.0).
"""H3-Tiny pruned and MeanFlow few-step inference on canonical MiniMax-H3 components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model.models.denoisers.h3_tiny import load_h3_tiny_transformer
from worldfoundry.base_models.diffusion_model.schedulers.h3_tiny import (
    H3_TINY_AUDIO_SHIFT,
    H3_TINY_FEWSTEP_DEFAULT_STEPS,
    H3_TINY_FULLSTEP_DEFAULT_STEPS,
    H3_TINY_REFERENCE_TIMESTEP,
    H3_TINY_VIDEO_SHIFT,
    H3TinySchedule,
)
from worldfoundry.core.io.paths import hfd_root_path
from worldfoundry.operators.h3_tiny_operator import (
    H3_TINY_DEFAULT_NUM_FRAMES,
    H3_TINY_DEFAULT_SEED,
    H3_TINY_FPS,
    H3TinyOperator,
)
from worldfoundry.pipelines.minimax._minimax_h3 import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from worldfoundry.pipelines.minimax.pipeline_minimax_h3 import (
    NativeMiniMaxH3Pipeline,
    _resolve_component_devices,
)

_PATCH_H = 2
_PATCH_W = 2
_VIDEO_LATENT_CHANNELS = 24
_AUDIO_LATENT_CHANNELS = 32
_AUDIO_CHANNELS = 2
_WORKFLOWS = ("ref2va", "fl2va", "t2va")


@dataclass(frozen=True)
class H3TinyVariantSpec:
    """Published H3-Tiny checkpoint variant contract."""

    variant_id: str
    repo_id: str
    revision: str
    hfd_dirname: str
    default_workflow: str
    endpoint_mode: str
    default_steps: int
    partition: str


H3_TINY_VARIANTS: dict[str, H3TinyVariantSpec] = {
    "17b-fewstep-ref2va": H3TinyVariantSpec(
        variant_id="17b-fewstep-ref2va",
        repo_id="H3-Tiny/MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1",
        revision="83aa8091053aeed77de0b94fe1a6450d112de512",
        hfd_dirname="H3-Tiny--MiniMax-H3-Tiny-17B-FewStep-Ref2VA-v0.1",
        default_workflow="ref2va",
        endpoint_mode="meanflow",
        default_steps=H3_TINY_FEWSTEP_DEFAULT_STEPS,
        partition="Ref2VA",
    ),
    "17b-pruned-ref2va": H3TinyVariantSpec(
        variant_id="17b-pruned-ref2va",
        repo_id="H3-Tiny/MiniMax-H3-Tiny-17B-Pruned-Ref2VA",
        revision="e0ce7d0de6becc72e2de98f4cafe29c4b1e1c445",
        hfd_dirname="H3-Tiny--MiniMax-H3-Tiny-17B-Pruned-Ref2VA",
        default_workflow="ref2va",
        endpoint_mode="none",
        default_steps=H3_TINY_FULLSTEP_DEFAULT_STEPS,
        partition="Ref2VA",
    ),
    "17b-pruned-fl2va": H3TinyVariantSpec(
        variant_id="17b-pruned-fl2va",
        repo_id="H3-Tiny/MiniMax-H3-Tiny-17B-Pruned-FL2VA",
        revision="e026f495b8816b87b8441dee82f718867e9f5d42",
        hfd_dirname="H3-Tiny--MiniMax-H3-Tiny-17B-Pruned-FL2VA",
        default_workflow="fl2va",
        endpoint_mode="none",
        default_steps=H3_TINY_FULLSTEP_DEFAULT_STEPS,
        partition="FL2VA",
    ),
    "14b-okd-ref2va": H3TinyVariantSpec(
        variant_id="14b-okd-ref2va",
        repo_id="H3-Tiny/MiniMax-H3-Tiny-14B-OKD-Ref2VA-v0.1",
        revision="30b4ca94a49929602ba665d21ff50169fd815826",
        hfd_dirname="H3-Tiny--MiniMax-H3-Tiny-14B-OKD-Ref2VA-v0.1",
        default_workflow="ref2va",
        endpoint_mode="none",
        default_steps=H3_TINY_FULLSTEP_DEFAULT_STEPS,
        partition="Ref2VA",
    ),
}

_VARIANT_ALIASES: dict[str, str] = {
    "h3-tiny": "17b-fewstep-ref2va",
    "minimax-h3-tiny": "17b-fewstep-ref2va",
    "h3-tiny-17b-fewstep": "17b-fewstep-ref2va",
    "h3-tiny-17b-pruned": "17b-pruned-ref2va",
    "h3-tiny-17b-fl2va": "17b-pruned-fl2va",
    "h3-tiny-14b-okd": "14b-okd-ref2va",
}


def _resolve_variant(variant: str | None) -> H3TinyVariantSpec:
    raw = "17b-fewstep-ref2va" if variant is None else str(variant).strip().lower()
    canonical = _VARIANT_ALIASES.get(raw, raw)
    spec = H3_TINY_VARIANTS.get(canonical)
    if spec is None:
        raise ValueError(f"Unsupported H3-Tiny variant {variant!r}; expected one of {sorted(H3_TINY_VARIANTS)}")
    return spec


class H3TinyPipeline(NativeMiniMaxH3Pipeline):
    """H3-Tiny pruned and MeanFlow few-step pipeline on shared MiniMax-H3 components."""

    MODEL_ID = "h3-tiny"
    MODEL_PATH_OPTION = "weights_path"
    DEFAULT_VARIANT = "17b-fewstep-ref2va"
    DEFAULT_NUM_INFERENCE_STEPS = H3_TINY_FEWSTEP_DEFAULT_STEPS
    DEFAULT_SHORT_EDGE = 768
    DEFAULT_DURATION_SECONDS = 5.0
    DEFAULT_SEED = H3_TINY_DEFAULT_SEED

    def __init__(
        self,
        *,
        variant: str = DEFAULT_VARIANT,
        workflow: str | None = None,
        endpoint_mode: str | None = None,
        weights_path: str | Path | None = None,
        base_model_path: str | Path | None = None,
        processor: Any = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        spec = _resolve_variant(variant)
        resolved_workflow = spec.default_workflow if workflow is None else str(workflow).strip().lower()
        if resolved_workflow not in _WORKFLOWS:
            raise ValueError(f"H3-Tiny workflow must be one of {_WORKFLOWS}, got {workflow!r}")
        super().__init__(model_id=model_id or self.MODEL_ID, processor=processor, **kwargs)
        self.variant = spec.variant_id
        self.workflow = resolved_workflow
        inferred_mode = getattr(self.transformer, "endpoint_mode", None)
        self.endpoint_mode = str(endpoint_mode or inferred_mode or spec.endpoint_mode)
        self.weights_path = None if weights_path is None else Path(weights_path).expanduser().resolve()
        self.base_model_path = None if base_model_path is None else Path(base_model_path).expanduser().resolve()
        self.operator = H3TinyOperator()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        component_devices: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> H3TinyPipeline:
        """Load an H3-Tiny transformer checkpoint with shared MiniMax-H3 encoders and VAEs."""
        options = cls._runtime_options(
            model_path,
            None if required_components is None else dict(required_components),
            kwargs,
        )
        cls._strip_framework_loading_options(options)
        variant_raw = options.pop("variant", options.pop("variant_id", cls.DEFAULT_VARIANT))
        spec = _resolve_variant(variant_raw)
        workflow = str(options.pop("workflow", spec.default_workflow)).strip().lower()
        if workflow not in _WORKFLOWS:
            raise ValueError(f"H3-Tiny workflow must be one of {_WORKFLOWS}, got {workflow!r}")
        endpoint_mode = options.pop("endpoint_mode", None)
        weights_path = options.pop("weights_path", None)
        base_model_path = options.pop("base_model_path", options.pop("model_path", None))
        explicit_partition = options.pop("partition", None)
        partition = (
            str(explicit_partition)
            if explicit_partition is not None
            else ("Ref2VA" if workflow == "ref2va" else "FL2VA")
        )
        processor_path = options.pop("processor_path", None)
        injected_transformer = options.pop("transformer", None)
        injected_processor = options.pop("processor", None)

        if weights_path is None:
            weights_path = hfd_root_path(spec.hfd_dirname)
        weights_root = Path(weights_path).expanduser().resolve()
        if base_model_path is None:
            base_model_path = hfd_root_path("MiniMaxAI--MiniMax-H3")
        base_root = Path(base_model_path).expanduser().resolve()
        if explicit_partition is None and not (base_root / partition).is_dir():
            for fallback_partition in ("Ref2VA", "FL2VA"):
                if (base_root / fallback_partition).is_dir():
                    partition = fallback_partition
                    break

        device_overrides = options.pop("component_devices", component_devices)
        placements = _resolve_component_devices(device, device_overrides)
        transformer = (
            injected_transformer
            if injected_transformer is not None
            else load_h3_tiny_transformer(
                weights_root,
                endpoint_mode=endpoint_mode,
                device=placements["transformer"],
            )
        )
        resolved_mode = str(getattr(transformer, "endpoint_mode", endpoint_mode or spec.endpoint_mode))

        base_options = dict(options)
        base_options.update(partition=partition, transformer=transformer)
        base = NativeMiniMaxH3Pipeline.from_pretrained(
            model_path=str(base_root),
            required_components=base_options,
            device=device,
            model_id=cls.MODEL_ID,
            component_devices=placements,
        )
        processor = (
            injected_processor
            if injected_processor is not None
            else cls._load_processor(
                Path(processor_path).expanduser()
                if processor_path is not None
                else cls._processor_root(base_root, partition)
            )
        )
        return cls(
            transformer=base.transformer,
            video_vae=base.video_vae,
            audio_vae=base.audio_vae,
            text_encoder=base.text_encoder,
            tokenizer=base.tokenizer,
            processor=processor,
            device=base.device,
            component_devices=base.component_devices,
            variant=spec.variant_id,
            workflow=workflow,
            endpoint_mode=resolved_mode,
            weights_path=weights_root,
            base_model_path=base_root,
            model_id=model_id or cls.MODEL_ID,
        )

    @torch.inference_mode()
    def generate(
        self,
        *,
        prompt_embeds: torch.Tensor,
        task: str = "ref2va",
        short_edge: int = DEFAULT_SHORT_EDGE,
        aspect_ratio: str = "16:9",
        duration_seconds: float = DEFAULT_DURATION_SECONDS,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        num_inference_steps: int | None = None,
        seed: int = DEFAULT_SEED,
        flow_shift: float | None = None,
        audio_flow_shift: float | None = None,
        keyframe_frame_indices: tuple[int, ...] | None = None,
        keyframe_cond_rows: torch.Tensor | None = None,
        ref_blocks: list[dict[str, Any]] | None = None,
        audio_ref_rows: torch.Tensor | None = None,
        text_token_tags: torch.Tensor | None = None,
        keyframe_noise_aug: float = H3_TINY_REFERENCE_TIMESTEP,
        return_latent_rows: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        """Run the coupled H3-Tiny denoise loop with per-step ``(t, r)`` endpoint conditioning."""
        if extra:
            raise TypeError(f"Unsupported H3TinyPipeline.generate options: {sorted(extra)}")
        if self.transformer is None:
            raise RuntimeError("H3-Tiny transformer is not initialized")

        steps = (
            (H3_TINY_FEWSTEP_DEFAULT_STEPS if self.endpoint_mode == "meanflow" else H3_TINY_FULLSTEP_DEFAULT_STEPS)
            if num_inference_steps is None
            else int(num_inference_steps)
        )

        device = torch.device(self.device)
        plan = self._resolve_plan(
            task=task,
            short_edge=short_edge,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            height=height,
            width=width,
            num_frames=num_frames,
        )

        if prompt_embeds.ndim == 2:
            prompt_batched = prompt_embeds[None]
        elif prompt_embeds.ndim == 3 and prompt_embeds.shape[0] == 1:
            prompt_batched = prompt_embeds
        else:
            raise ValueError(f"prompt_embeds must be [L, D] or [1, L, D], got {list(prompt_embeds.shape)}")
        prompt_batched = prompt_batched.to(device=device)
        text_len = int(prompt_batched.shape[1])

        if task == "ref2va":
            if ref_blocks is None:
                raise ValueError("ref2va requires ref_blocks")
            packed = minimax_h3_packed_sequence_ref2va_blocks(
                text_len=text_len,
                latent_t=plan["latent_t"],
                latent_h=plan["latent_h"],
                latent_w=plan["latent_w"],
                audio_t=plan["audio_t"],
                ref_blocks=ref_blocks,
            )
        else:
            include_keyframe = task == "fl2va"
            packed = minimax_h3_packed_sequence(
                text_len=text_len,
                latent_t=plan["latent_t"],
                latent_h=plan["latent_h"],
                latent_w=plan["latent_w"],
                audio_t=plan["audio_t"],
                include_keyframe_cond=include_keyframe,
                keyframe_frame_indices=keyframe_frame_indices if include_keyframe else None,
                frame_count=plan["frame_count"] if include_keyframe else None,
            )

        video_indices = packed["img_pos"].to(device=device, dtype=torch.long)
        audio_indices = packed["audio_pos"].to(device=device, dtype=torch.long)
        text_indices = packed["text_pos"].to(device=device, dtype=torch.long)
        used = int(text_indices.numel() + video_indices.numel() + audio_indices.numel())

        token_tags = packed["token_tags"][:used].clone()
        if text_token_tags is not None:
            supplied = text_token_tags.reshape(-1).to(dtype=token_tags.dtype, device=token_tags.device)
            if int(supplied.numel()) != text_len:
                raise ValueError(f"text_token_tags length {int(supplied.numel())} != prompt length {text_len}")
            token_tags[:text_len] = supplied
        token_tags = token_tags.to(device=device)
        position_ids = packed["img_position_ids"][:used].to(device=device, dtype=torch.float32)

        n_target_video = plan["latent_t"] * (plan["latent_h"] // _PATCH_H) * (plan["latent_w"] // _PATCH_W)
        n_cond_video = int(video_indices.numel()) - n_target_video
        n_target_audio = plan["audio_t"] * _AUDIO_CHANNELS
        n_cond_audio = int(audio_indices.numel()) - n_target_audio

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        target_video = minimax_h3_patchify_video_latent(
            torch.randn(
                1,
                _VIDEO_LATENT_CHANNELS,
                plan["latent_t"],
                plan["latent_h"],
                plan["latent_w"],
                generator=generator,
                dtype=torch.float32,
            ),
            patch_size=(1, _PATCH_H, _PATCH_W),
        ).to(device=device)
        target_audio = torch.randn(
            n_target_audio,
            _AUDIO_LATENT_CHANNELS,
            generator=generator,
            dtype=torch.float32,
        ).to(device=device)

        if n_cond_video > 0:
            if keyframe_cond_rows is None or int(keyframe_cond_rows.shape[0]) != n_cond_video:
                raise ValueError(f"Expected {n_cond_video} conditioning video rows, got {keyframe_cond_rows}")
            video_rows = torch.cat([keyframe_cond_rows.to(device=device, dtype=torch.float32), target_video], dim=0)
        else:
            video_rows = target_video

        if n_cond_audio > 0:
            if audio_ref_rows is None or int(audio_ref_rows.shape[0]) != n_cond_audio:
                raise ValueError(f"Expected {n_cond_audio} conditioning audio rows, got {audio_ref_rows}")
            audio_rows = torch.cat([audio_ref_rows.to(device=device, dtype=torch.float32), target_audio], dim=0)
        else:
            audio_rows = target_audio

        schedule = H3TinySchedule(
            num_steps=steps,
            video_shift=H3_TINY_VIDEO_SHIFT if flow_shift is None else float(flow_shift),
            audio_shift=H3_TINY_AUDIO_SHIFT if audio_flow_shift is None else float(audio_flow_shift),
        )
        row_plan = schedule.row_plan(
            video_indices=video_indices,
            audio_indices=audio_indices,
            num_condition_video_rows=n_cond_video,
            num_condition_audio_rows=n_cond_audio,
            num_text_tokens=text_len,
            keyframe_noise_aug=float(keyframe_noise_aug),
            device=device,
        )

        for step_index, (timestep, endpoint_timestep, timestep_indices) in enumerate(row_plan):
            video_vel, audio_vel = self.transformer(
                hidden_states=video_rows[None],
                audio_hidden_states=audio_rows[None],
                encoder_hidden_states=prompt_batched,
                timestep=timestep,
                endpoint_timestep=endpoint_timestep,
                timestep_indices=timestep_indices,
                token_tags=token_tags,
                position_ids=position_ids,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
            )
            video_rows[n_cond_video:] = schedule.step(
                video_vel[0, n_cond_video:].float(),
                video_rows[n_cond_video:],
                step_index,
                audio=False,
            )
            audio_rows[n_cond_audio:] = schedule.step(
                audio_vel[0, n_cond_audio:].float(),
                audio_rows[n_cond_audio:],
                step_index,
                audio=True,
            )

        video_target = video_rows[n_cond_video:]
        audio_target = audio_rows[n_cond_audio:]
        result: dict[str, Any] = {
            "plan": plan,
            "task": task,
            "endpoint_mode": self.endpoint_mode,
            "num_inference_steps": steps,
        }
        if return_latent_rows:
            result["video_rows"] = video_target
            result["audio_rows"] = audio_target
            return result

        video_latent = minimax_h3_unpatchify_video_tokens(
            video_target,
            latent_shape=(
                plan["latent_t"],
                plan["latent_h"] // _PATCH_H,
                plan["latent_w"] // _PATCH_W,
                _VIDEO_LATENT_CHANNELS,
            ),
            patch_size=(1, _PATCH_H, _PATCH_W),
        )
        audio_latent = minimax_h3_unpack_audio_tokens(
            audio_target,
            audio_t=int(audio_target.shape[0]),
            audio_channel=_AUDIO_CHANNELS,
        )
        result["video_latent"] = video_latent
        result["audio_latent"] = audio_latent
        if isinstance(self.video_vae, torch.nn.Module):
            result["video"] = self._decode_video(video_latent.to(self.component_devices["video_vae"]))
        if isinstance(self.audio_vae, torch.nn.Module):
            result["audio"] = self._decode_audio(audio_latent.to(self.component_devices["audio_vae"]))
        return result

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
        """Generate an H3-Tiny video+audio clip and optionally write its MP4 artifact."""
        if video is not None:
            raise ValueError("H3-Tiny in-tree pipeline currently accepts image references/keyframes, not video input.")

        request = dict(operator_kwargs or {})
        request.update(kwargs)
        request.pop("task_name", None)
        resolved_prompt = prompt if prompt is not None else request.pop("prompt", None)

        validated = self.operator.validate_request(
            prompt=resolved_prompt,
            task=request.pop("task", None),
            workflow=request.pop("workflow", None),
            images=images if images is not None else request.pop("images", None),
            image=request.pop("image", None),
            last_image=request.pop("last_image", None),
            references=request.pop("references", request.pop("reference_images", None)),
            num_frames=request.pop("num_frames", H3_TINY_DEFAULT_NUM_FRAMES),
            num_inference_steps=request.pop("num_inference_steps", request.pop("steps", None)),
            endpoint_mode=self.endpoint_mode,
            seed=request.pop("seed", self.DEFAULT_SEED),
            fps=request.pop("fps", H3_TINY_FPS),
            height=request.pop("height", None),
            width=request.pop("width", None),
            short_edge=request.pop("short_edge", self.DEFAULT_SHORT_EDGE),
            aspect_ratio=request.pop("aspect_ratio", "16:9"),
            duration_seconds=request.pop("duration_seconds", None),
        )
        flow_shift = float(request.pop("flow_shift", H3_TINY_VIDEO_SHIFT))
        audio_flow_shift = float(request.pop("audio_flow_shift", H3_TINY_AUDIO_SHIFT))
        return_latent_rows = bool(request.pop("return_latent_rows", False))
        if request:
            raise TypeError(f"Unsupported H3-Tiny call options: {sorted(request)}")

        task = validated["task"]
        if task == "ref2va" and self.workflow != "ref2va":
            raise ValueError("Load a Ref2VA H3-Tiny checkpoint (workflow='ref2va') for reference conditioning.")
        if task != "ref2va" and self.workflow == "ref2va":
            raise ValueError("Load an FL2VA H3-Tiny checkpoint (workflow='fl2va' or 't2va') for text/keyframe tasks.")

        plan = self._resolve_plan(
            task=validated["model_task"],
            short_edge=validated["short_edge"],
            aspect_ratio=validated["aspect_ratio"],
            duration_seconds=validated["duration_seconds"],
            height=validated["height"],
            width=validated["width"],
            num_frames=validated["num_frames"],
        )

        raw_visuals = validated["references"] or validated["keyframes"]
        presentation = [
            self._fit_image(self._as_image(item), height=plan["height"], width=plan["width"]) for item in raw_visuals
        ]
        prompt_embeds, text_tags = self._encode_presentation(validated["prompt"], presentation)

        condition_rows = None
        keyframe_indices = validated["keyframe_indices"]
        ref_blocks = None
        if validated["keyframes"]:
            encoded = [self._encode_image(img) for img in presentation]
            condition_rows = torch.cat(
                [self._patchify_condition(latent, seed=validated["seed"]) for latent in encoded], dim=0
            )
        elif validated["references"]:
            encoded = [self._encode_image(img) for img in presentation]
            condition_rows = torch.cat(
                [self._patchify_condition(latent, seed=validated["seed"]) for latent in encoded], dim=0
            )
            ref_blocks = [
                {"kind": "image", "latent_h": int(latent.shape[-2]), "latent_w": int(latent.shape[-1])}
                for latent in encoded
            ]

        result = self.generate(
            prompt_embeds=prompt_embeds,
            task=validated["model_task"],
            short_edge=validated["short_edge"],
            aspect_ratio=validated["aspect_ratio"],
            duration_seconds=validated["duration_seconds"],
            height=validated["height"],
            width=validated["width"],
            num_frames=validated["num_frames"],
            num_inference_steps=validated["num_inference_steps"],
            seed=validated["seed"],
            flow_shift=flow_shift,
            audio_flow_shift=audio_flow_shift,
            keyframe_frame_indices=keyframe_indices,
            keyframe_cond_rows=condition_rows,
            ref_blocks=ref_blocks,
            text_token_tags=text_tags,
            return_latent_rows=return_latent_rows,
        )
        result.update(
            model_id=self.MODEL_ID,
            variant=self.variant,
            workflow=self.workflow,
            endpoint_mode=self.endpoint_mode,
            fps=H3_TINY_FPS,
            inference_steps=validated["num_inference_steps"],
        )
        destination = Path(output_path or Path(output_dir or ".") / "h3-tiny.mp4").expanduser()
        if "video" in result:
            result["output_path"] = self._write_mp4_with_audio(
                video=result["video"],
                audio=result.get("audio"),
                output_path=destination,
                fps=H3_TINY_FPS,
            )
            result["artifact_path"] = result["output_path"]
        return result if return_dict else result.get("artifact_path", result)


__all__ = ["H3_TINY_VARIANTS", "H3TinyPipeline", "H3TinyVariantSpec"]
