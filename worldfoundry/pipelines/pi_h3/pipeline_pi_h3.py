# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hansheng Chen.
# Adapted from Lakonik/LakonLab (Apache-2.0), revision 2ee2b334374ec91b7cee371a85a524bf4ba644d3:
#   lakonlab/pipelines/pipeline_minimax_h3.py
#   demo/example_pih3_pipeline.py
#   demo/example_pih3_vae_pipeline.py
"""pi-H3 2-step and 4-step MiniMax-H3 pipeline with optional DWD pixel decoding."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model.models.denoisers.pi_h3 import load_pi_h3_transformer
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.pi_h3.sol_dwd_head import (
    RGBColorEncoder,
)
from worldfoundry.base_models.diffusion_model.schedulers.pi_h3 import (
    PI_H3_DEFAULT_AUDIO_SHIFT,
    PI_H3_DEFAULT_FINAL_STEP_SIZE_SCALE,
    PI_H3_DEFAULT_VIDEO_SHIFT,
    PI_H3_MAX_RAW_T,
    pi_h3_flow_euler_sigmas,
    pi_h3_flow_euler_step,
)
from worldfoundry.core.io.paths import hfd_root_path
from worldfoundry.operators.pi_h3_operator import (
    PI_H3_SUPPORTED_FPS,
    PI_H3_VARIANT_SPECS,
    PiH3Operator,
    normalize_pi_h3_variant,
)
from worldfoundry.pipelines.minimax._minimax_h3 import (
    MiniMaxH3DenoiseBranch,
    minimax_h3_packed_sequence,
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from worldfoundry.pipelines.minimax._minimax_h3.denoise_loop import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
)
from worldfoundry.pipelines.minimax.pipeline_minimax_h3 import (
    NativeMiniMaxH3Pipeline,
    _resolve_component_devices,
)

_VIDEO_LATENT_CHANNELS = 24
_AUDIO_LATENT_CHANNELS = 32
_AUDIO_CHANNELS = 2
_PATCH_H = 2
_PATCH_W = 2


class PiH3Pipeline(NativeMiniMaxH3Pipeline):
    """LakonLab pi-H3 2/4-step MiniMax-H3 pipeline with optional Sol-H3 DWD head."""

    MODEL_ID = "pi-h3"
    MODEL_PATH_OPTION = "base_model_path"
    DEFAULT_VARIANT = "4step-dwd"
    DEFAULT_NUM_INFERENCE_STEPS = 4
    DEFAULT_SHORT_EDGE = 768
    DEFAULT_ASPECT_RATIO = "16:9"
    DEFAULT_DURATION_SECONDS = 5.0
    DEFAULT_FLOW_SHIFT = PI_H3_DEFAULT_VIDEO_SHIFT
    DEFAULT_AUDIO_FLOW_SHIFT = PI_H3_DEFAULT_AUDIO_SHIFT

    def __init__(
        self,
        *,
        variant: str = DEFAULT_VARIANT,
        adapter_path: str | Path | None = None,
        base_model_path: str | Path | None = None,
        color_encoder: RGBColorEncoder | None = None,
        operator: PiH3Operator | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_variant = normalize_pi_h3_variant(variant)
        super().__init__(model_id=model_id or self.MODEL_ID, **kwargs)
        self.variant = resolved_variant
        self.use_dwd = bool(PI_H3_VARIANT_SPECS[resolved_variant]["dwd"])
        self.adapter_path = None if adapter_path is None else Path(adapter_path).expanduser().resolve()
        self.base_model_path = None if base_model_path is None else Path(base_model_path).expanduser().resolve()
        self.color_encoder = color_encoder if color_encoder is not None else RGBColorEncoder(use_affine_norm=True)
        self.operator = operator if operator is not None else PiH3Operator(variant=resolved_variant)

    @classmethod
    def _load_pruned_video_vae(cls, root: Path, *, device: str, required: bool) -> Any:
        vae_dir = root / "vae"
        if vae_dir.is_dir():
            from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_hyperflow import (
                load_hyperflow_video_vae,
            )

            return load_hyperflow_video_vae(vae_dir).to(device).eval()
        if (root / "video_vae").is_dir():
            return cls._load_video_vae(root, device=device)
        if required:
            raise FileNotFoundError(f"MiniMax-H3 video VAE directory not found under {root}")
        return None

    @classmethod
    def _load_pruned_audio_vae(cls, root: Path, *, device: str) -> Any:
        audio_dir = cls._subdir(root, "audio_vae")
        from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_hyperflow import (
            load_hyperflow_audio_vae,
        )

        return load_hyperflow_audio_vae(audio_dir).to(device).eval()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        component_devices: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> PiH3Pipeline:
        """Load ``MiniMax-H3-Pruned`` and one ``Lakonik/pi-H3`` adapter variant."""

        options = cls._runtime_options(
            model_path,
            None if required_components is None else dict(required_components),
            kwargs,
        )
        cls._strip_framework_loading_options(options)
        for key in ("runtime_profile", "pipeline_binding", "variant_id"):
            options.pop(key, None)

        variant = normalize_pi_h3_variant(str(options.pop("variant", cls.DEFAULT_VARIANT)))
        variant_spec = PI_H3_VARIANT_SPECS[variant]
        use_dwd = bool(variant_spec["dwd"])

        base_model_path = options.pop("base_model_path", options.pop("model_path", None))
        if base_model_path is None:
            base_model_path = hfd_root_path("multimodalart--MiniMax-H3-Pruned")
        base_root = Path(str(base_model_path)).expanduser().resolve()

        adapter_path = options.pop("adapter_path", None)
        adapter_root = options.pop("adapter_root", None)
        if adapter_path is not None:
            resolved_adapter_path = Path(str(adapter_path)).expanduser().resolve()
        elif adapter_root is not None:
            resolved_adapter_path = Path(str(adapter_root)).expanduser().resolve() / variant_spec["subfolder"]
        else:
            resolved_adapter_path = (hfd_root_path("Lakonik--pi-H3") / variant_spec["subfolder"]).resolve()

        transformer_subfolder = str(options.pop("transformer_subfolder", "transformer"))
        processor_path = options.pop("processor_path", None)
        dtype_raw = options.pop("dtype", torch.bfloat16)
        dtype = getattr(torch, dtype_raw) if isinstance(dtype_raw, str) else dtype_raw
        lora_scale = float(options.pop("lora_scale", 1.0))
        load_video_vae = options.pop("load_video_vae", None)

        transformer = options.pop("transformer", None)
        video_vae = options.pop("video_vae", None)
        audio_vae = options.pop("audio_vae", None)
        text_encoder = options.pop("text_encoder", None)
        tokenizer = options.pop("tokenizer", None)
        processor = options.pop("processor", None)
        placements = _resolve_component_devices(device, options.pop("component_devices", component_devices))

        if transformer is None:
            transformer_dir = cls._subdir(base_root, transformer_subfolder)
            transformer = load_pi_h3_transformer(
                transformer_dir,
                resolved_adapter_path,
                device=placements["transformer"],
                dtype=dtype,
                lora_scale=lora_scale,
            )

        if video_vae is None and load_video_vae is not False:
            video_vae = cls._load_pruned_video_vae(
                base_root,
                device=placements["video_vae"],
                required=not use_dwd,
            )
        if audio_vae is None:
            audio_vae = cls._load_pruned_audio_vae(base_root, device=placements["audio_vae"])
        if text_encoder is None:
            text_encoder = cls._load_text_encoder(base_root, device=placements["text_encoder"])
        if tokenizer is None:
            tokenizer = cls._load_tokenizer(base_root)
        if processor is None:
            proc_root = (
                Path(str(processor_path)).expanduser().resolve()
                if processor_path is not None
                else cls._processor_root(base_root, None)
            )
            if proc_root.is_dir():
                processor = cls._load_processor(proc_root)

        return cls(
            transformer=transformer,
            video_vae=video_vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            device=placements["transformer"],
            component_devices=placements,
            variant=variant,
            adapter_path=resolved_adapter_path,
            base_model_path=base_root,
            model_id=model_id or cls.MODEL_ID,
        )

    @torch.inference_mode()
    def generate(
        self,
        *,
        prompt_embeds: torch.Tensor,
        task: str = "t2va",
        short_edge: int = DEFAULT_SHORT_EDGE,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        duration_seconds: float = DEFAULT_DURATION_SECONDS,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        num_inference_steps: int | None = None,
        seed: int = 0,
        flow_shift: float | None = None,
        audio_flow_shift: float | None = None,
        max_raw_t: float | None = None,
        min_raw_t: float = 0.0,
        final_step_size_scale: float = PI_H3_DEFAULT_FINAL_STEP_SIZE_SCALE,
        keyframe_frame_indices: tuple[int, ...] | None = None,
        keyframe_cond_rows: torch.Tensor | None = None,
        text_token_tags: torch.Tensor | None = None,
        return_latent_rows: bool = False,
        **unused_kwargs: Any,
    ) -> dict[str, Any]:
        """Run the coupled pi-H3 FlowEulerODE denoise loop with optional Sol-H3 DWD decoding."""

        if unused_kwargs:
            raise TypeError(f"Unsupported pi-H3 generate options: {sorted(unused_kwargs)}")
        if self.transformer is None:
            raise RuntimeError("pi-H3 transformer is not initialized.")

        device = torch.device(self.device)
        steps = int(PI_H3_VARIANT_SPECS[self.variant]["steps"] if num_inference_steps is None else num_inference_steps)
        resolved_max_raw_t = PI_H3_MAX_RAW_T[steps] if max_raw_t is None and steps in PI_H3_MAX_RAW_T else max_raw_t
        v_shift = float(self.DEFAULT_FLOW_SHIFT if flow_shift is None else flow_shift)
        a_shift = float(self.DEFAULT_AUDIO_FLOW_SHIFT if audio_flow_shift is None else audio_flow_shift)

        plan = self._resolve_plan(
            task=task,
            short_edge=short_edge,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            height=height,
            width=width,
            num_frames=num_frames,
        )

        include_keyframe = task == "fl2va" or keyframe_cond_rows is not None
        text_len = int(prompt_embeds.shape[0])
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
        branch_tags = packed["token_tags"].clone()
        if text_token_tags is not None:
            supplied_tags = text_token_tags.reshape(-1).to(dtype=branch_tags.dtype, device=branch_tags.device)
            if int(supplied_tags.numel()) != text_len:
                raise ValueError(f"text_token_tags length {int(supplied_tags.numel())} != prompt length {text_len}")
            branch_tags[:text_len] = supplied_tags

        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=prompt_embeds,
            token_tags=branch_tags,
            device=device,
        )

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        n_audio_rows = int(packed["audio_pos"].shape[0])
        initial_video = minimax_h3_patchify_video_latent(
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
        )
        audio_generator = torch.Generator(device="cpu").manual_seed(int(seed))
        initial_audio = torch.randn(
            n_audio_rows,
            _AUDIO_LATENT_CHANNELS,
            generator=audio_generator,
            dtype=torch.float32,
        )

        target_noise = initial_video.to(device=device, dtype=torch.float32, copy=True)
        if keyframe_cond_rows is not None:
            cond_anchor = keyframe_cond_rows.to(device=device, dtype=torch.float32)
            if int(cond_anchor.shape[0]) != branch.video_target_start:
                raise ValueError(
                    f"keyframe_cond_rows {int(cond_anchor.shape[0])} != layout cond rows {branch.video_target_start}"
                )
            video_rows = torch.cat([cond_anchor, target_noise], dim=0)
        elif branch.video_target_start != 0:
            raise ValueError(f"layout has {branch.video_target_start} cond rows but keyframe_cond_rows is None")
        else:
            video_rows = target_noise
        audio_rows = initial_audio.to(device=device, dtype=torch.float32, copy=True)

        sigmas_video = pi_h3_flow_euler_sigmas(
            steps,
            shift=v_shift,
            max_raw_t=resolved_max_raw_t,
            min_raw_t=min_raw_t,
            final_step_size_scale=final_step_size_scale,
        )
        sigmas_audio = pi_h3_flow_euler_sigmas(
            steps,
            shift=a_shift,
            max_raw_t=resolved_max_raw_t,
            min_raw_t=min_raw_t,
            final_step_size_scale=final_step_size_scale,
        )
        video_timesteps = [1.0 - sigma for sigma in sigmas_video[:-1]]
        audio_timesteps = [1.0 - sigma for sigma in sigmas_audio[:-1]]
        timestep_plan = branch.prepare_timestep_plan(
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
            imgvid_cond_noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
            audio_ref_cond_noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
        )

        run_dwd = self.use_dwd and bool(getattr(self.transformer, "has_dwd_head", False))
        if self.use_dwd and not run_dwd:
            raise RuntimeError(f"pi-H3 variant {self.variant!r} requires a transformer with a SolH3DWDHead.")

        dwd_features: torch.Tensor | None = None
        video_target_slice = branch.video_target_slice
        audio_target_slice = branch.audio_target_slice

        for step in range(steps):
            fk = branch.forward_kwargs(
                video_rows=video_rows,
                audio_rows=audio_rows,
                step_timesteps=timestep_plan[step],
            )
            if run_dwd and step == steps - 1:
                (v_video, v_audio), features = self.transformer(**fk, return_features=True)
                dwd_features = features["image"]
            else:
                v_video, v_audio = self.transformer(**fk)

            video_rows[video_target_slice] = pi_h3_flow_euler_step(
                video_rows[video_target_slice],
                v_video,
                sigma=sigmas_video[step],
                sigma_next=sigmas_video[step + 1],
                prediction_type="v",
            )
            audio_rows[audio_target_slice] = pi_h3_flow_euler_step(
                audio_rows[audio_target_slice],
                v_audio[audio_target_slice],
                sigma=sigmas_audio[step],
                sigma_next=sigmas_audio[step + 1],
                prediction_type="v",
            )

        video_target = video_rows[video_target_slice]
        audio_target = audio_rows[audio_target_slice]
        result: dict[str, Any] = {
            "plan": plan,
            "task": task,
            "variant": self.variant,
            "sigmas_video": sigmas_video,
            "sigmas_audio": sigmas_audio,
        }
        if return_latent_rows:
            result["video_rows"] = video_target
            result["audio_rows"] = audio_target
            if dwd_features is not None:
                result["dwd_features"] = dwd_features
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

        if run_dwd:
            assert dwd_features is not None
            token_grid = (plan["latent_t"], plan["latent_h"] // _PATCH_H, plan["latent_w"] // _PATCH_W)
            dwd_pixels = self.transformer.forward_dwd_head(dwd_features.unsqueeze(0), token_grid)
            result["dwd_pixels"] = dwd_pixels
            result["video"] = self.color_encoder.decode_to_unit_range(dwd_pixels.float())
        elif isinstance(self.video_vae, torch.nn.Module) or (
            self.video_vae is not None and (hasattr(self.video_vae, "decode_base") or hasattr(self.video_vae, "decode"))
        ):
            result["video"] = self._decode_video(video_latent.to(self.component_devices["video_vae"]))

        if isinstance(self.audio_vae, torch.nn.Module) or (
            self.audio_vae is not None and hasattr(self.audio_vae, "decode")
        ):
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
        """Generate a 2/4-step pi-H3 audio-video clip and optionally write its MP4 artifact."""

        if video is not None:
            raise ValueError("pi-H3 accepts text prompts and optional first/last keyframe images, not a video input.")
        request = dict(operator_kwargs or {})
        request.update(kwargs)
        request.pop("task_name", None)
        resolved_prompt = prompt if prompt is not None else request.pop("prompt", None)
        image = request.pop("image", None)
        last_image = request.pop("last_image", None)
        task = request.pop("task", request.pop("workflow", "auto"))
        requested_variant = request.pop("variant", None)
        num_inference_steps = request.pop("num_inference_steps", request.pop("steps", None))
        num_frames = request.pop("num_frames", None)
        duration_seconds = request.pop("duration_seconds", None)
        fps = request.pop("fps", PI_H3_SUPPORTED_FPS)
        height = request.pop("height", None)
        width = request.pop("width", None)
        short_edge = request.pop("short_edge", self.DEFAULT_SHORT_EDGE)
        aspect_ratio = request.pop("aspect_ratio", self.DEFAULT_ASPECT_RATIO)
        seed = request.pop("seed", 0)
        return_latent_rows = bool(request.pop("return_latent_rows", False))

        validated = self.operator.validate_request(
            prompt=resolved_prompt,
            images=images,
            image=image,
            last_image=last_image,
            task=task,
            variant=requested_variant,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            duration_seconds=duration_seconds,
            fps=fps,
            height=height,
            width=width,
            short_edge=short_edge,
            aspect_ratio=aspect_ratio,
            seed=seed,
            extra=request,
        )
        if validated["variant"] != self.variant:
            raise ValueError(
                f"Pipeline was loaded with variant {self.variant!r}, cannot serve requested variant "
                f"{validated['variant']!r}."
            )

        resolved_task = validated["task"]
        keyframes = [self._as_image(item) for item in validated["keyframes"]]
        model_task = "fl2va" if keyframes else "t2va"
        plan = self._resolve_plan(
            task=model_task,
            short_edge=validated["short_edge"],
            aspect_ratio=validated["aspect_ratio"],
            duration_seconds=validated["duration_seconds"],
            height=validated["height"],
            width=validated["width"],
            num_frames=validated["num_frames"],
        )
        fitted = [self._fit_image(img, height=plan["height"], width=plan["width"]) for img in keyframes]
        prompt_embeds, text_tags = self._encode_presentation(validated["prompt"], fitted)

        condition_rows = None
        keyframe_indices = None
        if fitted:
            encoded = [self._encode_image(img) for img in fitted]
            condition_rows = torch.cat(
                [self._patchify_condition(latent, seed=validated["seed"]) for latent in encoded],
                dim=0,
            )
            keyframe_indices = {"i2va": (0,), "l2va": (-1,), "fl2va": (0, -1)}[resolved_task]

        result = self.generate(
            prompt_embeds=prompt_embeds,
            task=model_task,
            short_edge=validated["short_edge"],
            aspect_ratio=validated["aspect_ratio"],
            duration_seconds=validated["duration_seconds"],
            height=validated["height"],
            width=validated["width"],
            num_frames=validated["num_frames"],
            num_inference_steps=validated["num_inference_steps"],
            seed=validated["seed"],
            keyframe_frame_indices=keyframe_indices,
            keyframe_cond_rows=condition_rows,
            text_token_tags=text_tags,
            return_latent_rows=return_latent_rows,
        )
        result.update(
            model_id=self.MODEL_ID,
            variant=self.variant,
            dwd=self.use_dwd,
            fps=PI_H3_SUPPORTED_FPS,
            inference_steps=validated["num_inference_steps"],
        )
        should_write = (output_path is not None or output_dir is not None or not return_dict) and "video" in result
        if should_write:
            destination = Path(output_path or Path(output_dir or ".") / "pi-h3.mp4").expanduser()
            result["output_path"] = self._write_mp4_with_audio(
                video=result["video"],
                audio=result.get("audio"),
                output_path=destination,
                fps=PI_H3_SUPPORTED_FPS,
            )
            result["artifact_path"] = result["output_path"]
        return result if return_dict else result.get("artifact_path", result)


__all__ = ["PiH3Pipeline"]
