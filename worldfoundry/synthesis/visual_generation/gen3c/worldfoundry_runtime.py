"""Compatibility facade over WorldFoundry's native GEN3C pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.pipelines.gen3c.pipeline_gen3c import Gen3CPipeline

from .runtime_env import DEFAULT_GEN3C_MOGE1_REPO, DEFAULT_GEN3C_NEGATIVE_PROMPT


class Gen3CRuntime:
    """Keep the historical synthesis API without owning a second runtime."""

    def __init__(
        self,
        model_root: str | None = None,
        checkpoint_dir: str | None = None,
        moge_pretrained: str | None = None,
        device: str = "cuda",
        defaults: Mapping[str, Any] | None = None,
        *,
        pipeline: Gen3CPipeline | None = None,
    ) -> None:
        self.model_root = model_root or "worldfoundry-native-diffusion"
        self.checkpoint_dir = checkpoint_dir
        self.moge_pretrained = moge_pretrained or DEFAULT_GEN3C_MOGE1_REPO
        self.device = device
        self.defaults = dict(defaults or {})
        self.pipeline = pipeline or Gen3CPipeline.from_pretrained(
            checkpoint_dir,
            device=device,
            moge_pretrained=moge_pretrained,
            torch_dtype=self.defaults.get("torch_dtype", self.defaults.get("dtype")),
            offload_mode=self.defaults.get("offload_mode", "block"),
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str | None,
        args: Any = None,
        device: str | None = None,
        checkpoint_dir: str | None = None,
        moge_path: str | None = None,
        moge_pretrained: str | None = None,
        default_trajectory: str = "left",
        default_camera_rotation: str = "center_facing",
        default_movement_distance: float = 0.3,
        guidance: float = 1.0,
        num_steps: int = 35,
        num_video_frames: int = 121,
        fps: int = 24,
        height: int = 704,
        width: int = 1280,
        seed: int = 1,
        negative_prompt: str = DEFAULT_GEN3C_NEGATIVE_PROMPT,
        **kwargs: Any,
    ) -> "Gen3CRuntime":
        del args
        if moge_path is not None and str(moge_path).strip():
            raise ValueError("MoGe code is native; pass `moge_pretrained` for weights only.")
        source = checkpoint_dir or pretrained_model_path
        defaults = {
            "default_trajectory": default_trajectory,
            "default_camera_rotation": default_camera_rotation,
            "default_movement_distance": default_movement_distance,
            "guidance": guidance,
            "num_steps": num_steps,
            "num_video_frames": num_video_frames,
            "fps": fps,
            "height": height,
            "width": width,
            "seed": seed,
            "negative_prompt": negative_prompt,
            **kwargs,
        }
        resolved_device = device or "cuda"
        pipeline = Gen3CPipeline.from_pretrained(
            source,
            device=resolved_device,
            moge_pretrained=moge_pretrained,
            torch_dtype=defaults.get("torch_dtype", defaults.get("weight_dtype", defaults.get("dtype"))),
            offload_mode=defaults.get("offload_mode", "block"),
        )
        return cls(
            checkpoint_dir=source,
            moge_pretrained=moge_pretrained,
            device=resolved_device,
            defaults=defaults,
            pipeline=pipeline,
        )

    def predict(
        self,
        image: Any,
        prompt: str = "",
        trajectory: str | None = None,
        camera_rotation: str | None = None,
        movement_distance: float | None = None,
        output_dir: str | None = None,
        scene_name: str = "gen3c_scene",
        negative_prompt: str | None = None,
        return_dict: bool = False,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> Any:
        del show_progress
        output_root = (
            Path(output_dir).expanduser().resolve() / scene_name
            if output_dir is not None
            else Path(tempfile.mkdtemp(prefix="gen3c_")) / scene_name
        )
        output_root.mkdir(parents=True, exist_ok=True)

        def option(name: str, fallback: Any) -> Any:
            return kwargs.pop(name, self.defaults.get(name, fallback))

        # These flags configured the removed official subprocess. Native component
        # loading and camera rendering own the equivalent lifecycle now.
        for legacy_name in (
            "disable_prompt_upsampler",
            "disable_guardrail",
            "disable_prompt_encoder",
            "offload_diffusion_transformer",
            "offload_tokenizer",
            "offload_text_encoder_model",
            "offload_prompt_upsampler",
            "offload_guardrail_models",
            "num_gpus",
            "save_buffer",
            "filter_points_threshold",
            "foreground_masking",
            "noise_aug_strength",
            "moge_pretrained",
        ):
            kwargs.pop(legacy_name, None)

        selected_trajectory = trajectory or option("default_trajectory", "left")
        selected_rotation = camera_rotation or option("default_camera_rotation", "center_facing")
        selected_distance = (
            movement_distance
            if movement_distance is not None
            else float(option("default_movement_distance", 0.3))
        )
        fps = int(option("fps", 24))
        result = self.pipeline(
            images=image,
            prompt=prompt,
            trajectory=str(selected_trajectory),
            camera_rotation=str(selected_rotation),
            movement_distance=float(selected_distance),
            negative_prompt=negative_prompt or option("negative_prompt", DEFAULT_GEN3C_NEGATIVE_PROMPT),
            height=int(option("height", 704)),
            width=int(option("width", 1280)),
            num_frames=int(kwargs.pop("num_frames", option("num_video_frames", 121))),
            num_inference_steps=int(kwargs.pop("num_inference_steps", option("num_steps", 35))),
            guidance_scale=float(kwargs.pop("guidance_scale", option("guidance", 1.0))),
            fps=fps,
            seed=int(option("seed", 1)),
            output_path=output_root / "video.mp4",
            return_dict=True,
            **kwargs,
        )
        video = result["video"]
        compatibility_result = {
            **result,
            "frames": video,
            "generated_video_path": result.get("artifact_path"),
            "output_dir": str(output_root),
            "scene_name": scene_name,
            "camera_rotation": str(selected_rotation),
            "movement_distance": float(selected_distance),
            "fps": fps,
            "prompt": prompt,
            "negative_prompt": negative_prompt or self.defaults.get("negative_prompt"),
            "num_video_frames": int(video.shape[2]) if getattr(video, "ndim", 0) == 5 else 121,
            "checkpoint_dir": self.checkpoint_dir,
            "model_root": self.model_root,
            "moge_pretrained": self.moge_pretrained,
        }
        return compatibility_result if return_dict else video


__all__ = ["Gen3CRuntime"]
