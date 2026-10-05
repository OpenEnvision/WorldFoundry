"""SANA-WM product adapter over the canonical native diffusion runner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from ..native_diffusion import NativeVisualDiffusionPipeline


class SanaWMPipeline(NativeVisualDiffusionPipeline):
    """Resident camera-controlled world generation without a private runtime."""

    MODEL_ID = "sana-wm"
    OWNER = "SANA-WM"
    CHECKPOINT_ROLES = ("dit", "text-encoder", "tokenizer", "codec")
    PRIMARY_CHECKPOINT_ROLE = "dit"
    GENERATION_TYPE = "i2v"
    ACCEPTS_IMAGES = True
    REQUIRES_IMAGES = True
    ACCEPTS_INTERACTIONS = True
    DEFAULT_HEIGHT = 704
    DEFAULT_WIDTH = 1280
    DEFAULT_NUM_FRAMES = 161
    DEFAULT_NUM_INFERENCE_STEPS = 60
    DEFAULT_GUIDANCE_SCALE = 5.0
    DEFAULT_FPS = 16
    DEFAULT_NEGATIVE_PROMPT = ""
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 9.8}
    NUM_INFERENCE_STEP_ALIASES = ("step", "infer_steps")
    REQUEST_INPUT_DEFAULTS = {
        "camera_to_world": None,
        "camera_actions": None,
        "intrinsics": None,
    }
    REQUEST_INPUT_ALIASES = {
        "camera_poses": "camera_to_world",
        "action": "camera_actions",
        "camera_action": "camera_actions",
        "interaction_signal": "camera_actions",
        "interactions": "camera_actions",
    }

    @classmethod
    def _checkpoint_overrides(
        cls,
        model_path: str | Mapping[str, Any] | None,
        options: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        explicit = options.get("checkpoint_overrides")
        if isinstance(explicit, Mapping):
            return {str(name): value for name, value in explicit.items()}

        source = options.get(
            "checkpoint_path",
            options.get(
                "checkpoint_dir",
                options.get("pretrained_model_path", options.get("model_path", model_path)),
            ),
        )
        if not isinstance(source, (str, Path)):
            checkpoint_root = os.getenv("WORLDFOUNDRY_CKPT_DIR") or os.getenv("WORLDEVALS_CKPT_DIR")
            if not checkpoint_root:
                return super()._checkpoint_overrides(model_path, options)
            source = Path(checkpoint_root).expanduser() / "Efficient-Large-Model--SANA-WM_bidirectional"

        source_value = str(source)
        if "://" in source_value:
            return super()._checkpoint_overrides(model_path, options)
        sana_root = Path(source_value).expanduser()
        gemma_source = options.get("gemma_path") or options.get("text_encoder_path")
        gemma_root = (
            Path(gemma_source).expanduser()
            if isinstance(gemma_source, (str, Path))
            else sana_root.parent / "Efficient-Large-Model--gemma-2-2b-it"
        )
        codec_source = options.get("codec_path") or options.get("vae_path")
        codec_root = (
            Path(codec_source).expanduser()
            if isinstance(codec_source, (str, Path))
            else sana_root
        )
        return {
            "dit": str(sana_root),
            "text-encoder": str(gemma_root),
            "tokenizer": str(gemma_root),
            "codec": str(codec_root),
        }

    def __init__(self, *, native_pipeline, device: str, model_id: str | None = None) -> None:
        super().__init__(native_pipeline=native_pipeline, device=device, model_id=model_id)
        self._realtime_config: dict[str, Any] | None = None

    def prepare_realtime(self) -> dict[str, Any]:
        return {
            "realtime_spec": {
                "supports_prompt_updates": True,
                "supports_camera_actions": True,
                "resident": True,
                "runtime": "worldfoundry-native-diffusion",
            },
            "runtime_info": {
                "model_id": self.model_id,
                "device": str(self.native_pipeline.device),
                "dtype": str(self.native_pipeline.dtype),
            },
        }

    def configure_realtime(
        self,
        images: Any,
        prompt: str = "",
        seed: int = 42,
        fps: int = 16,
        window_frames: int | None = None,
        step: int = 60,
        cfg_scale: float = 5.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(images, (str, Path)):
            with Image.open(images) as source:
                images = source.convert("RGB")
        if not isinstance(images, Image.Image):
            raise ValueError("SANA-WM realtime requires a PIL image or image path")
        if not str(prompt).strip():
            raise ValueError("SANA-WM realtime requires a prompt")
        self._realtime_config = {
            "images": images.copy(),
            "prompt": str(prompt),
            "seed": int(seed),
            "fps": int(fps),
            "num_frames": int(window_frames or self.DEFAULT_NUM_FRAMES),
            "num_inference_steps": int(step),
            "guidance_scale": float(cfg_scale),
            **kwargs,
        }
        return self.prepare_realtime()

    def stream_realtime(
        self,
        interactions: Sequence[str] | None = None,
        prompt: str | None = None,
        realtime_segments: Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if realtime_segments:
            raise ValueError("use frame-rate interactions or explicit camera_to_world poses")
        if self._realtime_config is None:
            raise RuntimeError("SANA-WM realtime session is not configured")
        options = {**self._realtime_config, **kwargs}
        if prompt is not None:
            options["prompt"] = prompt
        if seed is not None:
            options["seed"] = int(seed)
        return self(interactions=interactions, return_dict=True, **options)

    def reset_realtime(self) -> None:
        self._realtime_config = None

    def __call__(
        self,
        images: Any = None,
        prompt: str = "",
        interactions: Sequence[str] | None = None,
        fps: int = 16,
        num_frames: int = 161,
        step: int = 60,
        cfg_scale: float = 5.0,
        seed: int = 42,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        return super().__call__(
            prompt=prompt,
            images=images,
            interactions=interactions,
            fps=fps,
            num_frames=int(kwargs.pop("window_frames", num_frames)),
            num_inference_steps=step,
            guidance_scale=cfg_scale,
            seed=seed,
            return_dict=return_dict,
            **kwargs,
        )


class SanaWMStreamingPipeline(SanaWMPipeline):
    """Chunk-causal SANA-WM with the released streaming refiner and VAE."""

    MODEL_ID = "sana-wm-streaming"
    CHECKPOINT_ROLES = (
        "dit",
        "text-encoder",
        "tokenizer",
        "codec",
        "refiner",
        "refiner-connectors",
        "refiner-text-encoder",
    )
    DEFAULT_NUM_FRAMES = 241
    DEFAULT_NUM_INFERENCE_STEPS = 4
    DEFAULT_GUIDANCE_SCALE = 1.0
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 8.0}
    REQUEST_INPUT_DEFAULTS = {
        "camera_to_world": None,
        "camera_actions": None,
        "intrinsics": None,
        "refiner_seed": 42,
    }

    @classmethod
    def _checkpoint_overrides(
        cls,
        model_path: str | Mapping[str, Any] | None,
        options: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        explicit = options.get("checkpoint_overrides")
        if isinstance(explicit, Mapping):
            return {str(name): value for name, value in explicit.items()}

        source = options.get(
            "checkpoint_path",
            options.get(
                "checkpoint_dir",
                options.get("pretrained_model_path", options.get("model_path", model_path)),
            ),
        )
        if not isinstance(source, (str, Path)):
            checkpoint_root = os.getenv("WORLDFOUNDRY_CKPT_DIR") or os.getenv(
                "WORLDEVALS_CKPT_DIR"
            )
            if not checkpoint_root:
                return None
            source = Path(checkpoint_root).expanduser() / "Efficient-Large-Model--SANA-WM_streaming"

        source_value = str(source)
        if "://" in source_value:
            return {name: source_value for name in cls.CHECKPOINT_ROLES}
        streaming_root = Path(source_value).expanduser()
        gemma_source = options.get("gemma_path") or options.get("text_encoder_path")
        gemma_root = (
            Path(gemma_source).expanduser()
            if isinstance(gemma_source, (str, Path))
            else streaming_root.parent / "Efficient-Large-Model--gemma-2-2b-it"
        )
        return {
            "dit": str(streaming_root),
            "text-encoder": str(gemma_root),
            "tokenizer": str(gemma_root),
            "codec": str(streaming_root),
            "refiner": str(streaming_root),
            "refiner-connectors": str(streaming_root),
            "refiner-text-encoder": str(streaming_root),
        }

    def configure_realtime(
        self,
        images: Any,
        prompt: str = "",
        seed: int = 42,
        fps: int = 16,
        window_frames: int | None = None,
        step: int = 4,
        cfg_scale: float = 1.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return super().configure_realtime(
            images=images,
            prompt=prompt,
            seed=seed,
            fps=fps,
            window_frames=window_frames or self.DEFAULT_NUM_FRAMES,
            step=step,
            cfg_scale=cfg_scale,
            **kwargs,
        )

    def __call__(
        self,
        images: Any = None,
        prompt: str = "",
        interactions: Sequence[str] | None = None,
        fps: int = 16,
        num_frames: int = 241,
        step: int = 4,
        cfg_scale: float = 1.0,
        seed: int = 42,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        return super().__call__(
            images=images,
            prompt=prompt,
            interactions=interactions,
            fps=fps,
            num_frames=num_frames,
            step=step,
            cfg_scale=cfg_scale,
            seed=seed,
            return_dict=return_dict,
            **kwargs,
        )


__all__ = ["SanaWMPipeline", "SanaWMStreamingPipeline"]
