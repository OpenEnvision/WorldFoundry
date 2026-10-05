"""Public SkyReels-V3 pipeline backed by WorldFoundry native diffusion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from worldfoundry.core import load_pil_image
from worldfoundry.core.io import load_yaml, resolve_data_path
from worldfoundry.pipelines.native_diffusion_video import NativeTextToVideoPipeline


class SkyReelsV3Pipeline(NativeTextToVideoPipeline):
    """Reference-to-video adapter for the native SkyReels-V3 recipe."""

    MODEL_ID = "skyreels-v3"
    OWNER = "SkyReels-V3"
    CHECKPOINT_ROLES = ("dit", "text-encoder", "tokenizer", "vae")
    GENERATION_TYPE = "i2v"
    ACCEPTS_IMAGES = True
    REQUIRES_IMAGES = True
    DEFAULT_HEIGHT = 720
    DEFAULT_WIDTH = 1280
    DEFAULT_NUM_FRAMES = 121
    DEFAULT_NUM_INFERENCE_STEPS = 8
    DEFAULT_GUIDANCE_SCALE = 1.0
    DEFAULT_FPS = 24
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 5.0}
    REQUEST_INPUT_DEFAULTS = {"image_guidance_scale": 1.0}

    @staticmethod
    def _first_reference(images: Any) -> Any:
        if isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)):
            if not images:
                raise ValueError("SkyReels-V3 requires at least one reference image")
            return images[0]
        return images

    @classmethod
    def _resolution_dimensions(cls, images: Any, resolution: str) -> tuple[int, int]:
        config_path = resolve_data_path(
            "models",
            "runtime",
            "configs",
            "skyreels_v3",
            "aspect_ratios.yaml",
        )
        values = load_yaml(config_path)
        table = values.get(str(resolution).upper()) if isinstance(values, dict) else None
        if not isinstance(table, dict) or not table:
            raise ValueError(f"unsupported SkyReels-V3 resolution: {resolution!r}")
        image = load_pil_image(cls._first_reference(images), first_sequence_item=False)
        target_ratio = image.height / image.width
        height, width = min(
            table.values(),
            key=lambda size: abs((float(size[0]) / float(size[1])) - target_ratio),
        )
        return int(height) // 16 * 16, int(width) // 16 * 16

    def __call__(
        self,
        prompt: str | list[str] = "",
        images: Any = None,
        *,
        task_type: str = "reference_to_video",
        duration: int | None = None,
        resolution: str = "720P",
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> Any:
        normalized_task = str(task_type).strip().lower().replace("-", "_")
        if normalized_task != "reference_to_video":
            raise ValueError(
                "the native SkyReels-V3 integration currently exposes only reference_to_video"
            )
        if images is None:
            raise ValueError("SkyReels-V3 reference_to_video requires images")
        if (height is None) != (width is None):
            raise ValueError("SkyReels-V3 height and width must be provided together")
        if height is None:
            height, width = self._resolution_dimensions(images, resolution)
        actual_fps = int(fps or self.DEFAULT_FPS)
        if num_frames is None and duration is not None:
            num_frames = int(duration) * actual_fps + 1
        return super().__call__(
            prompt=prompt,
            images=images,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=actual_fps,
            **kwargs,
        )


__all__ = ["SkyReelsV3Pipeline"]
