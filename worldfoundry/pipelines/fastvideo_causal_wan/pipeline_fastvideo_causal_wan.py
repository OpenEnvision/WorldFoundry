"""Product adapter for FastVideo's CausalWan2.2 rollout."""

from __future__ import annotations

from typing import Any

from PIL import Image

from ..native_diffusion import NativeVisualDiffusionPipeline


def _best_output_size(width: int, height: int, expected_area: int) -> tuple[int, int]:
    """FastVideo's aspect-preserving Wan canvas, aligned to 16-pixel patches."""
    if min(width, height, expected_area) <= 0:
        raise ValueError("CausalWan source size and target area must be positive")
    ratio = width / height
    ideal_width = (expected_area * ratio) ** 0.5
    ideal_height = expected_area / ideal_width
    width_first = int(ideal_width // 16 * 16)
    height_first = int(ideal_height // 16 * 16)
    if min(width_first, height_first) < 16:
        raise ValueError("CausalWan target canvas is too small for 16-pixel alignment")
    height_from_width = int(expected_area / width_first // 16 * 16)
    width_from_height = int(expected_area / height_first // 16 * 16)
    if min(height_from_width, width_from_height) < 16:
        raise ValueError("CausalWan target canvas is too small for 16-pixel alignment")
    ratio_width_first = width_first / height_from_width
    ratio_height_first = width_from_height / height_first
    if max(ratio / ratio_width_first, ratio_width_first / ratio) < max(
        ratio / ratio_height_first, ratio_height_first / ratio
    ):
        return width_first, height_from_width
    return width_from_height, height_first


class FastVideoCausalWanPipeline(NativeVisualDiffusionPipeline):
    """Eight-step, first-frame-conditioned CausalWan2.2 image-to-video pipeline."""

    MODEL_ID = "fastvideo-causal-wan2.2-i2v-14b"
    OWNER = "FastVideo CausalWan2.2"
    CHECKPOINT_ROLES = ("high-dit", "low-dit", "text-encoder", "tokenizer", "vae")
    GENERATION_TYPE = "i2v"
    ACCEPTS_IMAGES = True
    REQUIRES_IMAGES = True
    DEFAULT_HEIGHT = 480
    DEFAULT_WIDTH = 832
    DEFAULT_NUM_FRAMES = 81
    DEFAULT_NUM_INFERENCE_STEPS = 8
    DEFAULT_GUIDANCE_SCALE = 1.0
    DEFAULT_FPS = 16
    DEFAULT_NEGATIVE_PROMPT = ""
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 12.0}

    def __call__(
        self,
        prompt: str | list[str] = "",
        images: Any = None,
        **kwargs: Any,
    ) -> Any:
        source = images
        if source is None:
            for key in ("ref_image_path", "image_path", "input_path"):
                if kwargs.get(key) is not None:
                    source = kwargs.pop(key)
                    break
        if source is None:
            return super().__call__(prompt=prompt, images=images, **kwargs)

        from worldfoundry.core import load_pil_image

        image = load_pil_image(source)
        target_width = int(kwargs.get("width") or self.DEFAULT_WIDTH)
        target_height = int(kwargs.get("height") or self.DEFAULT_HEIGHT)
        if target_width <= 0 or target_height <= 0:
            raise ValueError("CausalWan target width and height must be positive")
        output_width, output_height = _best_output_size(
            image.width, image.height, target_width * target_height
        )
        scale = max(output_width / image.width, output_height / image.height)
        resized = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
        left = (resized.width - output_width) // 2
        top = (resized.height - output_height) // 2
        cropped = resized.crop((left, top, left + output_width, top + output_height))
        return super().__call__(
            prompt=prompt,
            images=cropped,
            width=output_width,
            height=output_height,
            **{key: value for key, value in kwargs.items() if key not in {"width", "height"}},
        )


__all__ = ["FastVideoCausalWanPipeline"]
