"""WorldFoundry pipeline for Stable Video Infinity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.stable_video_infinity.synthesis import (
    StableVideoInfinitySynthesis,
)
from worldfoundry.synthesis.visual_generation.stable_video_infinity.worldfoundry_runtime import (
    normalize_prompt_stream,
)


class StableVideoInfinityPipeline(PipelineABC):
    """Generate long I2V sequences from one image and a prompt stream."""

    MODEL_ID = "stable-video-infinity"
    SYNTHESIS_CLS = StableVideoInfinitySynthesis

    def __init__(
        self,
        *,
        synthesis_model: StableVideoInfinitySynthesis | None = None,
        device: str = "cuda",
    ) -> None:
        super().__init__(
            model_id=self.MODEL_ID,
            synthesis_model=synthesis_model,
            device=str(device),
        )
        self.model_name = self.MODEL_ID
        self.generation_type = "i2v"

    @classmethod
    def from_pretrained(  # ty: ignore[invalid-method-override]  # model-specific loader arguments, passed by keyword
        cls,
        model_path: Any = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        lazy: bool = True,
        **kwargs: Any,
    ) -> StableVideoInfinityPipeline:
        options: dict[str, Any] = {}
        pretrained_model_path = model_path
        if isinstance(model_path, Mapping):
            options.update(model_path)
            pretrained_model_path = None
        elif isinstance(pretrained_model_path, str) and not pretrained_model_path.strip():
            # Workspace uses an empty model_ref when the catalog-backed default
            # checkpoint should be selected.  Do not let that empty value
            # replace the staged SVI LoRA path from runtime_defaults.yaml.
            pretrained_model_path = None
        options.update(required_components or {})
        options.update(kwargs)
        options.setdefault("device", device)
        synthesis_model = cls._component("SYNTHESIS_CLS").from_pretrained(
            pretrained_model_path=pretrained_model_path,
            device=device,
            lazy=lazy,
            generator_overrides=options,
        )
        return cls(synthesis_model=synthesis_model, device=device)

    @staticmethod
    def process(
        prompt: str | Sequence[str],
        images: Any,
    ) -> dict[str, Any]:
        if images is None:
            raise ValueError("Stable Video Infinity requires an input image.")
        prompts = normalize_prompt_stream(prompt)
        return {
            "prompt": prompts[0] if isinstance(prompt, str) else prompts,
            "images": images,
        }

    def __call__(
        self,
        prompt: str | Sequence[str],
        images: Any = None,
        *,
        prompt_stream: str | Sequence[str] | None = None,
        output_path: str | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        if self.synthesis_model is None:
            raise RuntimeError("SVI is not loaded. Call from_pretrained() first.")
        effective_prompt = prompt if prompt_stream is None else prompt_stream
        request = self.process(prompt=effective_prompt, images=images)
        result = self.synthesis_model.predict(
            prompt=request["prompt"],
            images=request["images"],
            output_path=output_path,
            fps=fps,
            return_dict=True,
            **kwargs,
        )
        if return_dict:
            return result
        return result["video"]

    def run_pipeline_invocation(self, invocation: Any) -> Mapping[str, Any]:
        """Preserve prompt lists that the generic text normalizer would stringify."""

        inputs = invocation.request.inputs
        prompt = inputs.get("prompt_stream")
        if prompt is None:
            prompt = inputs.get("prompts")
        if prompt is None:
            prompt = inputs.get("prompt", invocation.prompt)
        return self(
            prompt=prompt,
            images=invocation.image,
            output_path=invocation.output_path,
            return_dict=True,
            **dict(invocation.pipeline_kwargs),
        )


SVIPipeline = StableVideoInfinityPipeline

__all__ = ["SVIPipeline", "StableVideoInfinityPipeline"]
