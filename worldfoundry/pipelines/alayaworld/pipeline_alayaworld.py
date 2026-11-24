"""Unified WorldFoundry pipeline for AlayaWorld."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.alayaworld import AlayaWorldSynthesis


class AlayaWorldPipeline(PipelineABC):
    """Generate camera-controlled long-horizon video from one image."""

    MODEL_ID = "alayaworld"
    SYNTHESIS_CLS = AlayaWorldSynthesis

    def __init__(
        self,
        *,
        synthesis_model: AlayaWorldSynthesis | None = None,
        device: str = "cuda",
    ) -> None:
        super().__init__(model_id=self.MODEL_ID, synthesis_model=synthesis_model, device=device)
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
    ) -> AlayaWorldPipeline:
        options: dict[str, Any] = {}
        checkpoint_path = model_path
        if isinstance(model_path, Mapping):
            options.update(model_path)
            checkpoint_path = None
        elif isinstance(model_path, str) and not model_path.strip():
            checkpoint_path = None
        options.update(required_components or {})
        options.update(kwargs)
        synthesis_model = cls._component("SYNTHESIS_CLS").from_pretrained(
            pretrained_model_path=checkpoint_path,
            device=device,
            lazy=lazy,
            generator_overrides=options,
        )
        return cls(synthesis_model=synthesis_model, device=device)

    @staticmethod
    def process(prompt: str, images: Any) -> dict[str, Any]:
        if images is None:
            raise ValueError("AlayaWorld requires an initial image.")
        return {"prompt": str(prompt or ""), "images": images}

    def __call__(
        self,
        prompt: str,
        images: Any = None,
        *,
        output_path: str | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        if self.synthesis_model is None:
            raise RuntimeError("AlayaWorld is not loaded. Call from_pretrained() first.")
        request = self.process(prompt, images)
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

    def close(self) -> None:
        """Release model resources owned by this pipeline."""

        if self.synthesis_model is not None:
            close = getattr(self.synthesis_model, "close", None)
            if callable(close):
                close()


__all__ = ["AlayaWorldPipeline"]
