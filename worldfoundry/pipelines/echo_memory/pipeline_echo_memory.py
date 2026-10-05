"""Unified pipelines for independently registered Echo-Memory models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.echo_memory import (
    EchoMemoryContextK1Synthesis,
    EchoMemorySynthesis,
)


class EchoMemoryPipeline(PipelineABC):
    """Common I2V interface; subclasses pin the model and checkpoint recipe."""

    ABSTRACT_PIPELINE = True
    SYNTHESIS_CLS: type[EchoMemorySynthesis]

    def __init__(
        self,
        *,
        synthesis_model: EchoMemorySynthesis | None = None,
        device: str = "cuda",
    ) -> None:
        if self.MODEL_ID is None:
            raise TypeError("EchoMemoryPipeline must be instantiated through a model-specific subclass")
        super().__init__(model_id=self.MODEL_ID, synthesis_model=synthesis_model, device=device)
        self.model_name = self.MODEL_ID
        self.generation_type = "i2v"

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        lazy: bool = True,
        **kwargs: Any,
    ) -> "EchoMemoryPipeline":
        options: dict[str, Any] = {}
        checkpoint_path = model_path
        if isinstance(model_path, Mapping):
            loader_options = dict(model_path)
            nested_model_path = loader_options.pop("model_path", None)
            options.update(loader_options)
            checkpoint_path = None
            if isinstance(nested_model_path, Mapping):
                options.update(nested_model_path)
            elif nested_model_path is not None:
                checkpoint_path = nested_model_path
        elif isinstance(model_path, str) and not model_path.strip():
            checkpoint_path = None
        options.update(required_components or {})
        options.update(kwargs)
        requested_model = options.pop("model_id", cls.MODEL_ID)
        if requested_model != cls.MODEL_ID:
            raise ValueError(f"{cls.__name__} is bound to {cls.MODEL_ID!r}, got {requested_model!r}")
        for loader_key in (
            "pipeline_binding",
            "profile_id",
            "runtime_profile",
            "variant_id",
        ):
            options.pop(loader_key, None)
        synthesis = cls.SYNTHESIS_CLS.from_pretrained(
            pretrained_model_path=checkpoint_path,
            device=device,
            lazy=lazy,
            generator_overrides=options,
        )
        return cls(synthesis_model=synthesis, device=device)

    @staticmethod
    def process(prompt: str, images: Any) -> dict[str, Any]:
        if images is None:
            raise ValueError("Echo-Memory requires an initial image")
        return {"prompt": str(prompt or ""), "images": images}

    def __call__(
        self,
        prompt: str,
        images: Any = None,
        *,
        output_path: str | None = None,
        fps: int | None = None,
        num_frames: int | None = None,
        frames: int | None = None,
        width: int | None = None,
        height: int | None = None,
        num_chunks: int | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        camera_trajectory: Any = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        if self.synthesis_model is None:
            raise RuntimeError(f"{self.MODEL_ID} is not loaded; call from_pretrained() first")
        request = self.process(prompt, images)
        result = self.synthesis_model.predict(
            prompt=request["prompt"],
            images=request["images"],
            output_path=output_path,
            fps=fps,
            return_dict=True,
            num_frames=num_frames if num_frames is not None else frames,
            width=width,
            height=height,
            num_chunks=num_chunks,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            camera_trajectory=camera_trajectory,
            **kwargs,
        )
        return result if return_dict else result["video"]

    def close(self) -> None:
        if self.synthesis_model is not None:
            close = getattr(self.synthesis_model, "close", None)
            if callable(close):
                close()


class EchoMemoryContextK1Pipeline(EchoMemoryPipeline):
    MODEL_ID = "echo-memory-context-k1"
    SYNTHESIS_CLS = EchoMemoryContextK1Synthesis


__all__ = [
    "EchoMemoryContextK1Pipeline",
    "EchoMemoryPipeline",
]
