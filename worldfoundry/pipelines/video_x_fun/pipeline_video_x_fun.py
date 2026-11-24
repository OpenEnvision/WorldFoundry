"""Independent WorldFoundry pipelines for VideoX-Fun camera checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.video_x_fun import (
    Wan21Fun1P3BCameraSynthesis,
    Wan21Fun14BCameraSynthesis,
    Wan22Fun5BCameraSynthesis,
    Wan22FunA14BCameraSynthesis,
)


class _WanFunCameraPipeline(PipelineABC):
    SYNTHESIS_CLS = None

    def __init__(self, synthesis_model: Any, *, device: str = "cuda") -> None:
        self.synthesis_model = synthesis_model
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ):
        del model_id
        options = cls._runtime_options(model_path, dict(required_components or {}), kwargs)
        checkpoint = options.pop("checkpoint_path", options.pop("model_path", None))
        cls._strip_framework_loading_options(options)
        for key in ("required_components", "runtime_profile", "variant_id", "pipeline_binding", "repo_root"):
            options.pop(key, None)
        return cls(cls._component("SYNTHESIS_CLS").from_pretrained(checkpoint, device=device, **options), device=device)

    def __call__(
        self,
        prompt: str = "",
        images: Any = None,
        output_path: Any = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("operator_kwargs", None)
        explicit_image_path = kwargs.pop("image_path", None)
        result = self.synthesis_model.predict(
            prompt=prompt,
            image_path=images or explicit_image_path,
            output_path=output_path,
            return_dict=True,
            **kwargs,
        )
        return result if return_dict else result.get("video")


class Wan21Fun1P3BCameraPipeline(_WanFunCameraPipeline):
    MODEL_ID = "wan21-fun-1p3b-cam"
    MODEL_PATH_OPTION = "checkpoint_path"
    SYNTHESIS_CLS = Wan21Fun1P3BCameraSynthesis


class Wan21Fun14BCameraPipeline(_WanFunCameraPipeline):
    MODEL_ID = "wan21-fun-14b-cam"
    MODEL_PATH_OPTION = "checkpoint_path"
    SYNTHESIS_CLS = Wan21Fun14BCameraSynthesis


class Wan22Fun5BCameraPipeline(_WanFunCameraPipeline):
    MODEL_ID = "wan22-fun-5b-cam"
    MODEL_PATH_OPTION = "checkpoint_path"
    SYNTHESIS_CLS = Wan22Fun5BCameraSynthesis


class Wan22FunA14BCameraPipeline(_WanFunCameraPipeline):
    MODEL_ID = "wan22-fun-a14b-cam"
    MODEL_PATH_OPTION = "checkpoint_path"
    SYNTHESIS_CLS = Wan22FunA14BCameraSynthesis


__all__ = [
    "Wan21Fun1P3BCameraPipeline",
    "Wan21Fun14BCameraPipeline",
    "Wan22Fun5BCameraPipeline",
    "Wan22FunA14BCameraPipeline",
]
