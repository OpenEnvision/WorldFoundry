"""Dedicated HY-World 2.0 trajectory-render pipeline for materialized scenes."""

from __future__ import annotations

from typing import Any, Mapping

from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.hunyuan_world.hy_world_2p0_worldgen_synthesis import (
    HYWorld2WorldgenSynthesis,
)


class HYWorld2WorldgenPipeline(PipelineABC):
    MODEL_ID = "hyworld-worldgen"

    def __init__(self, synthesis_model: HYWorld2WorldgenSynthesis, *, device: str = "cuda") -> None:
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
    ) -> "HYWorld2WorldgenPipeline":
        del model_id
        options = cls._runtime_options(model_path, dict(required_components or {}), kwargs)
        scene = options.pop("scene_path", None)
        model_scene = options.pop("model_path", None)
        repo_scene = options.pop("repo_root", None)
        if scene is None:
            scene = model_scene or repo_scene
        cls._strip_framework_loading_options(options)
        for key in ("required_components", "runtime_profile", "variant_id", "pipeline_binding"):
            options.pop(key, None)
        return cls(HYWorld2WorldgenSynthesis.from_pretrained(scene, device=device, **options), device=device)

    def __call__(
        self,
        prompt: str = "",
        images: Any = None,
        output_path: Any = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        del prompt
        kwargs.pop("operator_kwargs", None)
        kw_image_path = kwargs.pop("image_path", None)
        result = self.synthesis_model.predict(
            image_path=images or kw_image_path,
            output_path=output_path,
            return_dict=True,
            **kwargs,
        )
        return result if return_dict else result.get("video")


__all__ = ["HYWorld2WorldgenPipeline"]
