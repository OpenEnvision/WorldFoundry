"""Independent WorldFoundry pipelines for minWM Action2V variants."""

from __future__ import annotations

from typing import Any, Mapping

from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.minwm import (
    MinWMHYAction2VSynthesis,
    MinWMWanAction2VSynthesis,
)


class _MinWMPipeline(PipelineABC):
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
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        if model_path and not isinstance(model_path, Mapping):
            options["checkpoint_path"] = model_path
        options.update(dict(required_components or {}))
        options.update(kwargs)
        checkpoint = options.pop("checkpoint_path", options.pop("model_path", None))
        cls._strip_framework_loading_options(options)
        for key in ("required_components", "runtime_profile", "variant_id", "pipeline_binding", "repo_root"):
            options.pop(key, None)
        return cls(cls.SYNTHESIS_CLS.from_pretrained(checkpoint, device=device, **options), device=device)

    def __call__(
        self,
        prompt: str = "",
        images: Any = None,
        output_path: Any = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        kw_image_path = kwargs.pop("image_path", None)
        result = self.synthesis_model.predict(
            prompt=prompt,
            image_path=images or kw_image_path,
            output_path=output_path,
            return_dict=True,
            **kwargs,
        )
        return result if return_dict else result.get("video")


class MinWMHYAction2VPipeline(_MinWMPipeline):
    MODEL_ID = "minwm-hy-action2v"
    SYNTHESIS_CLS = MinWMHYAction2VSynthesis


class MinWMWanAction2VPipeline(_MinWMPipeline):
    MODEL_ID = "minwm-wan-action2v"
    SYNTHESIS_CLS = MinWMWanAction2VSynthesis


__all__ = ["MinWMHYAction2VPipeline", "MinWMWanAction2VPipeline"]
