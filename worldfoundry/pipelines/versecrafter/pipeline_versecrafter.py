"""Independent WorldFoundry pipeline for VerseCrafter."""

from __future__ import annotations

from typing import Any, Mapping

from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.versecrafter import VerseCrafterSynthesis


class VerseCrafterPipeline(PipelineABC):
    MODEL_PATH_OPTION = "checkpoint_path"

    def __init__(self, synthesis_model: VerseCrafterSynthesis, *, device: str = "cuda") -> None:
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
    ) -> "VerseCrafterPipeline":
        del model_id
        options = cls._runtime_options(model_path, dict(required_components or {}), kwargs)
        checkpoint = options.pop("checkpoint_path", options.pop("model_path", None))
        cls._strip_framework_loading_options(options)
        for key in ("required_components", "runtime_profile", "variant_id", "pipeline_binding", "repo_root"):
            options.pop(key, None)
        return cls(VerseCrafterSynthesis.from_pretrained(checkpoint, device=device, **options), device=device)

    def __call__(
        self,
        prompt: str = "",
        images: Any = None,
        output_path: Any = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("operator_kwargs", None)
        kw_image_path = kwargs.pop("image_path", None)
        result = self.synthesis_model.predict(
            prompt=prompt,
            image_path=images or kw_image_path,
            output_path=output_path,
            return_dict=True,
            **kwargs,
        )
        return result if return_dict else result.get("video")


__all__ = ["VerseCrafterPipeline"]
