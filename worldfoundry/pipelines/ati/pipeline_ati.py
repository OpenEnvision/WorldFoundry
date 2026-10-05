"""Independent WorldFoundry pipeline for ATI-Wan2.1."""

from __future__ import annotations

from typing import Any, Mapping

from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.ati import ATISynthesis


class ATIPipeline(PipelineABC):
    MODEL_ID = "ati-wan21-14b"

    def __init__(self, synthesis_model: ATISynthesis, *, device: str = "cuda") -> None:
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
    ) -> "ATIPipeline":
        del model_id
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        if model_path is not None and not isinstance(model_path, Mapping):
            options["checkpoint_path"] = model_path
        options.update(dict(required_components or {}))
        options.update(kwargs)
        checkpoint = options.pop("checkpoint_path", options.pop("model_path", None))
        cls._strip_framework_loading_options(options)
        for key in ("required_components", "runtime_profile", "variant_id", "pipeline_binding", "repo_root"):
            options.pop(key, None)
        return cls(ATISynthesis.from_pretrained(checkpoint, device=device, **options), device=device)

    def __call__(
        self,
        prompt: str = "",
        images: Any = None,
        image: Any = None,
        interactions: Any = None,
        output_path: Any = None,
        return_dict: bool = False,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        del interactions
        if operator_kwargs is not None and not isinstance(operator_kwargs, Mapping):
            raise TypeError("operator_kwargs must be a mapping")
        runtime_kwargs = dict(operator_kwargs or {})
        runtime_kwargs.update(kwargs)
        image_value = image if image is not None else images
        result = self.synthesis_model.predict(
            prompt=prompt,
            image=image_value,
            output_path=output_path,
            return_dict=True,
            **runtime_kwargs,
        )
        return result if return_dict else result.get("video")


__all__ = ["ATIPipeline"]
