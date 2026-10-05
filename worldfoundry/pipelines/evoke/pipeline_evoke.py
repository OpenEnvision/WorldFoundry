"""Independent WorldFoundry pipeline for Evoke."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.evoke import EvokeSynthesis


class EvokePipeline(PipelineABC):
    """Route standard WorldFoundry media arguments to the bundled Evoke CLI."""

    MODEL_ID = "evoke"
    MODEL_PATH_OPTION = "checkpoint_path"

    def __init__(self, synthesis_model: EvokeSynthesis, *, device: str = "cuda") -> None:
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
    ) -> "EvokePipeline":
        del model_id
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        if model_path is not None and not isinstance(model_path, Mapping):
            options["checkpoint_path"] = model_path
        options.update(dict(required_components or {}))
        options.update(kwargs)
        checkpoint = options.pop(
            "checkpoint_path",
            options.pop("model_path", options.pop("pretrained_model_path", None)),
        )
        cls._strip_framework_loading_options(options)
        for key in (
            "required_components",
            "runtime_profile",
            "variant_id",
            "pipeline_binding",
            "repo_root",
        ):
            options.pop(key, None)
        synthesis = EvokeSynthesis.from_pretrained(checkpoint, device=device, **options)
        return cls(synthesis, device=device)

    def __call__(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        output_path: Any = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("operator_kwargs", None)
        kw_image = kwargs.pop("image_path", None)
        kw_video = kwargs.pop("video_path", None)
        image_path = images if images is not None else kw_image
        video_path = video if video is not None else kw_video
        result = self.synthesis_model.predict(
            prompt=prompt,
            image_path=image_path,
            video_path=video_path,
            output_path=output_path,
            return_dict=True,
            **kwargs,
        )
        return result if return_dict else result.get("video")


__all__ = ["EvokePipeline"]
