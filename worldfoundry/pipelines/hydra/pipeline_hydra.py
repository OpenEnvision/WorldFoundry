"""Independent WorldFoundry pipeline for HyDRA."""

from __future__ import annotations

from typing import Any, Mapping

from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.hydra import HydraSynthesis


class HydraPipeline(PipelineABC):
    """Source-video and camera-trajectory conditioned HyDRA pipeline."""

    MODEL_ID = "hydra"

    def __init__(self, synthesis_model: HydraSynthesis, *, device: str = "cuda") -> None:
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
    ) -> "HydraPipeline":
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
        synthesis = HydraSynthesis.from_pretrained(checkpoint, device=device, **options)
        return cls(synthesis, device=device)

    def __call__(
        self,
        prompt: str = "",
        video: Any = None,
        output_path: Any = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        kw_video_path = kwargs.pop("video_path", None)
        video_path = video or kw_video_path
        result = self.synthesis_model.predict(
            prompt=prompt,
            video_path=video_path,
            output_path=output_path,
            return_dict=True,
            **kwargs,
        )
        return result if return_dict else result.get("video")


__all__ = ["HydraPipeline"]
