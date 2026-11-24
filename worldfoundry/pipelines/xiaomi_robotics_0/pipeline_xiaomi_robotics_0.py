"""WorldFoundry pipeline for Xiaomi-Robotics-0 action traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.action_generation.xiaomi_robotics_0 import XiaomiRobotics0Synthesis


class XiaomiRobotics0Pipeline(PipelineABC):
    """Thin pipeline over the in-tree, checkpoint-backed VLA runtime."""

    MODEL_ID = "xiaomi-robotics-0"
    generation_type = "vla_policy"

    def __init__(
        self,
        *,
        synthesis_model: XiaomiRobotics0Synthesis,
        device: str = "cuda",
        model_id: str = MODEL_ID,
    ) -> None:
        super().__init__(model_id=model_id, synthesis_model=synthesis_model, device=device)

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> XiaomiRobotics0Pipeline:
        """Resolve profile/checkpoint options without loading model weights."""

        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        if model_path is not None and not isinstance(model_path, Mapping):
            options["checkpoint_path"] = str(model_path)
        options.update(required_components or {})
        options.update(kwargs)
        resolved_model_id = str(options.get("model_id") or model_id or cls.MODEL_ID)
        synthesis = XiaomiRobotics0Synthesis.from_pretrained(
            {**options, "model_id": resolved_model_id},
            device=device,
        )
        return cls(synthesis_model=synthesis, device=device, model_id=resolved_model_id)

    def __call__(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] | Any | None = None,
        ref_image_path: str | Path | None = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate an action-trace artifact from instruction, images, and state."""

        observation_value = kwargs.pop("observation", None) or {}
        if not isinstance(observation_value, Mapping):
            raise TypeError("observation must be a mapping")
        observation = dict(operator_kwargs or {})
        observation.update(observation_value)
        if images is None and ref_image_path is not None:
            images = ref_image_path
        action_context: Sequence[Any]
        if interactions is None:
            action_context = ()
        elif isinstance(interactions, Sequence) and not isinstance(interactions, (str, bytes, bytearray)):
            action_context = interactions
        else:
            action_context = (interactions,)
        result = self.synthesis_model.predict(
            prompt=str(prompt or ""),
            images=images,
            video=video,
            interactions=action_context,
            output_path=output_path,
            fps=fps,
            observation=observation,
            **kwargs,
        )
        if return_dict:
            return result
        return result.get("artifact_path") or result

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        """The policy emits one action chunk, so streaming delegates to one call."""

        return self(*args, **kwargs)


__all__ = ["XiaomiRobotics0Pipeline"]
