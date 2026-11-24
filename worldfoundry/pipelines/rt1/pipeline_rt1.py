"""Rt1 visual generation pipeline module."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ...operators.rt1_operator import RT1Operator
from ...synthesis.action_generation.memory import ActionTraceMemory, RT1Memory
from ...synthesis.action_generation.rt1 import RT1Synthesis
from ..pipeline_utils import PipelineABC


class RT1Pipeline(PipelineABC):
    """WorldFoundry VLA/policy pipeline for Robotics Transformer RT-1 artifacts."""

    MODEL_ID = "rt-1"
    OPERATOR_CLS = RT1Operator
    MEMORY_CLS = RT1Memory
    SYNTHESIS_CLS = RT1Synthesis
    MEMORY_RECORD_TYPE = "runtime_result"
    RESERVED_SYNTHESIS_KEYS = frozenset({"prompt", "images", "video", "actions", "extra_inputs"})
    generation_type = "vla_policy"

    def __init__(
        self,
        operators: RT1Operator | None = None,
        synthesis_model: RT1Synthesis | None = None,
        memory_module: ActionTraceMemory | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        operator: RT1Operator | None = None,
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        resolved_operator = operators if operators is not None else operator
        self.model_id = model_id or self.MODEL_ID
        self.synthesis_model = synthesis_model
        self.operators = resolved_operator
        self.operator = resolved_operator
        self.memory_module = memory_module if memory_module is not None else self._component("MEMORY_CLS")()
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> RT1Pipeline:
        """Load the pipeline from pretrained checkpoints and configurations."""
        options: dict[str, Any] = {}
        if isinstance(model_path, dict):
            options.update(model_path)
        elif model_path is not None:
            options["repo_root"] = str(model_path)
        options.update(required_components or {})
        options.update(kwargs)

        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)
        synthesis_model = cls._component("SYNTHESIS_CLS").from_pretrained(
            {**options, "model_id": resolved_model_id},
            device=device,
        )
        profile = getattr(synthesis_model, "profile", None)
        operators = cls._component("OPERATOR_CLS")(input_schema=dict(getattr(profile, "input_schema", {}) or {}))

        return cls(
            operators=operators,
            synthesis_model=synthesis_model,
            memory_module=cls._component("MEMORY_CLS")(),
            device=device,
            model_id=resolved_model_id,
        )

    def process(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] | Any | None = None,
        ref_image_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        if self.operators is None:
            raise RuntimeError(f"{self.__class__.__name__} operator is not initialized.")

        self.operators.get_interaction(interactions)
        try:
            interaction = self.operators.process_interaction()
        finally:
            self.operators.delete_last_interaction()

        return {
            **self.operators.process_prompt(prompt, **kwargs),
            **self.operators.process_perception(
                images=images,
                video=video,
                ref_image_path=ref_image_path,
                **kwargs,
            ),
            **interaction,
        }

    def __call__(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] | Any | None = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute the complete pipeline generation flow."""
        if self.synthesis_model is None:
            raise RuntimeError(f"{self.__class__.__name__} synthesis_model is not initialized.")

        processed = self.process(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            ref_image_path=kwargs.pop("ref_image_path", None),
            **kwargs.pop("operator_kwargs", {}),
        )
        model_specific = {key: value for key, value in processed.items() if key not in self.RESERVED_SYNTHESIS_KEYS}
        synthesis_kwargs = {**processed.get("extra_inputs", {}), **model_specific, **kwargs}
        result = self.synthesis_model.predict(
            prompt=processed["prompt"],
            images=processed["images"],
            video=processed["video"],
            interactions=processed["actions"],
            output_path=output_path,
            fps=fps,
            **synthesis_kwargs,
        )
        self.memory_module.record(
            result,
            metadata={"type": self.MEMORY_RECORD_TYPE, "model_id": self.model_id},
        )
        if return_dict:
            return result
        return result.get("artifact_path") or result

    def stream(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] | Any | None = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Stream visual generation outputs chunk by chunk."""
        if images is None and video is None:
            previous = self.memory_module.select()
            if isinstance(previous, dict):
                images = previous.get("artifact_path")
        return self(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            return_dict=return_dict,
            **kwargs,
        )

    def get_operator(self) -> RT1Operator:
        """Get operator for RT1Pipeline."""
        if self.operators is None:
            raise RuntimeError(f"{self.__class__.__name__} operator is not initialized.")
        return self.operators

    def get_synthesis_model(self) -> RT1Synthesis:
        """Get synthesis model for RT1Pipeline."""
        if self.synthesis_model is None:
            raise RuntimeError(f"{self.__class__.__name__} synthesis_model is not initialized.")
        return self.synthesis_model
