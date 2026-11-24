"""Echo Infinity visual generation pipeline module."""

from __future__ import annotations

from typing import Any

from ...operators.runtime_video_operator import RuntimeVideoOperator
from ...synthesis.visual_generation.echo_infinity.synthesis import EchoInfinitySynthesis
from ...synthesis.visual_generation.memory.video import VideoArtifactMemory
from ..pipeline_utils import PipelineABC


class EchoInfinityPipeline(PipelineABC):
    """WorldFoundry pipeline for Echo-Infinity text-to-video generation."""

    SYNTHESIS_CLS = EchoInfinitySynthesis

    def __init__(
        self,
        operator: RuntimeVideoOperator | None = None,
        synthesis_model: EchoInfinitySynthesis | None = None,
        memory_module: VideoArtifactMemory | None = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.synthesis_model = synthesis_model
        self.generation_type = "t2v"
        self.model_name = "echo-infinity"
        self.operator = operator or RuntimeVideoOperator(generation_type=self.generation_type)
        self.memory_module = memory_module or VideoArtifactMemory(model_id=self.model_name)
        self.device = device

    @classmethod
    def from_pretrained(  # ty: ignore[invalid-method-override]  # model-specific loader arguments, passed by keyword
        cls,
        model_path: str | None = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        lazy: bool = True,
        **kwargs: Any,
    ) -> EchoInfinityPipeline:
        """Load the pipeline from pretrained checkpoints and configurations."""
        runtime_overrides = dict(required_components or {})
        runtime_overrides.update(kwargs)
        runtime_overrides.setdefault("device", device)
        synthesis_model = cls._component("SYNTHESIS_CLS").from_pretrained(
            pretrained_model_path=model_path,
            device=device,
            lazy=lazy,
            generator_overrides=runtime_overrides,
        )
        return cls(
            operator=RuntimeVideoOperator(generation_type=synthesis_model.generation_type),
            synthesis_model=synthesis_model,
            memory_module=VideoArtifactMemory(model_id="echo-infinity"),
            device=device,
        )

    def process(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        del kwargs
        self.operator.get_interaction(prompt)
        try:
            interaction = self.operator.process_interaction()
        finally:
            self.operator.delete_last_interaction()
        return {"prompt": interaction["processed_prompt"]}

    def __call__(
        self,
        prompt: str,
        output_path: str | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ):
        """Execute the complete pipeline generation flow."""
        if self.synthesis_model is None:
            raise RuntimeError("Synthesis model is not loaded. Use from_pretrained() first.")
        processed = self.process(prompt=prompt)
        result = self.synthesis_model.predict(
            prompt=processed["prompt"],
            output_path=output_path,
            fps=fps,
            return_dict=True,
            **kwargs,
        )
        if return_dict:
            return result
        return result["video"]


__all__ = ["EchoInfinityPipeline"]
