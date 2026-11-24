"""Allegro Ti2V visual generation pipeline module."""

from __future__ import annotations

from typing import Any

from ...operators.runtime_video_operator import RuntimeVideoOperator
from ...synthesis.visual_generation.allegro.allegro_ti2v_synthesis import AllegroTi2VSynthesis
from ...synthesis.visual_generation.memory.video import VideoArtifactMemory
from ..pipeline_utils import PipelineABC


class AllegroTi2VPipeline(PipelineABC):
    """Independent WorldFoundry pipeline for AllegroTi2V."""

    SYNTHESIS_CLS = AllegroTi2VSynthesis

    def __init__(
        self,
        operator: RuntimeVideoOperator | None = None,
        synthesis_model=None,
        memory_module: VideoArtifactMemory | None = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.synthesis_model = synthesis_model
        self.generation_type = getattr(synthesis_model, "generation_type", "i2v")
        self.model_name = getattr(synthesis_model, "model_name", None)
        self.operator = operator or RuntimeVideoOperator(generation_type=self.generation_type)
        self.memory_module = memory_module or VideoArtifactMemory(model_id="allegro-ti2v")
        self.device = device

    @classmethod
    def from_pretrained(  # ty: ignore[invalid-method-override]  # model-specific loader arguments, passed by keyword
        cls,
        model_path: str | None = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        lazy: bool = True,
        **kwargs,
    ) -> AllegroTi2VPipeline:
        """Load the pipeline from pretrained checkpoints and configurations."""
        required_components = dict(required_components or {})
        generator_overrides = dict(required_components)
        generator_overrides.update(kwargs)

        synthesis_model = cls._component("SYNTHESIS_CLS").from_pretrained(
            pretrained_model_path=model_path,
            device=device,
            lazy=lazy,
            generator_overrides=generator_overrides,
        )
        return cls(
            operator=RuntimeVideoOperator(generation_type=synthesis_model.generation_type),
            synthesis_model=synthesis_model,
            memory_module=VideoArtifactMemory(model_id="allegro-ti2v"),
            device=device,
        )

    def process(self, prompt: str = "", images=None, **kwargs) -> dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        del kwargs
        self.operator.get_interaction(prompt)
        try:
            interaction = self.operator.process_interaction()
        finally:
            self.operator.delete_last_interaction()
        perception = self.operator.process_perception(images=images)
        return {
            "prompt": interaction["processed_prompt"],
            "images": perception["images"],
        }

    def __call__(
        self,
        prompt: str = "",
        images=None,
        output_path: str | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        **kwargs,
    ):
        """Execute the complete pipeline generation flow."""
        if self.synthesis_model is None:
            raise RuntimeError("Synthesis model is not loaded. Use from_pretrained() first.")

        processed = self.process(prompt=prompt, images=images)
        result = self.synthesis_model.predict(
            prompt=processed["prompt"],
            images=processed["images"],
            output_path=output_path,
            fps=fps,
            return_dict=True,
            **kwargs,
        )
        if return_dict:
            return result
        return result["video"]

    def stream(
        self,
        prompt: str = "",
        images=None,
        output_path: str | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        **kwargs,
    ):
        """Stream visual generation outputs chunk by chunk."""
        if self.memory_module is None:
            raise ValueError("memory_module is not initialized")

        current_images = images
        if current_images is None and self.generation_type == "i2v":
            current_images = self.memory_module.select(prefer_type="image")
            if current_images is None:
                raise ValueError("stream() for i2v models requires an initial image on the first call.")
        elif current_images is not None:
            self.memory_module.record(current_images, metadata={"kind": "input_image"})

        result = self(
            prompt=prompt,
            images=current_images,
            output_path=output_path,
            fps=fps,
            return_dict=True,
            **kwargs,
        )
        self.memory_module.record(
            result["video"],
            metadata={
                "prompt": prompt,
                "model_name": self.model_name,
                "generation_type": self.generation_type,
            },
        )
        if return_dict:
            return result
        return result["video"]

    def get_operator(self) -> RuntimeVideoOperator:
        """Get operator for AllegroTi2VPipeline."""
        return self.operator

    def get_synthesis_model(self):
        """Get synthesis model for AllegroTi2VPipeline."""
        return self.synthesis_model
