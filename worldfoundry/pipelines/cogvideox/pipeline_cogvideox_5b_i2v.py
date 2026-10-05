"""Cogvideox 5B I2V visual generation pipeline module."""

from __future__ import annotations

from ...synthesis.visual_generation.memory.video import VideoArtifactMemory
from ..pipeline_utils import PipelineABC
from typing import Any, Dict, Optional

from ...operators.runtime_video_operator import RuntimeVideoOperator
from ...synthesis.visual_generation.cogvideox.cogvideox_5b_i2v_synthesis import CogVideoX5bI2VSynthesis


DEFAULT_COGVIDEOX_5B_I2V_PROMPT = (
    "A young man holding a bright sparkler gently turns his head and smiles as "
    "sparks scatter through the night air, with a subtle cinematic camera dolly, "
    "stable facial details, and natural motion."
)


class CogVideoX5bI2VPipeline(PipelineABC):
    """Independent WorldFoundry pipeline for CogVideoX5bI2V."""

    SYNTHESIS_CLS = CogVideoX5bI2VSynthesis

    def __init__(
        self,
        operator: Optional[RuntimeVideoOperator] = None,
        synthesis_model=None,
        memory_module: Optional[VideoArtifactMemory] = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.synthesis_model = synthesis_model
        self.generation_type = getattr(synthesis_model, "generation_type", "i2v")
        self.model_name = getattr(synthesis_model, "model_name", None)
        self.operator = operator or RuntimeVideoOperator(generation_type=self.generation_type)
        self.memory_module = memory_module or VideoArtifactMemory(model_id='cogvideox-5b-i2v')
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[str] = None,
        required_components: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        lazy: bool = True,
        **kwargs,
    ) -> "CogVideoX5bI2VPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        required_components = dict(required_components or {})
        generator_overrides = dict(required_components)
        generator_overrides.update(kwargs)

        synthesis_model = cls.SYNTHESIS_CLS.from_pretrained(
            pretrained_model_path=model_path,
            device=device,
            lazy=lazy,
            generator_overrides=generator_overrides,
        )
        return cls(
            operator=RuntimeVideoOperator(generation_type=synthesis_model.generation_type),
            synthesis_model=synthesis_model,
            memory_module=VideoArtifactMemory(model_id='cogvideox-5b-i2v'),
            device=device,
        )

    def process(self, prompt: str | None = None, images=None, **kwargs) -> Dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        del kwargs
        prompt = prompt or DEFAULT_COGVIDEOX_5B_I2V_PROMPT
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
        prompt: str | None = None,
        images=None,
        output_path: Optional[str] = None,
        fps: Optional[int] = None,
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
        prompt: str | None = None,
        images=None,
        output_path: Optional[str] = None,
        fps: Optional[int] = None,
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
                "prompt": result.get("prompt", prompt or DEFAULT_COGVIDEOX_5B_I2V_PROMPT),
                "model_name": self.model_name,
                "generation_type": self.generation_type,
            },
        )
        if return_dict:
            return result
        return result["video"]

    def get_operator(self) -> RuntimeVideoOperator:
        """Get operator for CogVideoX5bI2VPipeline."""
        return self.operator

    def get_synthesis_model(self):
        """Get synthesis model for CogVideoX5bI2VPipeline."""
        return self.synthesis_model
