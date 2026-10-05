"""Warp As History visual generation pipeline module."""

from __future__ import annotations

from ...synthesis.visual_generation.memory.video import VideoArtifactMemory
from pathlib import Path
from typing import Any, Dict, Optional

from ..pipeline_utils import PipelineABC
from ...synthesis.visual_generation.warp_as_history.variants import (
    get_warp_as_history_variant,
)
from ...operators.runtime_video_operator import RuntimeVideoOperator
from ...synthesis.visual_generation.warp_as_history.warp_as_history_synthesis import (
    WarpAsHistorySynthesis,
)


class WarpAsHistoryPipeline(PipelineABC):
    """WorldFoundry pipeline wrapper for Warp-as-History."""

    MODEL_ID = "warp-as-history"
    SYNTHESIS_CLS = WarpAsHistorySynthesis

    def __init__(
        self,
        *,
        model_id: str | None = None,
        operator: Optional[RuntimeVideoOperator] = None,
        synthesis_model: Optional[WarpAsHistorySynthesis] = None,
        memory_module: Optional[VideoArtifactMemory] = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        requested_model_id = model_id or self.MODEL_ID
        self.variant = get_warp_as_history_variant(requested_model_id)
        self.model_id = self.variant.model_id
        self.synthesis_model = synthesis_model
        self.generation_type = self.variant.task
        self.model_name = self.variant.display_name
        self.operator = operator or RuntimeVideoOperator(generation_type="i2v")
        self.operators = self.operator
        self.memory_module = memory_module or VideoArtifactMemory(model_id='warp-as-history')
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        model_id: str | None = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> "WarpAsHistoryPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        del lazy
        options: Dict[str, Any] = {}
        if isinstance(model_path, dict):
            options.update(model_path)
        elif model_path is not None:
            options["model_path"] = str(model_path)
        options.update(required_components or {})
        options.update(kwargs)
        resolved_model_id = str(
            options.get("model_id")
            or options.get("variant")
            or options.get("profile_id")
            or model_id
            or cls.MODEL_ID
        )
        synthesis_model = cls.SYNTHESIS_CLS.from_pretrained(
            {**options, "model_id": resolved_model_id},
            device=device,
        )
        return cls(
            model_id=resolved_model_id,
            operator=RuntimeVideoOperator(generation_type="i2v"),
            synthesis_model=synthesis_model,
            memory_module=VideoArtifactMemory(model_id='warp-as-history'),
            device=device,
        )

    def process(self, prompt: str | None = None, images: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        if prompt is None:
            prompt = ""
        self.operator.get_interaction(prompt)
        try:
            interaction = self.operator.process_interaction()
        finally:
            self.operator.delete_last_interaction()
        return {
            "prompt": interaction["processed_prompt"],
            "images": images,
            "extra_inputs": dict(kwargs),
        }

    def __call__(
        self,
        prompt: str | None = None,
        images: Any = None,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        fps: int | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        seed: int | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ):
        """Execute the complete pipeline generation flow."""
        if self.synthesis_model is None:
            raise RuntimeError("Warp-as-History synthesis model is not loaded. Use from_pretrained() first.")
        if output_path is None and output_dir is not None:
            output_path = Path(output_dir) / f"{self.model_id}.mp4"
        operator_kwargs = dict(kwargs.pop("operator_kwargs", {}))
        ref_image_path = kwargs.pop("ref_image_path", None)
        if images is None and ref_image_path is not None:
            images = ref_image_path
        for framework_key in ("sample_id", "task_name"):
            kwargs.pop(framework_key, None)
            operator_kwargs.pop(framework_key, None)
        processed = self.process(prompt=prompt, images=images, **operator_kwargs)
        result = self.synthesis_model.predict(
            prompt=processed["prompt"],
            images=processed["images"],
            output_path=output_path,
            fps=fps,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            **processed["extra_inputs"],
            **kwargs,
        )
        self.memory_module.record(result, metadata={"type": "warp_as_history_result", "model_id": self.model_id})
        if return_dict:
            return result
        return result.get("artifact_path") or result.get("generated_video_path") or result

    def stream(
        self,
        prompt: str | None = None,
        images: Any = None,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        fps: int | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        seed: int | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ):
        """Stream visual generation outputs chunk by chunk."""
        if images is None:
            previous = self.memory_module.select(prefer_type="image")
            if previous is not None:
                images = previous
        return self(
            prompt=prompt,
            images=images,
            output_path=output_path,
            output_dir=output_dir,
            fps=fps,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            return_dict=return_dict,
            **kwargs,
        )

    def get_operator(self) -> RuntimeVideoOperator:
        """Get operator for WarpAsHistoryPipeline."""
        return self.operator

    def get_synthesis_model(self) -> WarpAsHistorySynthesis:
        """Get synthesis model for WarpAsHistoryPipeline."""
        if self.synthesis_model is None:
            raise RuntimeError("Warp-as-History synthesis model is not loaded.")
        return self.synthesis_model
