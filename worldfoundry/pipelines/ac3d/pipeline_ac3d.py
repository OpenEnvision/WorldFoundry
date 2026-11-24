"""Ac3D visual generation pipeline module."""

from __future__ import annotations

from typing import Any

from ...operators.ac3d_operator import AC3DOperator
from ...synthesis.visual_generation.ac3d.ac3d_synthesis import AC3DSynthesis
from ..pipeline_utils import PipelineABC


class AC3DPipeline(PipelineABC):
    """WorldFoundry pipeline for AC3D camera-controlled video generation."""

    MODEL_ID = "ac3d"
    OPERATOR_CLS = AC3DOperator
    SYNTHESIS_CLS = AC3DSynthesis
    MEMORY_TARGET = None

    @classmethod
    def _uses_component_contract(cls) -> bool:
        """Determine whether the pipeline class implements the component contract."""
        return True

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> AC3DPipeline:
        """Load the pipeline from pretrained checkpoints and configurations."""
        options = cls._runtime_options(model_path, required_components, kwargs)
        resolved_model_id = cls._resolve_model_id(options, model_id=model_id)
        synthesis_model = AC3DSynthesis.from_pretrained(
            {**options, "model_id": resolved_model_id},
            device=device,
        )
        return cls(
            model_id=resolved_model_id,
            operators=AC3DOperator(),
            synthesis_model=synthesis_model,
            memory_module=None,
            device=device,
        )


__all__ = ["AC3DPipeline"]
