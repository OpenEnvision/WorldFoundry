"""Kairos visual generation pipeline module."""

from __future__ import annotations

from typing import Any

from ...operators.kairos_operator import KairosOperator
from ...synthesis.visual_generation.kairos import KairosSynthesis
from ..pipeline_utils import PipelineABC


class KairosPipeline(PipelineABC):
    """WorldFoundry pipeline for Kairos Sensenova generation."""

    MODEL_ID = "kairos-sensenova"
    MODEL_PATH_OPTION = "models_root"
    OPERATOR_CLS = KairosOperator
    SYNTHESIS_CLS = KairosSynthesis
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
    ) -> KairosPipeline:
        """Load the pipeline from pretrained checkpoints and configurations."""
        options = cls._runtime_options(model_path, required_components, kwargs)
        resolved_model_id = cls._resolve_model_id(options, model_id=model_id)
        synthesis_model = KairosSynthesis.from_pretrained(
            {**options, "model_id": resolved_model_id},
            device=device,
        )
        return cls(
            model_id=resolved_model_id,
            operators=KairosOperator(),
            synthesis_model=synthesis_model,
            memory_module=None,
            device=device,
        )


__all__ = ["KairosPipeline"]
