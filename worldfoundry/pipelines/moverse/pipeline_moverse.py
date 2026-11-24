"""WorldFoundry pipeline for MoVerse."""

from __future__ import annotations

from ..component_pipelines import ComponentPipeline


class MoVersePipeline(ComponentPipeline):
    """Generate a panoramic Gaussian world and camera-conditioned video."""

    MODEL_ID = "moverse"
    MODEL_PATH_OPTION = "moverse_path"
    OPERATOR_TARGET = "worldfoundry.operators.moverse_operator:MoVerseOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.visual_generation.memory.runtime:RuntimeMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.visual_generation.moverse:MoVerseSynthesis"
    generation_type = "image_to_navigable_world"


__all__ = ["MoVersePipeline"]
