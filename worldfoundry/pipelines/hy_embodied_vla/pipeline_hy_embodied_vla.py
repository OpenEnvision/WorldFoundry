"""Component-contract pipeline for the isolated Hy-VLA synthesis runtime."""

from __future__ import annotations

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class HyEmbodiedVLAPipeline(ComponentPipeline):
    """WorldFoundry VLA policy pipeline for Tencent Hy-Embodied-0.5-VLA."""

    MODEL_ID = "hy-embodied-vla"
    MODEL_PATH_OPTION = "checkpoint"
    generation_type = "vla_policy"
    OPERATOR_TARGET = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = (
        "worldfoundry.synthesis.action_generation.hy_embodied_vla.hy_embodied_vla_synthesis:HyEmbodiedVLASynthesis"
    )


__all__ = ["HyEmbodiedVLAPipeline"]
