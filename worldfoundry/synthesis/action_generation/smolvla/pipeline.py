"""Pipeline registration local to the SmolVLA integration."""

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class SmolVLAPipeline(ComponentPipeline):
    """WorldFoundry component pipeline for SmolVLA."""

    MODEL_ID = "smolvla"
    OPERATOR_TARGET = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.action_generation.smolvla.smolvla_synthesis:SmolVLASynthesis"
    generation_type = "vla_policy"


__all__ = ["SmolVLAPipeline"]
