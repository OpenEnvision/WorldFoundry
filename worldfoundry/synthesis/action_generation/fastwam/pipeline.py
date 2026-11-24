"""Pipeline registration for the in-tree FastWAM policy."""

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class FastWAMPipeline(ComponentPipeline):
    """Component pipeline backed by the native FastWAM runtime."""

    MODEL_ID = "fastwam"
    OPERATOR_TARGET = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.action_generation.fastwam.fastwam_synthesis:FastWAMSynthesis"
    generation_type = "world_action_model"


__all__ = ["FastWAMPipeline"]
