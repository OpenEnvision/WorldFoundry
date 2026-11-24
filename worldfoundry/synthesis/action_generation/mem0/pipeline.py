"""Pipeline registration local to the Mem-0 integration."""

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class Mem0Pipeline(ComponentPipeline):
    MODEL_ID = "mem-0"
    OPERATOR_TARGET = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.action_generation.mem0.mem0_synthesis:Mem0Synthesis"
    generation_type = "memory_vla_policy"


__all__ = ["Mem0Pipeline"]
