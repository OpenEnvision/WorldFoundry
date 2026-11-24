"""Pipeline registration for the in-tree ABot-M0 policy."""

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class ABotM0Pipeline(ComponentPipeline):
    """Component pipeline backed by the native ABot-M0 runtime."""

    MODEL_ID = "abot-m0"
    OPERATOR_TARGET = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.action_generation.abot_m0.abot_m0_synthesis:ABotM0Synthesis"
    generation_type = "vla_policy"


__all__ = ["ABotM0Pipeline"]
