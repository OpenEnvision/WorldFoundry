"""Pipeline registration kept local to the Spirit-v1.5 integration."""

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class SpiritV15Pipeline(ComponentPipeline):
    """WorldFoundry component pipeline for Spirit-v1.5."""

    MODEL_ID = "spirit-v1.5"
    OPERATOR_TARGET = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.action_generation.spirit_v15.spirit_v15_synthesis:SpiritV15Synthesis"
    generation_type = "vla_policy"


__all__ = ["SpiritV15Pipeline"]
