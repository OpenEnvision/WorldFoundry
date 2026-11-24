"""Pipeline registration kept local to the RDT-1B integration."""

from worldfoundry.pipelines.component_pipelines import ComponentPipeline


class RDT1BPipeline(ComponentPipeline):
    """WorldFoundry component pipeline for RDT-1B."""

    MODEL_ID = "rdt-1b"
    OPERATOR_TARGET = "worldfoundry.operators.official_policy_operator:OfficialPolicyOperator"
    MEMORY_TARGET = "worldfoundry.synthesis.action_generation.memory:ActionTraceMemory"
    SYNTHESIS_TARGET = "worldfoundry.synthesis.action_generation.rdt_1b.rdt_1b_synthesis:RDT1BSynthesis"
    generation_type = "diffusion_transformer_policy"


__all__ = ["RDT1BPipeline"]
