"""Astronex causal and bidirectional inference through the official sampler."""

from worldfoundry.pipelines.official_world import OfficialWorldPipeline
from worldfoundry.synthesis.visual_generation.astronex_world import AstronexWorldRuntime


class AstronexWorldPipeline(OfficialWorldPipeline):
    MODEL_ID = "astronex-world"
    RUNTIME_CLS = AstronexWorldRuntime
