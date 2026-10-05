"""Separate v1.1 inference route; the original AlayaWorld pipeline is unchanged."""

from worldfoundry.pipelines.official_world import OfficialWorldPipeline
from worldfoundry.synthesis.visual_generation.alayaworld_v11 import AlayaWorldV11Runtime


class AlayaWorldV11Pipeline(OfficialWorldPipeline):
    MODEL_ID = "alayaworld-v1.1"
    RUNTIME_CLS = AlayaWorldV11Runtime
