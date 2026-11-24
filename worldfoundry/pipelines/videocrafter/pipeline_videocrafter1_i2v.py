"""Videocrafter1 I2V visual generation pipeline module."""

from __future__ import annotations

from ...synthesis.visual_generation.videocrafter.videocrafter1_i2v_synthesis import VideoCrafter1I2VSynthesis
from .base import VideoCrafterPipelineBase


class VideoCrafter1I2VPipeline(VideoCrafterPipelineBase):
    """WorldFoundry pipeline for VideoCrafter1 image-to-video."""

    SYNTHESIS_CLS = VideoCrafter1I2VSynthesis
    MEMORY_MODEL_ID = "videocrafter1-i2v"


__all__ = ["VideoCrafter1I2VPipeline"]
