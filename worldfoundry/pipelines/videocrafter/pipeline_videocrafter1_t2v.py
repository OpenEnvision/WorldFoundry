"""Videocrafter1 T2V visual generation pipeline module."""

from __future__ import annotations

from ...synthesis.visual_generation.videocrafter.videocrafter1_t2v_synthesis import VideoCrafter1T2VSynthesis
from .base import VideoCrafterPipelineBase


class VideoCrafter1T2VPipeline(VideoCrafterPipelineBase):
    """WorldFoundry pipeline for VideoCrafter1 text-to-video."""

    SYNTHESIS_CLS = VideoCrafter1T2VSynthesis
    MEMORY_MODEL_ID = "videocrafter1-t2v"


__all__ = ["VideoCrafter1T2VPipeline"]
