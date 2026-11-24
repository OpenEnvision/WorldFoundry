"""Videocrafter2 T2V visual generation pipeline module."""

from __future__ import annotations

from ...synthesis.visual_generation.videocrafter.videocrafter2_t2v_synthesis import VideoCrafter2T2VSynthesis
from .base import VideoCrafterPipelineBase


class VideoCrafter2T2VPipeline(VideoCrafterPipelineBase):
    """WorldFoundry pipeline for VideoCrafter2 text-to-video."""

    SYNTHESIS_CLS = VideoCrafter2T2VSynthesis
    MEMORY_MODEL_ID = "videocrafter2-t2v"


__all__ = ["VideoCrafter2T2VPipeline"]
