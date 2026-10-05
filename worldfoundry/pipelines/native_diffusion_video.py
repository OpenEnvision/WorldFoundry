"""Compatibility import for video products using the unified visual adapter."""

from __future__ import annotations

from .native_diffusion import NativeVisualDiffusionPipeline


class NativeTextToVideoPipeline(NativeVisualDiffusionPipeline):
    """Semantic video alias; all implementation lives in the visual adapter."""


__all__ = ["NativeTextToVideoPipeline"]
