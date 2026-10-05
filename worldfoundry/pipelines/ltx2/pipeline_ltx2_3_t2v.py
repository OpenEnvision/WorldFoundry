"""LTX-2.3 text-to-video pipeline."""

from __future__ import annotations

from .pipeline_ltx2_3_i2v import LTX23I2VPipeline
from ...synthesis.visual_generation.ltx2.ltx2_3_t2v_synthesis import LTX23T2VSynthesis


class LTX23T2VPipeline(LTX23I2VPipeline):
    """Runnable text-only LTX-2.3 pipeline backed by its own native recipe."""

    MODEL_ID = "ltx-2.3-t2v"
    SYNTHESIS_CLS = LTX23T2VSynthesis


__all__ = ["LTX23T2VPipeline"]
