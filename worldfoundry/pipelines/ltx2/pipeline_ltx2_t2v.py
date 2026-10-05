"""LTX-2 text-to-video pipeline."""

from .pipeline_ltx2_3_i2v import LTX23I2VPipeline
from ...synthesis.visual_generation.ltx2.ltx2_t2v_synthesis import LTX2T2VSynthesis


class LTX2T2VPipeline(LTX23I2VPipeline):
    """Expose LTX-2 text-only generation through the native runtime."""

    MODEL_ID = "ltx-2-t2v"
    SYNTHESIS_CLS = LTX2T2VSynthesis


__all__ = ["LTX2T2VPipeline"]
