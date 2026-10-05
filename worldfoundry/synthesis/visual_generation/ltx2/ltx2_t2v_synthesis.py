"""LTX-2 text-to-video synthesis wrapper."""

from .ltx2_3_t2v_synthesis import LTX23T2VSynthesis


class LTX2T2VSynthesis(LTX23T2VSynthesis):
    """Select the LTX-2 19B checkpoint and native text-to-video recipe."""

    MODEL_NAME = "ltx2_t2v"
    RUNTIME_CONFIG_KEY = MODEL_NAME


__all__ = ["LTX2T2VSynthesis"]
