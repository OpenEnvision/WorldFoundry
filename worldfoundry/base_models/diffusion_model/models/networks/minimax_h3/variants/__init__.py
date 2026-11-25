"""MiniMax-H3 inference variants."""

from .h3_tiny import H3TinyDiTModel, H3TinyMeanFlowTimeEmbedder, LowRankAdalnProj
from .pi_h3 import PiH3DiTModel, PiH3PrunedTimeEmbedder, RGBColorEncoder, SolH3DWDHead
from .pika import load_pika_h3_flex_adapter

__all__ = [
    "H3TinyDiTModel",
    "H3TinyMeanFlowTimeEmbedder",
    "LowRankAdalnProj",
    "PiH3DiTModel",
    "PiH3PrunedTimeEmbedder",
    "RGBColorEncoder",
    "SolH3DWDHead",
    "load_pika_h3_flex_adapter",
]
