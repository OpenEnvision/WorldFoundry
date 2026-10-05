"""Declarative Wan 2.2 architecture and inference geometry data.

Index for Wan 2.2 / A14B identities: dual-expert T2V and I2V A14B,
TI2V 5B (Wan2.2 VAE38), speech-to-video 14B, and Animate 14B.
``WAN_CONFIGS`` maps short names; ``SIZE_CONFIGS`` / ``SUPPORTED_SIZES``
list legal pixel rectangles per identity.

Shared T5 / dtype / 81-frame defaults live in :mod:`.shared`.
Dual-expert graphs add low/high-noise checkpoints and a sigma boundary.
"""

from .animate_14b import animate_14B
from .i2v_a14b import i2v_A14B
from .s2v_14b import s2v_14B
from .t2v_a14b import t2v_A14B
from .ti2v_5b import ti2v_5B

WAN_CONFIGS = {
    "t2v-A14B": t2v_A14B,
    "i2v-A14B": i2v_A14B,
    "ti2v-5B": ti2v_5B,
    "animate-14B": animate_14B,
    "s2v-14B": s2v_14B,
}

SIZE_CONFIGS = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
    "704*1280": (704, 1280),
    "1280*704": (1280, 704),
    "1024*704": (1024, 704),
    "704*1024": (704, 1024),
}

MAX_AREA_CONFIGS = {name: height * width for name, (height, width) in SIZE_CONFIGS.items()}

SUPPORTED_SIZES = {
    "t2v-A14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "i2v-A14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "ti2v-5B": ("704*1280", "1280*704"),
    "s2v-14B": (
        "720*1280",
        "1280*720",
        "480*832",
        "832*480",
        "1024*704",
        "704*1024",
        "704*1280",
        "1280*704",
    ),
    "animate-14B": ("720*1280", "1280*720"),
}

__all__ = [
    "MAX_AREA_CONFIGS",
    "SIZE_CONFIGS",
    "SUPPORTED_SIZES",
    "WAN_CONFIGS",
    "animate_14B",
    "i2v_A14B",
    "s2v_14B",
    "t2v_A14B",
    "ti2v_5B",
]
