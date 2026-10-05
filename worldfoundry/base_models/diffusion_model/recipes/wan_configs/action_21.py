"""Wan 2.1 action-recipe lookup tables (T2V / I2V / T2I 14B + 1.3B).

Re-exports the official Wan 2.1 EasyDict graphs and maps them to the
short names used by action-conditioned runners (``t2v-14B``,
``i2v-14B``, …).  Also lists supported pixel sizes and max-area
caps for those identities.

``t2i-14B`` is a deepcopy of ``t2v-14B`` with a distinct display name.
Architecture numbers stay in :mod:`.wan21`; this file is the index.
"""

from __future__ import annotations

import copy

from worldfoundry.base_models.diffusion_model.recipes.wan_configs.wan21.i2v_14b_upstream import (
    i2v_14B,
)
from worldfoundry.base_models.diffusion_model.recipes.wan_configs.wan21.t2v_14b import (
    t2v_14B,
)
from worldfoundry.base_models.diffusion_model.recipes.wan_configs.wan21.t2v_1p3b import (
    t2v_1_3B,
)

t2i_14B = copy.deepcopy(t2v_14B)
t2i_14B.__name__ = "Config: Wan T2I 14B"

WAN_CONFIGS = {
    "t2v-14B": t2v_14B,
    "t2v-1.3B": t2v_1_3B,
    "i2v-14B": i2v_14B,
    "t2i-14B": t2i_14B,
}

SIZE_CONFIGS = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
    "1024*1024": (1024, 1024),
}

MAX_AREA_CONFIGS = {
    "720*1280": 720 * 1280,
    "1280*720": 1280 * 720,
    "480*832": 480 * 832,
    "832*480": 832 * 480,
}

SUPPORTED_SIZES = {
    "t2v-14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "t2v-1.3B": ("480*832", "832*480"),
    "i2v-14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "t2i-14B": tuple(SIZE_CONFIGS.keys()),
}

__all__ = [
    "MAX_AREA_CONFIGS",
    "SIZE_CONFIGS",
    "SUPPORTED_SIZES",
    "WAN_CONFIGS",
    "i2v_14B",
    "t2i_14B",
    "t2v_14B",
    "t2v_1_3B",
]
