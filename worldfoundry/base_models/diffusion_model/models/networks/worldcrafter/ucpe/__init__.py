# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
from .bridge import (
    build_ucpe_attention_kwargs_for_chunk,
    enable_ucpe_inference_sdpa_attention,
    load_ucpe_camera_adapter_weights,
    patch_worldcrafter_transformer_ucpe,
)

__all__ = [
    "build_ucpe_attention_kwargs_for_chunk",
    "enable_ucpe_inference_sdpa_attention",
    "load_ucpe_camera_adapter_weights",
    "patch_worldcrafter_transformer_ucpe",
]
