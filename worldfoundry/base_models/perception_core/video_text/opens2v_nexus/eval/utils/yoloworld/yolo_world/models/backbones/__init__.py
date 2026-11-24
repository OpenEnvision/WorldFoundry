# Copyright (c) Tencent Inc. All rights reserved.
# YOLO Multi-Modal Backbone (Vision Language)
# Vision: YOLOv8 CSPDarknet
# Language: CLIP Text Encoder (12-layer transformer)
from .mm_backbone import (
    HuggingCLIPLanguageBackbone,
    HuggingVisionBackbone,
    MultiModalYOLOBackbone,
    PseudoLanguageBackbone,
)

__all__ = [
    "HuggingCLIPLanguageBackbone",
    "HuggingVisionBackbone",
    "MultiModalYOLOBackbone",
    "PseudoLanguageBackbone",
]
