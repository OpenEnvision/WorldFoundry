# This file includes code originally from the Metric3D repository:
# https://github.com/YvanYin/Metric3D
# Licensed under the BSD-2 License. See repository-root THIRD-PARTY-NOTICES for details.

"""Module for base_models -> three_dimensions -> depth -> metric3d -> model -> backbones -> __init__.py functionality."""

from .ConvNeXt import convnext_base, convnext_large, convnext_small, convnext_tiny, convnext_xlarge
from .ViT_DINO_reg import vit_giant2_reg, vit_large_reg, vit_small_reg

__all__ = [
    "convnext_xlarge",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "convnext_tiny",
    "vit_small_reg",
    "vit_large_reg",
    "vit_giant2_reg",
]
