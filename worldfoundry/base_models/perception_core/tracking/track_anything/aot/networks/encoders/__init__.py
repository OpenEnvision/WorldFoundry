# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See repository-root THIRD-PARTY-NOTICES for details.

"""Module for base_models -> perception_core -> tracking -> track_anything -> aot -> networks -> encoders -> __init__.py functionality."""

from torch import nn

from ..layers.normalization import FrozenBatchNorm2d
from .resnet import ResNet50


def build_encoder(name, frozen_bn=True, freeze_at=-1):
    """Build encoder.

    Args:
        name: The name.
        frozen_bn: The frozen bn.
        freeze_at: The freeze at.
    """
    if frozen_bn:
        BatchNorm = FrozenBatchNorm2d
    else:
        BatchNorm = nn.BatchNorm2d

    if name == "resnet50":
        return ResNet50(16, BatchNorm, freeze_at=freeze_at)
    else:
        raise NotImplementedError
