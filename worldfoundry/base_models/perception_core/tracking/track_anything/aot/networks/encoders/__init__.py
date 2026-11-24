# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See repository-root THIRD-PARTY-NOTICES for details.

from torch import nn

from ..layers.normalization import FrozenBatchNorm2d
from .resnet import ResNet50


def build_encoder(name, frozen_bn=True, freeze_at=-1):
    BatchNorm = FrozenBatchNorm2d if frozen_bn else nn.BatchNorm2d

    if name == "resnet50":
        return ResNet50(16, BatchNorm, freeze_at=freeze_at)
    else:
        raise NotImplementedError
