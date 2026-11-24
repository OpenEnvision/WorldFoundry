# Copyright (c) Tencent Inc. All rights reserved.
# Basic brick modules for PAFPN based on CSPLayers

from .yolo_bricks import (
    CSPLayerWithTwoConv,
    ImagePoolingAttentionModule,
    MaxSigmoidAttnBlock,
    MaxSigmoidCSPLayerWithTwoConv,
    RepConvMaxSigmoidCSPLayerWithTwoConv,
    RepMaxSigmoidCSPLayerWithTwoConv,
)

__all__ = [
    "CSPLayerWithTwoConv",
    "ImagePoolingAttentionModule",
    "MaxSigmoidAttnBlock",
    "MaxSigmoidCSPLayerWithTwoConv",
    "RepConvMaxSigmoidCSPLayerWithTwoConv",
    "RepMaxSigmoidCSPLayerWithTwoConv",
]
