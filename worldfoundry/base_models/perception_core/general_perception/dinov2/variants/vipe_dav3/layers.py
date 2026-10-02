# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""DINO-style layers assembled from shared WorldFoundry neural-network primitives."""

import logging
from typing import Callable, Optional

from torch import nn

from worldfoundry.base_models.perception_core.general_perception.dinov2.variants.depth_anything_v3.layers.attention import (
    Attention,
)
from worldfoundry.base_models.perception_core.general_perception.dinov2.variants.depth_anything_v3.layers.block import (
    Block,
)
from worldfoundry.core.attention.rotary.rope_2d import PositionGetter, RotaryPositionEmbedding2D
from worldfoundry.core.nn import SwiGLUFFN
from worldfoundry.core.nn.blocks.stochastic_depth import drop_add_residual_stochastic_depth, get_branges_scales

logger = logging.getLogger("dinov2")
XFORMERS_AVAILABLE = False


class SwiGLUFFNFused(SwiGLUFFN):
    """DINO-compatible fused SwiGLU width adjustment."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        del act_layer, drop
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


__all__ = [
    "SwiGLUFFN",
    "SwiGLUFFNFused",
    "Block",
    "PositionGetter",
    "RotaryPositionEmbedding2D",
]
