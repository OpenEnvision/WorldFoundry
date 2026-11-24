# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# rCM: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""rCM's Wan decoder keeps nearest-exact interpolation in the resident dtype."""

from typing import Any

from torch import nn

from ..reference_21 import Decoder3d as WanDecoder3d
from ..reference_21 import Resample


class Decoder3d(WanDecoder3d):
    """Use native BF16 upsampling without the reference decoder's FP32 temporary."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for layer in self.upsamples:
            if isinstance(layer, Resample) and layer.mode in ("upsample2d", "upsample3d"):
                layer.resample[0] = nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact")
