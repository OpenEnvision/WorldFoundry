# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size,
        patch_size,
        out_channels,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, bias=False)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=False)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x
