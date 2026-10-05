# Copyright 2024 MIT Han Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Conv2d + optional norm/act helper used by SANA MBConvPreGLU.

:class:`ConvLayer` wraps ``nn.Conv2d`` with same-padding, dropout, and
registry-built norm / activation.  It is the building block of the
PyTorch MBConv + pre-GLU reference that the Triton fused FFN mirrors.

Input / output are channel-first ``B C H W`` feature maps.  Norm names
follow :mod:`.norm` (``bn2d``, ``ln2d``, …); act names follow
:mod:`~worldfoundry.base_models.diffusion_model.models.networks.sana.activation`.
"""

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.networks.sana.activation import build_act, get_act_name

from worldfoundry.core.nn import get_same_padding
from .norm import build_norm, get_norm_name


class ConvLayer(nn.Module):
    """Conv2d with optional fused norm and activation for MBConv stacks."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        padding: int or None = None,
        use_bias=False,
        dropout=0.0,
        norm="bn2d",
        act="relu",
    ):
        """Init.

        Args:
            in_dim: The in dim.
            out_dim: The out dim.
            kernel_size: The kernel size.
            stride: The stride.
            dilation: The dilation.
            groups: The groups.
            padding: The padding.
            use_bias: The use bias.
            dropout: The dropout.
            norm: The norm.
            act: The act.
        """
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.use_bias = use_bias

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_dim)
        self.act = build_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x
