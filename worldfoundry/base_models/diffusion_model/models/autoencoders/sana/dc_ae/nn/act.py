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

"""Activation factories used by Sana DC-AE blocks.

:func:`build_act` maps config strings (SiLU, ReLU, …) onto
``nn.Module`` constructors consumed by MBConv / residual
stages in :mod:`.ops` and :mod:`.ops_3d`.

No tensor layout of its own — activations follow the
incoming BCHW / BCTHW feature map.

This is DC-AE graph math, not a runner Protocol adapter.
"""

from functools import partial
from typing import Optional

import torch.nn as nn

from worldfoundry.core.configuration import build_kwargs_from_config

__all__ = ["build_act"]


# register activation function here
REGISTERED_ACT_DICT: dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}


def build_act(name: str, **kwargs) -> Optional[nn.Module]:
    """Build act.

    Args:
        name: The name.

    Returns:
        The return value.
    """
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None
