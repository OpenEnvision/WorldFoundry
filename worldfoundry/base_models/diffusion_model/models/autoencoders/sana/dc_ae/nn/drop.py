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

"""Drop-path / dropout helpers for DC-AE residual stages.

:func:`apply_drop_func` / :func:`apply_droppath` and
:class:`DropPathResidualBlock` implement stochastic depth
used when building EfficientViT stages.

Applied to residual feature maps (BCHW or BCTHW) inside
:mod:`.ops`.  Inference typically leaves drop-path off.

Sana DC-AE training utility; not used by Wan / LTX VAEs.
"""

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from ..nn.ops import IdentityLayer, ResidualBlock
from worldfoundry.core.configuration import build_kwargs_from_config

__all__ = ["apply_drop_func"]


class _InferenceSchedulerStub:
    PROGRESS = 1.0


Scheduler = _InferenceSchedulerStub


def apply_drop_func(network: nn.Module, drop_config: Optional[dict[str, Any]]) -> None:
    """Apply drop func.

    Args:
        network: The network.
        drop_config: The drop config.

    Returns:
        The return value.
    """
    if drop_config is None:
        return

    drop_lookup_table = {
        "droppath": apply_droppath,
    }

    drop_func = drop_lookup_table[drop_config["name"]]
    drop_kwargs = build_kwargs_from_config(drop_config, drop_func)

    drop_func(network, **drop_kwargs)


def apply_droppath(
    network: nn.Module,
    drop_prob: float,
    linear_decay=True,
    scheduled=True,
    skip=0,
) -> None:
    """Apply droppath.

    Args:
        network: The network.
        drop_prob: The drop prob.
        linear_decay: The linear decay.
        scheduled: The scheduled.
        skip: The skip.

    Returns:
        The return value.
    """
    all_valid_blocks = []
    for m in network.modules():
        for name, sub_module in m.named_children():
            if isinstance(sub_module, ResidualBlock) and isinstance(sub_module.shortcut, IdentityLayer):
                all_valid_blocks.append((m, name, sub_module))
    all_valid_blocks = all_valid_blocks[skip:]
    for i, (m, name, sub_module) in enumerate(all_valid_blocks):
        prob = drop_prob * (i + 1) / len(all_valid_blocks) if linear_decay else drop_prob
        new_module = DropPathResidualBlock(
            sub_module.main,
            sub_module.shortcut,
            sub_module.post_act,
            sub_module.pre_norm,
            prob,
            scheduled,
        )
        m._modules[name] = new_module


class DropPathResidualBlock(ResidualBlock):
    """Drop path residual block implementation."""
    def __init__(
        self,
        main: nn.Module,
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
        ######################################
        drop_prob: float = 0,
        scheduled=True,
    ):
        """Init.

        Args:
            main: The main.
            shortcut: The shortcut.
            post_act: The post act.
            pre_norm: The pre norm.
            drop_prob: The drop prob.
            scheduled: The scheduled.
        """
        super().__init__(main, shortcut, post_act, pre_norm)

        self.drop_prob = drop_prob
        self.scheduled = scheduled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: The x.

        Returns:
            The return value.
        """
        if not self.training or self.drop_prob == 0 or not isinstance(self.shortcut, IdentityLayer):
            return ResidualBlock.forward(self, x)
        else:
            drop_prob = self.drop_prob
            if self.scheduled:
                drop_prob *= np.clip(Scheduler.PROGRESS, 0, 1)
            keep_prob = 1 - drop_prob

            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()  # binarize

            res = self.forward_main(x) / keep_prob * random_tensor + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
            return res
