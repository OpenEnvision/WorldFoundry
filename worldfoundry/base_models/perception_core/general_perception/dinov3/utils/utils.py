# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Module for base_models -> perception_core -> general_perception -> dinov3 -> utils -> utils.py functionality."""

import logging
import random
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from worldfoundry.core.nn.blocks.module_utils import named_apply

logger = logging.getLogger("dinov3")


def cat_keep_shapes(x_list: List[Tensor]) -> Tuple[Tensor, List[Tuple[int]], List[int]]:
    """Cat keep shapes.

    Args:
        x_list: The x list.

    Returns:
        The return value.
    """
    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def uncat_with_shapes(flattened: Tensor, shapes: List[Tuple[int]], num_tokens: List[int]) -> List[Tensor]:
    """Uncat with shapes.

    Args:
        flattened: The flattened.
        shapes: The shapes.
        num_tokens: The num tokens.

    Returns:
        The return value.
    """
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    shapes_adjusted = [shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes]
    outputs_reshaped = [o.reshape(shape) for o, shape in zip(outputs_splitted, shapes_adjusted)]
    return outputs_reshaped


def named_replace(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    """Named replace.

    Args:
        fn: The fn.
        module: The module.
        name: The name.
        depth_first: The depth first.
        include_root: The include root.

    Returns:
        The return value.
    """
    if not depth_first and include_root:
        module = fn(module=module, name=name)
    for child_name_o, child_module in list(module.named_children()):
        child_name = ".".join((name, child_name_o)) if name else child_name_o
        new_child = named_replace(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
        setattr(module, child_name_o, new_child)

    if depth_first and include_root:
        module = fn(module=module, name=name)
    return module


def count_parameters(module: nn.Module) -> int:
    """Count parameters.

    Args:
        module: The module.

    Returns:
        The return value.
    """
    c = 0
    for m in module.parameters():
        c += m.nelement()
    return c
