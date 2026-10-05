# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Module for base_models -> perception_core -> segment -> sam3 -> perflib -> fused.py functionality."""

import torch

addmm_act_op = torch.ops.aten._addmm_activation


def addmm_act(activation, linear, mat1):
    """Addmm act.

    Args:
        activation: The activation.
        linear: The linear.
        mat1: The mat1.
    """
    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")
    self = linear.bias.detach()
    mat2 = linear.weight.detach()
    dtype = (
        torch.get_autocast_dtype(mat1.device.type)
        if torch.is_autocast_enabled(mat1.device.type)
        else mat1.dtype
    )
    self = self.to(dtype)
    mat1 = mat1.to(dtype)
    mat2 = mat2.to(dtype)
    mat1_flat = mat1.view(-1, mat1.shape[-1])
    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))
    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))
    raise ValueError(f"Unexpected activation {activation}")
