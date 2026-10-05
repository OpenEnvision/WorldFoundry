# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
"""Losslessly store BF16-origin UCPE weights; preserve FP32 linear operations."""

import types
import torch
from torch import nn
from torch.nn import functional as F


def _forward(self, x):
    # Keep the input and operator unchanged. No BF16 GEMM substitution.
    return F.linear(
        x, self.weight.float(), None if self.bias is None else self.bias.float()
    )


def compact_ucpe(model):
    modules = []
    parameters = []
    for block in model.blocks:
        camera = block.cam_self_attn
        supported = set()
        for layer in camera.modules():
            if isinstance(layer, nn.Linear):
                modules.append(layer)
                for p in layer.parameters(recurse=False):
                    if p.dtype != torch.float32:
                        raise ValueError(
                            f"Expected FP32 UCPE parameters, got {p.dtype}"
                        )
                    if not torch.equal(p, p.bfloat16().float()):
                        raise ValueError(
                            "UCPE storage conversion would lose information"
                        )
                    parameters.append(p)
                    supported.add(id(p))
        if {id(p) for p in camera.parameters()} != supported:
            raise ValueError("Unsupported UCPE parameter type")
    saved = sum(p.numel() * 2 for p in parameters)
    for p in parameters:
        p.data = p.data.bfloat16()
    for layer in modules:
        layer.forward = types.MethodType(_forward, layer)
    return dict(
        modules=len(modules),
        tensors=len(parameters),
        bytes_saved=saved,
        storage_dtype="bfloat16",
        linear_weight_compute_dtype="float32",
        all_weights_exact=True,
    )
