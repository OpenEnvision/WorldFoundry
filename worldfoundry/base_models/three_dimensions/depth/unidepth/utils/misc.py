"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""


import torch
import torch.nn.functional as F


@torch.jit.script
def max_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return torch.stack(tensors, dim=-1).max(dim=-1).values

def last_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    return tensors[-1]

def first_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    return tensors[0]

@torch.jit.script
def softmax_stack(tensors: list[torch.Tensor], temperature: float = 1.0) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return F.softmax(torch.stack(tensors, dim=-1) / temperature, dim=-1).sum(dim=-1)

@torch.jit.script
def mean_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return torch.stack(tensors, dim=-1).mean(dim=-1)

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
