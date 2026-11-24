"""Activation functions required by SHARP inference."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch

from .types import ActivationType


class ActivationPair(NamedTuple):
    forward: Callable[[torch.Tensor], torch.Tensor]
    inverse: Callable[[torch.Tensor], torch.Tensor]


def inverse_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(tensor / (1.0 - tensor))


def inverse_softplus(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    tensor = tensor.clamp_min(eps)
    return tensor + torch.log(-torch.expm1(-tensor))


def create_activation_pair(activation_type: ActivationType) -> ActivationPair:
    if activation_type == "linear":
        return ActivationPair(lambda value: value, lambda value: value)
    if activation_type == "exp":
        return ActivationPair(torch.exp, torch.log)
    if activation_type == "sigmoid":
        return ActivationPair(torch.sigmoid, inverse_sigmoid)
    if activation_type == "softplus":
        return ActivationPair(torch.nn.functional.softplus, inverse_softplus)
    raise ValueError(f"Unsupported activation function: {activation_type}.")
