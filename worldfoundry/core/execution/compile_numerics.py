# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Native normalization boundaries for inference compilation.

Opaque LayerNorm and RMSNorm operators retain vendor reductions inside compiled
regions. Module preparation changes bound forwards, not module types, parameters,
or hooks. Autograd uses the original tensor implementations. Opaque operators
around the exact diffusion kernels live in :mod:`worldfoundry.core.kernels.compiled_ops`.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MethodType
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from worldfoundry.core.nn.transformer.dit import RMSNorm


_CompileFunction = TypeVar("_CompileFunction", bound=Callable[..., Any])


def nested_compile_region(function: _CompileFunction) -> _CompileFunction:
    """Apply the optional PyTorch regional-compile hint without changing eager math."""
    decorate = getattr(torch.compiler, "nested_compile_region", None)
    return function if decorate is None else decorate(function)


@torch.library.custom_op(
    "worldfoundry::compile_layer_norm",
    mutates_args=(),
    schema="(Tensor value, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, "
    "float eps, bool amp_enabled, ScalarType amp_dtype) -> Tensor",
)
def _layer_norm(
    value: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    with torch.autocast(value.device.type, enabled=amp_enabled, dtype=amp_dtype):
        return F.layer_norm(value, normalized_shape, weight, bias, eps)


@_layer_norm.register_fake
def _layer_norm_fake(
    value: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    with torch.autocast(value.device.type, enabled=amp_enabled, dtype=amp_dtype):
        return F.layer_norm(value, normalized_shape, weight, bias, eps)


@torch.library.custom_op(
    "worldfoundry::compile_rms_norm",
    mutates_args=(),
    schema="(Tensor value, Tensor? weight, float eps, str compute_mode) -> Tensor",
)
def _rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    compute_mode: str,
) -> torch.Tensor:
    from worldfoundry.core.nn.transformer.dit import _rms_norm_reference

    return _rms_norm_reference(value, weight, eps, compute_mode)


@_rms_norm.register_fake
def _rms_norm_fake(
    value: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    compute_mode: str,
) -> torch.Tensor:
    from worldfoundry.core.nn.transformer.dit import _rms_norm_reference

    return _rms_norm_reference(value, weight, eps, compute_mode)


def _layer_norm_forward(module: torch.nn.LayerNorm, input: torch.Tensor) -> torch.Tensor:
    if not torch.compiler.is_compiling() or torch.is_grad_enabled():
        return torch.nn.LayerNorm.forward(module, input)
    return _layer_norm(
        input,
        list(module.normalized_shape),
        module.weight,
        module.bias,
        module.eps,
        torch.is_autocast_enabled(input.device.type),
        torch.get_autocast_dtype(input.device.type),
    )


def _rms_norm_forward(module: RMSNorm, value: torch.Tensor) -> torch.Tensor:
    from worldfoundry.core.nn.transformer.dit import RMSNorm

    if not torch.compiler.is_compiling() or torch.is_grad_enabled():
        return RMSNorm.forward(module, value)
    return _rms_norm(value, module.weight, module.eps, module.compute_mode)


def preserve_normalization_reductions(region: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    """Prepare native LayerNorm/RMSNorm inference math and return changed modules.

    Only canonical, unmodified forwards are prepared. Hooks still run through
    the existing modules; eager execution and autograd retain their contracts.
    Deleting the returned modules' instance ``forward`` attributes restores
    their original class methods if compilation preparation falls back.
    """
    from worldfoundry.core.nn.transformer.dit import RMSNorm

    prepared: list[torch.nn.Module] = []
    for module in region.modules():
        if "forward" in module.__dict__:
            continue
        # Instance ``forward`` attributes shadow the class methods until deleted.
        if type(module) is torch.nn.LayerNorm:
            cast(Any, module).forward = MethodType(_layer_norm_forward, module)
            prepared.append(module)
        elif type(module) is RMSNorm:
            cast(Any, module).forward = MethodType(_rms_norm_forward, module)
            prepared.append(module)
    return tuple(prepared)
