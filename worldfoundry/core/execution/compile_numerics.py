# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Native arithmetic boundaries for inference compilation.

Opaque operators retain vendor reductions and existing exact diffusion kernels.
Module preparation changes bound forwards, not module types, parameters, or hooks.
Autograd uses the original tensor implementations.
"""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from worldfoundry.core.nn.transformer.dit import RMSNorm


@torch.library.custom_op(
    "worldfoundry::compile_layer_norm",
    mutates_args=(),
    schema="(Tensor value, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, "
    "float eps, bool amp_enabled, ScalarType amp_dtype) -> Tensor",
)
def _layer_norm(
    value: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    with torch.autocast(value.device.type, enabled=amp_enabled, dtype=amp_dtype):
        return F.layer_norm(value, normalized_shape, weight, bias, eps)


@_layer_norm.register_fake
def _layer_norm_fake(
    value: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
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
    weight: Optional[torch.Tensor],
    eps: float,
    compute_mode: str,
) -> torch.Tensor:
    from worldfoundry.core.nn.transformer.dit import _rms_norm_reference

    return _rms_norm_reference(value, weight, eps, compute_mode)


@_rms_norm.register_fake
def _rms_norm_fake(
    value: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    compute_mode: str,
) -> torch.Tensor:
    from worldfoundry.core.nn.transformer.dit import _rms_norm_reference

    return _rms_norm_reference(value, weight, eps, compute_mode)


@torch.library.custom_op("worldfoundry::compile_layer_norm_scale_shift", mutates_args=())
def compiled_layer_norm_scale_shift(
    value: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    upcast: bool,
    backend: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    """Keep native LayerNorm and scale/shift rounding in a compiled region."""
    from worldfoundry.core.kernels.diffusion import layer_norm_scale_shift

    with torch.autocast(value.device.type, enabled=amp_enabled, dtype=amp_dtype):
        return layer_norm_scale_shift(value, scale, shift, eps, upcast=upcast, backend=backend)


@compiled_layer_norm_scale_shift.register_fake
def _layer_norm_scale_shift_fake(
    value: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    upcast: bool,
    backend: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    from worldfoundry.core.kernels.diffusion import _layer_norm_scale_shift_torch

    with torch.autocast(value.device.type, enabled=amp_enabled, dtype=amp_dtype):
        return _layer_norm_scale_shift_torch(value, scale, shift, eps, upcast)


@torch.library.custom_op("worldfoundry::compile_residual_gate", mutates_args=())
def compiled_residual_gate(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    """Keep the native multiply-then-add boundary without FMA contraction."""
    from worldfoundry.core.kernels.diffusion import residual_gate_add

    return residual_gate_add(residual, update, gate, backend=backend)


@compiled_residual_gate.register_fake
def _residual_gate_fake(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    return residual + update * gate


@torch.library.custom_op("worldfoundry::compile_residual_gate_inplace", mutates_args=("residual",))
def compiled_residual_gate_inplace(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
) -> None:
    """Preserve the native mutating gate without hiding its input mutation."""
    from worldfoundry.core.kernels.diffusion import residual_gate_add_

    residual_gate_add_(residual, update, gate)


@compiled_residual_gate_inplace.register_fake
def _residual_gate_inplace_fake(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
) -> None:
    return None


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
        if type(module) is torch.nn.LayerNorm:
            setattr(module, "forward", MethodType(_layer_norm_forward, module))
            prepared.append(module)
        elif type(module) is RMSNorm:
            setattr(module, "forward", MethodType(_rms_norm_forward, module))
            prepared.append(module)
    return tuple(prepared)
