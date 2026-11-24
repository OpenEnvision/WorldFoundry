# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Opaque operators that keep the exact diffusion kernels intact under compilation.

Each operator wraps one kernel from :mod:`.diffusion` so Inductor cannot fuse,
reorder or contract its arithmetic inside a compiled region. Fake
implementations provide shape and dtype propagation; autograd uses the original
tensor implementations.
"""

from __future__ import annotations

import torch


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


__all__ = ["compiled_layer_norm_scale_shift", "compiled_residual_gate", "compiled_residual_gate_inplace"]
