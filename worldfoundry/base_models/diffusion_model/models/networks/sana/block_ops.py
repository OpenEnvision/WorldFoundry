"""SANA block glue using shared kernels without changing checkpoint modules.

The dispatcher retains PyTorch for autograd, unsupported inputs, small eager
workloads, and compiled graphs. Custom normalization and stochastic depth
keep their own forward semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from worldfoundry.core.kernels import layer_norm_scale_shift, residual_gate_add, scale_shift
from worldfoundry.core.nn.blocks.layers import DropPath


@dataclass(frozen=True, slots=True)
class SanaBlockFusionPolicy:
    """Per-model eager fusion; small workloads and autograd keep original math."""

    backend: str = "auto"
    min_elements: int = 12 * 1024**2
    fuse_layer_norm: bool = False
    repack_tokens: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "torch", "triton"}:
            raise ValueError("SANA block fusion backend must be auto, torch or triton")
        if isinstance(self.min_elements, bool) or not isinstance(self.min_elements, int):
            raise TypeError("min_elements must be an integer")
        if self.min_elements < 0:
            raise ValueError("min_elements must be non-negative")
        if not isinstance(self.fuse_layer_norm, bool):
            raise TypeError("fuse_layer_norm must be bool")
        if not isinstance(self.repack_tokens, bool):
            raise TypeError("repack_tokens must be bool")

    def eligible(self, value: torch.Tensor) -> bool:
        return not torch.is_grad_enabled() and value.numel() >= self.min_elements


def modulated_norm(
    value: torch.Tensor,
    norm: nn.Module,
    shift: torch.Tensor,
    scale: torch.Tensor,
    *,
    frames: int | None = None,
    policy: SanaBlockFusionPolicy | None = None,
) -> torch.Tensor:
    """Apply ``norm(value) * (1 + scale) + shift`` with eligible fusion.

    Only the standard affine-free last-dimension LayerNorm may be bypassed.
    In particular, FP32LayerNorm subclasses and offload wrappers still run.
    ``frames`` broadcasts ``[B, F, 1, C]`` modulation over flat video tokens.
    Custom norms still receive the original flat input.
    """

    fused = policy is not None and policy.eligible(value)
    if (
        fused
        and policy.fuse_layer_norm
        and type(norm) is nn.LayerNorm
        and not norm.elementwise_affine
        and norm.normalized_shape == (value.shape[-1],)
    ):
        shaped = value if frames is None else value.reshape(value.shape[0], frames, -1, value.shape[-1])
        result = layer_norm_scale_shift(shaped, scale, shift, eps=norm.eps, backend=policy.backend)
    else:
        normalized = norm(value)
        shaped = normalized if frames is None else normalized.reshape(value.shape[0], frames, -1, value.shape[-1])
        result = scale_shift(shaped, scale, shift, backend=policy.backend) if fused else shaped * (1 + scale) + shift
    return result.reshape(value.shape)


def gated_residual(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    drop_path: nn.Module,
    *,
    frames: int | None = None,
    policy: SanaBlockFusionPolicy | None = None,
) -> torch.Tensor:
    """Apply ``residual + drop_path(gate * update)`` preserving training order."""

    shaped_residual = residual
    shaped_update = update
    if frames is not None:
        shaped_residual = residual.reshape(residual.shape[0], frames, -1, residual.shape[-1])
        shaped_update = update.reshape(shaped_residual.shape)
    if (
        policy is not None
        and policy.eligible(residual)
        and (type(drop_path) is nn.Identity or (type(drop_path) is DropPath and not drop_path.training))
    ):
        return residual_gate_add(shaped_residual, shaped_update, gate, backend=policy.backend).reshape(residual.shape)
    return residual + drop_path((gate * shaped_update).reshape(residual.shape))


def prepare_sana_block_input(blocks: nn.ModuleList, value: torch.Tensor) -> torch.Tensor:
    """Convert token layout only under a separately opted-in numerical budget.

    Patch embedding yields transposed tokens; the native pointwise path keeps
    that stride through every residual. Repacking once allows the existing
    contiguous fusion kernels to execute without a copy at every block. Layout
    changes also change vendor reduction schedules, so defaults retain strides.
    """
    if value.is_cuda and any(
        (policy := getattr(block, "_worldfoundry_block_fusion", None)) is not None
        and policy.repack_tokens
        and policy.backend != "torch"
        and policy.eligible(value)
        for block in blocks
    ):
        return value.contiguous()
    return value


def run_sana_blocks(blocks: nn.ModuleList, value: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """One block-stack seam for stateless image/video graphs and request caches."""
    cache = kwargs.pop("feature_cache", None)
    step = kwargs.pop("feature_cache_step", 0)
    total_steps = kwargs.pop("feature_cache_total_steps", None)
    value = prepare_sana_block_input(blocks, value)
    if cache is None:
        for block in blocks:
            value = block(value, *args, **kwargs)
        return value
    return cache.run_blocks(
        step,
        value,
        lambda index, hidden: blocks[index](hidden, *args, **kwargs),
        block_count=len(blocks),
        total_steps=total_steps,
    )
