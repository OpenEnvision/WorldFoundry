"""Forward-only Triton normalization kernels used by Helios inference."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from diffusers.models.normalization import FP32LayerNorm, LayerNorm, RMSNorm

from .fp32_rmsnorm import FP32RMSNorm
from .utils import calculate_settings, torch_gpu_device


def replace_all_norms_with_flash_norms(model):
    """Patch supported normalization modules with inference-only kernels."""

    patched_count = {"LayerNorm": 0, "RMSNorm": 0}
    for module in model.modules():
        if isinstance(module, (LayerNorm, FP32LayerNorm)) and module.elementwise_affine:
            module.forward = (lambda self, x: flash_layernorm(self, x)).__get__(module, module.__class__)
            patched_count["LayerNorm"] += 1
        if isinstance(module, (torch.nn.RMSNorm, RMSNorm, FP32RMSNorm)):
            module.forward = (lambda self, x: flash_rms_layernorm(self, x)).__get__(module, module.__class__)
            patched_count["RMSNorm"] += 1

    print(f"Patched {patched_count['LayerNorm']} Flash_LayerNorm modules")
    print(f"Patched {patched_count['RMSNorm']} Flash_RMSNorm modules")
    return model


@triton.jit
def _layernorm_forward(
    output,
    output_row_stride,
    value,
    value_row_stride,
    weight,
    bias,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    value_row = tl.load(value + row_idx * value_row_stride + offsets, mask=mask, other=0).to(tl.float32)
    weight_row = tl.load(weight + offsets, mask=mask, other=0).to(tl.float32)
    bias_row = tl.load(bias + offsets, mask=mask, other=0).to(tl.float32)
    mean = tl.sum(value_row, axis=0) / n_cols
    centered = tl.where(mask, value_row - mean, 0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    normalized = centered * tl.rsqrt(variance + eps)
    tl.store(
        output + row_idx * output_row_stride + offsets,
        normalized * weight_row + bias_row,
        mask=mask,
    )


def flash_layernorm(layernorm, value: torch.Tensor) -> torch.Tensor:
    if layernorm.bias is None:
        return torch.nn.functional.layer_norm(
            value.float(),
            layernorm.normalized_shape,
            layernorm.weight.float(),
            None,
            layernorm.eps,
        ).to(value.dtype)

    shape = value.shape
    matrix = value.reshape(-1, shape[-1])
    output = torch.empty_like(matrix)
    block_size, num_warps = calculate_settings(matrix.shape[1])
    eps = getattr(layernorm, "variance_epsilon", layernorm.eps)
    with torch_gpu_device(matrix.device):
        _layernorm_forward[(matrix.shape[0],)](
            output,
            output.stride(0),
            matrix,
            matrix.stride(0),
            layernorm.weight,
            layernorm.bias,
            matrix.shape[1],
            eps,
            block_size=block_size,
            num_warps=num_warps,
        )
    return output.reshape(shape)


@triton.jit
def _rmsnorm_forward(
    output,
    output_row_stride,
    value,
    value_row_stride,
    weight,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols
    value_row = tl.load(value + row_idx * value_row_stride + offsets, mask=mask, other=0).to(tl.float32)
    weight_row = tl.load(weight + offsets, mask=mask, other=0)
    inverse_rms = tl.rsqrt(tl.sum(value_row * value_row, axis=0) / n_cols + eps)
    normalized = (value_row * inverse_rms).to(weight_row.dtype)
    tl.store(
        output + row_idx * output_row_stride + offsets,
        normalized * weight_row,
        mask=mask,
    )


@torch.compiler.disable
def flash_rms_layernorm(layernorm, value: torch.Tensor) -> torch.Tensor:
    shape = value.shape
    matrix = value.reshape(-1, shape[-1])
    output = torch.empty_like(matrix)
    block_size, num_warps = calculate_settings(matrix.shape[1])
    eps = getattr(layernorm, "variance_epsilon", layernorm.eps)
    with torch_gpu_device(matrix.device):
        _rmsnorm_forward[(matrix.shape[0],)](
            output,
            output.stride(0),
            matrix,
            matrix.stride(0),
            layernorm.weight,
            matrix.shape[1],
            eps,
            block_size=block_size,
            num_warps=num_warps,
        )
    return output.reshape(shape)
