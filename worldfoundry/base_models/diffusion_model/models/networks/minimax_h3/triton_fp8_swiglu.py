"""Fused Triton SwiGLU + row-wise FP8 activation quantization for MiniMax H3.

Eliminates materializing the ``[T, ffn_hidden_size]`` BF16 SwiGLU intermediate
tensor in HBM between ``mlp.fc1`` and FP8 ``mlp.fc2`` (for H3: ``14336`` wide,
saving ~1.1 GiB of DRAM read/write per layer across 50 layers at 768p).

Preserves exact ``nn.Module`` attribute names (``fc1``, ``fc2``) and
checkpoint key compatibility while delegating ``fc2``'s FP8 GEMM directly to
:meth:`Float8Linear._forward_quantized`.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from worldfoundry.core.acceleration.quantization.linear import (
    Float8Linear,
    _fp8_linear_eligible,
    _is_compiling,
)
from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

from .ops import silu_mul

configure_persistent_compile_cache(namespace="minimax-h3-fp8-swiglu")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

_FP8_TL_DTYPE = {
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e5m2: tl.float8e5,
}


@triton.jit
def _swiglu_quantize_rowwise_fp8_tiled_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    ffn_hidden,
    fp8_max,
    tiny,
    in_row_stride,
    out_row_stride,
    BLOCK_N: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
):
    """Two-pass register-safe tiled SwiGLU (``gate | up``) + rowwise FP8 quantize.

    Pass 1 streams ``BLOCK_N`` tiles of ``gate`` and ``up`` to compute the
    per-row ``amax`` of ``round_in_dtype(silu(gate) * up)``.
    Pass 2 reloads L2-resident ``gate``/``up`` tiles, quantizes by ``1/scale``,
    and stores FP8 codes directly into ``output_ptr``.
    """

    row = tl.program_id(0).to(tl.int64)
    gate_row = input_ptr + row * in_row_stride
    up_row = gate_row + ffn_hidden
    out_row = output_ptr + row * out_row_stride
    in_elem_ty = input_ptr.dtype.element_ty

    amax = tl.full((), 0.0, tl.float32)
    for start in tl.range(0, ffn_hidden, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < ffn_hidden
        gate = tl.load(gate_row + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_row + cols, mask=mask, other=0.0).to(tl.float32)
        silu_gate = gate * tl.sigmoid(gate)
        # Match silu_and_mul's fp32 product -> input-dtype rounding before FP8 cast.
        val = (silu_gate * up).to(in_elem_ty).to(tl.float32)
        val = tl.where(mask, val, 0.0)
        amax = tl.maximum(amax, tl.max(tl.abs(val), axis=0))

    fp8_max_f = tl.cast(fp8_max, tl.float32)
    tiny_f = tl.cast(tiny, tl.float32)
    scale = tl.maximum(amax / fp8_max_f, tiny_f)
    inv_scale = 1.0 / scale
    tl.store(scale_ptr + row, scale)

    for start in tl.range(0, ffn_hidden, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < ffn_hidden
        gate = tl.load(gate_row + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_row + cols, mask=mask, other=0.0).to(tl.float32)
        val = (gate * tl.sigmoid(gate) * up).to(in_elem_ty).to(tl.float32)
        quantized = tl.minimum(tl.maximum(val * inv_scale, -fp8_max_f), fp8_max_f)
        tl.store(out_row + cols, quantized.to(FP8_DTYPE), mask=mask)


def swiglu_quantize_rowwise_fp8_triton(
    hidden: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU + rowwise FP8 quantization for ``[rows, 2 * ffn_hidden]``."""

    if dtype not in _FP8_TL_DTYPE:
        raise ValueError(f"unsupported FP8 dtype: {dtype}")
    if hidden.device.type != "cuda" or hidden.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("fused SwiGLU FP8 quantization requires CUDA fp16/bf16 input")
    if hidden.ndim != 2 or hidden.shape[-1] % 2 != 0:
        raise ValueError(f"expected 2D [rows, 2 * ffn_hidden], got {tuple(hidden.shape)}")

    source = hidden if hidden.stride(1) == 1 else hidden.contiguous()
    rows = int(source.shape[0])
    ffn_hidden = int(source.shape[1]) // 2
    codes = torch.empty((rows, ffn_hidden), dtype=dtype, device=source.device)
    scales = torch.empty((rows, 1), dtype=torch.float32, device=source.device)
    if rows == 0 or ffn_hidden == 0:
        return codes, scales

    fp8_max = float(torch.finfo(dtype).max)
    tiny = float(torch.finfo(torch.float32).tiny)
    block_n = min(triton.next_power_of_2(ffn_hidden), 2048)
    num_warps = 4 if block_n <= 512 else 8
    _swiglu_quantize_rowwise_fp8_tiled_kernel[(rows,)](
        source,
        codes,
        scales,
        ffn_hidden,
        fp8_max,
        tiny,
        source.stride(0),
        codes.stride(0),
        BLOCK_N=block_n,
        FP8_DTYPE=_FP8_TL_DTYPE[dtype],
        num_warps=num_warps,
    )
    return codes, scales


class FusedFP8SwiGLUMLP(nn.Module):
    """Drop-in replacement for :class:`MiniMaxH3MLP` fusing SwiGLU into FP8 ``fc2``."""

    def __init__(self, fc1: nn.Module, fc2: Float8Linear) -> None:
        super().__init__()
        self.fc1 = fc1
        self.fc2 = fc2
        self.fused_calls = 0
        self.fallback_calls = 0
        self.request_fused_calls = 0
        self.request_fallback_calls = 0
        self._request_window_active = False

    def reset_request_window(self) -> None:
        self.request_fused_calls = 0
        self.request_fallback_calls = 0
        self._request_window_active = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        down = cast(Float8Linear, self.fc2)
        ffn_hidden = down.in_features
        # Eligibility mirrors Float8Linear: CUDA, bf16/fp16, rowwise, no grad,
        # not compiling, and GEMM work above gate (0 when dense fallback dropped).
        min_work = 0.0 if down.weight is None else None
        rows = hidden.numel() // hidden.shape[-1] if hidden.numel() else 0
        eligible = (
            not _is_compiling()
            and rows > 0
            and down.low_precision_enabled
            and down.scaling == "rowwise"
            and hidden.shape[-1] == 2 * ffn_hidden
            and hidden.device.type == "cuda"
            and hidden.dtype in {torch.float16, torch.bfloat16}
            and not torch.is_grad_enabled()
            and _fp8_linear_eligible(
                hidden[..., :ffn_hidden],
                down.out_features,
                down._hardware_eligible,
                min_gemm_work=min_work,
            )
        )
        if not eligible:
            if not _is_compiling():
                self.fallback_calls += 1
                self.request_fallback_calls += 1
            return down(silu_mul(hidden))

        codes, scales = swiglu_quantize_rowwise_fp8_triton(
            hidden.reshape(rows, 2 * ffn_hidden),
            down.fp8_dtype,
        )
        out_shape = (*hidden.shape[:-1], ffn_hidden)
        result = down._forward_quantized(codes, scales, out_shape, hidden.dtype)
        if not _is_compiling():
            self.fused_calls += 1
            self.request_fused_calls += 1
        return result


def fuse_minimax_h3_fp8_swiglu_ffns(model: nn.Module) -> int:
    """Replace eligible ``MiniMaxH3MLP`` modules with :class:`FusedFP8SwiGLUMLP`."""

    from .dit import MiniMaxH3MLP

    count = 0
    for name, child in list(model.named_children()):
        if isinstance(child, FusedFP8SwiGLUMLP):
            continue
        if (
            isinstance(child, MiniMaxH3MLP)
            and isinstance(child.fc2, Float8Linear)
            and child.fc2.scaling == "rowwise"
            and isinstance(child.fc1, (nn.Linear, Float8Linear))
            and child.fc1.out_features == 2 * child.fc2.in_features
        ):
            setattr(model, name, FusedFP8SwiGLUMLP(child.fc1, child.fc2))
            count += 1
        else:
            count += fuse_minimax_h3_fp8_swiglu_ffns(child)
    return count


__all__ = [
    "FusedFP8SwiGLUMLP",
    "fuse_minimax_h3_fp8_swiglu_ffns",
    "swiglu_quantize_rowwise_fp8_triton",
]
