# Copyright 2023 Meta
# SPDX-License-Identifier: BSD-3-Clause
"""Native PyTorch NVFP4 packing and Blackwell linear execution.

Responsibility: pack weights/activations as NVFP4 (1x16 E4M3 local
scale + tensorwise decode) and run Blackwell ``_scaled_mm_v2``, with a
dense fallback on any other GPU.

This module is not a trainer and is not the Triton encoder
(``triton_nvfp4.py`` is an optional activation path). Packed payloads
are inference buffers. OOM must propagate; other scaled-mm failures
blacklist the device for this process so the next call uses dense.

Public surface:
- :func:`quantize_nvfp4` / :func:`dequantize_nvfp4`
- :func:`nvfp4_swizzle_scales` / :func:`nvfp4_unswizzle_scales`
- :class:`NVFP4Linear` / :func:`replace_linear_with_nvfp4`

The FP4 RNE encoding, two-level scaling, and scale swizzle are minimal
adaptations of TorchAO. See the third-party notices for the pinned source.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry.core.kernels.capabilities import kernel_device_profile

FP4_MAX = 6.0
FP8_E4M3_MAX = 448.0
_NVFP4_KNOWN_CAPABILITIES = {(10, 0), (10, 3), (12, 0), (12, 1)}
_FAILED_DEVICES: set[str] = set()


# ──────────────────────────────────────────────────────────────────────────
# Bit packing — RNE E2M1 codes and two-level scales, no CUDA required
# ──────────────────────────────────────────────────────────────────────────


def _is_compiling() -> bool:
    """Skip telemetry writes inside Dynamo so counters are not graph outputs."""
    compiler = getattr(torch, "compiler", None)
    predicate = getattr(compiler, "is_compiling", None)
    return bool(predicate()) if callable(predicate) else False


def _n_ones(bits: int) -> int:
    """Bitmask of ``bits`` ones; used for FP32 → E2M1 field widths."""
    return (1 << bits) - 1


def _f32_to_floatx_unpacked(value: torch.Tensor, exponent_bits: int, mantissa_bits: int) -> torch.Tensor:
    """Encode float32 with round-to-nearest-even and finite saturation."""

    if value.dtype != torch.float32:
        raise TypeError("FP4 encoding expects float32 input")
    fp32_exponent_bits, fp32_mantissa_bits = 8, 23
    fp32_exponent_bias = _n_ones(fp32_exponent_bits - 1)
    exponent_bias = _n_ones(exponent_bits - 1)
    max_int = _n_ones(exponent_bits + mantissa_bits)
    sign_mask = 1 << (exponent_bits + mantissa_bits)
    magic_adder = _n_ones(fp32_mantissa_bits - mantissa_bits - 1)
    max_normal = 2 ** (_n_ones(exponent_bits) - exponent_bias) * (
        _n_ones(mantissa_bits + 1) / (2**mantissa_bits)
    )
    min_normal = 2 ** (1 - exponent_bias)
    denorm_exponent = (
        fp32_exponent_bias - exponent_bias + fp32_mantissa_bits - mantissa_bits + 1
    )
    denorm_mask_int = denorm_exponent << fp32_mantissa_bits
    denorm_mask_float = torch.tensor(
        denorm_mask_int,
        device=value.device,
        dtype=torch.int32,
    ).view(torch.float32)

    bits = value.view(torch.int32)
    sign = bits & 0x80000000
    magnitude = (bits ^ sign).view(torch.float32)
    saturated = magnitude >= max_normal
    denormal = (~saturated) & (magnitude < min_normal)
    normal = ~(saturated | denormal)

    denormal_value = (magnitude + denorm_mask_float).view(torch.int32)
    denormal_value = (denormal_value - denorm_mask_int).to(torch.uint8)

    normal_value = magnitude.view(torch.int32)
    mantissa_odd = (normal_value >> (fp32_mantissa_bits - mantissa_bits)) & 1
    normal_value = normal_value + (
        ((exponent_bias - fp32_exponent_bias) << fp32_mantissa_bits) + magic_adder
    )
    normal_value = normal_value + mantissa_odd
    normal_value = (normal_value >> (fp32_mantissa_bits - mantissa_bits)).to(torch.uint8)

    encoded = torch.full_like(magnitude, max_int, dtype=torch.uint8)
    encoded = torch.where(denormal, denormal_value, encoded)
    encoded = torch.where(normal, normal_value, encoded)
    encoded_sign = sign >> (
        fp32_mantissa_bits + fp32_exponent_bits - mantissa_bits - exponent_bits
    )
    encoded_sign = encoded_sign.to(torch.uint8) & sign_mask
    return encoded | encoded_sign


def _pack_uint4(unpacked: torch.Tensor) -> torch.Tensor:
    """Pack adjacent E2M1 codes into uint8; even index occupies the low nibble."""
    if unpacked.dtype != torch.uint8 or unpacked.shape[-1] % 2:
        raise ValueError("unpacked FP4 codes must be uint8 with an even final dimension")
    shape = unpacked.shape
    flat = unpacked.contiguous().view(-1)
    packed = flat[::2] | (flat[1::2] << 4)
    return packed.view(*shape[:-1], shape[-1] // 2)


def _unpack_uint4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_pack_uint4` for the reference dequantizer."""
    raw = packed.contiguous().view(torch.uint8)
    unpacked = torch.stack((raw & 0xF, raw >> 4), dim=-1)
    return unpacked.view(*packed.shape[:-1], packed.shape[-1] * 2)


def _ceil_div(value: int, divisor: int) -> int:
    """Pad-block count for the cuBLAS 128×4 swizzle."""
    return (value + divisor - 1) // divisor


def nvfp4_swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    """Convert ``[rows, K/16]`` scales to cuBLAS 128x4 blocked layout."""

    if scales.ndim != 2:
        raise ValueError("NVFP4 scales must be a 2D matrix")
    rows, columns = scales.shape
    row_blocks = _ceil_div(rows, 128)
    column_blocks = _ceil_div(columns, 4)
    padded_rows, padded_columns = row_blocks * 128, column_blocks * 4
    if (rows, columns) == (padded_rows, padded_columns):
        padded = scales
    else:
        padded = torch.zeros(
            (padded_rows, padded_columns),
            dtype=scales.dtype,
            device=scales.device,
        )
        padded[:rows, :columns] = scales
    blocks = padded.view(row_blocks, 128, column_blocks, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def nvfp4_unswizzle_scales(scales: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    """Inverse of :func:`nvfp4_swizzle_scales`, used for validation."""

    row_blocks = _ceil_div(rows, 128)
    column_blocks = _ceil_div(columns, 4)
    rearranged = scales.view(row_blocks * column_blocks, 32, 16)
    blocks = rearranged.reshape(row_blocks * column_blocks, 32, 4, 4).transpose(1, 2)
    padded = blocks.reshape(row_blocks, column_blocks, 128, 4).permute(0, 2, 1, 3)
    return padded.reshape(row_blocks * 128, column_blocks * 4)[:rows, :columns]


# ──────────────────────────────────────────────────────────────────────────
# Quantize / dequantize — portable reference used when Triton is unavailable
# ──────────────────────────────────────────────────────────────────────────


def _global_decode_scale(value: torch.Tensor) -> torch.Tensor:
    """Tensorwise scale so local E4M3 blocks stay inside ``FP8_E4M3_MAX * FP4_MAX``."""
    amax = value.detach().float().abs().amax()
    candidate = amax / (FP8_E4M3_MAX * FP4_MAX)
    return torch.where(amax > 0, candidate, torch.ones_like(candidate)).reshape(1)


def quantize_nvfp4(
    value: torch.Tensor,
    *,
    global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a 2D tensor using NVFP4 1x16 + tensorwise decode scales."""

    if value.ndim != 2 or value.shape[-1] % 16:
        raise ValueError("NVFP4 quantization expects [rows, K] with K divisible by 16")
    if not value.is_floating_point():
        raise TypeError("NVFP4 quantization expects floating-point input")
    contiguous = value.detach().float().contiguous()
    rows, width = contiguous.shape
    decode_global = _global_decode_scale(contiguous) if global_scale is None else global_scale.float().reshape(1)
    blocks = contiguous.view(rows, width // 16, 16)
    block_amax = blocks.abs().amax(dim=-1)
    local = (block_amax / FP4_MAX) / decode_global
    local_fp8 = local.clamp(
        min=torch.finfo(torch.float8_e4m3fn).tiny,
        max=FP8_E4M3_MAX,
    ).to(torch.float8_e4m3fn)
    normalized = blocks / (decode_global * local_fp8.float()).unsqueeze(-1)
    normalized = normalized.clamp(-FP4_MAX, FP4_MAX).reshape(rows, width)
    packed = _pack_uint4(_f32_to_floatx_unpacked(normalized, 2, 1))
    return packed, local_fp8, decode_global


def dequantize_nvfp4(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference dequantization for validation and unsupported hardware."""

    codes = _unpack_uint4(packed).long()
    positive = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=packed.device,
    )
    values = positive[codes & 0x7]
    values = torch.where((codes & 0x8).bool(), -values, values)
    rows, width = values.shape
    scales = block_scales.float().view(rows, width // 16, 1) * global_scale.float()
    return (values.view(rows, width // 16, 16) * scales).view(rows, width).to(dtype)


# ──────────────────────────────────────────────────────────────────────────
# Hardware gate — known Blackwell CCs + CUDA ≥ 12.8 + ATen scaled-mm
# ──────────────────────────────────────────────────────────────────────────


def _cuda_version_at_least_12_8() -> bool:
    """NVFP4 kernels shipped with CUDA 12.8; a missing version string is ineligible."""
    try:
        major, minor = (int(item) for item in str(torch.version.cuda).split(".")[:2])
    except (TypeError, ValueError):
        return False
    return (major, minor) >= (12, 8)


def _nvfp4_hardware_eligible(device: torch.device) -> bool:
    """False on ROCm, unknown SM, a prior scaled-mm failure, or a missing ATen op."""
    if device.type != "cuda" or getattr(torch.version, "hip", None) is not None:
        return False
    if str(device) in _FAILED_DEVICES or not _cuda_version_at_least_12_8():
        return False
    if not hasattr(torch, "float4_e2m1fn_x2") or not callable(getattr(F, "scaled_mm", None)):
        return False
    profile = kernel_device_profile(device)
    return profile.compute_capability in _NVFP4_KNOWN_CAPABILITIES


# ──────────────────────────────────────────────────────────────────────────
# NVFP4Linear — packed weights + optional dense copy; OOM never falls back
# ──────────────────────────────────────────────────────────────────────────


class NVFP4Linear(nn.Module):
    """Dynamic-activation NVFP4 linear with a dense cross-GPU fallback."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        keep_dense_fallback: bool = True,
    ) -> None:
        """Require ``in % 32 == 0`` and ``out % 16 == 0`` (cuBLAS NVFP4 tile)."""
        super().__init__()
        if weight.ndim != 2 or weight.shape[1] % 32 or weight.shape[0] % 16:
            raise ValueError("NVFP4Linear requires in_features % 32 == 0 and out_features % 16 == 0")
        packed, raw_scales, global_scale = quantize_nvfp4(weight)
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.low_precision_enabled = True
        self.register_buffer("weight_fp4", packed.contiguous())
        self.register_buffer("weight_scale", nvfp4_swizzle_scales(raw_scales).contiguous())
        self.register_buffer("weight_global_scale", global_scale.to(device=weight.device))
        dense = weight.detach().clone() if keep_dense_fallback else None
        self.register_buffer("weight", dense)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self._hardware_eligible = _nvfp4_hardware_eligible(weight.device)
        self._worldfoundry_quantization_layer = True
        self.low_precision_kernel_calls = 0
        self.dense_fallback_calls = 0
        self.request_low_precision_kernel_calls = 0
        self.request_dense_fallback_calls = 0
        self._request_window_active = False
        self.last_fallback_reason: str | None = None

    def reset_request_window(self) -> None:
        """Clear request-local execution receipts while retaining lifetime totals."""

        self.request_low_precision_kernel_calls = 0
        self.request_dense_fallback_calls = 0
        self._request_window_active = True
        self.last_fallback_reason = None

    @classmethod
    def from_linear(cls, layer: nn.Linear, *, keep_dense_fallback: bool = True) -> "NVFP4Linear":
        """Copy weight/bias from a dense ``nn.Linear``; the source layer is not mutated."""
        return cls(layer.weight, layer.bias, keep_dense_fallback=keep_dense_fallback)

    def _apply(self, fn, recurse: bool = True):
        """Move device with the tree but keep packed FP4/E4M3/FP32 scales off ``to(dtype=)``."""
        # NVFP4's packed payload, E4M3 block scales, and FP32 tensor scale have
        # fixed storage dtypes.  A normal ``module.to(dtype=...)`` must only
        # change the dense fallback/bias compute dtype, not this metadata.
        weight_fp4 = self.weight_fp4
        weight_scale = self.weight_scale
        weight_global_scale = self.weight_global_scale
        result = super()._apply(fn, recurse=recurse)
        target_device = self.weight_fp4.device
        self.weight_fp4 = weight_fp4.to(device=target_device)
        self.weight_scale = weight_scale.to(device=target_device)
        self.weight_global_scale = weight_global_scale.to(device=target_device)
        self._hardware_eligible = _nvfp4_hardware_eligible(self.weight_fp4.device)
        return result

    def _dense(self, input: torch.Tensor, reason: str) -> torch.Tensor:
        """Exact ``F.linear`` on the retained dense copy; fatal if that copy was dropped."""
        if self.weight is None:
            raise RuntimeError(
                "NVFP4 is unavailable and no dense fallback was retained: " + reason
            )
        if not _is_compiling():
            self.dense_fallback_calls += 1
            self.request_dense_fallback_calls += 1
            self.last_fallback_reason = reason
        bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
        return F.linear(input, self.weight.to(dtype=input.dtype), bias)

    def _fallback_reason(self, input: torch.Tensor) -> str:
        """First eligibility gate that rejected this call, for request-window logs."""
        if not self.low_precision_enabled:
            return "low-precision execution disabled by quality policy"
        if not self._hardware_eligible:
            return "device or PyTorch build does not expose an eligible NVFP4 scaled-mm kernel"
        if torch.is_grad_enabled():
            return "NVFP4 inference kernel is disabled while gradients are enabled"
        if input.device != self.weight_fp4.device:
            return "input and packed NVFP4 weight are on different devices"
        if input.dtype not in {torch.float16, torch.bfloat16}:
            return f"NVFP4 kernel requires fp16/bf16 input, got {input.dtype}"
        return "NVFP4 eligibility check rejected the call"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Blackwell scaled-mm when eligible; dense otherwise. OOM never falls back."""
        if input.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {input.shape[-1]}")
        if input.numel() == 0:
            return input.new_empty((*input.shape[:-1], self.out_features))
        eligible = (
            self.low_precision_enabled
            and self._hardware_eligible
            and not torch.is_grad_enabled()
            and input.device == self.weight_fp4.device
            and input.dtype in {torch.float16, torch.bfloat16}
        )
        if not eligible:
            return self._dense(input, self._fallback_reason(input))

        original_shape = input.shape
        flattened = input.reshape(-1, self.in_features).contiguous()
        input_global_scale = _global_decode_scale(flattened)
        try:
            from worldfoundry.core.acceleration.quantization.triton_nvfp4 import triton_quantize_nvfp4

            input_fp4, input_scale = triton_quantize_nvfp4(flattened, input_global_scale)
        except Exception as exc:
            message = str(exc).casefold()
            if "out of memory" in message or "alloc_failed" in message:
                raise
            input_fp4, input_raw_scale, input_global_scale = quantize_nvfp4(
                flattened,
                global_scale=input_global_scale,
            )
            input_scale = nvfp4_swizzle_scales(input_raw_scale).contiguous()
        fp4_dtype = torch.float4_e2m1fn_x2
        try:
            # Call the ATen operator directly: PyTorch 2.10's public Python
            # ``F.scaled_mm`` wrapper is not Dynamo-allowlisted, while the
            # underlying operator is graph-safe.  Values follow the public
            # ScalingType/SwizzleType enum contract in that release:
            # BlockWise1x16=2, TensorWise=0, SWIZZLE_32_4_4=1.
            output = torch.ops.aten._scaled_mm_v2.default(
                input_fp4.view(fp4_dtype),
                self.weight_fp4.view(fp4_dtype).t(),
                [input_scale, input_global_scale],
                [2, 0],
                [1],
                [self.weight_scale, self.weight_global_scale],
                [2, 0],
                [1],
                None,
                input.dtype,
                [],
                False,
            )
        except RuntimeError as exc:
            message = str(exc).casefold()
            if "out of memory" in message or "alloc_failed" in message:
                raise
            if torch.compiler.is_compiling():
                raise
            _FAILED_DEVICES.add(str(input.device))
            self._hardware_eligible = False
            return self._dense(input, f"NVFP4 scaled-mm runtime failure: {exc}")
        if self.bias is not None:
            output = output + self.bias.to(dtype=input.dtype)
        if not _is_compiling():
            self.low_precision_kernel_calls += 1
            self.request_low_precision_kernel_calls += 1
        return output.reshape(*original_shape[:-1], self.out_features)

    def runtime_report(self) -> dict[str, object]:
        """Request-window counters when active; lifetime totals otherwise."""
        low_precision_calls = int(
            self.request_low_precision_kernel_calls
            if self._request_window_active
            else self.low_precision_kernel_calls
        )
        dense_calls = int(
            self.request_dense_fallback_calls
            if self._request_window_active
            else self.dense_fallback_calls
        )
        if low_precision_calls and dense_calls:
            effective = "mixed-nvfp4-and-dense"
        elif low_precision_calls:
            effective = "nvfp4-scaled-mm"
        elif dense_calls:
            effective = "dense"
        else:
            effective = "pending"
        return {
            "format": "nvfp4",
            "storage": "nvfp4-1x16",
            "effective": effective,
            "hardware_eligible": bool(self._hardware_eligible),
            "low_precision_kernel_calls": low_precision_calls,
            "packed_weight_calls": 0,
            "dense_compute_calls": dense_calls,
            "dense_fallback_calls": dense_calls,
            "lifetime_low_precision_kernel_calls": int(self.low_precision_kernel_calls),
            "lifetime_dense_fallback_calls": int(self.dense_fallback_calls),
            "last_fallback_reason": self.last_fallback_reason,
            "dense_fallback_retained": self.weight is not None,
        }

    def extra_repr(self) -> str:
        """Print whether the dense fallback buffer is still present."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, dense_fallback={self.weight is not None}"
        )


# ──────────────────────────────────────────────────────────────────────────
# In-place replace — after checkpoint load; packed weights are not parameters
# ──────────────────────────────────────────────────────────────────────────


def replace_linear_with_nvfp4(
    module: nn.Module,
    *,
    min_features: int = 1024,
    keep_dense_fallback: bool = True,
    exclude: tuple[str, ...] = (),
) -> int:
    """Replace aligned child linears with :class:`NVFP4Linear` in-place.

    Apply this transform after loading the dense checkpoint and placing its
    modules.  The packed weights intentionally become inference buffers and
    are not optimizer/FSDP parameters.
    """

    replaced = 0
    for name, child in tuple(module.named_children()):
        if any(pattern and pattern in name for pattern in exclude):
            continue
        if isinstance(child, nn.Linear):
            if (
                child.in_features >= min_features
                and child.out_features >= min_features
                and child.in_features % 32 == 0
                and child.out_features % 16 == 0
            ):
                setattr(
                    module,
                    name,
                    NVFP4Linear.from_linear(child, keep_dense_fallback=keep_dense_fallback),
                )
                replaced += 1
            continue
        replaced += replace_linear_with_nvfp4(
            child,
            min_features=min_features,
            keep_dense_fallback=keep_dense_fallback,
            exclude=exclude,
        )
    return replaced


__all__ = [
    "NVFP4Linear",
    "dequantize_nvfp4",
    "nvfp4_swizzle_scales",
    "nvfp4_unswizzle_scales",
    "quantize_nvfp4",
    "replace_linear_with_nvfp4",
]
