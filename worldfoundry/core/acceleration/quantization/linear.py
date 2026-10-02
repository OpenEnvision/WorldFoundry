"""In-tree low-precision Linear layers with portable dense fallback.

Responsibility: replace ``nn.Linear`` with FP8 or packed INT8/INT4
modules that keep a dense fallback so the same checkpoint runs on
GPUs without a scaled-mm kernel. The master switch is
:func:`set_low_precision_enabled` so a quality gate can disable every
approximation at once.

This module is not a trainer, an optimizer, or an NVFP4 implementation
(that lives in ``nvfp4.py``). Packed weights are inference buffers, not
FSDP/optimizer parameters. Counters are skipped under
``torch.compile`` so Dynamo does not treat them as graph outputs.

Public surface:
- :class:`Float8Linear` / :class:`WeightOnlyLinear`
- :func:`replace_linear_with_float8` / :func:`replace_linear_with_weight_only`
- :func:`set_low_precision_enabled`
- :func:`quantization_runtime_report` / :func:`reset_quantization_runtime_window`

FP8 replacements via :func:`replace_linear_with_float8`.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry.core.kernels.capabilities import kernel_device_profile

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
# H100 measurements show that dynamic activation quantization dominates small
# GEMMs (the 1024x2048x2048 profile is slower than BF16). ``M*K*N`` above five
# billion reliably selects the measured profitable shapes while retaining an
# environment override for machine-specific calibration.
_FP8_MIN_GEMM_WORK = 5.0e9
# The in-tree dynamic W8A8 path wins on the measured Wan SP4 projections
# (M=27280, K>=3072, N>=3072), but loses badly on 8192x1024x1024 because
# activation quantization dominates.  Keep a conservative default and expose
# an override so each deployment can calibrate its own accelerator/toolchain.
_INT8_MIN_GEMM_WORK = 1.0e11


# ──────────────────────────────────────────────────────────────────────────
# Work gates — small GEMMs lose to activation quant; env overrides are once-only
# ──────────────────────────────────────────────────────────────────────────


def _fp8_min_gemm_work() -> float:
    """Return the process-local FP8 work gate, reading the environment once."""

    cached = getattr(_fp8_min_gemm_work, "_cached", None)
    if cached is not None:
        return cached
    raw = os.environ.get("WORLDFOUNDRY_FP8_MIN_GEMM_FLOP")
    try:
        value = _FP8_MIN_GEMM_WORK if raw is None else float(raw)
    except ValueError as error:
        raise ValueError("WORLDFOUNDRY_FP8_MIN_GEMM_FLOP must be a non-negative number") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError("WORLDFOUNDRY_FP8_MIN_GEMM_FLOP must be a non-negative finite number")
    _fp8_min_gemm_work._cached = value
    return value


def _int8_min_gemm_work() -> float:
    """Return the calibrated dynamic-W8A8 work gate for this process."""

    cached = getattr(_int8_min_gemm_work, "_cached", None)
    if cached is not None:
        return cached
    raw = os.environ.get("WORLDFOUNDRY_INT8_MIN_GEMM_FLOP")
    try:
        value = _INT8_MIN_GEMM_WORK if raw is None else float(raw)
    except ValueError as error:
        raise ValueError(
            "WORLDFOUNDRY_INT8_MIN_GEMM_FLOP must be a non-negative number"
        ) from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            "WORLDFOUNDRY_INT8_MIN_GEMM_FLOP must be a non-negative finite number"
        )
    _int8_min_gemm_work._cached = value
    return value


def _int8_linear_profitable(
    input: torch.Tensor,
    out_features: int,
    *,
    min_gemm_work: float | None = None,
) -> bool:
    """True when ``M*K*N`` meets the calibrated INT8 floor (0 disables the gate)."""
    threshold = (
        _int8_min_gemm_work() if min_gemm_work is None else float(min_gemm_work)
    )
    return threshold <= 0 or input.numel() * int(out_features) >= threshold


def _scaled_mm_modern() -> bool:
    """Return whether this PyTorch exposes the scale-tensor ``_scaled_mm`` API."""

    scaled_mm = getattr(torch, "_scaled_mm", None)
    if not callable(scaled_mm):
        return False
    version = torch.__version__.split("+", 1)[0].split(".")
    try:
        major, minor = int(version[0]), int(version[1])
    except (IndexError, ValueError):
        return False
    return (major, minor) >= (2, 5)


# ──────────────────────────────────────────────────────────────────────────
# FP8 quantizers — tensorwise vs rowwise; Triton is optional, Torch is truth
# ──────────────────────────────────────────────────────────────────────────


def _scaled_mm_supports_rowwise() -> bool:
    """Return whether row-wise scale tensors are supported by ``_scaled_mm``.

    Row-wise scaling was added with the modern PyTorch 2.5 API.  Hardware
    eligibility is checked separately for the tensor's actual CUDA device.
    """

    return _scaled_mm_modern()


def _quantize_tensorwise_fp8(
    value: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One FP32 decode scale for the whole tensor; used when rowwise ``_scaled_mm`` is absent."""
    if dtype not in _FP8_DTYPES:
        raise ValueError(f"unsupported FP8 dtype: {dtype}")
    fp8_max = float(torch.finfo(dtype).max)
    scale = (value.detach().float().abs().amax() / fp8_max).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (value.float() / scale).clamp(-fp8_max, fp8_max).to(dtype)
    return quantized, scale.reshape(1).float()


def _quantize_rowwise_fp8(
    value: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D tensor with one FP32 decode scale per row."""

    if dtype not in _FP8_DTYPES:
        raise ValueError(f"unsupported FP8 dtype: {dtype}")
    if value.ndim != 2:
        raise ValueError("row-wise FP8 quantization expects a 2D tensor")
    if value.device.type == "cuda" and value.numel():
        try:
            from worldfoundry.core.acceleration.quantization.triton_fp8 import quantize_rowwise_fp8_triton

            return quantize_rowwise_fp8_triton(value, dtype)
        except (ImportError, RuntimeError, ValueError):
            # Triton is an acceleration provider, not a correctness dependency.
            # Unsupported shapes/toolchains retain the portable Torch path.
            pass
    fp8_max = float(torch.finfo(dtype).max)
    scale = (
        value.detach().float().abs().amax(dim=1, keepdim=True) / fp8_max
    ).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (value.float() / scale).clamp(-fp8_max, fp8_max).to(dtype)
    return quantized, scale.float()


def _fp8_hardware_eligible(device: torch.device) -> bool:
    """SM + ``_scaled_mm`` probe; does not mean this GEMM shape will take the kernel."""
    if device.type != "cuda":
        return False
    profile = kernel_device_profile(device)
    return profile.supports_fp8 and callable(getattr(torch, "_scaled_mm", None))


def _fp8_linear_eligible(
    input: torch.Tensor,
    out_features: int,
    hardware_eligible: bool,
    *,
    min_gemm_work: float | None = None,
) -> bool:
    """Inference-only: CUDA, fp16/bf16, 16-aligned widths, and enough GEMM work."""
    if input.device.type != "cuda" or torch.is_grad_enabled():
        return False
    if input.dtype not in {torch.float16, torch.bfloat16}:
        return False
    if input.shape[-1] % 16 or out_features % 16:
        return False
    threshold = _fp8_min_gemm_work() if min_gemm_work is None else float(min_gemm_work)
    if threshold > 0 and input.numel() * out_features < threshold:
        return False
    return hardware_eligible


def _is_compiling() -> bool:
    """Skip counter updates inside Dynamo so telemetry tensors do not become graph outputs."""
    compiler = getattr(torch, "compiler", None)
    predicate = getattr(compiler, "is_compiling", None)
    return bool(predicate()) if callable(predicate) else False


def _fp8_fallback_reason(
    input: torch.Tensor,
    out_features: int,
    hardware_eligible: bool,
    *,
    enabled: bool,
    min_gemm_work: float | None = None,
) -> str:
    """Explain the first FP8 eligibility gate that rejected one call."""

    if not enabled:
        return "low-precision execution disabled by quality policy"
    if input.device.type != "cuda":
        return f"FP8 kernel requires CUDA, got {input.device.type}"
    if torch.is_grad_enabled():
        return "FP8 inference kernel is disabled while gradients are enabled"
    if input.dtype not in {torch.float16, torch.bfloat16}:
        return f"FP8 kernel requires fp16/bf16 input, got {input.dtype}"
    if input.shape[-1] % 16 or out_features % 16:
        return "FP8 kernel requires input/output widths divisible by 16"
    if not hardware_eligible:
        return "device or PyTorch build does not expose an eligible FP8 scaled-mm kernel"
    threshold = _fp8_min_gemm_work() if min_gemm_work is None else float(min_gemm_work)
    if threshold > 0 and input.numel() * out_features < threshold:
        return f"GEMM work is below the calibrated FP8 threshold ({threshold:g})"
    return "FP8 eligibility check rejected the call"


def _autocast_linear_input(input: torch.Tensor) -> torch.Tensor:
    """Mirror ``nn.Linear`` input casting for this custom FP8 module.

    CUDA autocast recognizes built-in linear operators, but it does not cast a
    custom module's arguments before ``forward``. Wan feeds FP32 conditioner
    tensors inside a BF16 autocast region, so explicitly select that region's
    dtype before checking FP8 eligibility.
    """

    if input.dtype != torch.float32 or input.device.type not in {"cuda", "cpu"}:
        return input
    try:
        enabled = torch.is_autocast_enabled(input.device.type)
    except TypeError:  # PyTorch < 2.4 compatibility
        enabled = torch.is_autocast_enabled()
    if not enabled:
        return input
    try:
        dtype = torch.get_autocast_dtype(input.device.type)
    except AttributeError:  # PyTorch < 2.4 compatibility
        dtype = (
            torch.get_autocast_gpu_dtype()
            if input.device.type == "cuda"
            else torch.get_autocast_cpu_dtype()
        )
    if dtype not in {torch.float16, torch.bfloat16}:
        return input
    return input.to(dtype=dtype)


# ──────────────────────────────────────────────────────────────────────────
# Float8Linear — store FP8 weights; dense copy is the portability contract
# ──────────────────────────────────────────────────────────────────────────


class Float8Linear(nn.Module):
    """Inference-only FP8 linear with an exact dense fallback.

    Quantized weights are stored in row-major ``[out, in]`` form and viewed as
    the column-major right GEMM operand at execution. Dynamic activation
    quantization happens on-device. A dense weight copy is retained by default
    so the same checkpoint remains runnable on V100, T4, A100, ROCm and CPU.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        keep_dense_fallback: bool = True,
        scaling: str = "auto",
        use_fast_accum: bool = True,
    ) -> None:
        """Quantize a 2-D ``[out, in]`` weight; ``auto`` scaling prefers rowwise on PyTorch ≥ 2.5."""
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("weight must have [out_features, in_features] shape")
        if fp8_dtype not in _FP8_DTYPES:
            raise ValueError(f"unsupported FP8 dtype: {fp8_dtype}")
        requested_scaling = str(scaling).lower()
        if requested_scaling not in {"auto", "rowwise", "tensorwise"}:
            raise ValueError(f"unsupported FP8 scaling mode: {scaling!r}")
        resolved_scaling = (
            "rowwise"
            if requested_scaling == "rowwise"
            or (requested_scaling == "auto" and _scaled_mm_supports_rowwise())
            else "tensorwise"
        )
        if resolved_scaling == "rowwise":
            quantized, scale = _quantize_rowwise_fp8(weight.detach(), fp8_dtype)
            # _scaled_mm consumes B as [K,N] and its row-wise B scale as [1,N].
            scale = scale.transpose(0, 1).contiguous()
        else:
            quantized, scale = _quantize_tensorwise_fp8(weight.detach(), fp8_dtype)
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.fp8_dtype = fp8_dtype
        self.scaling = resolved_scaling
        self.use_fast_accum = bool(use_fast_accum)
        self.low_precision_enabled = True
        self.register_buffer("weight_fp8", quantized.contiguous())
        self.register_buffer("weight_scale", scale.to(device=weight.device))
        dense = weight.detach().clone() if keep_dense_fallback else None
        self.register_buffer("weight", dense)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self._hardware_eligible = _fp8_hardware_eligible(weight.device)
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
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        keep_dense_fallback: bool = True,
        scaling: str = "auto",
        use_fast_accum: bool = True,
    ) -> "Float8Linear":
        """Copy weight/bias from a dense ``nn.Linear``; the source layer is not mutated."""
        return cls(
            layer.weight,
            layer.bias,
            fp8_dtype=fp8_dtype,
            keep_dense_fallback=keep_dense_fallback,
            scaling=scaling,
            use_fast_accum=use_fast_accum,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """FP8 scaled-mm when eligible; dense ``F.linear`` otherwise (fatal if no fallback)."""
        if input.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {input.shape[-1]}")
        if input.numel() == 0:
            return input.new_empty((*input.shape[:-1], self.out_features))
        input = _autocast_linear_input(input)
        # Without a retained dense payload the FP8 path is the only runnable
        # representation, so the performance gate must not turn into a runtime
        # failure. Hardware/dtype/alignment checks still apply.
        min_gemm_work = 0.0 if self.weight is None else None
        eligible = self.low_precision_enabled and _fp8_linear_eligible(
            input,
            self.out_features,
            self._hardware_eligible,
            min_gemm_work=min_gemm_work,
        )
        if not eligible:
            reason = _fp8_fallback_reason(
                input,
                self.out_features,
                self._hardware_eligible,
                enabled=self.low_precision_enabled,
                min_gemm_work=min_gemm_work,
            )
            if self.weight is None:
                raise RuntimeError(
                    "FP8 is unavailable for this workload and no dense fallback was retained "
                    f"(device={input.device}, dtype={input.dtype}, grad={torch.is_grad_enabled()}, "
                    f"in={self.in_features}, out={self.out_features}, "
                    f"hardware_eligible={self._hardware_eligible})"
                )
            if not _is_compiling():
                self.dense_fallback_calls += 1
                self.request_dense_fallback_calls += 1
                self.last_fallback_reason = reason
            bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
            return F.linear(input, self.weight.to(dtype=input.dtype), bias)

        original_shape = input.shape
        flattened = input.reshape(-1, self.in_features).contiguous()
        if self.scaling == "rowwise":
            input_fp8, input_scale = _quantize_rowwise_fp8(flattened, self.fp8_dtype)
        else:
            input_fp8, input_scale = _quantize_tensorwise_fp8(flattened, self.fp8_dtype)
        # A contiguous [N, K] weight transposes to the column-major [K, N]
        # layout required by torch._scaled_mm/cuBLASLt without another copy.
        weight_mat = self.weight_fp8.t()
        bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
        output = torch._scaled_mm(
            input_fp8,
            weight_mat,
            input_scale,
            self.weight_scale,
            bias=bias,
            out_dtype=input.dtype,
            use_fast_accum=self.use_fast_accum,
        )
        if isinstance(output, tuple):
            output = output[0]
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
            effective = "mixed-fp8-and-dense"
        elif low_precision_calls:
            effective = "fp8-scaled-mm"
        elif dense_calls:
            effective = "dense"
        else:
            effective = "pending"
        return {
            "format": "fp8",
            "storage": f"{self.fp8_dtype}",
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

    def _apply(self, fn, recurse: bool = True):
        """Move device with the tree but keep FP8 codes and FP32 scales off ``to(dtype=)``."""
        # ``Module.to(dtype=...)`` applies its dtype conversion to every
        # floating buffer.  FP8 payloads and their FP32 decode scale are
        # storage-format metadata, not model-compute tensors, so preserve the
        # original values/dtypes while still following the requested device.
        weight_fp8 = self.weight_fp8
        weight_scale = self.weight_scale
        result = super()._apply(fn, recurse=recurse)
        target_device = self.weight_fp8.device
        self.weight_fp8 = weight_fp8.to(device=target_device)
        self.weight_scale = weight_scale.to(device=target_device)
        self._hardware_eligible = _fp8_hardware_eligible(self.weight_fp8.device)
        return result

    def extra_repr(self) -> str:
        """Print scaling and whether the dense fallback buffer is still present."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, fp8_dtype={self.fp8_dtype}, "
            f"scaling={self.scaling}, fast_accum={self.use_fast_accum}, "
            f"dense_fallback={self.weight is not None}"
        )


# ──────────────────────────────────────────────────────────────────────────
# In-place replace — after checkpoint load; packed weights are not parameters
# ──────────────────────────────────────────────────────────────────────────


def replace_linear_with_float8(
    module: nn.Module,
    *,
    min_features: int = 1024,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    keep_dense_fallback: bool = True,
    exclude: tuple[str, ...] = (),
    scaling: str = "auto",
    use_fast_accum: bool = True,
    _prefix: str = "",
) -> int:
    """Replace eligible child ``nn.Linear`` modules in-place.

    Apply it after loading the dense checkpoint and placing its modules.  This
    explicit inference transform turns quantized weights into buffers, avoids
    hidden runtime monkey-patching, and returns the number of replacements.
    """

    replaced = 0
    for name, child in tuple(module.named_children()):
        qualified_name = f"{_prefix}.{name}" if _prefix else name
        if any(pattern and pattern in qualified_name for pattern in exclude):
            continue
        if isinstance(child, nn.Linear):
            if (
                child.in_features >= min_features
                and child.out_features >= min_features
                and child.in_features % 16 == 0
                and child.out_features % 16 == 0
            ):
                setattr(
                    module,
                    name,
                    Float8Linear.from_linear(
                        child,
                        fp8_dtype=fp8_dtype,
                        keep_dense_fallback=keep_dense_fallback,
                        scaling=scaling,
                        use_fast_accum=use_fast_accum,
                    ),
                )
                replaced += 1
            continue
        replaced += replace_linear_with_float8(
            child,
            min_features=min_features,
            fp8_dtype=fp8_dtype,
            keep_dense_fallback=keep_dense_fallback,
            exclude=exclude,
            scaling=scaling,
            use_fast_accum=use_fast_accum,
            _prefix=qualified_name,
        )
    return replaced


# ──────────────────────────────────────────────────────────────────────────
# INT4 nibble pack — two signed codes per uint8; +8 bias matches unpack
# ──────────────────────────────────────────────────────────────────────────


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack paired INT4 codes into uint8; ``+8`` maps the signed range onto a nibble."""
    if values.dtype != torch.int8 or values.shape[-1] % 2:
        raise ValueError("INT4 codes must be int8 with an even final dimension")
    encoded = (values + 8).to(torch.uint8)
    return (encoded[..., 0::2] | (encoded[..., 1::2] << 4)).contiguous()


def _unpack_int4(values: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_pack_int4`; low nibble is the even index."""
    if values.dtype != torch.uint8:
        raise TypeError("packed INT4 storage must use uint8")
    low = (values & 0xF).to(torch.int8) - 8
    high = (values >> 4).to(torch.int8) - 8
    return torch.stack((low, high), dim=-1).flatten(start_dim=-2)


# ──────────────────────────────────────────────────────────────────────────
# Weight-only INT8/INT4 — W8A8 kernel when per-channel; else dequant + dense
# ──────────────────────────────────────────────────────────────────────────


class WeightOnlyLinear(nn.Module):
    """INT8/INT4 weight-only Linear with an optional W8A8 Triton path.

    INT8 per-output-channel weights can execute through the same dynamic
    per-token W8A8 Triton kernel family used by LightX2V. Groupwise INT8 and
    INT4 remain portable compressed formats: they dequantize one layer into
    the input compute dtype before a dense ``linear``. Runtime telemetry keeps
    those cases separate so compressed storage never masquerades as a low-bit
    GEMM.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        bits: int,
        group_size: int = 0,
        keep_dense_fallback: bool = False,
        use_kernel: bool = True,
    ) -> None:
        """Pack INT8/INT4 weights; ``group_size=0`` is the per-channel W8A8 layout."""
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("weight must have [out_features, in_features] shape")
        if bits not in {4, 8}:
            raise ValueError("weight-only quantization supports only 4 or 8 bits")
        if int(group_size) < 0:
            raise ValueError("group_size must be non-negative")
        self.bits = int(bits)
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        # group_size=0 means one scale per output channel. That layout is the
        # one consumed directly by the W8A8 kernel; explicit positive group
        # sizes preserve the portable groupwise storage contract.
        self.group_size = int(group_size) or self.in_features
        self.per_channel = int(group_size) == 0 or self.group_size >= self.in_features
        self.use_kernel = bool(use_kernel)
        self.padded_in_features = int(
            math.ceil(self.in_features / self.group_size) * self.group_size
        )
        if self.bits == 4 and self.padded_in_features % 2:
            self.padded_in_features += self.group_size

        source = weight.detach().float()
        if self.padded_in_features != self.in_features:
            source = F.pad(source, (0, self.padded_in_features - self.in_features))
        groups = source.reshape(self.out_features, -1, self.group_size)
        qmax = 127 if self.bits == 8 else 7
        scale = (groups.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        quantized = torch.round(groups / scale).clamp(-qmax, qmax).to(torch.int8)
        quantized = quantized.reshape(self.out_features, self.padded_in_features)
        packed = quantized.contiguous() if self.bits == 8 else _pack_int4(quantized)

        self.low_precision_enabled = True
        self._worldfoundry_quantization_layer = True
        self.register_buffer("qweight", packed)
        self.register_buffer("weight_scale", scale.squeeze(-1).contiguous())
        dense = weight.detach().clone() if keep_dense_fallback else None
        self.register_buffer("weight", dense)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.low_precision_kernel_calls = 0
        self.packed_weight_calls = 0
        self.dense_policy_calls = 0
        self.dense_fallback_calls = 0
        self.request_low_precision_kernel_calls = 0
        self.request_packed_weight_calls = 0
        self.request_dense_policy_calls = 0
        self.request_dense_fallback_calls = 0
        self._request_window_active = False
        self.last_kernel_provider: str | None = None
        self.last_policy_reason: str | None = None
        self.last_fallback_reason: str | None = None

    def reset_request_window(self) -> None:
        """Clear request-local execution receipts while retaining lifetime totals."""

        self.request_low_precision_kernel_calls = 0
        self.request_packed_weight_calls = 0
        self.request_dense_policy_calls = 0
        self.request_dense_fallback_calls = 0
        self._request_window_active = True
        self.last_kernel_provider = None
        self.last_policy_reason = None
        self.last_fallback_reason = None

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        bits: int,
        group_size: int = 0,
        keep_dense_fallback: bool = False,
        use_kernel: bool = True,
    ) -> "WeightOnlyLinear":
        """Copy weight/bias from a dense ``nn.Linear``; the source layer is not mutated."""
        return cls(
            layer.weight,
            layer.bias,
            bits=bits,
            group_size=group_size,
            keep_dense_fallback=keep_dense_fallback,
            use_kernel=use_kernel,
        )

    def _dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        """Expand packed codes to the input compute dtype, dropping group padding."""
        codes = self.qweight if self.bits == 8 else _unpack_int4(self.qweight)
        groups = codes.reshape(self.out_features, -1, self.group_size).to(torch.float32)
        weight = groups * self.weight_scale.unsqueeze(-1)
        return weight.reshape(self.out_features, self.padded_in_features)[
            :, : self.in_features
        ].to(dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """W8A8 Triton when eligible; else dequant+dense, or retained dense under policy."""
        if input.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {input.shape[-1]}")
        if input.numel() == 0:
            return input.new_empty((*input.shape[:-1], self.out_features))
        input = _autocast_linear_input(input)
        if not self.low_precision_enabled:
            if self.weight is None:
                raise RuntimeError(
                    f"INT{self.bits} execution was disabled but no dense fallback was retained"
                )
            if not _is_compiling():
                self.dense_fallback_calls += 1
                self.request_dense_fallback_calls += 1
                self.last_fallback_reason = "low-precision execution disabled by quality policy"
            bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
            return F.linear(input, self.weight.to(dtype=input.dtype), bias)

        kernel_reason = self._int8_kernel_fallback_reason(input)
        if (
            kernel_reason is None
            and self.weight is not None
            and not _int8_linear_profitable(input, self.out_features)
        ):
            reason = (
                "GEMM work is below the calibrated INT8 threshold "
                f"({_int8_min_gemm_work():g})"
            )
            if not _is_compiling():
                self.dense_policy_calls += 1
                self.request_dense_policy_calls += 1
                self.last_policy_reason = reason
            bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
            return F.linear(input, self.weight.to(dtype=input.dtype), bias)
        if kernel_reason is None:
            try:
                from worldfoundry.core.kernels.quantized_gemm import (
                    int8_gemm_bias_triton,
                    int8_gemm_triton,
                    int8_quantize_triton,
                )

                try:
                    from vllm import _custom_ops as vllm_ops

                    quantized_input, input_scale, _ = vllm_ops.scaled_int8_quant(
                        input.contiguous(),
                        scale=None,
                        azp=None,
                        symmetric=True,
                    )
                    provider = "vllm-activation+triton-gemm"
                except (ImportError, RuntimeError, AttributeError):
                    quantized_input, input_scale = int8_quantize_triton(
                        input.contiguous()
                    )
                    provider = "triton-activation+triton-gemm"
                weight_scale = self.weight_scale.reshape(self.out_features).float()
                if self.bias is None:
                    output = int8_gemm_triton(
                        quantized_input,
                        self.qweight,
                        input_scale.float(),
                        weight_scale,
                        output_dtype=input.dtype,
                    )
                else:
                    output = int8_gemm_bias_triton(
                        quantized_input,
                        self.qweight,
                        self.bias.float(),
                        input_scale.float(),
                        weight_scale,
                        output_dtype=input.dtype,
                    )
                if not _is_compiling():
                    self.low_precision_kernel_calls += 1
                    self.request_low_precision_kernel_calls += 1
                    self.last_kernel_provider = provider
                    self.last_fallback_reason = None
                return output
            except (ImportError, RuntimeError, AssertionError, ValueError) as error:
                kernel_reason = f"INT8 Triton kernel failed: {type(error).__name__}: {error}"

        if not _is_compiling():
            self.packed_weight_calls += 1
            self.request_packed_weight_calls += 1
            self.last_fallback_reason = kernel_reason
        bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
        return F.linear(input, self._dequantized_weight(input.dtype), bias)

    def _int8_kernel_fallback_reason(self, input: torch.Tensor) -> str | None:
        """First policy gate that rejects W8A8; ``None`` means try the Triton path."""
        if self.bits != 8:
            return "INT4 has no qualified low-bit GEMM kernel"
        if not self.use_kernel:
            return "INT8 kernel execution was disabled by policy"
        if not self.per_channel or int(self.weight_scale.shape[1]) != 1:
            return "INT8 W8A8 kernel requires per-output-channel weight scales"
        if input.device.type != "cuda":
            return f"INT8 W8A8 kernel requires CUDA, got {input.device.type}"
        if torch.is_grad_enabled():
            return "INT8 W8A8 kernel is disabled while gradients are enabled"
        if input.dtype not in {torch.float16, torch.bfloat16}:
            return f"INT8 W8A8 kernel requires fp16/bf16 input, got {input.dtype}"
        if self.in_features % 16 or self.out_features % 16:
            return "INT8 W8A8 kernel requires input/output widths divisible by 16"
        profile = kernel_device_profile(input.device)
        if not profile.supports_triton:
            return "device is not qualified for the in-tree Triton INT8 kernel"
        return None

    def _apply(self, fn, recurse: bool = True):
        """Move device with the tree but keep packed codes and scales off ``to(dtype=)``."""
        qweight = self.qweight
        scale = self.weight_scale
        result = super()._apply(fn, recurse=recurse)
        target_device = self.qweight.device
        self.qweight = qweight.to(device=target_device)
        self.weight_scale = scale.to(device=target_device)
        return result

    def runtime_report(self) -> dict[str, object]:
        """Distinguish kernel / dequant / calibrated-dense / policy fallback for logs."""
        if self._request_window_active:
            low_precision_calls = int(self.request_low_precision_kernel_calls)
            packed_calls = int(self.request_packed_weight_calls)
            policy_calls = int(self.request_dense_policy_calls)
            fallback_calls = int(self.request_dense_fallback_calls)
        else:
            low_precision_calls = int(self.low_precision_kernel_calls)
            packed_calls = int(self.packed_weight_calls)
            policy_calls = int(self.dense_policy_calls)
            fallback_calls = int(self.dense_fallback_calls)
        if low_precision_calls and (packed_calls or fallback_calls):
            effective = f"mixed-int{self.bits}-kernel-and-dense"
        elif low_precision_calls and policy_calls:
            effective = "int8-triton-w8a8+calibrated-dense"
        elif low_precision_calls:
            effective = "int8-triton-w8a8"
        elif packed_calls and fallback_calls:
            effective = f"mixed-int{self.bits}-weight-only-and-dense"
        elif packed_calls:
            effective = f"int{self.bits}-weight-only-dequantize+dense-gemm"
        elif fallback_calls:
            effective = "dense"
        elif policy_calls:
            effective = "calibrated-dense"
        else:
            effective = "pending"
        return {
            "format": f"int{self.bits}",
            "storage": (
                f"per-channel-int{self.bits}"
                if self.per_channel
                else f"groupwise-int{self.bits}"
            ),
            "group_size": self.group_size,
            "effective": effective,
            "hardware_eligible": bool(
                self.bits == 8
                and self.use_kernel
                and self.per_channel
                and kernel_device_profile(self.qweight.device).supports_triton
            ),
            "kernel_provider": self.last_kernel_provider,
            "low_precision_kernel_calls": low_precision_calls,
            "packed_weight_calls": packed_calls,
            "dense_policy_calls": policy_calls,
            "dense_compute_calls": packed_calls + policy_calls + fallback_calls,
            "dense_fallback_calls": fallback_calls,
            "lifetime_low_precision_kernel_calls": int(self.low_precision_kernel_calls),
            "lifetime_packed_weight_calls": int(self.packed_weight_calls),
            "lifetime_dense_policy_calls": int(self.dense_policy_calls),
            "lifetime_dense_fallback_calls": int(self.dense_fallback_calls),
            "last_policy_reason": self.last_policy_reason,
            "last_fallback_reason": self.last_fallback_reason,
            "dense_fallback_retained": self.weight is not None,
        }


def replace_linear_with_weight_only(
    module: nn.Module,
    *,
    bits: int,
    min_features: int = 1024,
    group_size: int = 0,
    keep_dense_fallback: bool = False,
    use_kernel: bool = True,
    exclude: tuple[str, ...] = (),
    _prefix: str = "",
) -> int:
    """Replace eligible linears with portable packed INT8/INT4 storage."""

    replaced = 0
    for name, child in tuple(module.named_children()):
        qualified_name = f"{_prefix}.{name}" if _prefix else name
        if any(pattern and pattern in qualified_name for pattern in exclude):
            continue
        if isinstance(child, nn.Linear):
            if child.in_features >= min_features and child.out_features >= min_features:
                setattr(
                    module,
                    name,
                    WeightOnlyLinear.from_linear(
                        child,
                        bits=bits,
                        group_size=group_size,
                        keep_dense_fallback=keep_dense_fallback,
                        use_kernel=use_kernel,
                    ),
                )
                replaced += 1
            continue
        replaced += replace_linear_with_weight_only(
            child,
            bits=bits,
            min_features=min_features,
            group_size=group_size,
            keep_dense_fallback=keep_dense_fallback,
            use_kernel=use_kernel,
            exclude=exclude,
            _prefix=qualified_name,
        )
    return replaced


# ──────────────────────────────────────────────────────────────────────────
# Module-wide telemetry / quality switch — walk installed quantized linears
# ──────────────────────────────────────────────────────────────────────────


def quantization_runtime_report(module: nn.Module) -> dict[str, object] | None:
    """Aggregate actual execution telemetry from installed quantized linears."""

    named_modules = getattr(module, "named_modules", None)
    if not callable(named_modules):
        return None
    layers: list[dict[str, object]] = []
    for name, child in named_modules():
        if not bool(getattr(child, "_worldfoundry_quantization_layer", False)):
            continue
        reporter = getattr(child, "runtime_report", None)
        if not callable(reporter):
            continue
        layer_report = dict(reporter())
        layer_report["module"] = name
        layers.append(layer_report)
    if not layers:
        return None

    formats = sorted({str(layer["format"]) for layer in layers})
    low_precision_calls = sum(int(layer["low_precision_kernel_calls"]) for layer in layers)
    packed_calls = sum(int(layer["packed_weight_calls"]) for layer in layers)
    policy_calls = sum(int(layer.get("dense_policy_calls") or 0) for layer in layers)
    dense_calls = sum(int(layer["dense_compute_calls"]) for layer in layers)
    fallback_calls = sum(int(layer["dense_fallback_calls"]) for layer in layers)
    lifetime_low_precision_calls = sum(
        int(layer.get("lifetime_low_precision_kernel_calls") or 0)
        for layer in layers
    )
    lifetime_packed_calls = sum(
        int(layer.get("lifetime_packed_weight_calls") or 0) for layer in layers
    )
    lifetime_policy_calls = sum(
        int(layer.get("lifetime_dense_policy_calls") or 0) for layer in layers
    )
    lifetime_fallback_calls = sum(
        int(layer.get("lifetime_dense_fallback_calls") or 0) for layer in layers
    )
    executed = low_precision_calls + packed_calls + policy_calls + fallback_calls
    if executed == 0:
        effective = "pending"
    elif low_precision_calls and (packed_calls or fallback_calls):
        effective = "mixed-low-precision-kernel-and-dense"
    elif low_precision_calls and policy_calls:
        effective = "+".join(formats) + "-kernel+calibrated-dense"
    elif low_precision_calls:
        effective = "+".join(formats) + "-kernel"
    elif packed_calls and fallback_calls:
        effective = "mixed-packed-weight-and-dense"
    elif packed_calls:
        effective = "+".join(formats) + "-weight-only-dequantize+dense-gemm"
    elif policy_calls:
        effective = "calibrated-dense"
    else:
        effective = "dense"
    fallback_reasons = sorted(
        {
            str(layer["last_fallback_reason"])
            for layer in layers
            if layer.get("last_fallback_reason")
        }
    )
    policy_reasons = sorted(
        {
            str(layer["last_policy_reason"])
            for layer in layers
            if layer.get("last_policy_reason")
        }
    )
    return {
        "formats": formats,
        "effective": effective,
        "layers": len(layers),
        "low_precision_kernel_calls": low_precision_calls,
        "packed_weight_calls": packed_calls,
        "dense_policy_calls": policy_calls,
        "dense_compute_calls": dense_calls,
        "dense_fallback_calls": fallback_calls,
        "lifetime_low_precision_kernel_calls": lifetime_low_precision_calls,
        "lifetime_packed_weight_calls": lifetime_packed_calls,
        "lifetime_dense_policy_calls": lifetime_policy_calls,
        "lifetime_dense_fallback_calls": lifetime_fallback_calls,
        "policy_reasons": policy_reasons,
        "fallback_reasons": fallback_reasons,
        "layer_reports": layers,
    }


def reset_quantization_runtime_window(module: nn.Module) -> int:
    """Begin a request-local telemetry window for installed quantized linears."""

    modules = getattr(module, "modules", None)
    if not callable(modules):
        return 0
    reset = 0
    for child in modules():
        if not bool(getattr(child, "_worldfoundry_quantization_layer", False)):
            continue
        resetter = getattr(child, "reset_request_window", None)
        if callable(resetter):
            resetter()
            reset += 1
    return reset


def set_low_precision_enabled(module: nn.Module, enabled: bool) -> int:
    """Toggle in-tree FP8/NVFP4 modules for dense boundary denoising steps."""

    updated = 0
    for child in module.modules():
        if hasattr(child, "low_precision_enabled"):
            child.low_precision_enabled = bool(enabled)
            updated += 1
    return updated


__all__ = [
    "Float8Linear",
    "WeightOnlyLinear",
    "quantization_runtime_report",
    "reset_quantization_runtime_window",
    "replace_linear_with_float8",
    "replace_linear_with_weight_only",
    "set_low_precision_enabled",
]
