"""Explicit projection fusion and precision schedules for resident MHA.

Checkpoint geometry, normalization, position embeddings and request ownership
belong to model adapters. This module owns only derived projections and SDPA.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from worldfoundry.core.acceleration.quantization.linear import Float8Linear, WeightOnlyLinear


@dataclass(frozen=True)
class MHASchedule:
    fusion: str = "none"
    projection_precision: str = "native"
    output_precision: str = "native"
    sdpa_backend: str = "torch"
    quantized_sdpa: bool = False
    use_tma: bool = False

    def __post_init__(self):
        if self.fusion not in {"none", "kv", "qkv"}:
            raise ValueError("MHA fusion must be none, kv or qkv")
        allowed = {"native", "fp8_e4m3", "fp8_e5m2", "int8"}
        if self.projection_precision not in allowed or self.output_precision not in allowed:
            raise ValueError("invalid MHA projection precision")
        if self.sdpa_backend not in {"torch", "cudnn", "fa2"}:
            raise ValueError("MHA SDPA backend must be torch, cudnn or fa2")
        if type(self.quantized_sdpa) is not bool or type(self.use_tma) is not bool:
            raise TypeError("MHA quantized_sdpa and use_tma must be booleans")
        if self.quantized_sdpa and self.sdpa_backend == "torch":
            raise ValueError("FP8 SDPA requires the explicit cudnn or fa2 provider")
        if self.use_tma and (self.sdpa_backend != "fa2" or self.quantized_sdpa):
            raise ValueError("TMA schedule requires native-precision fa2; FP8 uses pointer FA2 or cuDNN")

    @property
    def approximate(self):
        return self.quantized_sdpa or self.projection_precision != "native" or self.output_precision != "native"


class ScheduledProjection:
    """Nonpersistent execution weights, retaining canonical source modules."""

    def __init__(self, sources, precision):
        self.sources = tuple(sources)
        self.precision = precision
        self.widths = tuple(module.out_features for module in sources)
        self._versions = self._fingerprint()
        self.quantized = None
        self.calls = 0
        if len(sources) > 1:
            self.weight = torch.cat([source.weight.detach() for source in sources]).contiguous()
            self.bias = torch.cat([source.bias.detach() for source in sources]) if sources[0].bias is not None else None
        else:
            self.weight, self.bias = sources[0].weight.detach(), sources[0].bias
        if precision.startswith("fp8"):
            dtype = torch.float8_e4m3fn if precision == "fp8_e4m3" else torch.float8_e5m2
            from worldfoundry.core.acceleration.quantization.fp8_linear import TritonFloat8Linear

            provider = Float8Linear if precision == "fp8_e4m3" else TritonFloat8Linear
            self.quantized = provider(
                self.weight,
                self.bias,
                fp8_dtype=dtype,
                scaling="rowwise",
                use_fast_accum=False,
                keep_dense_fallback=False,
            )
            self.quantized.weight = self.weight
        elif precision == "int8":
            self.quantized = WeightOnlyLinear(
                self.weight, self.bias, bits=8, group_size=0, keep_dense_fallback=False, use_kernel=True
            )

    def _fingerprint(self):
        return tuple(
            (id(v), v.data_ptr(), v.dtype, v.device, None if v.is_inference() else v._version)
            for module in self.sources
            for v in module.parameters(recurse=False)
        )

    def __call__(self, x):
        if self._versions != self._fingerprint():
            raise RuntimeError("MHA source projections changed; uninstall and reinstall the schedule")
        if self.quantized is not None:
            output = self.quantized(x)
        elif len(self.sources) == 1:
            output = self.sources[0](x)
        else:
            output = F.linear(x, self.weight, self.bias)
        self.calls += 1
        return output.split(self.widths, dim=-1)

    def reset_request_window(self):
        self.calls = 0
        if self.quantized is not None:
            self.quantized.reset_request_window()

    def report(self):
        return {
            "calls": self.calls,
            "fusion_width": len(self.sources),
            "precision": self.precision,
            "quantization": None if self.quantized is None else self.quantized.runtime_report(),
        }


def scheduled_sdpa(q, k, v, *, num_heads, schedule):
    """Projected BLC tensors; no mask, state, implicit provider or silent fallback."""
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        raise RuntimeError("scheduled MHA requires eager no-grad inference")
    if q.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("scheduled MHA CUDA Graph capture is unvalidated")
    if type(num_heads) is not int or num_heads <= 0:
        raise ValueError("MHA requires a positive integer head count")
    if any(tensor.ndim != 3 or not tensor.numel() for tensor in (q, k, v)):
        raise ValueError("MHA requires nonempty projected BLC tensors")
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[2] % num_heads:
        raise ValueError("MHA projection geometry or head count does not match")
    if any(tensor.device != q.device or tensor.dtype != q.dtype for tensor in (k, v)):
        raise ValueError("MHA projections must share device and dtype")
    if q.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError("MHA requires floating-point projections")
    original_dtype = q.dtype
    projected = [tensor.reshape(tensor.shape[0], tensor.shape[1], num_heads, -1) for tensor in (q, k, v)]
    if schedule.quantized_sdpa:
        # Explicit upstream precision contract: raw FP8 Q/K/V, no calibration.
        projected = [tensor.to(torch.float8_e4m3fn) for tensor in projected]
    if schedule.sdpa_backend == "fa2":
        if schedule.use_tma:
            from .backends.triton_tma import triton_tma_sdpa

            result = triton_tma_sdpa(*projected)
        else:
            from .backends.triton_fa2 import flash_attention_2

            result = flash_attention_2(*projected, output_dtype=original_dtype)
    else:
        q, k, v = [tensor.transpose(1, 2) for tensor in projected]
        if schedule.quantized_sdpa:
            from .backends.native_fp8 import native_cudnn_fp8_sdpa

            result = native_cudnn_fp8_sdpa(q, k, v).to(original_dtype)
        elif schedule.sdpa_backend == "cudnn":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                result = F.scaled_dot_product_attention(q, k, v)
        else:
            result = F.scaled_dot_product_attention(q, k, v)
        result = result.transpose(1, 2)
    return result.reshape(result.shape[0], result.shape[1], -1)
