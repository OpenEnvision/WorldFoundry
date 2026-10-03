"""Calibrated smoothing, low-rank decomposition and real packed W4A4 execution."""

from __future__ import annotations

import torch
from torch import nn

from .calibration import projection_digest
from .linear import _autocast_linear_input


def calibrate_svdquant(source: nn.Linear, activation_max: torch.Tensor, *, rank: int = 32, seed: int = 0) -> dict:
    if type(rank) is not int or rank < 16 or rank % 16 or rank > min(source.weight.shape):
        raise ValueError("SVDQuant rank must be a positive multiple of 16 within the matrix dimensions")
    if source.in_features % 64 or source.out_features % 16:
        raise ValueError("SVDQuant requires input width divisible by 64 and output width by 16")
    if activation_max.shape != (source.in_features,) or not bool(torch.isfinite(activation_max).all()):
        raise ValueError("invalid SVDQuant calibration channel maxima")
    if bool((activation_max < 0).any()) or not bool(torch.isfinite(source.weight).all()):
        raise ValueError("SVDQuant requires finite weights and nonnegative channel maxima")
    if source.bias is not None and not bool(torch.isfinite(source.bias).all()):
        raise ValueError("SVDQuant requires finite bias")
    weight = source.weight.detach().float()
    maxima = activation_max.to(device=weight.device, dtype=torch.float32).clamp_min(1e-6)
    smooth = (maxima / weight.abs().amax(0).clamp_min(1e-6)).sqrt().clamp(1e-3, 1e3)
    scaled = weight * smooth
    # Local deterministic randomized SVD, without changing process RNG state.
    generator = torch.Generator(device=weight.device).manual_seed(seed)
    width = min(rank + 16, min(weight.shape))
    basis = scaled @ torch.randn(
        (source.in_features, width), generator=generator, device=weight.device, dtype=torch.float32
    )
    for _ in range(4):
        basis = torch.linalg.qr(basis, mode="reduced")[0]
        right = torch.linalg.qr(scaled.T @ basis, mode="reduced")[0]
        basis = scaled @ right
    basis = torch.linalg.qr(basis, mode="reduced")[0]
    u, singular, vh = torch.linalg.svd(basis.T @ scaled, full_matrices=False)
    up = (basis @ u[:, :rank]) * singular[:rank].sqrt()
    down = singular[:rank, None].sqrt() * vh[:rank]
    residual = scaled - up @ down
    grouped = residual.reshape(source.out_features, -1, 64)
    scales = (grouped.abs().amax(-1) / 7).clamp_min(1e-8)
    values = (grouped / scales[..., None]).round().clamp(-7, 7).to(torch.int8).reshape_as(weight)
    qweight = ((values[:, 0::2] & 15) | ((values[:, 1::2] & 15) << 4)).to(torch.uint8)
    return {
        "source_digest": projection_digest(source),
        "rank": rank,
        "group_size": 64,
        "smooth": smooth.cpu(),
        "qweight": qweight.cpu(),
        "scales": scales.cpu(),
        "down": (down / smooth).cpu(),
        "up": up.cpu(),
    }


def validate_svdquant_state(source, state):
    required = {"source_digest", "rank", "group_size", "smooth", "qweight", "scales", "down", "up"}
    if not isinstance(state, dict) or set(state) != required or state["group_size"] != 64:
        raise ValueError("invalid packed SVDQuant state schema")
    n, k = source.weight.shape
    rank = state["rank"]
    if type(rank) is not int or rank < 16 or rank % 16 or rank > min(n, k) or k % 64 or n % 16:
        raise ValueError("invalid SVDQuant rank or projection geometry")
    if state["source_digest"] != projection_digest(source):
        raise ValueError("SVDQuant calibration does not match source weights")
    for name, shape, dtype in (
        ("smooth", (k,), torch.float32),
        ("qweight", (n, k // 2), torch.uint8),
        ("scales", (n, k // 64), torch.float32),
        ("down", (rank, k), torch.float32),
        ("up", (n, rank), torch.float32),
    ):
        value = state[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f"invalid SVDQuant {name} shape or dtype")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite SVDQuant {name}")
        if name in {"smooth", "scales"} and not bool((value > 0).all()):
            raise ValueError(f"SVDQuant {name} must be positive")


class PackedSVDQuantLinear(nn.Module):
    """Strict inference provider; no dense-weight fallback is advertised as INT4."""

    _worldfoundry_quantization_layer = True

    def __init__(self, source: nn.Linear, state: dict):
        super().__init__()
        validate_svdquant_state(source, state)
        object.__setattr__(self, "_source", source)
        self.in_features, self.out_features = source.in_features, source.out_features
        for name in ("smooth", "qweight", "scales", "down", "up"):
            dtype = source.weight.dtype if name in {"down", "up"} else state[name].dtype
            value = state[name].to(device=source.weight.device, dtype=dtype).contiguous()
            if name in {"down", "up"} and not bool(torch.isfinite(value).all()):
                raise ValueError("SVDQuant factors overflow the source compute dtype")
            self.register_buffer(name, value)
        self.register_buffer("bias", None if source.bias is None else source.bias.detach())
        self._versions = self._fingerprint()
        self.calls = self.request_calls = 0
        self.eval()

    def _fingerprint(self):
        return tuple(
            (id(v), v.data_ptr(), v.dtype, v.device, None if v.is_inference() else v._version)
            for v in self._source.parameters()
        )

    def _apply(self, *args, **kwargs):
        raise RuntimeError("uninstall svdquant before changing placement or dtype")

    def train(self, mode=True):
        if mode:
            raise RuntimeError("uninstall svdquant before training")
        return super().train(False)

    def forward(self, input):
        if torch.is_grad_enabled() or torch.compiler.is_compiling():
            raise RuntimeError("SVDQuant requires eager no-grad inference")
        if self._versions != self._fingerprint():
            raise RuntimeError("SVDQuant source weights changed; recalibrate and reinstall")
        input = _autocast_linear_input(input)
        if input.device != self.qweight.device or input.device.type != "cuda":
            raise ValueError("packed SVDQuant requires colocated CUDA tensors")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SVDQuant CUDA Graph capture is not validated")
        if input.dtype not in {torch.bfloat16, torch.float16} or input.dtype != self.down.dtype:
            raise ValueError("packed SVDQuant requires matching FP16/BF16 input and factors")
        if input.ndim < 2 or input.shape[-1] != self.in_features or not input.numel():
            raise ValueError("invalid SVDQuant input geometry")
        if torch.cuda.get_device_capability(input.device)[0] < 8:
            raise ValueError("packed SVDQuant requires SM80 or newer")
        from .triton_svdquant import packed_svdquant

        result = packed_svdquant(input, self.smooth, self.qweight, self.scales, self.down, self.up, self.bias)
        self.calls += 1
        self.request_calls += 1
        return result

    def reset_request_window(self):
        self.request_calls = 0

    def runtime_report(self):
        return {
            "format": "svdquant-int4",
            "low_precision_kernel_calls": self.request_calls,
            "packed_weight_calls": 0,
            "dense_compute_calls": 0,
            "dense_fallback_calls": 0,
            "lifetime_low_precision_kernel_calls": self.calls,
            "last_fallback_reason": None,
            "native_packed_int4_calls": self.request_calls,
            "low_rank_gemm_calls": self.request_calls,
            "provider": "triton-int4-storage-int8-dot-fused-low-rank",
            "rank": self.down.shape[0],
        }
