"""Checkpoint-bound, channel-calibrated FP8 convolution execution state."""

import math

import torch
from torch import nn

from .calibration import projection_digest


def calibrate_fp8_convolution(source, activation_max, *, margin=2.0):
    validate_convolution(source)
    if not math.isfinite(margin) or margin < 1:
        raise ValueError("FP8 calibration margin must be finite and at least one")
    if activation_max.shape != (source.in_channels,) or not bool(torch.isfinite(activation_max).all()):
        raise ValueError("FP8 calibration requires finite per-input-channel maxima")
    if bool((activation_max < 0).any()):
        raise ValueError("FP8 calibration maxima cannot be negative")
    weight = source.weight.detach().float()
    input_scale = (activation_max.to(weight.device).float() * margin / 448).clamp_min(1e-8)
    adjusted = weight * input_scale.reshape(1, -1, *([1] * (weight.ndim - 2)))
    weight_scale = (adjusted.flatten(1).abs().amax(1) / 448).clamp_min(1e-8)
    quantized = (adjusted / weight_scale.reshape(-1, *([1] * (weight.ndim - 1)))).clamp(-448, 448)
    return {
        "source_digest": projection_digest(source),
        "input_scale": input_scale.cpu(),
        "weight_scale": weight_scale.cpu(),
        "qweight": quantized.to(torch.float8_e4m3fn).cpu(),
        "geometry": convolution_geometry(source),
        "margin": float(margin),
    }


def convolution_geometry(source):
    return {
        **{name: tuple(getattr(source, name)) for name in ("kernel_size", "stride", "padding", "dilation")},
        "groups": source.groups,
        "padding_mode": source.padding_mode,
        "causal_padding": tuple(getattr(source, "_padding", ())),
    }


def validate_convolution(source):
    if not isinstance(source, (nn.Conv2d, nn.Conv3d)) or source.groups != 1 or source.padding_mode != "zeros":
        raise ValueError("FP8 convolution requires ungrouped zero-padded Conv2d/Conv3d")
    if isinstance(source.padding, str) or source.weight.dtype != torch.float32:
        raise ValueError("FP8 codec convolution requires explicit padding and FP32 canonical weights")
    if not all(bool(torch.isfinite(value).all()) for value in source.parameters(recurse=False)):
        raise ValueError("FP8 convolution source parameters must be finite")


def validate_fp8_convolution_state(source, state):
    validate_convolution(source)
    keys = {"source_digest", "input_scale", "weight_scale", "qweight", "geometry", "margin"}
    if not isinstance(state, dict) or set(state) != keys:
        raise ValueError("invalid FP8 convolution calibration schema")
    if state["source_digest"] != projection_digest(source) or state["geometry"] != convolution_geometry(source):
        raise ValueError("FP8 calibration does not match source weights or convolution geometry")
    if not isinstance(state["margin"], float) or not math.isfinite(state["margin"]) or state["margin"] < 1:
        raise ValueError("invalid FP8 calibration margin")
    for name, shape, dtype in (
        ("qweight", source.weight.shape, torch.float8_e4m3fn),
        ("input_scale", (source.in_channels,), torch.float32),
        ("weight_scale", (source.out_channels,), torch.float32),
    ):
        value = state[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or value.dtype != dtype:
            raise ValueError(f"invalid FP8 convolution {name} geometry or dtype")
        if not bool(torch.isfinite(value.float()).all()):
            raise ValueError(f"nonfinite FP8 convolution {name}")
        if name.endswith("scale") and not bool((value > 0).all()):
            raise ValueError(f"FP8 convolution {name} must be positive")


class CalibratedFP8Convolution:
    """Callable for ``_conv_forward``; causal padding/cache stay in the owner."""

    def __init__(self, source, state):
        validate_fp8_convolution_state(source, state)
        self.source = source
        self.qweight = state["qweight"].to(source.weight.device).contiguous()
        self.input_scale = state["input_scale"].to(source.weight.device).contiguous()
        self.weight_scale = state["weight_scale"].to(source.weight.device).contiguous()
        self.clipped = torch.zeros((), device=source.weight.device, dtype=torch.int64)
        self.versions = self._fingerprint()
        self.geometry = convolution_geometry(source)
        self.calls = self.request_calls = 0

    def _fingerprint(self):
        return tuple(
            (id(v), v.data_ptr(), v.dtype, v.device, None if v.is_inference() else v._version)
            for v in self.source.parameters(recurse=False)
        )

    def __call__(self, input, weight, bias):
        if torch.is_grad_enabled() or torch.compiler.is_compiling() or self.source.training:
            raise RuntimeError("FP8 convolution requires eager eval no-grad inference")
        if weight is not self.source.weight or bias is not self.source.bias or self.versions != self._fingerprint():
            raise RuntimeError("FP8 source parameters changed; recalibrate and reinstall")
        if self.geometry != convolution_geometry(self.source):
            raise RuntimeError("FP8 convolution geometry changed; recalibrate and reinstall")
        if input.device.type != "cuda" or input.device != self.qweight.device or input.dtype != torch.float32:
            raise ValueError("FP8 codec convolution requires resident CUDA FP32 inputs")
        if torch.is_autocast_enabled("cuda") or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("FP8 codec autocast and CUDA Graph capture are unsupported")
        if torch.cuda.get_device_capability(input.device)[0] < 9:
            raise ValueError("FP8 codec convolution requires SM90 or newer")
        if input.ndim != weight.ndim or input.shape[1] != self.source.in_channels or not input.numel():
            raise ValueError("invalid FP8 convolution input geometry")
        from .triton_fp8_conv import fp8_convolution

        output = fp8_convolution(
            input,
            self.qweight,
            self.input_scale,
            self.weight_scale,
            bias,
            stride=self.source.stride,
            padding=self.source.padding,
            dilation=self.source.dilation,
            clipped=self.clipped,
        )
        self.calls += 1
        self.request_calls += 1
        return output

    def reset_request_window(self):
        self.request_calls = 0
        self.clipped.zero_()

    def report(self):
        return {
            "kernel_calls": self.request_calls,
            "lifetime_kernel_calls": self.calls,
            "clipped_input_operands": int(self.clipped.item()),
            "dense_fallback_calls": 0,
        }
