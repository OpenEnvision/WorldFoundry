"""Implicit FP8 convolution versus an independent quantized convolution."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from worldfoundry.core.acceleration.quantization.fp8_conv import (
    CalibratedFP8Convolution,
    calibrate_fp8_convolution,
    validate_fp8_convolution_state,
)


def test_fp8_state_binds_geometry_and_weights():
    source = nn.Conv3d(3, 5, 3, padding=1).eval()
    state = calibrate_fp8_convolution(source, torch.ones(3))
    validate_fp8_convolution_state(source, state)
    source.stride = (2, 2, 2)
    with pytest.raises(ValueError, match="geometry"):
        validate_fp8_convolution_state(source, state)
    with pytest.raises(ValueError, match="margin"):
        calibrate_fp8_convolution(source, torch.ones(3), margin=float("nan"))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dimensions", [2, 3])
def test_fp8_implicit_gemm_matches_reference(dimensions):
    torch.manual_seed(10)
    conv = nn.Conv2d if dimensions == 2 else nn.Conv3d
    source = conv(3, 5, 3, stride=2, padding=1, device="cuda").eval()
    x = torch.randn((2, 3, 13, 17) if dimensions == 2 else (2, 3, 5, 13, 17), device="cuda")
    maxima = x.abs().amax(dim=(0, *range(2, x.ndim)))
    state = calibrate_fp8_convolution(source, maxima)
    executor = CalibratedFP8Convolution(source, state)
    with torch.no_grad():
        actual = executor(x, source.weight, source.bias)
        scale = executor.input_scale.reshape(1, -1, *([1] * dimensions))
        qx = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        operation = F.conv2d if dimensions == 2 else F.conv3d
        prior = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cudnn.allow_tf32 = False
            ref = operation(qx, executor.qweight.float(), stride=source.stride, padding=source.padding)
        finally:
            torch.backends.cudnn.allow_tf32 = prior
        ref = ref * executor.weight_scale.reshape(1, -1, *([1] * dimensions))
        ref += source.bias.reshape(1, -1, *([1] * dimensions))
    torch.testing.assert_close(actual, ref, atol=2e-5, rtol=2e-5)
    assert executor.report()["kernel_calls"] == 1
    assert executor.report()["clipped_input_operands"] == 0
    with torch.no_grad():
        executor(x * 100, source.weight, source.bias)
    assert executor.report()["clipped_input_operands"] > 0
    executor.reset_request_window()
    assert executor.report()["kernel_calls"] == 0
    with torch.no_grad():
        source.weight.add_(1)
        with pytest.raises(RuntimeError, match="parameters changed"):
            executor(x, source.weight, source.bias)
