# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Activation helpers retain decomposed rounding and clamped detector odds."""

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.core.nn.blocks.layers import (
    InputFP32LayerNorm,
    Mish,
    PreNormFeedForward,
    QuickGELU,
    TwoLayerReLU,
    inverse_sigmoid,
)
from worldfoundry.core.nn.diffusion.convolution import zero_module

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mish_retains_decomposed_low_precision_arithmetic(dtype: torch.dtype) -> None:
    torch.manual_seed(31)
    value = torch.randn(1000, dtype=dtype) * 4
    actual = Mish()(value)
    torch.testing.assert_close(actual, value * torch.tanh(F.softplus(value)), atol=0, rtol=0)
    assert Mish().state_dict() == {}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_quick_gelu_retains_clip_multiplication_order(dtype: torch.dtype) -> None:
    value = torch.linspace(-10, 10, 1000, dtype=dtype)
    activation = QuickGELU()
    torch.testing.assert_close(activation(value), value * torch.sigmoid(1.702 * value), rtol=0, atol=0)
    assert activation.state_dict() == {}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_inverse_sigmoid_clamps_before_forming_odds(dtype: torch.dtype) -> None:
    value = torch.tensor([-1.0, 0.0, 0.1, 0.5, 0.9, 1.0, 2.0], dtype=dtype)
    clamped = value.clamp(min=0, max=1)
    expected = torch.log(clamped.clamp(min=1e-3) / (1 - clamped).clamp(min=1e-3))
    torch.testing.assert_close(inverse_sigmoid(value), expected, rtol=0, atol=0)


def test_zero_module_preserves_buffers_and_parameter_grad_flags() -> None:
    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    module[1].running_mean.fill_(3)
    assert zero_module(module) is module
    assert all(torch.count_nonzero(p).item() == 0 for p in module.parameters())
    assert all(p.requires_grad for p in module.parameters())
    assert torch.equal(module[1].running_mean, torch.full((4,), 3.0))


def test_pre_norm_feedforward_preserves_sequential_slots_and_dropout_order() -> None:
    layer = PreNormFeedForward(8, 12, dropout=0.25)
    assert list(layer.state_dict()) == [
        "net.0.weight",
        "net.0.bias",
        "net.1.weight",
        "net.1.bias",
        "net.4.weight",
        "net.4.bias",
    ]
    assert [type(child) for child in layer.net] == [
        torch.nn.LayerNorm,
        torch.nn.Linear,
        torch.nn.GELU,
        torch.nn.Dropout,
        torch.nn.Linear,
        torch.nn.Dropout,
    ]
    x = torch.randn(2, 3, 8)
    torch.manual_seed(21)
    expected = layer.net(x)
    torch.manual_seed(21)
    torch.testing.assert_close(layer(x), expected, atol=0, rtol=0)


def test_two_layer_relu_keeps_native_parameter_names() -> None:
    layer = TwoLayerReLU(8, 12, 4)
    assert list(layer.state_dict()) == ["layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias"]
    x = torch.randn(2, 3, 8)
    torch.testing.assert_close(layer(x), layer.layer2(F.relu(layer.layer1(x))), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_input_fp32_layer_norm_keeps_native_affine_precision(dtype: torch.dtype) -> None:
    layer = InputFP32LayerNorm(8)
    x = torch.randn(2, 3, 8, dtype=dtype)
    expected = F.layer_norm(x.float(), (8,), layer.weight, layer.bias, layer.eps).type_as(x)
    torch.testing.assert_close(layer(x=x), expected, rtol=0, atol=0)
    layer.to(dtype)
    with pytest.raises(RuntimeError, match="mixed dtype"):
        layer(x)
