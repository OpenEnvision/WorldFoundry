"""Functional fusion preserves eager intermediate dtype rounding and policies."""

import os

import pytest
import torch

from worldfoundry.core.kernels import layer_norm_scale_shift, residual_gate_add, scale_shift
from worldfoundry.core.kernels.registry import kernel_dispatch_receipt_scope


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("update_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("modulation_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_eager_rounding_and_explicit_model_backend(monkeypatch, dtype, update_dtype, modulation_dtype):
    # Explicit model configuration must take precedence over process defaults.
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "torch")
    torch.manual_seed(102)
    residual = torch.randn(2, 6, 2240, dtype=dtype, device="cuda")
    update = torch.randn(residual.shape, dtype=update_dtype, device="cuda")
    scale = torch.randn(2, 1, 2240, dtype=update_dtype, device="cuda")
    shift = torch.randn(2, 1, 2240, dtype=modulation_dtype, device="cuda")
    gate = torch.randn_like(shift)
    expected_gate = residual + update * gate
    expected_affine = residual * (1 + scale) + shift
    receipt = {}
    with torch.inference_mode(), kernel_dispatch_receipt_scope(receipt):
        actual_gate = residual_gate_add(residual, update, gate, backend="triton")
        actual_affine = scale_shift(residual, scale, shift, backend="triton")
    torch.testing.assert_close(actual_gate, expected_gate, rtol=0, atol=0)
    torch.testing.assert_close(actual_affine, expected_affine, rtol=0, atol=0)
    assert len(receipt["dispatches"]) == 2
    assert all(item["accelerated"] and item["backend"] == "triton" for item in receipt["dispatches"])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape,modulation_shape", [
    ((3, 2240), (1, 2240)),
    ((2, 3, 2240), (2, 1, 2240)),
    ((2, 3, 4, 2240), (1, 3, 1, 2240)),
    ((2, 3, 4, 2, 2240), (2, 1, 4, 1, 2240)),
])
def test_rounding_with_strided_modulation_across_supported_ranks(shape, modulation_shape):
    torch.manual_seed(205)
    x = torch.randn(shape, dtype=torch.float16, device="cuda")
    update = torch.randn_like(x)
    storage_shape = (*modulation_shape[:-1], modulation_shape[-1] * 2)
    scale, shift, gate = (torch.randn(storage_shape, dtype=torch.bfloat16, device="cuda")[..., ::2] for _ in range(3))
    originals = [value.clone() for value in (x, update, scale, shift, gate)]
    receipt = {}
    with torch.inference_mode(), kernel_dispatch_receipt_scope(receipt):
        actual_gate = residual_gate_add(x, update, gate, backend="triton")
        actual_affine = scale_shift(x, scale, shift, backend="triton")
    torch.testing.assert_close(actual_gate, x + update * gate, rtol=0, atol=0)
    torch.testing.assert_close(actual_affine, x * (1 + scale) + shift, rtol=0, atol=0)
    for actual, original in zip((x, update, scale, shift, gate), originals):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert all(item["accelerated"] and item["backend"] == "triton" for item in receipt["dispatches"])


def test_explicit_backend_keeps_autograd_fallback_and_process_policy(monkeypatch):
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "torch")
    x = torch.randn(2, 6, requires_grad=True)
    update = torch.randn_like(x, requires_grad=True)
    gate = torch.randn(1, 6, requires_grad=True)
    expected = x + update * gate
    expected_grads = torch.autograd.grad(expected.sum(), (x, update, gate))
    receipt = {}
    with kernel_dispatch_receipt_scope(receipt):
        actual = residual_gate_add(x, update, gate, backend="triton")
    actual_grads = torch.autograd.grad(actual.sum(), (x, update, gate))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert receipt["dispatches"][-1]["backend"] == "torch"
    assert receipt["dispatches"][-1]["fallback"]
    assert os.environ["WORLDFOUNDRY_KERNEL_BACKEND"] == "torch"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("upcast", [False, True])
@pytest.mark.parametrize("features", [2240, 5120])
def test_layer_norm_modulation_preserves_vendor_bits_and_upcast(dtype, mixed, upcast, features):
    torch.manual_seed(102)
    x = torch.randn(2, 6, features, dtype=dtype, device="cuda")
    modulation_dtype = torch.float32 if mixed else dtype
    scale, shift = (torch.randn(2, 1, features, dtype=modulation_dtype, device="cuda") for _ in range(2))
    normalized = torch.nn.functional.layer_norm(x.float() if upcast else x, (features,), eps=1e-6)
    if upcast:
        normalized = normalized.to(x.dtype)
    expected = normalized * (1 + scale) + shift
    receipt = {}
    with torch.inference_mode(), kernel_dispatch_receipt_scope(receipt):
        actual = layer_norm_scale_shift(x, scale, shift, upcast=upcast, backend="triton")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(receipt["dispatches"]) == 1
    assert receipt["dispatches"][0]["implementation"] == "triton_layer_norm_scale_shift"
    assert receipt["dispatches"][0]["accelerated"]
