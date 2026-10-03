"""Functional fusion preserves eager intermediate dtype rounding and policies."""

import pytest
import torch

from worldfoundry.core.kernels import residual_gate_add, scale_shift
from worldfoundry.core.kernels.registry import kernel_dispatch_receipt_scope


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mixed", [False, True])
def test_eager_rounding_and_explicit_model_backend(monkeypatch, dtype, mixed):
    # Explicit model configuration must take precedence over process defaults.
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "torch")
    torch.manual_seed(102)
    residual = torch.randn(2, 6, 2240, dtype=dtype, device="cuda")
    update = torch.randn_like(residual)
    modulation_dtype = torch.float32 if mixed else dtype
    scale, shift = (torch.randn(2, 1, 2240, dtype=modulation_dtype, device="cuda") for _ in range(2))
    gate = torch.randn_like(scale)
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
