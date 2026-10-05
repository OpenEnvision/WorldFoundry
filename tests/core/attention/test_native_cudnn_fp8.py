"""CPU contracts for the explicit cuDNN FP8 provider and unchanged dense defaults."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from worldfoundry.core.attention.backends import dispatch
from worldfoundry.core.attention.backends.native import NativeAttention
from worldfoundry.core.attention.backends.native_fp8 import native_cudnn_fp8_sdpa
from worldfoundry.core.attention.backends.probe import normalize_attention_backend, resolve_attention_backend


def test_fp8_backend_is_explicit_and_unavailable_cpu_requests_do_not_resolve_to_torch() -> None:
    assert normalize_attention_backend("cudnn_fp8") == "cudnn_fp8"
    assert resolve_attention_backend("auto", "cpu") == "torch"
    with pytest.raises(RuntimeError, match="cudnn_fp8.*unavailable"):
        resolve_attention_backend("cudnn_fp8", "cpu")


def test_fp8_dispatch_rejects_unsupported_requests_without_torch_fallback() -> None:
    value = torch.ones((1, 2, 16, 64))
    with patch.object(dispatch, "_invoke_torch_sdpa_audited", side_effect=AssertionError("dense fallback")):
        with pytest.raises(ValueError, match="CUDA"):
            dispatch.attention_forward(value, value, value, backend="cudnn_fp8")
        for options in ({"attn_mask": torch.ones(16, 16, dtype=torch.bool)}, {"compatibility_mode": True}):
            with pytest.raises(ValueError, match="does not support"):
                dispatch.attention_forward(value, value, value, backend="cudnn_fp8", **options)


def test_direct_fp8_provider_rejects_cpu_and_native_dense_backend_stays_correct() -> None:
    generator = torch.Generator().manual_seed(2)
    query, key, value = [torch.randn((1, 2, 16, 64), generator=generator) for _ in range(3)]
    with pytest.raises(ValueError, match="CUDA"):
        native_cudnn_fp8_sdpa(query, key, value)
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    torch.testing.assert_close(NativeAttention(backend="math")(query, key, value), expected, rtol=1e-5, atol=1e-6)


def test_fp8_policy_receipt_marks_numerical_approximation(monkeypatch):
    from worldfoundry.core.attention.backends import probe
    from worldfoundry.core.model_loading.optimize import AppliedOptimizations, apply_attention_policy

    class Seam(torch.nn.Module):
        def set_attention_backend(self, backend):
            self.backend = backend

    monkeypatch.setattr(probe, "resolve_attention_backend", lambda *args: "cudnn_fp8")
    model = Seam()
    report = apply_attention_policy(model, "cudnn_fp8", device="cpu")
    assert report.approximate and model.backend == "cudnn_fp8"
    applied = AppliedOptimizations()
    applied.record_attention(report)
    assert applied.quality_tier == "numerically-approximate"
