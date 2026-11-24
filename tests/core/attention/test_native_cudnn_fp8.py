"""CPU contracts for the explicit cuDNN FP8 provider and unchanged dense defaults."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch

from worldfoundry.core.attention.backends import dispatch
from worldfoundry.core.attention.backends.native import NativeAttention
from worldfoundry.core.attention.backends.native_fp8 import _Plan, native_cudnn_fp8_sdpa
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
        unsupported: list[dict[str, Any]] = [
            {"attn_mask": torch.ones(16, 16, dtype=torch.bool)},
            {"compatibility_mode": True},
        ]
        for options in unsupported:
            with pytest.raises(ValueError, match="does not support"):
                dispatch.attention_forward(value, value, value, backend="cudnn_fp8", **options)


def test_direct_fp8_provider_rejects_cpu_and_native_dense_backend_stays_correct() -> None:
    generator = torch.Generator().manual_seed(2)
    query, key, value = [torch.randn((1, 2, 16, 64), generator=generator) for _ in range(3)]
    with pytest.raises(ValueError, match="CUDA"):
        native_cudnn_fp8_sdpa(query, key, value)
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    torch.testing.assert_close(NativeAttention(backend="math")(query, key, value), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("edge_data", [False, True], ids=["cuda12", "cuda13"])
def test_fp8_graph_capture_matches_cuda_binding_signatures(edge_data: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    stream, parent_graph, child_graph, child_node = 7, 11, 17, 19
    dependencies = [23]
    edge_fields = (None,) if edge_data else ()
    runtime = SimpleNamespace(
        cudaError_t=SimpleNamespace(cudaSuccess=0),
        cudaStreamCaptureStatus=SimpleNamespace(cudaStreamCaptureStatusActive=1),
        cudaStreamUpdateCaptureDependenciesFlags=SimpleNamespace(cudaStreamSetCaptureDependencies=2),
        cudaGraphCreate=Mock(return_value=(0, child_graph)),
        cudaStreamGetCaptureInfo=Mock(return_value=(0, 1, 29, parent_graph, dependencies, *edge_fields, 1)),
        cudaGraphAddChildGraphNode=Mock(return_value=(0, child_node)),
        cudaStreamUpdateCaptureDependencies=Mock(return_value=(0,)),
        cudaGraphDestroy=Mock(return_value=(0,)),
    )
    # Only the binding boundary is stubbed; plan tensor preparation runs on CPU.
    plan = _Plan.__new__(_Plan)
    monkeypatch.setattr(plan, "runtime", runtime, raising=False)
    plan.device = torch.device("cpu")
    plan.output_shape = (1, 1, 2, 64)
    plan.output_stride = (128, 128, 64, 1)
    descriptors = [Mock(get_uid=Mock(return_value=index)) for index in range(12)]
    monkeypatch.setattr(plan, "descriptors", descriptors[:3], raising=False)
    monkeypatch.setattr(plan, "scales", descriptors[3:9], raising=False)
    monkeypatch.setattr(plan, "outputs", tuple(descriptors[9:]), raising=False)
    plan.graph = Mock(get_workspace_size=Mock(return_value=0))
    plan.cudnn = Mock()
    plan.handle = 31
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=stream))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    query = torch.ones(plan.output_shape)
    plan(query, query, query, 1.0, 1.0, 1.0, 1.0)

    runtime.cudaStreamGetCaptureInfo.assert_called_once_with(stream)
    runtime.cudaGraphAddChildGraphNode.assert_called_once_with(parent_graph, dependencies, 1, child_graph)
    runtime.cudaStreamUpdateCaptureDependencies.assert_called_once_with(stream, [child_node], *edge_fields, 1, 2)
    runtime.cudaGraphDestroy.assert_called_once_with(child_graph)


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
