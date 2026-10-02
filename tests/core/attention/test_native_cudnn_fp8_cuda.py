"""Actual H100 cuDNN FP8 numerical, stream, output-lifetime and graph contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch.nn import functional as F

from worldfoundry.core.attention.backends import dispatch
from worldfoundry.core.attention.backends.native import NativeAttention
from worldfoundry.core.attention.backends.native_fp8 import native_cudnn_fp8_sdpa
from worldfoundry.core.execution.graphs.inference_graph import InferenceCUDAGraphRunner

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for cuDNN FP8")]


@pytest.fixture
def cuda_device() -> torch.device:
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device)[0] < 9:
        pytest.skip("cuDNN FP8 requires Hopper or newer")
    pytest.importorskip("cudnn", reason="nvidia-cudnn-frontend required")
    pytest.importorskip("cuda.bindings.runtime", reason="cuda-bindings required")
    return device


def _inputs(device: torch.device, query_length: int, key_length: int, head_dim: int, seed: int = 43):
    generator = torch.Generator(device=device).manual_seed(seed)
    shapes = ((1, 2, query_length, head_dim), (1, 2, key_length, head_dim), (1, 2, key_length, head_dim))
    return tuple(
        (torch.randn(shape, device=device, generator=generator) * 0.5).to(torch.float8_e4m3fn) for shape in shapes
    )


def _assert_numerical_budget(
    actual: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale=None,
    descales=(1.0, 1.0, 1.0),
):
    # Independent dense FP32 math, reconstructed from the *quantized* inputs.
    # Budget includes two E4M3 rounding stages (S and O), plus BF16 restoration.
    # Derive an elementwise bound from 6.25% normal E4M3 rounding and its minimum
    # subnormal half-ULP. S error propagates as P @ abs(V), including cancellation.
    # A separate 5% L2 ceiling prevents a conservative local bound masking errors.
    query, key, value = (tensor.float() * factor for tensor, factor in zip((query, key, value), descales))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(query, key, value, scale=scale)
        weighted_absolute_value = F.scaled_dot_product_attention(query, key, value.abs(), scale=scale)
    actual = actual.float()
    assert torch.isfinite(actual).all()
    softmax_error_bound = 0.0625 * weighted_absolute_value + (2**-10 / 128) * value.abs().sum(-2, keepdim=True)
    output_error_bound = 0.0625 * (expected.abs() + softmax_error_bound) + (2**-10 * descales[2])
    bf16_error_bound = (2**-8) * (expected.abs() + softmax_error_bound + output_error_bound)
    absolute_error = (actual - expected).abs()
    assert (absolute_error <= softmax_error_bound + output_error_bound + bf16_error_bound).all(), (
        "cuDNN FP8 elementwise error exceeds the S/O representation rounding bound"
    )
    relative_error = (actual - expected).norm() / expected.norm()
    assert relative_error.item() < 0.05
    assert F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item() > 0.9987


@pytest.mark.parametrize("query_length,key_length,head_dim", [(32, 64, 64), (48, 80, 128), (16, 1024, 64)])
@torch.inference_mode()
def test_native_cudnn_fp8_matches_float_reference_across_shapes(cuda_device, query_length, key_length, head_dim):
    for seed in (43, 57, 113):
        query, key, value = _inputs(cuda_device, query_length, key_length, head_dim, seed)
        for amplitudes in ((1.0, 1.0, 1.0), (2.0, 3.0, 7.0)):
            # Explicit per-tensor quantization, never an unscaled arbitrary cast.
            originals = [tensor.float() * amplitude for tensor, amplitude in zip((query, key, value), amplitudes)]
            descales = tuple(tensor.abs().amax().item() / 448 for tensor in originals)
            query_scaled, key_scaled, value_scaled = (
                tensor.div(factor).to(query.dtype) for tensor, factor in zip(originals, descales)
            )
            for scale in (None, 0.2):
                actual = native_cudnn_fp8_sdpa(
                    query_scaled,
                    key_scaled,
                    value_scaled,
                    scale=scale,
                    descale_q=descales[0],
                    descale_k=descales[1],
                    descale_v=descales[2],
                )
                assert actual.dtype == torch.bfloat16
                assert actual.shape == query.shape
                _assert_numerical_budget(actual, query_scaled, key_scaled, value_scaled, scale, descales)


@torch.inference_mode()
def test_fp8_dispatch_uses_actual_provider_and_preserves_token_major_layout(cuda_device):
    query, key, value = _inputs(cuda_device, 32, 64, 64)
    dispatch.reset_attention_provider_runtime()
    actual = dispatch.attention_forward(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        q_pattern="b s n d",
        k_pattern="b s n d",
        v_pattern="b s n d",
        out_pattern="b s n d",
        backend="cudnn_fp8",
    )
    _assert_numerical_budget(actual.transpose(1, 2), query, key, value)
    report = dispatch.attention_provider_runtime_report()
    assert report["cudnn_fp8"]["attempts"] == report["cudnn_fp8"]["successes"] == 1
    assert report["cudnn_fp8"]["errors"] == report["cudnn_fp8"]["fallbacks"] == 0
    assert report.get("torch", {}).get("attempts", 0) == 0


@torch.inference_mode()
def test_fp8_graph_replay_observes_new_inputs_and_retained_outputs_stay_independent(cuda_device):
    query, key, value = _inputs(cuda_device, 32, 64, 64)
    runner = InferenceCUDAGraphRunner(native_cudnn_fp8_sdpa, warmup=1)
    first = runner(query, key, value)
    second_query, second_key, second_value = (
        tensor.float().mul(-0.7).to(tensor.dtype) for tensor in (query, key, value)
    )
    second = runner(second_query, second_key, second_value)
    _assert_numerical_budget(first, query, key, value)
    _assert_numerical_budget(second, second_query, second_key, second_value)
    assert first.data_ptr() != second.data_ptr()
    report = runner.report()
    assert report["graphs"] == report["capture"] == 1
    assert report["replay"] == 2
    assert report["capture_failed"] == report["eager"] == 0


@torch.inference_mode()
def test_fp8_provider_runs_on_worker_thread_and_two_streams_without_output_aliasing(cuda_device):
    query, key, value = _inputs(cuda_device, 32, 64, 64)
    producer = torch.cuda.current_stream(cuda_device)

    def execute_on_stream(scale):
        torch.cuda.set_device(cuda_device)
        stream = torch.cuda.Stream(device=cuda_device)
        stream.wait_stream(producer)
        with torch.cuda.stream(stream), torch.inference_mode():
            actual = native_cudnn_fp8_sdpa(query, key, value, scale=scale)
        stream.synchronize()
        return actual

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(execute_on_stream, 0.15)
        second_future = executor.submit(execute_on_stream, 0.2)
        first, second = first_future.result(), second_future.result()
    assert first.data_ptr() != second.data_ptr()
    _assert_numerical_budget(first, query, key, value, 0.15)
    _assert_numerical_budget(second, query, key, value, 0.2)
    native = NativeAttention(backend="cudnn_fp8")
    _assert_numerical_budget(native(query, key, value), query, key, value)


@torch.inference_mode()
def test_fp8_provider_rejects_invalid_dtype_gqa_alignment_and_scale(cuda_device):
    query, key, value = _inputs(cuda_device, 32, 64, 64)
    with pytest.raises(ValueError, match="raw torch.float8"):
        native_cudnn_fp8_sdpa(query.bfloat16(), key, value)
    with pytest.raises(ValueError, match="GQA"):
        native_cudnn_fp8_sdpa(query, key[:, :1], value[:, :1])
    storage = torch.empty(query.numel() + 1, device=cuda_device, dtype=query.dtype)
    with pytest.raises(ValueError, match="aligned"):
        native_cudnn_fp8_sdpa(storage[1:].view_as(query), key, value)
    for scale in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive and finite"):
            native_cudnn_fp8_sdpa(query, key, value, scale=scale)
        for argument in ("descale_q", "descale_k", "descale_v", "scale_o"):
            with pytest.raises(ValueError, match="positive and finite"):
                native_cudnn_fp8_sdpa(query, key, value, **{argument: scale})


@torch.inference_mode()
def test_fp8_cold_graph_capture_is_rejected_without_building_a_plan(cuda_device):
    query, key, value = _inputs(cuda_device, 33, 65, 64)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="warm up cudnn_fp8"):
        with torch.cuda.graph(graph):
            native_cudnn_fp8_sdpa(query, key, value, scale=0.193)


@torch.inference_mode()
def test_fp8_partial_compile_receipts_prove_provider_execution_without_claiming_graph_traces(cuda_device):
    query, key, value = _inputs(cuda_device, 32, 64, 64)
    dispatch.reset_attention_provider_runtime()

    def attention(query, key, value):
        return dispatch.attention_forward(query, key, value, backend="cudnn_fp8")

    compiled = torch.compile(attention, backend="eager", fullgraph=False)
    actual = compiled(query, key, value)
    _assert_numerical_budget(actual, query, key, value)
    report = dispatch.attention_provider_runtime_report()
    assert report["cudnn_fp8"]["attempts"] == report["cudnn_fp8"]["successes"] == 1
    assert report["cudnn_fp8"]["compiled_graph_traces"] == 0
    assert report.get("torch", {}).get("attempts", 0) == 0
