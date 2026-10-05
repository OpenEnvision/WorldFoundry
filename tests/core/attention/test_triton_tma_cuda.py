"""Actual H100 dense TMA numerical, alignment, worker and graph contracts."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest
import torch
from torch.nn import functional as F

from worldfoundry.core.attention.backends import dispatch
from worldfoundry.core.attention.backends.native import NativeAttention
from worldfoundry.core.execution.graphs.inference_graph import InferenceCUDAGraphRunner

triton = pytest.importorskip("triton")
tma = pytest.importorskip("worldfoundry.core.attention.backends.triton_tma")
is_triton_tma_supported, triton_tma_sdpa = tma.is_triton_tma_supported, tma.triton_tma_sdpa

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for TMA")]


@pytest.fixture
def cuda_device():
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.version.hip or torch.cuda.get_device_capability(device)[0] < 9:
        pytest.skip("NVIDIA Hopper or newer required for TMA")
    return device


def _inputs(device, query_length=37, key_length=53, head_dim=64, dtype=torch.bfloat16, seed=43):
    generator = torch.Generator(device=device).manual_seed(seed)
    shapes = ((1, query_length, 2, head_dim), (1, key_length, 2, head_dim), (1, key_length, 2, head_dim))
    return tuple(torch.randn(shape, device=device, dtype=dtype, generator=generator) for shape in shapes)


def _assert_dense_budget(actual, query, key, value, scale=None):
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(
            query.transpose(1, 2).float(),
            key.transpose(1, 2).float(),
            value.transpose(1, 2).float(),
            scale=scale,
        ).transpose(1, 2)
    assert actual.shape == query.shape
    assert actual.dtype == query.dtype
    assert torch.isfinite(actual).all()
    # Probability and output storage round once in the requested dense dtype;
    # dense BF16 is held below 0.6% L2, FP16 below 0.1%, with pointwise budgets.
    relative_budget, atol, rtol = (0.006, 0.006, 0.012) if query.dtype == torch.bfloat16 else (0.001, 0.0008, 0.002)
    torch.testing.assert_close(actual.float(), expected, rtol=rtol, atol=atol)
    assert ((actual.float() - expected).norm() / expected.norm()).item() < relative_budget


@pytest.mark.parametrize("shape", [(37, 53, 64), (129, 128, 128), (17, 67, 256), (1, 1, 16)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_dense_tma_matches_independent_fp32_sdpa_for_partial_tiles_and_head_widths(cuda_device, shape, dtype):
    for seed in (43, 57):
        query, key, value = _inputs(cuda_device, *shape, dtype, seed)
        assert is_triton_tma_supported(query, key, value)
        for scale in (None, 0.2, -0.12, 0.0):
            _assert_dense_budget(triton_tma_sdpa(query, key, value, scale=scale), query, key, value, scale)


@pytest.mark.parametrize("index", [0, 1, 2])
@torch.inference_mode()
def test_dense_tma_rejects_each_misaligned_input_pointer(cuda_device, index):
    tensors = list(_inputs(cuda_device))
    original = tensors[index]
    storage = torch.empty(original.numel() + 1, device=cuda_device, dtype=original.dtype)
    tensors[index] = storage[1:].view_as(original)
    assert not is_triton_tma_supported(*tensors)
    with pytest.raises(RuntimeError, match="base pointers and strides"):
        triton_tma_sdpa(*tensors)


@torch.inference_mode()
def test_dense_tma_preserves_padded_token_and_head_strides(cuda_device):
    query, key, value = _inputs(cuda_device, 74, 106, 64)
    query, key, value = (tensor[:, ::2] for tensor in (query, key, value))
    assert not query.is_contiguous()
    assert is_triton_tma_supported(query, key, value)
    _assert_dense_budget(triton_tma_sdpa(query, key, value), query, key, value)


@torch.inference_mode()
def test_dense_tma_replays_new_inputs_and_preserves_prior_output(cuda_device):
    query, key, value = _inputs(cuda_device)
    runner = InferenceCUDAGraphRunner(triton_tma_sdpa, warmup=1)
    first = runner(query, key, value)
    second_inputs = tuple(tensor.mul(-0.7) for tensor in (query, key, value))
    second = runner(*second_inputs)
    _assert_dense_budget(first, query, key, value)
    _assert_dense_budget(second, *second_inputs)
    assert first.data_ptr() != second.data_ptr()
    assert not torch.equal(first, second)
    report = runner.report()
    assert report["graphs"] == report["capture"] == 1
    assert report["replay"] == 2
    assert report["eager"] == report["capture_failed"] == 0


@torch.inference_mode()
def test_dense_tma_workers_and_streams_preserve_the_callers_descriptor_allocator(cuda_device):
    query, key, value = _inputs(cuda_device)
    triton_tma_sdpa(query, key, value)  # autotune before concurrent launches
    producer = torch.cuda.current_stream(cuda_device)

    def run(scale):
        torch.cuda.set_device(cuda_device)
        stream = torch.cuda.Stream(device=cuda_device)
        stream.wait_stream(producer)

        def preserve_allocator():
            def sentinel_allocator(*args):
                raise AssertionError("TMA kernel used the caller allocator")

            triton.set_allocator(sentinel_allocator)
            with torch.cuda.stream(stream), torch.inference_mode():
                actual = triton_tma_sdpa(query, key, value, scale=scale)
            assert any(value is sentinel_allocator for _, value in copy_context().items())
            stream.synchronize()
            return actual

        return copy_context().run(preserve_allocator)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run, scale) for scale in (0.15, 0.2)]
        first, second = [future.result() for future in futures]
    assert first.data_ptr() != second.data_ptr()
    _assert_dense_budget(first, query, key, value, 0.15)
    _assert_dense_budget(second, query, key, value, 0.2)


def test_dense_tma_rejects_raw_fp8_autograd_outer_stride_and_invalid_scales(cuda_device):
    query, key, value = _inputs(cuda_device)
    assert not is_triton_tma_supported(query.to(torch.float8_e4m3fn), key, value)
    with pytest.raises(RuntimeError, match="inference-only"):
        triton_tma_sdpa(query.requires_grad_(), key, value)
    query = query.detach()
    storage = torch.empty(5000, device=cuda_device, dtype=query.dtype)
    bad_stride_query = storage.as_strided(query.shape, (5000, 129, 64, 1))
    assert not is_triton_tma_supported(bad_stride_query, key, value)
    with pytest.raises(RuntimeError, match="strides"):
        triton_tma_sdpa(bad_stride_query, key, value)
    for scale in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            triton_tma_sdpa(query, key, value, scale=scale)


@torch.inference_mode()
def test_dense_tma_dispatch_receipt_and_native_attention_use_the_actual_provider(cuda_device):
    query, key, value = _inputs(cuda_device)
    dispatch.reset_attention_provider_runtime()
    actual = dispatch.attention_forward(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        backend="triton_tma",
    ).transpose(1, 2)
    _assert_dense_budget(actual, query, key, value)
    report = dispatch.attention_provider_runtime_report()
    assert report["triton_tma"]["attempts"] == report["triton_tma"]["successes"] == 1
    assert report["triton_tma"]["errors"] == report["triton_tma"]["fallbacks"] == 0
    assert report.get("torch", {}).get("attempts", 0) == 0
    _assert_dense_budget(NativeAttention(qkv_format="bshd", backend="triton_tma")(query, key, value), query, key, value)
