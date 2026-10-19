"""Actual scaled INT8 QK / FP8 PV kernels and finalized-key ownership."""

import pytest
import torch
from torch.nn import functional as F

from worldfoundry.core.attention.schedule import MHASchedule, scheduled_sdpa
from worldfoundry.core.attention.sequence.key_centering import streaming_key_mean


def test_scaled_schedule_requires_explicit_approximation():
    with pytest.raises(ValueError, match="quantized_sdpa=True"):
        MHASchedule(sdpa_backend="scaled_int8_fp8")
    assert MHASchedule(sdpa_backend="scaled_int8_fp8", quantized_sdpa=True).approximate


def test_finalized_key_sum_is_reused_and_rebuilt_after_window_roll():
    keys = torch.arange(1 * 8 * 2 * 4, dtype=torch.float32).reshape(1, 8, 2, 4)
    cache = {"k": keys.clone()}
    first = streaming_key_mean(cache, current_start=4, local_start=4, local_end=6, maximum=4)
    torch.testing.assert_close(first, cache["k"][:, 2:6].mean(dim=1, keepdim=True))
    history = cache["key_history_sum"]
    cache["k"][:, 4:6] += 100
    second = streaming_key_mean(cache, current_start=4, local_start=4, local_end=6, maximum=4)
    assert cache["key_history_sum"] is history
    torch.testing.assert_close(second, cache["k"][:, 2:6].mean(dim=1, keepdim=True))
    cache["k"][:, 2:6] = cache["k"][:, 4:8].clone()
    rolled = streaming_key_mean(cache, current_start=6, local_start=4, local_end=6, maximum=4)
    assert cache["key_history_sum"] is not history
    torch.testing.assert_close(rolled, cache["k"][:, 2:6].mean(dim=1, keepdim=True))
    independent = {"k": torch.zeros_like(keys)}
    assert (
        torch.count_nonzero(streaming_key_mean(independent, current_start=4, local_start=4, local_end=6, maximum=4))
        == 0
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 64, 31, 77), (1, 12, 128, 129, 211), (1, 2, 256, 31, 77)])
def test_scaled_kernel_matches_native_with_unequal_lengths_and_channel_scales(dtype, shape):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("SM90 required")
    from worldfoundry.core.attention.backends.scaled_int8_fp8 import scaled_int8_fp8_attention

    batch, heads, dim, length, context = shape
    torch.manual_seed(42)
    q = torch.randn(batch, length, heads, dim, dtype=dtype, device="cuda")
    k = torch.randn(batch, context, heads, dim, dtype=dtype, device="cuda") + 8
    v = torch.randn(batch, context, heads, dim, dtype=dtype, device="cuda")
    v *= torch.logspace(-2, 2, dim, device="cuda", dtype=dtype)
    with torch.no_grad():
        actual = scaled_int8_fp8_attention(q, k, v)
        expected = F.scaled_dot_product_attention(*[tensor.transpose(1, 2).float() for tensor in (q, k, v)]).transpose(
            1, 2
        )
        supplied_mean = scaled_int8_fp8_attention(q, k, v, key_mean=k.mean(dim=1, keepdim=True))
    assert actual.dtype == dtype and bool(actual.isfinite().all())
    assert (actual.float() - expected).norm() / expected.norm() < 0.06
    torch.testing.assert_close(actual, supplied_mean, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_schedule_and_generic_dispatch_execute_the_scaled_provider():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("SM90 required")
    from worldfoundry.core.attention.backends.dispatch import (
        attention_forward,
        attention_provider_runtime_report,
        reset_attention_provider_runtime,
    )

    q, k, v = [torch.randn(1, length, 128, device="cuda", dtype=torch.bfloat16) for length in (31, 77, 77)]
    reset_attention_provider_runtime()
    with torch.no_grad():
        schedule = MHASchedule(sdpa_backend="scaled_int8_fp8", quantized_sdpa=True)
        expected = scheduled_sdpa(q, k, v, num_heads=2, schedule=schedule)
        actual = attention_forward(
            q.reshape(1, 31, 2, 64),
            k.reshape(1, 77, 2, 64),
            v.reshape(1, 77, 2, 64),
            q_pattern="b s n d",
            k_pattern="b s n d",
            v_pattern="b s n d",
            out_pattern="b s n d",
            backend="scaled_int8_fp8",
            backend_options={"key_mean": k.reshape(1, 77, 2, 64).mean(dim=1, keepdim=True)},
        )
        with pytest.raises(ValueError, match="masks"):
            attention_forward(q, k, v, backend="scaled_int8_fp8", attn_mask=torch.ones(31, 77, device="cuda"))
    torch.testing.assert_close(actual.flatten(2), expected, rtol=0, atol=0)
    report = attention_provider_runtime_report()["scaled_int8_fp8"]
    assert report["successes"] == 1 and report["fallbacks"] == 0
