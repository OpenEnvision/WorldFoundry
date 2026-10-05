"""Explicit attention schedule admission and actual dense/FP8 pointer kernels."""

import pytest
import torch
from torch.nn import functional as F

from worldfoundry.core.attention.schedule import MHASchedule, scheduled_sdpa


@pytest.mark.parametrize(
    "options",
    [
        {"fusion": "typo"},
        {"sdpa_backend": "auto"},
        {"projection_precision": "int4"},
        {"quantized_sdpa": True},
        {"use_tma": True},
        {"quantized_sdpa": 1},
    ],
)
def test_schedule_rejects_ambiguous_or_unsupported_options(options):
    with pytest.raises((TypeError, ValueError)):
        MHASchedule(**options)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("quantized", [False, True])
def test_pointer_fa2_executes_unequal_sequence_lengths(quantized):
    torch.manual_seed(32)
    q = torch.randn(2, 31, 128, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(2, 77, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    schedule = MHASchedule(sdpa_backend="fa2", quantized_sdpa=quantized)
    with torch.no_grad():
        output = scheduled_sdpa(q, k, v, num_heads=2, schedule=schedule)
        operands = [value.reshape(2, -1, 2, 64).transpose(1, 2) for value in (q, k, v)]
        if quantized:
            operands = [value.to(torch.float8_e4m3fn).float() for value in operands]
        else:
            operands = [value.float() for value in operands]
        expected = F.scaled_dot_product_attention(*operands).transpose(1, 2).reshape_as(output)
    assert output.dtype == torch.bfloat16 and bool(torch.isfinite(output).all())
    relative = (output.float() - expected).norm() / expected.norm()
    # FP8 FA2 additionally quantizes the unnormalized probabilities in P@V.
    assert relative < (0.06 if quantized else 0.006)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("quantized", [False, True])
def test_pointer_fa2_wan_head_width_and_fused_projection_strides(quantized):
    torch.manual_seed(42)
    q, k = [torch.randn(1, 3600, 1536, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    v = torch.randn(1, 3600, 4608, device="cuda", dtype=torch.bfloat16)[..., 3072:]
    with torch.no_grad():
        output = scheduled_sdpa(
            q, k, v, num_heads=12, schedule=MHASchedule(sdpa_backend="fa2", quantized_sdpa=quantized)
        )
        reference = scheduled_sdpa(q, k, v, num_heads=12, schedule=MHASchedule())
    assert (output.float() - reference.float()).norm() / reference.float().norm() < (0.08 if quantized else 0.006)


@pytest.mark.parametrize("failure", ["heads", "width", "dtype", "empty"])
def test_schedule_rejects_inconsistent_projection_geometry(failure):
    q, k, v = [torch.ones(1, 3, 64) for _ in range(3)]
    heads = 1
    if failure == "heads":
        heads = 0
    elif failure == "width":
        k = k[..., :32]
    elif failure == "dtype":
        v = v.double()
    else:
        k, v = k[:, :0], v[:, :0]
    with torch.no_grad(), pytest.raises(ValueError):
        scheduled_sdpa(q, k, v, num_heads=heads, schedule=MHASchedule())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("provider", ["tma", "cudnn", "cudnn_fp8"])
def test_optional_schedule_provider_matches_torch(provider):
    if provider == "tma":
        import triton

        from worldfoundry.core.attention.backends.probe import _triton_tma_version_supported

        if not _triton_tma_version_supported(triton.__version__):
            pytest.skip("TMA requires Triton >= 3.5")
    if provider == "cudnn_fp8":
        pytest.importorskip("cudnn")
        pytest.importorskip("cuda.bindings.runtime")
    torch.manual_seed(41)
    q = torch.randn(1, 32, 128, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    schedule = MHASchedule(
        sdpa_backend="fa2" if provider == "tma" else "cudnn",
        use_tma=provider == "tma",
        quantized_sdpa=provider == "cudnn_fp8",
    )
    with torch.no_grad():
        actual = scheduled_sdpa(q, k, v, num_heads=2, schedule=schedule)
        expected = scheduled_sdpa(q, k, v, num_heads=2, schedule=MHASchedule())
    assert (actual.float() - expected.float()).norm() / expected.float().norm() < (
        0.09 if provider == "cudnn_fp8" else 0.01
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("precision", ["fp8_e4m3", "fp8_e5m2", "int8"])
def test_scheduled_projection_uses_real_low_precision_kernel(precision, monkeypatch):
    from worldfoundry.core.acceleration.quantization import linear
    from worldfoundry.core.attention.schedule import ScheduledProjection

    # Small test operands need an explicit work-gate override; this is numerical
    # and execution coverage, never profitability evidence for small matrices.
    monkeypatch.setattr(linear, "_fp8_min_gemm_work", lambda: 0.0)
    source = torch.nn.Linear(128, 128, device="cuda", dtype=torch.bfloat16).eval()
    x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
    projection = ScheduledProjection((source, source), precision)
    with torch.no_grad():
        output = projection(x)[0]
        expected = source(x)
    assert (output.float() - expected.float()).norm() / expected.float().norm() < 0.1
    report = projection.report()["quantization"]
    assert report["low_precision_kernel_calls"] == 1
    assert report["dense_compute_calls"] == report["dense_fallback_calls"] == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bias", [True, False])
def test_e5m2_masked_gemm_matches_independent_quantized_reference(bias):
    from worldfoundry.core.acceleration.quantization.fp8_linear import TritonFloat8Linear
    from worldfoundry.core.acceleration.quantization.linear import _quantize_rowwise_fp8

    source = torch.nn.Linear(80, 48, bias=bias, device="cuda", dtype=torch.bfloat16).eval()
    provider = TritonFloat8Linear(
        source.weight,
        source.bias,
        fp8_dtype=torch.float8_e5m2,
        scaling="rowwise",
        use_fast_accum=False,
        keep_dense_fallback=False,
    )
    x = torch.randn(2, 17, 80, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        output = provider(x)
        qx, scale = _quantize_rowwise_fp8(x.flatten(0, 1), torch.float8_e5m2)
        reference = (qx.float() @ provider.weight_fp8.float().T) * scale * provider.weight_scale
        if bias:
            reference += source.bias.float()
    torch.testing.assert_close(output, reference.to(output.dtype).reshape_as(output), rtol=0, atol=0)
